import argparse
import csv
import math
import os
import time

# One CPU thread per MPI rank. Otherwise every rank's JAX/BLAS starts a thread
# pool the size of the whole node, N ranks oversubscribe the cores, and even
# trivial XLA compiles take minutes ("Very slow compile?" warnings). Must be
# set before jax/numpy are imported.
for _var in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(_var, "1")
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")

import numpy as np
import jax.numpy as jnp
from mpi4py import MPI
from scipy.optimize import minimize

import esr.fitting.match as match
from esr.fitting.dm_likelihood import MIGHTEELikelihood
from combine_all_galaxies import galaxy_out_dir, max_param_total, resolve_fn_dir
from screen_global_local import load_allfcn_and_aifeyn


# Step 2 of the global/local scheme: for each (function, global-subset)
# candidate from screen_global_local.py, re-fit with the chosen SHAPE
# parameters shared by every galaxy, then price the result with the same
# description length rules as the all-local pipeline.
#
# Nested optimisation (as in the CLASH paper): an outer Nelder-Mead over the
# global parameters; at every outer step each galaxy's local parameters
# (remaining shape params, rho0, rs, Inc, D) are re-optimised with the
# globals held fixed. The full per-galaxy search (sign enumeration, DIRECT,
# multi-start) is NOT repeated in the inner loop -- each galaxy is
# warm-started from its own all-local best fit (and, after the first outer
# step, from its previous inner solution). If sharing a parameter helps at
# all, the shared value has to sit near most galaxies' own best values, so
# local polishing from there is enough.
#
# Galaxies are split across MPI ranks. Every rank runs the identical outer
# Nelder-Mead (deterministic), and the summed loss is an allreduce, so the
# ranks stay in lockstep. Works single-process too.
#
# Code length:
#   locals  -- match.compute_function_codelen per galaxy, with the global
#              entries left out of active_idx (held fixed, not priced), so
#              snapping and the integral fallback behave exactly as in
#              match.py; includes Inc/D as in the all-local run.
#   globals -- Fisher = sum over galaxies of each galaxy's d2(-logL)/dg2,
#              Delta* = sqrt(12/F*), cost log(2) + log(|g|/Delta*), floored
#              at log(2) (|g| = Delta*) like an unresolved parameter in
#              match.codelen_from_vector. If F* is unusable, Delta* comes
#              from match.get_sigma_from_integral on the summed likelihood.
#   DL_mixed = sum(-logL) + sum(local codelen) + global codelen + aifeyn.

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

BAD = 1e30


def log(msg):
    if rank == 0:
        print(msg, flush=True)


def load_candidates(path, start, end):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f, delimiter=";"))
    out = []
    for r in rows[start:end]:
        out.append({
            "comp": int(r["comp"]), "row": int(r["row"]), "function": r["function"],
            "global_idx": [int(j) for j in r["global_idx"].split(",")],
            "DL_local": float(r["DL_local"]), "DL_optimistic": float(r["DL_optimistic"]),
        })
    return out


def load_allocal_rows(comp, rows, galaxies, data_dir, base_run_name, use_physical_scale):
    """Per galaxy: {row: (p_full, inc, d, negloglike)} for the requested variant rows."""
    rho_tag = "rhoTrue" if use_physical_scale else "rhoFalse"
    mpt = max_param_total(comp, use_physical_scale)
    out = {}
    for galaxy in galaxies:
        path = os.path.join(galaxy_out_dir(data_dir, base_run_name, rho_tag, galaxy, comp),
                            f"codelen_matches_comp{comp}.dat")
        data = np.atleast_2d(np.genfromtxt(path))
        out[galaxy] = {
            r: (data[r, 3:3 + mpt].copy(), float(data[r, 3 + 2 * mpt]),
                float(data[r, 3 + 2 * mpt + 1]), float(data[r, 0]))
            for r in rows
        }
    return out


class GalaxyFit:
    """One galaxy's likelihood for one candidate function, with globals held fixed."""

    def __init__(self, likelihood, fcn, global_idx, p_full, inc, d, max_fun_params, n_extra):
        self.lk = likelihood
        self.fcn, eq, integrated = likelihood.run_sympify(fcn, tmax=5, try_integration=False)
        self.nshape = likelihood.nparam_shape
        eq_numpy = match.make_lambdified_eq(eq, self.nshape)

        self.shape_idx, self.pad_idx, extra_idx, _ = match.build_layout(
            self.nshape, max_fun_params, n_extra)
        mpt = max_fun_params + n_extra
        galaxy_idx = np.arange(mpt, mpt + likelihood.num_galaxy_params)
        self.extra_idx_ext = np.concatenate([extra_idx, galaxy_idx]).astype(int)
        self.active_idx_ext = np.concatenate([self.shape_idx, self.extra_idx_ext]).astype(int)
        self.global_idx = np.asarray(global_idx, dtype=int)
        self.local_idx = np.array([j for j in self.active_idx_ext if j not in set(global_idx)], dtype=int)
        # rho0 and rs are optimised in log10 (they must stay positive, and are ~1e7-1e18)
        self.log_idx = extra_idx

        self.p_ext = np.concatenate([np.asarray(p_full, dtype=float), [inc, d]])

        chi2_fcn = likelihood.get_wrapped_like(likelihood.get_loss(
            eq_numpy, integrated, value="evaluate", include_priors=True))
        self.hessian = likelihood.get_loss(eq_numpy, integrated, value="hessian", include_priors=True)
        lk, act = likelihood, self.active_idx_ext
        self.fop = lambda theta: chi2_fcn(jnp.asarray(theta, dtype=float)[act],
                                          lk.xvar, lk.yvar, lk.yerr_lo, lk.yerr_hi)

    def _theta(self, g, z):
        theta = self.p_ext.copy()
        theta[self.global_idx] = g
        theta[self.local_idx] = z
        is_log = np.isin(self.local_idx, self.log_idx)
        theta[self.local_idx[is_log]] = 10.0 ** z[is_log]
        return theta

    def _z(self):
        z = self.p_ext[self.local_idx].copy()
        is_log = np.isin(self.local_idx, self.log_idx)
        z[is_log] = np.log10(z[is_log])
        return z

    def loss(self, theta):
        v = float(self.fop(theta))
        return v if np.isfinite(v) else BAD

    def optimise_locals(self, g, maxiter):
        """Polish this galaxy's locals at fixed globals g; warm-start from the last solution."""
        # A single Nelder-Mead run often stalls short of the minimum here (same
        # g gave up to ~1 nat different sums depending on the warm start), so
        # restart it from its own result until it stops improving.
        z, best = self._z(), np.inf
        for _ in range(3):
            res = minimize(lambda z: self.loss(self._theta(g, z)), z, method="Nelder-Mead",
                           options={"maxiter": maxiter, "xatol": 1e-6, "fatol": 1e-6})
            z, improved = res.x, best - res.fun
            best = min(best, res.fun)
            if improved < 1e-3:
                break
        theta = self._theta(g, z)
        val = self.loss(theta)
        if val < BAD:
            self.p_ext = theta
        return val

    def codelen(self, g):
        """Local codelen (match.py rules, globals fixed) and this galaxy's global-Fisher diagonal."""
        theta = self.p_ext.copy()
        theta[self.global_idx] = g
        nll = self.loss(theta)
        H_act = np.asarray(self.hessian(jnp.asarray(theta[self.active_idx_ext], dtype=float),
                                        self.lk.xvar, self.lk.yvar, self.lk.yerr_lo,
                                        self.lk.yerr_hi), dtype=float)
        fish = np.zeros((len(theta), len(theta)))
        fish[np.ix_(self.active_idx_ext, self.active_idx_ext)] = H_act

        shape_local = np.array([j for j in self.shape_idx if j not in set(self.global_idx)], dtype=int)
        codelen, nll_final, p_final, _ = match.compute_function_codelen(
            self.fcn, theta, fish, shape_local, self.extra_idx_ext, self.local_idx,
            self.pad_idx, self.fop, nll, number_points=10**3,
        )
        self.p_ext = p_final
        return codelen, nll_final, np.diag(fish)[self.global_idx]


def total_loss(fits, g, maxiter):
    local = sum(f.optimise_locals(g, maxiter) for f in fits)
    return comm.allreduce(local, op=MPI.SUM)


def fit_candidate(cand, galaxies_mine, likelihoods, allocal, aifeyn, max_fun_params, n_extra,
                  inner_maxiter, outer_maxiter, trace=False):
    g_idx = cand["global_idx"]
    fits = []
    for galaxy in galaxies_mine:
        p_full, inc, d, _ = allocal[galaxy][cand["row"]]
        fits.append(GalaxyFit(likelihoods[galaxy], cand["function"], g_idx, p_full, inc, d,
                              max_fun_params, n_extra))

    # Outer start: median of the galaxies' own all-local values for each global param
    own = np.array([f.p_ext[g_idx] for f in fits]).reshape(-1, len(g_idx))
    all_own = np.concatenate(comm.allgather(own), axis=0)
    g0 = np.median(all_own, axis=0)
    step = np.maximum(np.abs(np.percentile(all_own, 75, axis=0) - np.percentile(all_own, 25, axis=0)),
                      0.1 * np.maximum(np.abs(g0), 1e-3))
    simplex = np.vstack([g0] + [g0 + step[j] * np.eye(len(g_idx))[j] for j in range(len(g_idx))])

    n_eval = [0]

    def outer(g):
        n_eval[0] += 1
        val = total_loss(fits, np.asarray(g, dtype=float), inner_maxiter)
        if trace:
            log(f"    outer {n_eval[0]}: g={np.asarray(g).tolist()}  sum(-logL)={val:.4f}")
        return val

    # fatol 0.05 nats: warm-started inner fits make the summed loss slightly
    # history-dependent (~1e-2 nats), so a tighter tolerance never triggers.
    res = minimize(outer, g0, method="Nelder-Mead",
                   options={"maxfev": outer_maxiter, "initial_simplex": simplex,
                            "xatol": 1e-4, "fatol": 5e-2})
    g_best = np.asarray(res.x, dtype=float)
    # Re-polish locals at the final globals (the last outer evaluation may not have been g_best)
    total_loss(fits, g_best, inner_maxiter)

    # Code length
    local_cl, local_nll, fisher_g = 0.0, 0.0, np.zeros(len(g_idx))
    for f in fits:
        cl, nll, fg = f.codelen(g_best)
        local_cl += cl
        local_nll += nll
        fisher_g += np.where(np.isfinite(fg), fg, np.nan)
    local_cl = comm.allreduce(local_cl, op=MPI.SUM)
    local_nll = comm.allreduce(local_nll, op=MPI.SUM)
    fisher_g = comm.allreduce(fisher_g, op=MPI.SUM)

    Sigma = np.full(len(g_idx), np.inf)
    ok = np.isfinite(fisher_g) & (fisher_g > 0)
    Sigma[ok] = 1.0 / np.sqrt(fisher_g[ok])
    used_integral = False
    if not np.all(ok):
        # Same fallback as match.py, on the summed likelihood with locals fixed
        used_integral = True

        def fop_total(gv):
            return comm.allreduce(sum(f.loss(_with_g(f, gv)) for f in fits), op=MPI.SUM)

        Sigma = match.get_sigma_from_integral(
            g_best.copy(), Sigma, fop_total, fop_total(g_best),
            active_idx=np.where(~ok)[0], fcn_i=cand["function"], number_points=10**3)
        Sigma[np.isnan(Sigma)] = np.inf

    Delta = np.sqrt(12.0) * Sigma
    global_cl = 0.0
    floored = []
    for j in range(len(g_idx)):
        if g_best[j] == 0:
            continue
        if not np.isfinite(Delta[j]) or Delta[j] <= 0 or abs(g_best[j]) < Delta[j]:
            floored.append(j)
            global_cl += math.log(2.0)
        else:
            global_cl += math.log(2.0) + math.log(abs(g_best[j]) / Delta[j])

    DL_mixed = local_nll + local_cl + global_cl + float(aifeyn[cand["row"]])
    return {
        **cand, "global_values": g_best.tolist(), "DL_mixed": DL_mixed,
        "negloglike_sum": local_nll, "codelen_local": local_cl, "codelen_global": global_cl,
        "aifeyn": float(aifeyn[cand["row"]]), "outer_evals": n_eval[0],
        "outer_converged": bool(res.success), "global_floored": floored,
        "global_integral_fallback": used_integral,
    }


def _with_g(fit, g):
    theta = fit.p_ext.copy()
    theta[fit.global_idx] = g
    return theta


def _parse_args():
    p = argparse.ArgumentParser(
        description="Nested global/local re-fit (step 2) of the candidates written "
                    "by screen_global_local.py. Run with mpirun to split galaxies "
                    "across ranks; --start/--end pick a slice of the candidate list "
                    "so a job array can share the work."
    )
    p.add_argument("--candidates", required=True)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=None)
    p.add_argument("--inner-maxiter", type=int, default=1000)
    p.add_argument("--outer-maxiter", type=int, default=200)
    p.add_argument("--trace", action="store_true", help="Print every outer evaluation.")
    p.add_argument("--galaxy-file", default="galaxy_names.txt")
    p.add_argument("--data-file", default="ALL_gbar_gobs_RAR_direct_phot_with_5_arc_radius_plus_Halpha_for_ESR.txt",
                   help="Catalogue the all-local fits being re-fitted were run on.")
    p.add_argument("--data-dir", default=os.getcwd(), help="Where fitting/output/... lives.")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--run-name", default="mightee")
    p.add_argument("--use-physical-scale", dest="use_physical_scale", action="store_true", default=True)
    p.add_argument("--no-physical-scale", dest="use_physical_scale", action="store_false")
    p.add_argument("--fn-library-dir", default=os.environ.get("ESR_FUNCTION_LIBRARY_DIR"))
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    fn_dir = resolve_fn_dir(args.fn_library_dir)
    rho_tag = "rhoTrue" if args.use_physical_scale else "rhoFalse"
    out_dir = args.out_dir or os.path.join(args.data_dir, "fitting", "output",
                                           f"output_{args.run_name}_{rho_tag}", "MIXED")
    if rank == 0:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.galaxy_file) as f:
        galaxies = [line.strip() for line in f if line.strip()]
    galaxies_mine = galaxies[rank::size]

    candidates = load_candidates(args.candidates, args.start, args.end)
    log(f"{len(candidates)} candidates, {len(galaxies)} galaxies over {size} rank(s)")

    n_extra = 2 if args.use_physical_scale else 0
    likelihoods = {
        g: MIGHTEELikelihood(data_file=args.data_file, name=g, run_name=f"mixed_{g}",
                             data_dir=os.getcwd(), use_physical_scale=args.use_physical_scale,
                             fn_library_dir=fn_dir)
        for g in galaxies_mine
    }

    end = args.end if args.end is not None else args.start + len(candidates)
    out_path = os.path.join(out_dir, f"mixed_results_{args.start}_{end}.csv")
    fields = ["comp", "row", "function", "global_idx", "global_values", "DL_mixed", "DL_local",
              "DL_optimistic", "negloglike_sum", "codelen_local", "codelen_global", "aifeyn",
              "outer_evals", "outer_converged", "global_floored", "global_integral_fallback",
              "seconds"]
    if rank == 0:
        with open(out_path, "w", newline="") as f:
            csv.writer(f, delimiter=";").writerow(fields)

    for comp in sorted({c["comp"] for c in candidates}):
        comp_cands = [c for c in candidates if c["comp"] == comp]
        _, aifeyn = load_allfcn_and_aifeyn(fn_dir, comp)
        mpt = max_param_total(comp, args.use_physical_scale)
        allocal = load_allocal_rows(comp, sorted({c["row"] for c in comp_cands}), galaxies_mine,
                                    args.data_dir, args.run_name, args.use_physical_scale)
        for cand in comp_cands:
            t0 = time.time()
            r = fit_candidate(cand, galaxies_mine, likelihoods, allocal, aifeyn,
                              mpt - n_extra, n_extra, args.inner_maxiter, args.outer_maxiter,
                              trace=args.trace)
            r["seconds"] = round(time.time() - t0, 1)
            log(f"comp={comp} row={cand['row']} {cand['function']} global={cand['global_idx']} "
                f"-> g={np.round(r['global_values'], 4).tolist()}  DL_mixed={r['DL_mixed']:.2f}  "
                f"(all-local {r['DL_local']:.2f}, optimistic {r['DL_optimistic']:.2f})  "
                f"{r['outer_evals']} outer evals, {r['seconds']}s")
            if rank == 0:
                with open(out_path, "a", newline="") as f:
                    csv.writer(f, delimiter=";").writerow([
                        r["comp"], r["row"], r["function"], ",".join(map(str, r["global_idx"])),
                        ",".join(f"{v:.8g}" for v in r["global_values"]), f"{r['DL_mixed']:.4f}",
                        f"{r['DL_local']:.4f}", f"{r['DL_optimistic']:.4f}",
                        f"{r['negloglike_sum']:.4f}", f"{r['codelen_local']:.4f}",
                        f"{r['codelen_global']:.4f}", f"{r['aifeyn']:.4f}", r["outer_evals"],
                        r["outer_converged"], ",".join(map(str, r["global_floored"])),
                        r["global_integral_fallback"], r["seconds"]])
    log(f"Saved: {out_path}")
