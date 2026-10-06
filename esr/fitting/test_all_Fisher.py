import numpy as np
import math
from scipy.optimize import minimize
import sympy
from mpi4py import MPI
import warnings
import os
import sys
import itertools
import numdifftools as nd
from scipy.stats import mode

import esr.fitting.test_all as test_all
from esr.fitting.sympy_symbols import *
import esr.generation.simplifier as simplifier
import jax
import jax.numpy as jnp

warnings.filterwarnings("ignore")

use_relative_dx = True              # CHANGE

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
    
def load_loglike(comp, likelihood, data_start, data_end, split=True):
    """Load results of optimisation completed by test_all.py

    Args:
        :comp (int): complexity of functions to consider
        :likelihood (fitting.likelihood object): object containing data, likelihood functions and file paths
        :data_start (int): minimum index of results we want to load (only if split=True)
        :data_end (int): maximum index of results we want to load (only if split=True)
        :split (bool, deault=True): whether to return subset of results given by data_start and data_end (True) or all data (False)

    Returns:
        :negloglike (list): list of minimum log-likelihoods
        :params (np.ndarray): list of parameters at maximum likelihood points. Shape = (nfun, nparam).
        :Nconv, Niter, times: as saved by test_all.py.
        :inc_fit, d_fit, stage2_chi2: stage-2 galaxy-nuisance-parameter results from
            test_all.py (nan-filled if likelihood.num_galaxy_params == 0).

    Column layout of negloglike_comp{comp}.dat (see test_all.py's main()):
        [chi2, params (max_param_total cols, shape+rho0/rs), Inc, D, stage2_chi2, Nconv, Niter, times]
    max_param_total is computed the same deterministic way test_all.py's main() does, rather
    than inferred from file width, so the params/Inc/D/stage2_chi2 columns split unambiguously.
    """
    if rank == 0:
        print(likelihood.out_dir + "/negloglike_comp"+str(comp)+".dat")
    data = np.genfromtxt(likelihood.out_dir + "/negloglike_comp"+str(comp)+".dat")
    data = np.atleast_2d(data)

    max_param_shape = int(max(4, np.floor((comp - 1) / 2)))
    n_extra = 2 if getattr(likelihood, "use_physical_scale", False) else 0
    max_param_total = max_param_shape + n_extra

    negloglike = np.atleast_1d(data[:, 0])
    params = np.atleast_2d(data[:, 1:1 + max_param_total])
    inc_fit = np.atleast_1d(data[:, 1 + max_param_total])
    d_fit = np.atleast_1d(data[:, 1 + max_param_total + 1])
    stage2_chi2 = np.atleast_1d(data[:, 1 + max_param_total + 2])
    Nconv = np.atleast_1d(data[:,-3])
    Niter = np.atleast_1d(data[:,-2])
    times = np.atleast_1d(data[:,-1])

    if split:
        negloglike = negloglike[data_start:data_end]               # Assuming same order of fcn and chi2 files
        params = params[data_start:data_end,:]
        inc_fit = inc_fit[data_start:data_end]
        d_fit = d_fit[data_start:data_end]
        stage2_chi2 = stage2_chi2[data_start:data_end]
        Nconv = Nconv[data_start:data_end]
        Niter = Niter[data_start:data_end]
        times = times[data_start:data_end]
    return negloglike, params, Nconv, Niter, times, inc_fit, d_fit, stage2_chi2

def _galaxy_param_term(val, fisher_diag_val):
    """MDL codelen contribution for a single galaxy nuisance parameter (Inc or D),
    floored at Delta=|theta| -- pure arithmetic on an already-computed Fisher value,
    no likelihood/Hessian call.

    Same floor semantics as codelen_from_vector: a snapped-to-zero value costs
    nothing (never actually happens for Inc/D in practice, but matches the
    convention exactly); an unresolved or non-finite Fisher_diag (the
    inclination/amplitude degeneracy documented elsewhere in this codebase for
    rho0/rs applies here too) is capped at Delta=|theta|, costing exactly
    log(2) and nothing more -- never nan, never negative.
    """
    if val == 0:
        return 0.0
    if (not np.isfinite(fisher_diag_val)) or fisher_diag_val <= 0:
        return math.log(2.0)
    Delta = math.sqrt(12.0 / fisher_diag_val)
    if Delta >= abs(val):
        return math.log(2.0)
    return math.log(2.0) + math.log(abs(val) / Delta)


def galaxy_codelen_from_fisher(inc_ML, d_ML, fisher_inc, fisher_d):
    """MDL codelen contribution for Inc, D given their own (already-computed)
    Fisher-diagonal entries -- see _galaxy_param_term for the floor semantics.
    """
    return _galaxy_param_term(inc_ML, fisher_inc) + _galaxy_param_term(d_ML, fisher_d)


def convert_params(fcn_i, eq, integrated, theta_ML, likelihood, negloglike, max_param=4,  max_fun_param = 4,
                    inc_ML=np.nan, d_ML=np.nan):
    """Compute Fisher, correct MLP and find parametric contirbution to description length for single function
    
    Args:
        :fcn_i (str): string representing function we wish to fit to data
        :eq (sympy object): sympy object for the function we wish to fit to data
        :integrated (bool): whether eq_numpy has already been integrated
        :theta_ML (list): the maximum likelihood values of the parameters
        :likelihood (fitting.likelihood object): object containing data, likelihood functions and file paths
        :negloglike (float): the minimum log-likelihood for this function
        :max_param (int, default=4): The maximum number of parameters considered. This sets the shapes of arrays used.
    
    Returns:
        :params (list): the corrected maximum likelihood values of the parameters
        :negloglike (float): the corrected minimum log-likelihood for this function
        :deriv (list): flattened version of the Hessian of -log(likelihood) at the maximum likelihood point
        :codelen (float): the parameteric contribution to the description length of this function
        :fisher_inc_out (float): Inc's own Fisher-diagonal entry (curvature of
            -log(likelihood) w.r.t. Inc alone), read off the single joint
            [shape,Inc,D] Hessian computed once, unconditionally, near the
            top of this function (same point every branch below uses) and
            threaded through every return path unchanged -- saved by main()
            so match.py can seed its own per-canonical-function cache
            instead of waiting on its first fresh Hessian call to discover
            whether Inc's Fisher is even usable. The shape-only Hessian used
            for the snap-to-zero search below is the top-left block of this
            same joint Hessian (fixing Inc/D and differentiating only the
            rest gives exactly that block), so only one Hessian is ever
            computed per function.
        :fisher_d_out (float): D's own Fisher-diagonal entry, same idea.
        :flag_reason (str): non-scoring plausibility flag, currently only set
            when the joint Hessian is indefinite after snap-to-zero (empty
            string otherwise). Does NOT set codelen=nan or exclude the
            function -- see the check's own comment for why -- saved by
            main() to its own flagged_indefinite_comp{N}.dat file, same
            "flag it, don't exclude it" pattern test_all.py's own
            flagged_comp{N}.dat already uses.

    """
    def get_deriv(Hmat, nparam, max_param=4, max_fun_param=4):
        Hmat_max = np.zeros((max_param,max_param))
        Hmat_max[:nparam, :nparam] = Hmat[:nparam, :nparam]
        Hmat_max[max_fun_param:, :nparam] = Hmat[nparam:, :nparam]
        Hmat_max[:nparam, max_fun_param:] = Hmat[:nparam, nparam:]
        Hmat_max[max_fun_param:,max_fun_param:] = Hmat[nparam:, nparam:]

        deriv = Hmat_max[np.triu_indices(max_param)]
        return deriv

    #Set number of params
    # nparam = simplifier.count_params([fcn_i], max_param)[0] #fun params
    #max_fun_params = 4
    #max_param = max_fun_params + likelihood.num_galaxy_params

    nshape = simplifier.count_params([fcn_i], max_fun_param)[0]
    n_extra = max(0, max_param - max_fun_param)
    nparam = nshape + n_extra
    num_galaxy_params = getattr(likelihood, "num_galaxy_params", 0)

    # Data
    xvar = likelihood.xvar
    yvar = likelihood.yvar
    # yerr = likelihood.yerr

    yerr_lo = likelihood.yerr_lo
    yerr_hi = likelihood.yerr_hi
        
    params = np.zeros(max_param)
    deriv = np.full(int((max_param) * (max_param + 1) / 2), np.nan)
    # Non-excluding plausibility flag (indefinite-after-snap-to-zero, below)
    # -- never turned into codelen=nan by itself, just recorded for main()
    # to write to its own file, same "flag it, don't exclude it" pattern
    # test_all.py's own flagged_comp{N}.dat already uses.
    flag_reason = ""

    # print(fcn_i, 'nparam', nparam, flush=True)

    try:
        if nshape == 0:
            eq_numpy = sympy.lambdify([x], eq, modules=["jax"])
        elif nshape > 1:
            all_a = ' '.join([f'a{i}' for i in range(nshape)])
            all_a = list(sympy.symbols(all_a, real=True))
            eq_numpy = sympy.lambdify([x] + all_a, eq, modules=["jax"])
        else:
            eq_numpy = sympy.lambdify([x, a0], eq, modules=["jax"])
    except Exception:
        print("BAD:", fcn_i, negloglike, np.isfinite(negloglike))
        Fisher_diag = np.nan
        deriv[:] = np.nan
        codelen = np.nan  # couldn't even build eq_numpy
        return params, negloglike, deriv, codelen, np.full(3 + 2 * max_param, np.nan), flag_reason

    theta_ML_compact_orig = jnp.concatenate(
        [theta_ML[:nshape], theta_ML[max_fun_param:max_fun_param + n_extra]]
    )
    shape_fit_joint = np.asarray(theta_ML_compact_orig, dtype=float)

    n_model = shape_fit_joint.shape[0]
    galaxy_hessian_extra = np.full(3 + 2 * max_param, np.nan)
    if num_galaxy_params == 0:
        Hmat_joint = None
        fisher_inc_out, fisher_d_out = np.nan, np.nan
        codelen_galaxy = 0.0
    else:
        theta_full = jnp.concatenate([
            jnp.atleast_1d(jnp.asarray(shape_fit_joint, dtype=jnp.float64)),
            jnp.array([inc_ML, d_ML], dtype=jnp.float64),
        ])
        joint_hessian_template = likelihood.get_loss(eq_numpy, integrated, value='hessian', include_priors=True)
        Hmat_joint = np.asarray(
            joint_hessian_template(theta_full, xvar, yvar, yerr_lo, yerr_hi), dtype=float
        )
        fisher_inc_out = float(Hmat_joint[n_model, n_model])
        fisher_d_out = float(Hmat_joint[n_model + 1, n_model + 1])
        codelen_galaxy = galaxy_codelen_from_fisher(inc_ML, d_ML, fisher_inc_out, fisher_d_out)

        galaxy_hessian_extra[0] = fisher_inc_out
        galaxy_hessian_extra[1] = fisher_d_out
        galaxy_hessian_extra[2] = float(Hmat_joint[n_model, n_model + 1])
        galaxy_hessian_extra[3:3 + n_model] = Hmat_joint[:n_model, n_model]
        galaxy_hessian_extra[3 + max_param:3 + max_param + n_model] = Hmat_joint[:n_model, n_model + 1]

    loss_template = likelihood.get_loss(
        eq_numpy, integrated, value='evaluate', include_priors=True,
        fixed_galaxy_params=(inc_ML, d_ML),
    )
    chi2_fcn = likelihood.get_wrapped_like(loss_template)


    if nparam ==0 :
        codelen = 0 + codelen_galaxy
        negloglike = float(chi2_fcn(
            jnp.asarray([], dtype=jnp.float64), xvar, yvar, yerr_lo, yerr_hi,
        ))
        deriv = np.zeros(int((max_param) * (max_param + 1) / 2))
        return params, negloglike, deriv, codelen, galaxy_hessian_extra, flag_reason

    # Get Hessian

    theta_ML = jnp.asarray(shape_fit_joint, dtype=jnp.float64)

    if num_galaxy_params == 0:
        hessian_template = likelihood.get_loss(
            eq_numpy, integrated, value='hessian', fixed_galaxy_params=(inc_ML, d_ML),
        )
        Hmat = hessian_template(theta_ML, xvar, yvar, yerr_lo, yerr_hi)
    else:
        Hmat = Hmat_joint[:n_model, :n_model]

    #Other related quantities
    Fisher_diag = jnp.diag(Hmat)
    Delta = np.sqrt(12./Fisher_diag)
    deriv = get_deriv(Hmat, nshape, max_param=max_param, max_fun_param=max_fun_param)
    Nsteps = abs(np.array(theta_ML))/Delta

    if (np.sum(Fisher_diag <= 0.) > 0.) or (np.sum(np.isnan(Fisher_diag)) > 0):
        codelen = np.nan
        # deriv = np.inf
        print(fcn_i, 'bad Fisher', flush=True)
        return params, negloglike, deriv, codelen, galaxy_hessian_extra, flag_reason
    k = nparam
    theta_ML_orig = np.copy(theta_ML)

    # See if we can snap any parameters to zero. fixed_galaxy_params=
    # (inc_ML, d_ML), so every negloglike evaluated from here on (the
    # snap-to-zero search, the fallback below, and whatever final theta_ML
    # gets reported) uses the true joint Inc/D, matching theta_ML being
    # shape_fit_joint.
    def fop(x):
        return chi2_fcn(jnp.array(x), xvar, yvar, yerr_lo, yerr_hi)

    negloglike_orig = float(fop(theta_ML_orig))

    # rho0/rs (the n_extra physical-scale slots, indices nshape:nparam) must
    # never be snapped to zero -- rho0=0 is an identically-zero density,
    # rs=0 is a singular scale radius, neither is a meaningful "we don't
    # need this parameter" statement the way it is for a flat-prior shape
    # coefficient
    snap_eligible = np.arange(nparam) < nshape

    if np.sum((Nsteps<1) & snap_eligible)>0:
        # First try setting any eligible (shape-only) parameter to 0 that
        # doesn't have at least one precision step, and recompute -log(L).
        theta_ML = theta_ML.at[(Nsteps<1) & snap_eligible].set(0)
        negloglike = fop(theta_ML)
        # print('HERE TOO', fcn_i, negloglike)
        # print(fcn_i, theta_ML, Delta, negloglike)
        # For the codelen, we effectively don't have the parameter that had Nsteps<1
        if np.isfinite(negloglike):
            k -= np.sum((Nsteps<1) & snap_eligible)
            kept_mask = ~((Nsteps<1) & snap_eligible)
        else:
            # Let's see if setting any of the parameters to zero is ok
            # Search from the most-parsimonious combination (dropping the
            # most parameters
            try_idx = np.arange(nparam)[(Nsteps < 1) & snap_eligible]
            found = False
            for r in reversed(range(1, len(try_idx))):
                for idx in itertools.combinations(try_idx, r):
                    theta_ML = np.copy(theta_ML_orig)
                    for idx_ in idx:
                        theta_ML[idx_] = 0.
                        # theta_ML = theta_ML.at[idx_].set(0.)
                    negloglike = fop(theta_ML)
                    if np.isfinite(negloglike):
                        found = True
                        break
                if found:
                    break
            kept_mask = np.ones(len(theta_ML), dtype=bool)
            if np.isfinite(negloglike):
                k -= len(idx)
                kept_mask[list(idx)] = 0
                # kept_mask = kept_mask.at[idx].set(0)
            else:
                theta_ML = theta_ML_orig
                negloglike = negloglike_orig
                k = nparam
            
        if k<0:
            print("This shouldn't have happened", flush=True)
            quit()
        elif k==0:
            # codelen_galaxy already reflects curvature at the true joint
            # optimum (computed once, above), independent of the separate
            # shape-only snap-to-zero decision reported here.
            codelen = 0 + codelen_galaxy
            return params, negloglike, deriv, codelen, galaxy_hessian_extra, flag_reason

        Fisher_diag = Fisher_diag[kept_mask]     # Only consider these parameters in the codelen
        theta_ML = theta_ML[kept_mask]

    else:
        kept_mask = np.ones(len(theta_ML), dtype=bool)
   
        negloglike = negloglike_orig

    Hmat_kept = np.asarray(Hmat)[np.ix_(kept_mask, kept_mask)]

    if Hmat_kept.size > 0 and not np.all(np.isfinite(Hmat_kept)):
        codelen = np.nan
        print(fcn_i, 'bad Fisher (non-finite Hessian after snap-to-zero)', flush=True)
        return params, negloglike, deriv, codelen, galaxy_hessian_extra, flag_reason
    # Joint indefiniteness (all-positive diagonal, but a negative eigenvalue
    # from an off-diagonal/joint-curvature direction -- e.g. rho0/rs trading
    # off against each other) does NOT exclude the function here. Unlike the
    # non-finite-Hessian check above, this can fire on a genuinely flat
    # (not broken) direction whose eigenvalue sits at the floating-point
    # noise floor -- CLASH's own test_all_Fisher.py never had this check at
    # all, only ever checking the Fisher diagonal, and the method's own
    # fallback for an unresolved parameter (match.py's get_sigma_from_integral)
    # is ALSO a one-parameter-at-a-time profile likelihood that can't resolve
    # a joint degeneracy either -- so excluding here is stricter than
    # anything the rest of the method can actually act on. Flag it (for
    # main() to write to its own file) and fall through to the same
    # diagonal-only codelen computation CLASH would use instead.
    if Hmat_kept.size > 0:
        min_eigval = np.min(np.linalg.eigvalsh(Hmat_kept))
        if min_eigval <= 0.:
            flag_reason = f"indefinite Hessian after snap-to-zero (min_eigval={min_eigval:.3e})"
            print(fcn_i, 'bad Fisher (indefinite after snap-to-zero) -- flagged, not excluded', flush=True)

    cutoff_Delta = np.copy(Delta)
    cutoff_Delta[Nsteps < 1] = np.abs(theta_ML_orig[Nsteps < 1])

    Delta_codelen = cutoff_Delta[kept_mask]
    codelen = -k/2.*math.log(3.) + np.sum(
        0.5*math.log(12.) - np.log(Delta_codelen) + np.log(abs(np.array(theta_ML)))
    )
    # New params after the setting to 0, padded to length max_param as always
    theta_ML = theta_ML_orig
    theta_ML[~kept_mask] = 0.
    Delta[~kept_mask] = 0.
    cutoff_Delta[~kept_mask] = 0.

    # Check if the function has any cutoffs in the likelihood
    if np.isinf(fop(theta_ML + cutoff_Delta)) or np.isinf(fop(theta_ML - cutoff_Delta)):
            codelen = np.nan
            deriv = np.nan*np.ones(deriv.shape)
            print(fcn_i, 'cutoffs', flush=True)
            return params, negloglike, deriv, codelen, galaxy_hessian_extra, flag_reason

    codelen = codelen + codelen_galaxy

    #Save the parameters
    params = np.zeros(max_param)
    params[:nshape] = theta_ML[:nshape]
    params[max_fun_param:max_fun_param + n_extra] = theta_ML[nshape:nshape + n_extra]
    # print('CODELEN', fcn_i, codelen, flush=True)
    return params, negloglike, deriv, codelen, galaxy_hessian_extra, flag_reason

def main(comp, likelihood, tmax=5, print_frequency=50, try_integration=False, max_fun_params=4):
    """Compute Fisher, correct MLP and find parametric contirbution to description length for all functions and save to file
    
    Args:
        :comp (int): complexity of functions to consider
        :likelihood (fitting.likelihood object): object containing data, likelihood functions and file paths
        :tmax (float, default=5.): maximum time in seconds to run any one part of simplification procedure for a given function
        :print_frequency (int, default=50): the status of the fits will be printed every ``print_frequency`` number of iterations
        :try_integration (bool, default=False): when likelihood requires integral, whether to try to analytically integrate (True) or just numerically integrate (False)
        
    Returns:
        None
    
    """

    if likelihood.is_mse:
        raise ValueError('Cannot use MSE with description length')
        
    if rank == 0:
        print('\nComputing Fisher', flush=True)

    if comp>=8:
        sys.setrecursionlimit(2000 + 500 * (comp - 8))

    fcn_list_proc, data_start, data_end = test_all.get_functions(comp, likelihood)
    (
        negloglike, params_proc, Nconv_proc, Niter_proc, times_proc,
        inc_fit_proc, d_fit_proc, stage2_chi2_proc,
    ) = load_loglike(comp, likelihood, data_start, data_end)

    max_param = params_proc.shape[1]


    codelen = np.zeros(len(fcn_list_proc))          # This is now only for this proc
    params = np.zeros([len(fcn_list_proc), max_param])
    deriv = np.zeros([len(fcn_list_proc), int(max_param* (max_param+1) / 2)])

    galaxy_hessian_all = np.full((len(fcn_list_proc), 3 + 2 * max_param), np.nan)
    # Non-scoring plausibility flags (indefinite-after-snap-to-zero, from
    # convert_params) -- same pattern as test_all.py's own flagged_comp{N}.dat:
    # written to its own file below, never fed into codelen/DL.
    flagged_rows = []
    print(len(fcn_list_proc), flush=True)
    for i in range(len(fcn_list_proc)):           # Consider all possible complexities
        if rank == 0 and ((i == 0) or ((i+1) % print_frequency == 0)):
            print(f'{i+1} of {len(fcn_list_proc)}', flush=True)
        if np.isnan(negloglike[i]) or np.isinf(negloglike[i]):
            codelen[i]=np.nan
            print('bad fucntion', fcn_list_proc[i], negloglike[i])
            continue

        theta_ML = params_proc[i,:]
        # try:
        fcn_i = fcn_list_proc[i].replace('\n', '')
        fcn_i = fcn_list_proc[i].replace('\'', '')
        fcn_i, eq, integrated = likelihood.run_sympify(fcn_i, tmax=tmax, try_integration=try_integration)
        params[i,:], negloglike[i], deriv[i,:], codelen[i], galaxy_hessian_all[i,:], flag_reason = convert_params(
            fcn_i, eq, integrated, theta_ML, likelihood, negloglike[i],
            max_param=max_param, max_fun_param=max_fun_params,
            inc_ML=inc_fit_proc[i], d_ML=d_fit_proc[i],
        )
        if flag_reason:
            flagged_rows.append(f"{fcn_i};{flag_reason}")
        # except NameError:
        #     # Occurs if function produced not implemented in numpy
        #     if try_integration:
        #         fcn_i = fcn_list_proc[i].replace('\n', '')
        #         fcn_i = fcn_list_proc[i].replace('\'', '')
        #         fcn_i, eq, integrated = likelihood.run_sympify(fcn_i, tmax=tmax, try_integration=False)
        #         params[i,:], negloglike[i], deriv[i,:], codelen[i] = convert_params(fcn_i, eq, integrated, theta_ML, likelihood, negloglike[i], max_param=max_param)
        #     else:
        #         params[i,:] = 0.
        #         deriv[i,:] = 0.
        #         codelen[i] = 0

        # except:
        #     print('bad function aqui', fcn_list_proc[i], negloglike[i])
        #     params[i,:] = 0.
        #     deriv[i,:] = 0.
        #     codelen[i] = 0

    # print('total codelen', codelen)
        

    # out_arr = np.transpose(np.vstack([codelen, negloglike] + [params[:,i] for i in range(max_param)], [Nconv_proc, Niter_proc, times_proc]))
    out_arr = np.vstack([codelen, negloglike] + [params[:, i] for i in range(max_param)] + [Nconv_proc, Niter_proc, times_proc])
    out_arr = np.transpose(out_arr)
    # print(negloglike)

    # out_arr_deriv = np.transpose(np.vstack([deriv[:,0], deriv[:,1], deriv[:,2], deriv[:,3], deriv[:,4], deriv[:,5], deriv[:,6], deriv[:,7], deriv[:,8], deriv[:,9]]))
    out_arr_deriv = np.transpose(np.vstack([deriv[:,i] for i in range(deriv.shape[1])]))

    # Combined shape + galaxy Hessian info, one row per unique function (same
    # order/indexing as codelen_comp{N}_deriv.dat, both keyed by the SAME
    # canonical index match.py's matches_proc[i] points into):
    #   [0 : deriv.shape[1]]                     shape/extra Hessian upper
    #                                             triangle (get_deriv's layout,
    #                                             unchanged)
    #   [deriv.shape[1] : deriv.shape[1]+3+2*max_param]
    #                                             galaxy Hessian info
    #                                             (fisher_inc, fisher_d,
    #                                             fisher_inc_d, cross_shape_inc
    #                                             [:max_param], cross_shape_d
    #                                             [:max_param] -- convert_params's
    #                                             galaxy_hessian_extra layout,
    #                                             unchanged)
    # Kept in one file (rather than a separate galaxy_fisher_comp{N}.dat)
    # since the two blocks are always written together, always the same
    # length, and always keyed by the same row -- see convert_params's own
    # docstring for why match.py wants this data at all.
    out_arr_deriv = np.hstack([out_arr_deriv, galaxy_hessian_all])
    np.savetxt(likelihood.temp_dir + '/codelen_deriv_'+str(comp)+'_'+str(rank)+'.dat', out_arr, fmt='%.7e')
    np.savetxt(likelihood.temp_dir + '/derivs_'+str(comp)+'_'+str(rank)+'.dat', out_arr_deriv, fmt='%.7e')
    comm.Barrier()

    if rank == 0:
        string = 'cat `find ' + likelihood.temp_dir + '/ -name "codelen_deriv_'+str(comp)+'_*.dat" | sort -V` > ' + likelihood.out_dir + '/codelen_comp'+str(comp)+'_deriv.dat'
        os.system(string)
        string = 'rm ' + likelihood.temp_dir + '/codelen_deriv_'+str(comp)+'_*.dat'
        os.system(string)

        string = 'cat `find ' + likelihood.temp_dir + '/ -name "derivs_'+str(comp)+'_*.dat" | sort -V` > ' + likelihood.out_dir + '/derivs_comp'+str(comp)+'.dat'
        os.system(string)
        string = 'rm ' + likelihood.temp_dir + '/derivs_'+str(comp)+'_*.dat'
        os.system(string)

    comm.Barrier()

    # Plausibility-flag list (indefinite-after-snap-to-zero, from
    # convert_params) -- same per-rank-write-then-rank-0-concatenate pattern
    # as test_all.py's own flagged_comp{N}.dat, into its own file (a
    # different name, since this is a different pipeline stage and a
    # different flag type) so it never touches the existing format
    # match.py/combine_DL.py already parse. Written every rank (even if
    # empty) so the find/cat/sort below behaves uniformly.
    with open(likelihood.temp_dir + '/flagged_indefinite_comp' + str(comp) + '_' + str(rank) + '.dat', 'w') as f:
        for row in flagged_rows:
            f.write(row + '\n')

    comm.Barrier()

    if rank == 0:
        string = 'cat `find ' + likelihood.temp_dir + '/ -name "flagged_indefinite_comp'+str(comp)+'_*.dat" | sort -V` > ' + likelihood.out_dir + '/flagged_indefinite_comp'+str(comp)+'.dat'
        os.system(string)
        string = 'rm ' + likelihood.temp_dir + '/flagged_indefinite_comp'+str(comp)+'_*.dat'
        os.system(string)

    comm.Barrier()

    return
