# The MIGHTEE ESR Fitting Pipeline — How It Works

**Purpose of this file:** a reference walkthrough of what each stage of the pipeline does, and
every non-obvious behaviour found and understood along the way (as of 2026-08-15, after the
self-consistency fixes). If you want to know what a specific line does, start here.

## The pipeline at a glance

```
For each galaxy (e.g. J022128.8-042448) and complexity comp (number of nodes in the function tree):

  1. test_all.py::main()            -- fit EVERY candidate function to this galaxy's rotation curve
       -> chi2_comp{N}weights_{rank}.dat / negloglike_comp{N}.dat

  2. test_all_Fisher.py::main()     -- Fisher matrix + description length for UNIQUE (canonical) functions
       -> derivs_comp{N}.dat, codelen_deriv_comp{N}.dat

  3. match.py::main()               -- extend that to EVERY function (incl. duplicate-group members,
                                        which are algebraic rewrites of a canonical form), reparametrize,
                                        snap-to-zero, add the galaxy (Inc/D) codelen term
       -> codelen_matches_comp{N}.dat

  4. combine_DL.py::main()          -- rank by total description length DL = -logL + codelen + aifeyn,
                                        picking the best duplicate-group member per unique function
       -> results_pretty_{N}.txt   (the human-readable ranking table)
```

Everything downstream depends on `dm_likelihood.py::MIGHTEELikelihood`, which defines the physics:
given a candidate density profile `rho(r) = rho0 * f(r/rs)` (an ESR-generated symbolic function `f`)
plus nuisance parameters Inclination and Distance, predict the rotation curve `v_circ(r)` and compare
to the observed one.

---

## 1. `test_all.py::optimise_fun_direct_nm` — fitting ONE function to ONE galaxy

This is the function that gets called once per candidate function, and is by far the most
mechanically complex piece. It runs in two stages: **stage 1** finds the best shape (+ `rho0`, `rs`)
with Inc/D held at their catalog values; **stage 2** takes the best shape(s) from stage 1 and lets
Inc/D move too, jointly with the shape, to find the true best fit.

### 1a. Sign combinations

ESR-generated functions have free coefficients (`a0`, `a1`, ...) whose *sign* isn't fixed by the
symbolic form. Rather than let a continuous optimizer discover signs (which local optimizers are bad
at — a sign flip is a discrete, non-local move), the code enumerates every `±1` combination of the
shape coefficients up front (`itertools.product([1,-1], repeat=nshape)`) and searches each sign
combo's continuous parameter space *separately and completely*. For `nshape=3` this is 8 combos; the
combinatorics stay small because ESR complexity is capped low enough that `nshape` rarely exceeds 4.

**Weird case — sign-combo commitment can miss the true optimum.** DIRECT's own search within each
sign combo is exhaustive-ish (see 1b), but the code never revisits "was the wrong combo even chosen
as promising" — a combo that looks mediocre on its own best point can still contain the true joint
optimum once Inc/D get to move in stage 2, and there's no mechanism to reconsider a combo once stage
1's coarse ranking has effectively deprioritized it. Confirmed directly on
`(pow(x,a0)-pow(Abs(a1),(1/x)))/x`: the true joint optimum lives in sign combo `(1,1,1,1)`, but
stage 1 (shape-only) rates that combo as mediocre (nll≈85.6) next to a different combo's better
stage-1 score, so `(1,1,1,1)` gets essentially no polish attention. The box-width tuning done today
(see below) cannot fix this — it's a level above where the box operates.

### 1b. DIRECT global search per sign combo

For each sign combo, `scipy.optimize.direct` (the DIviding RECTangles algorithm — deterministic,
derivative-free, no randomness) searches the internal coordinate space. Internal coordinates are
**log-magnitude**: a physical parameter `p` is represented as `x` where `p = sign * 10**x`, searched
over a box `[pmin, pmax]` (default **`[-6, 10]`**, i.e. `p` ranges roughly `1e-6` to `1e10` in
magnitude — see "the box" below for why this specific, asymmetric range). DIRECT is deterministic and
budget-capped (`direct_maxfun=direct_maxiter=3000` — empirically reproduces the old `100000/10000`
budget's best-fit value to 1e-5 relative, at 4-20x less wall time).

**Weird case — DIRECT's sampling is box-position-dependent, not just box-size-dependent.** DIRECT
adaptively partitions its search box; two boxes of the *same width* but a different center/position
produce genuinely different sample points, not "the same points plus more." This is why "the box"
(next section) affects different functions unpredictably rather than uniformly.

### 1c. The box — why `[-6, 10]`, not symmetric `[-8, 8]`

The default used to be a symmetric `[-8, 8]` (16 decades of magnitude either side of 1). Investigating
cases where a from-scratch random search beat this pipeline found the box was too narrow for
`rho0`/`rs` specifically to reach their true joint optimum in some cases — but *widening* the box
uniformly (tested `[-16,16]`) made ~25% of already-good functions measurably *worse* at the same
compute budget (DIRECT's sample points shift, not just add) — not a free win. Alicia's proposal,
verified extensively (13/13 originally-losing cases, a 20-function "already good" sample, all 37
comp=3+4 functions, and a final 24-function "good functions only" stress test): **keep the same total
width (16 decades) but shift it asymmetrically to `[-6, 10]`**, since params empirically needed more
headroom on the positive side, rarely the negative side. Net result: strong improvement, not a
strictly dominant one — one real ~1-nat regression survived across every test, on a function whose
own shape optimum is a numerically pathological float64-overflow escape (see 1e) in both boxes; which
of several equally-pathological alternate candidates a shifted box happens to also sample is
essentially arbitrary, not a meaningful search-quality difference.

### 1d. Building the `hybrid_starts` pool

Before polishing, the code assembles a pool of promising starting points from **five sources**, all
independent of `log_opt`... except the last:

1. **best-direct** — DIRECT's own single best point (per sign combo, then the best across combos).
2. **global pool** — up to `hybrid_global_n=10` diverse points from *all* DIRECT-sampled points
   across every sign combo, within `global_pool_delta_ll=5.0` nats of the global best, spaced at
   least `start_min_separation=1e-6` apart in internal coordinates.
3. **per-sign** — the best endpoint from *each* sign combo's own DIRECT run (so a combo that's
   globally mediocre still contributes its own best point).
4. **per-combo pool** — for the top `n_top_combos_for_pool=4` sign combos (by their own best score),
   a few more diverse points measured against *that combo's own* best, not the global one. Added
   because the global pool (item 2) silently contributes zero diversity to any combo whose own best
   is far from the global best — confirmed directly on a real regression case where this left the
   combo containing the true optimum with only its single per-sign point.
5. **random** — `n_random_polish_starts=10` fresh points drawn uniformly in
   `[-random_polish_log_range, random_polish_log_range] = [-3, 3]`, with random signs. These are the
   one source that's added *unconditionally*, regardless of `log_opt` — a design detail that mattered
   for the `log_opt=False` bug below.

### 1e. Polishing each start (Nelder-Mead + BFGS, both, always unbounded)

Every start in `hybrid_starts` gets polished twice — once with Nelder-Mead, once with BFGS (using
JAX-computed gradients) — and whichever converges better is kept. Neither algorithm dominates: BFGS's
gradient tracks narrow ridges NM's rank-only simplex comparisons can get pulled off of, but BFGS is
actively harmful on some other landscapes (e.g. confirmed harmful near the Inc prior wall in an
earlier design). Empirically BFGS wins on ~15% of starts — worth the extra cost, kept.

Polish is **fully unbounded** (as of 2026-08-15, after a brief period bounding it at `[-20,20]`).
Reverted because the bound cost a genuine minimum on `x/(x-pow(x,x)/x)` (that function's true joint
optimum sits at `rho0~1e22`, past any bound short of the actual, data-dependent ridge edge — any fixed
bound trades some functions for others the same unpredictable way widening DIRECT's own box does).

**Weird case — float64-overflow escapes.** Because polish is unbounded, Nelder-Mead/BFGS can walk a
parameter to the edge of float64 representable range (`~1.8e308`) or beyond (`inf`). Confirmed live: a
case where the best DIRECT point sat exactly at both box walls simultaneously, and unbounded polish
walked straight through into `rho0≈1.7977e+308` — literally float64's max value. This can produce an
excellent-looking loss built on a fragile, arguably-untrustworthy numeric escape rather than a "clean"
optimum. Per Alicia's explicit call: **judge the optimizer on whether it finds the true minimum of the
loss it's given, not on whether that minimum looks physically sensible** — the latter is a
priors/bounds question, deliberately out of scope for optimizer-trust purposes (see
`MEMORY.md`'s "weird answer vs. genuine minimum" entry).

**Weird case — the degenerate DM-vanishing escape (a related but distinct pathology).** Separately
from raw numeric overflow, `rho0->0` or `rs->huge` can make the dark-matter term's contribution to
predicted `v_circ²` vanish at every observed radius — the shape function then carries *no real
information*, and the fit compensates by leaning entirely on the (fixed) baryon term, inflated via
`Inc` pinned near its 90° prior wall and `D` dragged many sigma from its catalog prior. This is a
defect in the *objective* (nothing stops it from being profitable), not the search — confirmed present
in both DIRECT and a from-scratch random search alike, so no search-strategy change alone fixes it.
`dm_likelihood.py::degenerate_dm_penalty` (added 2026-08-14, see §6) is the actual fix: a smooth
log-barrier penalty, using the *best-explained* radius (not average/worst, so a function that's
baryon-dominated at some radii but genuinely DM-informative at even one radius escapes the penalty).

**Weird case (found + fixed 2026-08-15) — `log_opt=False` crashed.** `_loss_of_internal` (the
JAX-jitted BFGS objective) used to unconditionally apply the `sign*10**x` log-transform, even when
`log_opt=False` (where `signs=None` for every DIRECT-derived start and the internal coordinate already
*is* the physical value). `jnp.asarray(None, dtype=float)` raises, so this crashed on literally the
first BFGS call of the first polish start, every time — meaning `log_opt=False` would fail 100% of
functions in a real run. **Not live in production** (the one real entry point, `run_esr_mightee.py`,
always passes `log_opt=True` explicitly), but `main()`'s own default parameter was the crashing value.
Fixed by mirroring `dm_likelihood.py::wrapped_like`'s existing correct `if signs is None: p = x.copy()`
branch. A second, related bug was masked by this crash and only became reachable once fixed: the
final physical-value conversion checked the *global* `log_opt` flag rather than the winning start's
own `signs` — but the "random" starts (1d, item 5) always carry real signs regardless of `log_opt`, so
a winning random start under `log_opt=False` would have been saved un-converted (its raw log-magnitude
coordinate saved as if it were already physical). Fixed together.

### 1f. Selecting shape candidates for stage 2 (not just the single best)

After polishing, rather than keep only the single overall-best shape (by stage-1 loss), the code keeps
the top `n_shape_candidates=3` *distinct* polished shape solutions (deduplicated by
`shape_candidate_min_sep=2.0` decades of separation in internal coordinates — no loss-magnitude
cutoff, so a candidate that looks *worse* on stage-1 loss alone is exactly what this is meant to
rescue). Added 2026-08-14 after finding a basin that scored slightly worse on stage-1 loss alone was
worth 20-250 nats better once Inc/D were free to adapt to it in stage 2 — the shape-only ranking and
the joint ranking aren't the same ordering.

### 1g. Stage 2 — joint Inc/D polish (`galaxy_params_polish`)

For **each** of the (up to 3) shape candidates, `galaxy_params_polish` runs a fully joint,
unbounded Nelder-Mead optimization over `[shape/rho0/rs, Inc, D]` together — this time *with* the
Inc/D priors included in the loss (`include_priors=True`; see §6). It's tried from **6 fixed Inc
seeds** (catalog mean + `10, 25, 45, 65, 80`, deliberately avoiding the 0/90 boundary where seeds get
stuck), keeping whichever seed converges to the lowest actual loss; D is not multi-seeded (every
converged result in testing landed at essentially the same D regardless of Inc seed).

**Weird case — a single Inc seed isn't enough, but more isn't free either.** A single catalog-mean
seed can converge to a *non-stationary point*: Nelder-Mead reports "success" because its
mixed-scale simplex (dominated by huge `rho0`/`rs` values) satisfies its own tolerance while barely
moving Inc at all — confirmed directly (`a0+1/x`: single-seed stage2_chi2=88.74 vs. an 11-seed
sweep's genuine 37.59). But *more* seeds cost more: a brief period cut this to 1 seed (reasoning: with
a *good* shape already in hand, multi-seed Inc gain was noise-level, ~0.17-2.16 nats) regressed a full
comp=3 check by 1-10 nats on the best-fitting functions — restored to 6.

Whichever of the (shape-candidate × Inc-seed) combinations gives the lowest joint loss wins —
`winning_k` in the code. `optimise_fun_direct_nm` returns `(chi2_i, full_params, ..., inc_fit, d_fit,
stage2_chi2)`.

**Weird case (found + fixed 2026-08-15) — the returned shape didn't match the returned Inc/D
whenever a non-default candidate won.** `full_params` (the saved/returned shape) used to *always* be
candidate 0's shape (the single best-by-stage-1-loss point), regardless of which candidate actually
won stage 2 — so whenever `winning_k != 0`, the saved `(shape, Inc, D)` triple described a point that
was never jointly evaluated together. Confirmed live on a real case: the saved shape (candidate 0,
`a0~2.3e54`) paired with Inc/D that actually came from candidate 2 (`a0~-2.8e118`, a *completely
different* pathological escape point) — self-consistent-*looking* output, actually two different
optima glued together. Fixed: when `winning_k != 0`, save `shape_candidates[winning_k]` — the exact
seed that *produced* the winning Inc/D — not `galaxy_params_polish`'s *output* shape for that
candidate (tried first; doesn't work, since `galaxy_params_polish` isn't a fixed point of its own
result — re-seeding from its output landed on a nearby but measurably different optimum).

### Saving to disk

`main()` writes one row per function to `chi2_comp{N}weights_{rank}.dat` (concatenated across MPI
ranks into `negloglike_comp{N}.dat`): `[stage1_chi2, shape+rho0/rs params (padded), Inc, D,
stage2_chi2, Nconv, Niter, time]`. The shape params are saved in a **padded layout**:
`[shape values, padded with zeros up to max_param_shape=4, then rho0, rs]` — e.g. a 1-shape-parameter
function's saved row has 6 numbers: `[a0, 0, 0, 0, rho0, rs]`. This padding exists so every function
at a given complexity has the same-width row (comp's own `max_param_shape = max(4, floor((comp-1)/2))`
scales up for very high complexity) — but it is the direct cause of the padded-vs-compact confusion
described in §3.

---

## 2. `test_all_Fisher.py::convert_params` — Fisher matrix + description length, per unique function

Runs once per *unique* (canonical) function, computing the pieces that feed the final ranking.

### The Minimum Description Length (MDL) formula

Per the ESR paper (Bartlett, Desmond & Ferreira 2023): total description length
`DL = -logL(data | theta_MLE) + codelen(theta_MLE) + aifeyn(function)`. `codelen` is the cost of
specifying the fitted parameters to just enough precision that a reader could reconstruct
them — derived by Taylor-expanding `-logL` around the MLE and choosing an encoding resolution `Delta`
that minimizes total expected length. For a flat-prior parameter (the shape/`rho0`/`rs` block, true
ignorance), this reduces to `codelen = log(|theta|/Delta)` with `Delta = 1/sqrt(Fisher_diag)`. For
Inc/D (which have a *real* informative prior — the catalog value), the equivalent derivation gives
`codelen = -log(prior_density(theta_ML)) + 0.5*log(Fisher_diag)` — this is `galaxy_params_codelen`,
computed separately from the shape codelen and added on top.

**This is exactly why `-logL` and `codelen_galaxy` must be evaluated at the *same* point** — they're
two halves of one Taylor expansion around one MLE. Getting this wrong was 2026-08-15's critical fix
(§5).

### The Hessian, and the padding trap ("Bug #10", fixed 2026-07-30)

The Fisher matrix is `Hmat = hessian(-logL)` at the (now correctly joint, post-2026-08-15) shape
point. `get_deriv` reshuffles this into a fixed `max_param × max_param` layout (padding shape to 4
slots, extras after) so every function's saved Hessian has the same shape regardless of `nshape`.
**Weird case:** an earlier version sliced the *input* `theta_ML` as `theta_ML[:nparam]` — only correct
when `nshape == max_fun_param`; for the (common) case `nshape < 4`, this read *past* the real shape
values into the zero-padding and appended `rho0`/`rs` a second time on top, corrupting the Hessian for
nearly every low-complexity function. Fixed by explicitly taking `nshape` real values then the extras
from their true (not padding-adjacent) position.

### Snap-to-zero

If a shape parameter's magnitude is smaller than its own encoding resolution (`Nsteps = |theta|/Delta
< 1`), the data doesn't actually resolve it from zero — reporting a nonzero value for it would cost
more bits than it's worth. The code tries setting it (and combinations of such parameters) to exactly
zero and re-checking whether the loss stays finite, keeping whichever configuration gives the lowest
total description length. **Weird case — this is where a naive implementation produces *negative*
codelen**: `log(|theta|/Delta)` diverges to `-infinity` as `theta->0` if `Delta` isn't also capped.
Confirmed concretely on SPARC data: a raw codelen of -24.6 nats for an unresolved parameter that
should have scored +0.69 once correctly capped — the fix (`cutoff_Delta`) caps `Delta` at `|theta|`
for any parameter that still has `Nsteps<1` after the snap-to-zero search exhausts its options,
turning a runaway-negative artifact into a correctly-bounded zero-information contribution.

**Weird case — this cap also neutralizes information the codelen was supposed to carry.** Because
*every* unresolved parameter gets `Delta` capped at exactly `|theta|`, it always contributes exactly
`log(1) = 0` to the codelen — regardless of how differently unresolved two different parameters
actually are. Confirmed directly comparing a physically-sane NFW fit against a numerically pathological
one: both get *identical* per-parameter codelen contributions despite wildly different actual Fisher
uncertainties, because the safety cap erases the distinction. Not a bug to fix (the alternative,
unbounded negative codelen, is strictly worse) — a known, accepted limitation of the MDL formalism as
implemented here.

### The 2026-08-15 self-consistency fix

Before today, `convert_params` re-derived `shape_fit_joint`/`inc_ML`/`d_ML` (the true joint optimum —
see the 2026-08-07 fix below) but only ever *used* it for `galaxy_params_codelen`; the Hessian and the
reported `negloglike`/`params` still came from the pre-joint `theta_ML`, with Inc/D silently defaulted
to catalog (the only thing `get_loss(include_priors=False)` could do). Fixed by (a) adding
`fixed_galaxy_params` to `get_loss` so the pure likelihood can be evaluated at an explicit non-catalog
Inc/D, and (b) using `shape_fit_joint` everywhere in this function, not just for the galaxy codelen.
Quantified: on `a0+1/x`, the previously-reported `negloglike=77.18` was 48 nats worse than the true
joint value (28.76) — see the development notes for the full before/after regression.

### The 2026-08-07 fix this one builds on

`galaxy_params_polish` moves the shape parameters jointly with Inc/D, not just Inc/D alone — but its
return signature (at the time) only surfaced Inc/D. Pairing the resulting Inc/D with the *caller's*
pre-polish shape evaluated the galaxy-term Hessian at a point the optimizer never actually visited, so
its curvature had no reason to be positive-definite — root-caused as the cause of most
negative-Fisher-for-Inc cases seen at the time (`a0*x`: fisher_inc flips from -17.35 to +0.377 once
evaluated self-consistently). Fixed then by having `galaxy_params_polish` also return the shape it
actually landed on (`shape_fit_joint`), and by both `convert_params` and `match.py` re-deriving it
fresh (re-running the same deterministic optimization, guaranteed to reproduce the same Inc/D) rather
than trusting a stale value passed in.

---

## 3. `match.py::main()` — extending results to every duplicate-group member

ESR's function library contains many algebraically-equivalent rewrites of the same underlying
function (e.g. `a0*x` and `x/a0`, related by `a0 -> 1/a0`) — only *one* member per group (the
"canonical" form) actually gets fit by `test_all.py`; the rest are recovered by substitution. `match.py`
iterates over *every* function (including duplicates), looks up its canonical index, and reparametrizes
the canonical fit into that row's own symbolic form via `simplifier.convert_params` + a substitution
table (`inv_subs`).

### The reparametrization, and why it's cached

Since multiple rows share one canonical index, the expensive part (re-deriving `shape_fit_joint` via a
fresh `galaxy_params_polish` call — same 2026-08-07 mechanism as §2) is cached per canonical index, not
re-run per row; only the (cheap) symbolic reparametrization runs per row.

**Weird case (2026-08-07 incident, still the standing cautionary example) — a silently wrong
substitution mislabeled the winning function.** `simplifier.convert_params`'s `n=` argument was passed
`max_fun_params` (4) instead of the correct `max_param_total` (6, since MIGHTEE has 2 extra physical
params) for any row actually needing a real substitution — raising a shape-mismatch error, silently
caught, falling back to the *canonical* function's own params (wrong function, wrong point). This
alone caused `x/a0` to show a spuriously finite codelen while `a0*x`'s own, correctly-paired result
legitimately showed `codelen=nan` — `combine_DL.py`'s tie-break then picked the bogus finite value,
mislabeling the winning function for that canonical slot. Fixed by correcting the `n=` argument; the
underlying "no logging on the except path" gap this incident depended on being silent is exactly
what 2026-08-15's Finding B closed (see below).

### The 2026-08-15 self-consistency fix, and what it caught

Before today, this file computed the reparametrized shape **twice**, from two different sources: once
from `measured` (`test_all.py`'s raw, pre-joint saved shape — used for the shape/rho0/rs codelen and
the reported output `params`) and once from `shape_fit_joint` (used only for `galaxy_params_codelen`,
mirroring §2's own before-fix pattern). Fixed by unifying to one reparametrization, always from
`shape_fit_joint`, reused everywhere in the row — plus `fixed_galaxy_params` on the row's own
`loss_template`, and a fresh `negloglike_all[i] = fop(p)` computed once up front (previously, when a
row needed no snap-to-zero at all — the common case — `negloglike_all[i]` was *never* updated from the
stale value loaded off disk at the top of the loop; same class of gap, found and fixed in the same
pass).

**Bug discovered while verifying this fix — pre-existing, not introduced by it: `fop` needs the
*compact* parameter vector, but was being called with the *padded* one.** `chi2_fcn`/`get_pred`
read `rho0` at index `[nshape]` (the compact position, right after the real shape values) — but
match.py's `p` is padded (`rho0` fixed at index `[4]` regardless of `nshape`). Whenever `nshape<4`,
`fop(p)` was silently reading a padding zero as `rho0`, forcing `valid_scale=False` and reporting
`+inf` for a perfectly good point. This was **already present in the old code** but dormant — the old
code only called `fop` with the padded vector inside the snap-to-zero branch, which apparently never
triggered for comp=3's specific functions (every pre-2026-08-15 comp=3 `-logL` came from
`test_all.py`'s own untouched, correctly-compact computation, never from this function). Making the
fresh-negloglike computation unconditional (needed for the fix above) exposed it immediately. Fixed at
the true source — `fop` itself now compacts via `active_idx` before calling `chi2_fcn` — so its two
existing callers (`eval_total`, and `get_sigma_from_integral`'s boundary-search, both of which
correctly need the *padded* view for their own per-index bookkeeping) needed no changes.

### `eval_total` and the snap-to-zero comparison (this file's own version)

Same idea as §2's snap-to-zero, but explores subsets combinatorially (baseline no-snap, snap-all,
then every subset in between) rather than a single greedy pass, keeping whichever candidate gives the
lowest total `(negloglike + codelen)`. Same `Delta_capped` negative-codelen-prevention mechanism as §2.

**Weird case (found 2026-08-15, deliberately NOT fixed — different subsystem) —
`inv_subs=[nan]`.** All five comp=3 two-shape-parameter functions (`a0+a1`, `a0*a1`, `a0-a1`, `a0/a1`,
`pow(Abs(a0),a1)`) have a malformed substitution-table entry that makes `simplifier.convert_params`
raise, *regardless of what input values are passed* (confirmed by reproducing the identical failure
with arbitrary, unrelated inputs) — meaning this was **already silently hitting the reparametrization
fallback before today**, just invisibly (same bare `except`, no logging). Only became visible because
2026-08-15's Finding B added logging to that exact except block. The root cause is in
`inv_subs_3.txt` (the ESR function-matching data, generated by a different subsystem entirely, outside
the fitting pipeline) — not chased further today. **Practical consequence: don't trust either the old
or new reported value for these 5 functions until this is separately fixed** — both numbers come from
the same broken fallback path, just with different (both wrong) inputs.

---

## 4. `combine_DL.py::main()` — final ranking

For each *unique* canonical function, collects every duplicate-group member's own `(negloglike,
codelen)` row (computed by §3) and picks the member with the lowest total
`DL = negloglike + codelen + aifeyn` (`aifeyn` is a fixed per-operator complexity cost from the ESR
function-generation step, unrelated to the fitting). Writes `results_pretty_{N}.txt`.

**Why this tie-break matters:** it's the mechanism by which, e.g., `x/a0` vs. `a0*x` (§3's cautionary
example) compete to represent one canonical slot — a silently-wrong reparametrization in *either* row
can win the slot for the wrong reason, which is exactly why §3's self-consistency and logging fixes
matter for the table a reader actually sees.

---

## 5. `dm_likelihood.py::MIGHTEELikelihood` — the physics

### `get_pred` — predicted rotation velocity

```
v_circ(r)^2 = inc_ratio2 * baryon_term(r) + dm_term(r)

  baryon_term(r) = (D/distance_true) * v_bar(r)^2         -- fixed, from photometry, D-rescaled
  dm_term(r)     = G * M(<r) / r,  M(<r) from rho(r) = rho0 * f(r/rs)   -- the candidate function
  inc_ratio2     = (sin(deg2rad(Inc)) / sin(deg2rad(inc_true)))^2       -- catalog-relative correction
```

Confirmed directly (independent derivation, not just reading the code): if `Vobs` is deprojected from
a fixed line-of-sight velocity as `V_LOS/sin(i)`, moving the equivalent correction onto the *model*
side requires multiplying the model `v²` by `[sin(i_test)/sin(i_cat)]²` — growing toward 90°, exactly
the implemented direction. No sign error.

**Weird case (resolved 2026-08-24) — the correction applies to the baryon term only, not the whole
`v²`.** This is a deliberate, precedent-matched choice (verified identical in both
`CLASH_SPARC/dm_likelihood.py` and `SPARC/esr/fitting/dm_likelihood.py`) — not a MIGHTEE-specific
choice if it's wrong, a shared convention across three codebases. No citation to a specific
methodology paper was found in the repo, but the asymmetry has a structural justification confirmed
directly against the code (Alicia's reasoning, verified): `self.v_bar` (the baryon term) is not
measured directly, it's *derived* elsewhere (outside this codebase) from projected photometry,
deprojected to a face-on mass model using some assumed inclination — so if the catalog's assumed
inclination is off, `v_bar` itself is off, and correcting it by `inc_ratio2` compensates for exactly
that. The dark matter term never goes through an inclination-dependent derivation: `r_obs` (the
physical radius its density profile is integrated over) depends *only* on distance
(`r_obs = x * D/distance_true`, confirmed directly — no `Inc` anywhere in it), and `rho(r)` is a
freshly-evaluated candidate function, not a pre-derived catalog quantity. So the two terms are
asymmetric by construction, not by oversight — one inherits an inclination assumption baked in at
creation time, the other doesn't. Still no literal source-paper citation, but this is now a
structurally-grounded explanation, not just "matches precedent."

### `include_priors` — three related but distinct modes

- `include_priors=False`, `fixed_galaxy_params=None` (default): `a` is shape+`rho0`/`rs` only; Inc/D
  default to catalog. This is stage 1's own objective, and (before 2026-08-15) the *only* thing
  `convert_params`/`match.py` could evaluate for the ranking `-logL` — the root of the critical fix.
- `include_priors=False`, `fixed_galaxy_params=(Inc, D)` (added 2026-08-15): same shape-only `a`, but
  Inc/D come from the given values instead of catalog — still **no** prior cost added (that stays
  exclusively `galaxy_params_codelen`'s job, so nothing gets double-counted). This is what makes
  §2/§3's self-consistency fix possible.
- `include_priors=True`: `a` is `[shape/rho0/rs..., Inc, D]` together, and the loss adds the Inc/D
  prior terms. This is stage 2's own joint-polish objective (`galaxy_params_polish`), and also what
  `galaxy_params_codelen`'s own Hessian uses (so the prior's curvature floors the Fisher information
  even when the data itself is uninformative about Inc/D).

### The degenerate-DM penalty (added 2026-08-14)

`degenerate_dm_penalty(dm_v2, baryon_v2)`: a smooth log-barrier, `0` once the DM term's fractional
contribution to `v_circ²` at its *best-explained* radius clears a threshold (`0.05`), growing
unboundedly as it shrinks toward 0 — directly targets the "vanishing DM term, compensate via Inc/D"
pathology from §1. Verified: wired into the total loss (not discarded), correctly signed (pure
penalty, never a reward), and doesn't over-penalize genuinely DM-dominated fits (uses the max, not
mean/min, across radii, so a function only needs to carry real information *somewhere*).

**Weird case (found + fixed 2026-08-15) — two unguarded divisions risked NaN gradients, not
values.** `degenerate_dm_penalty`'s `dm_v2/(dm_v2+baryon_v2+1e-10)`, and `get_pred`'s own
`u = r_obs/rs`: both can hit `inf/inf` or `x/0` when `rho0`/`rs` escape to invalid values.
`get_pred`'s `mass = jnp.where(valid_scale, mass, jnp.inf)` correctly masks the *value* in both
cases — but `jnp.where` evaluates both of its branches, so the NaN produced computing the untaken
branch still poisons the *gradient*, a standard JAX gotcha. Confirmed live (fed real escape-point
params through `get_loss(value='grad')`, saw NaN, fixed, saw finite). Previously papered over by two
different, independent downstream guards (`nan_to_num` in `test_all.py`, an `isnan` check on
`Fisher_diag` in `test_all_Fisher.py`) rather than fixed at the source — any future caller of
`get_loss(value='grad'/'hessian')` that didn't replicate one of those guards would have gotten
silently corrupted gradients with no warning from this file itself. Fixed with the same
"compute-a-safe-value-first" idiom already used elsewhere in this file (`sin_inc_true_safe`).

### Priors on Inc, D

Truncated-Gaussian, anchored at catalog values (`inc_true`/`e_inc`, `distance_true`/`e_d`), correctly
signed (pulls fit *toward* catalog, verified — not a sign that would push away, which would be very
hard to notice from chi2 values alone). Bounds `[0°,90°]` for Inc, `[0,∞)` for D — unit-consistent
throughout (Inc always in degrees, converted to radians exactly once per use site; D always in kpc).

---

## Cross-cutting weird cases (span multiple files)

- **Padded vs. compact parameter layout** is the single most common source of subtle bugs in this
  pipeline — three genuinely separate bugs (test_all_Fisher's "Bug #10", 2026-07-30; match.py's `n=`
  argument, 2026-08-07; match.py's `fop` call, 2026-08-15) all trace back to the same root cause: some
  arrays are padded to a fixed width (`[shape (padded to 4), rho0, rs]`, used for on-disk storage and
  for `active_idx`-based bookkeeping) and some are compact (`[shape (nshape long), rho0, rs]`, used
  for anything actually calling into the likelihood/`eq_numpy`), and nothing in the type system
  distinguishes them. Whenever a new call site is added, check which convention it needs.
- **Self-consistency of `(shape, Inc, D)` triples** is the second recurring theme — any time these
  three are computed at genuinely different points (a stale save, a re-derivation seeded from the
  wrong thing, two separate reparametrizations of the same canonical index) and then combined as if
  they described one point, the result looks plausible but isn't real. Three instances found and
  fixed this way in one week (2026-08-07, plus two more today).
- **Silent `except Exception` fallbacks** are how every one of the above stayed invisible for as long
  as it did. The standing lesson (now applied everywhere in `match.py`): a fallback that silently
  substitutes a plausible-looking wrong value is worse than one that's loud about firing — log the
  exception, even in a hot loop, since the caught exceptions are rare precisely in the cases where
  they matter most.
