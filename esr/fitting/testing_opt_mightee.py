import numpy as np
import sympy
from scipy.optimize import minimize
from itertools import product
import jax.numpy as jnp
from esr.fitting.mightee_likelihood import MIGHTEELikelihood

# -------------------------------------------------
# Configuración fácil de cambiar
# -------------------------------------------------
DATA_FILE = "rar_with_ml_direct_phot.csv"
GALAXY_NAME = "J022128.8-042448"
RUN_NAME = "test_mightee"

FUNCTIONS_TO_TRY = [
    "a0/(1 + x/a1)",
    "a0*exp(-x/a1)",
    "a0/(x + a1)",
    "a0/(1 + pow(x, a1))",
]

METHODS_TO_TRY = [
    "BFGS",
    "Nelder-Mead",
    "Powell",
]

TRY_INTEGRATION = True
N_STARTS = 5
USE_SIGN_COMBINATIONS = True
MAX_PARAM = 4


# -------------------------------------------------
# Helpers
# -------------------------------------------------
def build_equation(eq_string, nparam):
    """
    Parse the ESR string and build a JAX-lambdified function.
    """
    eq_string = eq_string.replace("\n", "").replace("'", "")

    eq = sympy.sympify(
        eq_string,
        locals={
            "inv": inv,
            "square": square,
            "cube": cube,
            "sqrt": sqrt,
            "log": log,
            "exp": exp,
            "pow": pow,
            "x": x,
            "a0": a0,
            "a1": a1,
            "a2": a2,
        }
    )

    if nparam > 1:
        all_a = list(sympy.symbols(" ".join([f"a{i}" for i in range(nparam)]), real=True))
        eq_numpy = sympy.lambdify([x] + all_a, eq, modules=["jax"])
    elif nparam == 1:
        eq_numpy = sympy.lambdify([x, a0], eq, modules=["jax"])
    else:
        eq_numpy = sympy.lambdify(x, eq, modules=["jax"])

    return eq, eq_numpy


def random_initial_guess(nparam, low=-2.0, high=2.0):
    return np.random.uniform(low, high, size=nparam)


def run_scipy_optimize(loss_fn, x0, method="BFGS"):
    """
    loss_fn must return (loss, grad) for gradient-based methods.
    """
    if method in ["BFGS", "L-BFGS-B", "CG", "Newton-CG", "trust-constr"]:
        res = minimize(
            fun=lambda p: float(loss_fn(p)[0]),
            x0=x0,
            jac=lambda p: np.array(loss_fn(p)[1]),
            method=method,
            options={"maxiter": 2000}
        )
    else:
        res = minimize(
            fun=lambda p: float(loss_fn(p)[0]),
            x0=x0,
            method=method,
            options={"maxiter": 2000}
        )
    return res


def best_of_many_starts(loss_fn, nparam, method, n_starts=5, use_signs=True):
    """
    Try several random starts and optional sign combinations.
    Returns best scipy result.
    """
    best_res = None
    best_fun = np.inf

    sign_list = [None]
    if use_signs and nparam > 0:
        sign_list = list(product([1, -1], repeat=nparam))

    for signs in sign_list:
        for _ in range(n_starts):
            x0 = random_initial_guess(nparam)

            if signs is not None:
                x0 = np.abs(x0)

            res = run_scipy_optimize(loss_fn, x0, method=method)

            if res.fun < best_fun:
                best_fun = res.fun
                best_res = res

    return best_res


# -------------------------------------------------
# Main fitting loop
# -------------------------------------------------
def fit_one_function(likelihood, fcn_string, method="BFGS", try_integration=True, n_starts=5):
    """
    Fit one ESR function to one galaxy.
    """
    nparam = simplifier.count_params([fcn_string], MAX_PARAM)[0]

    fcn_string, eq, integrated = likelihood.run_sympify(
        fcn_string,
        try_integration=try_integration
    )

    # Build symbolic function/lambdified version
    _, eq_numpy = build_equation(fcn_string, nparam)

    # Loss with asymmetric errors and density positivity check
    loss_template = likelihood.get_loss(eq_numpy, integrated, value="value_and_grad")

    def loss_fn(params):
        return loss_template(
            jnp.array(params),
            likelihood.xvar,
            likelihood.yvar,
            likelihood.yerr_lo,
            likelihood.yerr_hi
        )

    # Optimize
    res = best_of_many_starts(
        loss_fn=loss_fn,
        nparam=nparam,
        method=method,
        n_starts=n_starts,
        use_signs=USE_SIGN_COMBINATIONS
    )

    return {
        "function": fcn_string,
        "method": method,
        "nparam": nparam,
        "integrated": integrated,
        "chi2": float(res.fun),
        "params": np.array(res.x),
        "success": bool(res.success),
        "message": res.message if hasattr(res, "message") else "",
    }


def fit_many_functions(likelihood, functions, methods):
    results = []

    for fcn in functions:
        for method in methods:
            print(f"\nFitting: {fcn} | method = {method}")
            out = fit_one_function(
                likelihood=likelihood,
                fcn_string=fcn,
                method=method,
                try_integration=TRY_INTEGRATION,
                n_starts=N_STARTS,
            )
            results.append(out)
            print("  chi2 =", out["chi2"])
            print("  params =", out["params"])

    return results


# -------------------------------------------------
# Example usage
# -------------------------------------------------
likelihood = MIGHTEELikelihood(
    data_file=DATA_FILE,
    name=GALAXY_NAME,
    run_name=RUN_NAME
)

results = fit_many_functions(likelihood, FUNCTIONS_TO_TRY, METHODS_TO_TRY)

# Best result
best = min(results, key=lambda d: d["chi2"])
print("\nBEST:")
print(best)