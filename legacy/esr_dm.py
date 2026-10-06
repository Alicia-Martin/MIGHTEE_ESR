import numpy as np
import os
import sys
from mpi4py import MPI
import pickle
import scipy
from sympy import *
import sympy
import time
import matplotlib.pyplot as plt
import jax.numpy as jnp
import pandas as pd
import itertools
from tabulate import tabulate

import esr.fitting.test_all
import esr.fitting.test_all_Fisher
import esr.fitting.match
import esr.fitting.combine_DL
import esr.fitting.plot
from esr.fitting.likelihood import Likelihood
from esr.generation.simplifier import time_limit
from esr.fitting.sympy_symbols import *
import esr.plotting.plot
import esr.generation.simplifier as simplifier
from esr.fitting.fit_single import fit_from_string
import jax_cosmo as jc
import jax
from tools import cumtrapz
from jax import jit
import math
from esr.fitting.dm_likelihood import DMLikelihood


import astropy
import astropy.units as apu

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# os.environ["OMP_NUM_THREADS"] = "1"
    
def run_fit_nfw(likelihood, try_integration):
        basis_functions = [["x", "a"],  # type0
                ["inv", "abs"],  # type1
                ["+", "*", "-", "/", "pow"]]  # type2

        logl_lcdm_cc, dl_lcdm_cc, labels, params = fit_from_string("a0/(x*(1+x)^2)",
                                                        basis_functions,
                                                        likelihood,
                                                        Niter=100,
                                                        Nconv=70,
                                                        try_integration = try_integration,
                                                        verbose=True,
                                                        log_opt=log_opt,
                                                        pmin=-1,
                                                        pmax= 10)
        
        # print(logl_lcdm_cc)

        return params

def run_fit_single(likelihood, method, try_integration, log_opt, rho0, optimise_scaling, optimise_reff, fn):
        basis_functions = [["x", "a"],  # type0
                ["inv", "abs", "log", "exp"],  # type1
                ["+", "*", "-", "/", "pow"]]  # type2
        
        #Functions to try
        # fn = "1/(a0 + x)"
        # fn = "pow(Abs(a0),(-x))"
        # fn = 'a0 + 1/x'
        # fn = "-x + 1/x"
        # fn = 'pow(Abs(a0), (1/x))'
        # fn = 'pow((Abs(a0)/x),(pow(x,x)))'
        # fn = 'a0 /(x + Abs(a1))'
        # fn = 'pow(Abs(a0 + x),a1)'
        # fn = 'a0/(x*(a1+x)^2)'
        # fn = 'a0 - Abs(a1)*x'
        # fn = 'pow(Abs(a0),(a1 + x))'

        #Optimise fun
        start = time.time()
        # fn, eq, integrated = likelihood.run_sympify(fn, try_integration=False)
        # eq = eq*likelihood.rho0
        # eq_numpy = sympy.lambdify([x, a0], eq, modules=["jax"])
        # create_grid(likelihood, eq_numpy, integrated)
        # sys.exit()
        logl_lcdm_cc, dl_lcdm_cc, labels, params, Niter, Nconv = fit_from_string(fn,
                                                        basis_functions,
                                                        likelihood,
                                                        method = method,
                                                        try_integration = False,
                                                        verbose=True,
                                                        log_opt=log_opt)
        end = time.time()
        time_total = end - start
        print('Time taken:', end - start)

        #get eq_numpy to compute likelihood
        fn, eq, integrated = likelihood.run_sympify(fn, try_integration=False)
        eq = eq*likelihood.rho0
        eq_numpy = sympy.lambdify([x, a0], eq, modules=["jax"])

        # nparam = likelihood.nparam
        # print('params', nparam)
        # max_fun_param = 4
        # theta_ML = jnp.append(params[:nparam], params[max_fun_param:])
        # grad_template = likelihood.get_loss(eq_numpy, integrated, value = 'grad')
        # grad = grad_template(theta_ML, likelihood.xvar, likelihood.yvar, likelihood.yerr)
        # print('grad', grad)

        # mass = likelihood.get_pred(likelihood.xvar, params, eq_numpy, integrated=integrated)*likelihood.rho0
        # velocity = likelihood.get_vel(params[2], params[3], params[4], params[0], params[1], mass, likelihood.xvar)
        # # print('mass', mass)
        # plt.plot(likelihood.xvar, likelihood.yvar)
        # plt.plot(likelihood.xvar, velocity)
        # plt.show()
        # params = [1.33965004e+06, -1.14379633e+01, 6.19497996e+01,  4.23307741e+00,  4.90349307e-01,  9.87322401e-01]
        # plot_results(likelihood, params, rho0, fn, optimise_scaling, optimise_reff)
        # params = [3.93575843e+06,0, 0, 0, 7.13069622e+01, 4.63515096e+00, 7.91427939e-01, 1.38925859e+00]


        # loss_template = likelihood.get_loss(eq_numpy, integrated, value = 'evaluate')
        # chi2_fcn =likelihood.get_wrapped_like(loss_template, False)

        # #sets pams to priors
        # p0 = np.arange(0, 10, 1)
        # p1 = np.arange(0, 10, 1)
        # p = [likelihood.distance_true, likelihood.inc_true, likelihood.gamma_disk_true, likelihood.gamma_gas_true]
        # negloglike = chi2_fcn(p, likelihood.xvar, likelihood.yvar, likelihood.yerr, signs= None)

        # compute_likelihood(likelihood, try_integration, params[0], params, eq_numpy, integrated)

        return params, logl_lcdm_cc, Niter, Nconv, time_total
        # return None, None, None, None, None

def create_grid(likelihood, eq_numpy, integrated):
    loss_template = likelihood.get_loss(eq_numpy, integrated, value = 'evaluate')
    chi2_fcn =likelihood.get_wrapped_like(loss_template, False)

    p0_range = np.arange(0, 10, 1)
    p1_range = np.arange(0, 10, 1)

    # Initialize an empty grid to store likelihood values
    likelihood_grid = np.zeros((len(p0_range), len(p1_range)))

    for i, p0 in enumerate(p0_range):
        for j, p1 in enumerate(p1_range):
            p = [p0, p1, likelihood.distance_true, likelihood.inc_true, likelihood.upsilon_disk_true, likelihood.upsilon_gas_true]
            print(p)
            p = [1.81238623e+00, -5.38413757e+00,  1.04811428e+02,  9.06619711e+00, 1.03872760e-01,  1.76500494e+00]
            negloglike = chi2_fcn(p, likelihood.xvar, likelihood.yvar, likelihood.yerr, signs=[1,1])
            likelihood_grid[i, j] = negloglike

    # Plot the heatmap
    plt.figure(figsize=(8, 6))
    plt.imshow(likelihood_grid)
    plt.colorbar(label='Negative Log-Likelihood')
    plt.xlabel('p1')
    plt.ylabel('p0')
    plt.title('Likelihood Grid')
    plt.grid(False)
    plt.show()

def compute_likelihood(likelihood, try_integration, a0, params, eq_numpy, integrated):
        # params = [3.93575843e+06, 7.13069622e+01, 4.63515096e+00, 7.91427939e-01, 1.38925859e+00]
        loss_template = likelihood.get_loss(eq_numpy, integrated, value = 'evaluate')
        chi2_fcn =likelihood.get_wrapped_like(loss_template, False)

        negs = jnp.array([])
        # a0s =jnp.arange( 0.001,  20,  10**-2)
        # a0s = jnp.linspace(params[0]-10, params[0] + 10, 1000)
        # a0s = jnp.append(a0s, params[0])
        nparams = likelihood.nparam
        # print(params)
        # print(likelihood.xvar)
        a0s = np.linspace(1*10**6, 6*10**6, 100)

        for a0 in a0s:
            p_params = np.copy(params)
            p = jnp.append(p_params[:nparams],p_params[4:])
            p = p.at[0].set(a0)
            negloglike = chi2_fcn(p, likelihood.xvar, likelihood.yvar, likelihood.yerr, signs= None)
            print(negloglike)
            negloglike = jnp.array(negloglike)
            negs =jnp.append(negs, negloglike)
        # print(negs)
        plt.plot(a0s,jnp.exp(-negs + jnp.min(negs)))
        # plt.xscale('log')
        # plt.plot(a0s,negs)
        # p = np.append(params[:nparams],params[4:])
        # min_neg = chi2_fcn(p, likelihood.xvar, likelihood.yvar, likelihood.yerr)
        # print('min', negs, min_neg)
        plt.show()

def plot_results(likelihood, a, rho0, fn, optimise_scaling, optimise_reff):

    if rho0 == None:
        rho0 =1
        Reff = 1
    else:
        Reff = likelihood.Reff

    fcn, eq, integrated = likelihood.run_sympify(fn,
                                            tmax=5,
                                            try_integration=False)

    eq_numpy = sympy.lambdify([x, a0], eq, modules=["jax"])
    print('eq', eq)

    nparam = likelihood.nparam

    if optimise_scaling and optimise_reff :
        if likelihood.Lbul == 0:
            gamma_bul = 0
            Inc, D, gamma_disk, gamma_gas, Reff, rho = a[-6:]
            params = a[:nparam]
        else:
            Inc, D, gamma_disk, gamma_gas, gamma_bul, Reff, rho = a[-7:]
            params = a[:nparam]

    elif optimise_scaling:
        Reff = 1
        if likelihood.Lbul == 0:
            gamma_bul = 0
            Inc, D, gamma_disk, gamma_gas, rho = a[-5:]
            params = a[:nparam]
        else:
            Inc, D, gamma_disk, gamma_gas, gamma_bul, rho = a[-6:]
            params = a[:nparam]

    elif optimise_reff:
        rho = 1
        if likelihood.Lbul == 0:
            gamma_bul = 0
            Inc, D, gamma_disk, gamma_gas, Reff = a[-5:]
            params = a[:nparam]
        else:
            Inc, D, gamma_disk, gamma_gas, gamma_bul, Reff = a[-6:]
            params = a[:nparam]

    else:
        rho = likelihood.rho0
        Reff = likelihood.Reff
        if likelihood.Lbul == 0:
            gamma_bul = 0
            Inc, D, gamma_disk, gamma_gas = a[-4:]
            params = a[:nparam]
        else:
            Inc, D, gamma_disk, gamma_gas, gamma_bul = a[-5:]
            params = a[:nparam]

    r = likelihood.xvar/Reff*D/likelihood.distance_true
    # print('r', r)
    # x, a, Reff, D, distance_true, eq_numpy, integrated
    print('params', params)
    ypred = likelihood.get_pred(r, [params], Reff, D, likelihood.distance_true, eq_numpy, integrated=integrated)*rho

    v = likelihood.get_vel(gamma_disk, gamma_bul, gamma_gas, Inc, D, ypred, r)

    plt.errorbar(r, likelihood.yvar, likelihood.yerr)
    plt.plot(r, v)
    plt.show()


def fit_function(name, options, fun, method):
    start = time.time()
    try_integration = False

    #OPTIONS
    #1. scale R by Reff and rho by a rho_0 from NFW
    #2. take reff and rho to be params to optimise
    #3. optimise reff
    #4 optimise rho
    #5. optimise in log scale
    #6. log space but optimise reff

    # options = 5

    # print('hola', method, options, fun, name)
    # options = options[0]

    if options == 1:
        likelihood_NFW = DMLikelihood('df.pkl', 'df1.pkl', None, False, False, True, name, name + '_local_params', data_dir=os.getcwd())
        params_nfw = run_fit_nfw(likelihood_NFW, try_integration)
        
        log_opt = False
        optimise_scaling = False
        optimise_reff = False
        Reff = True
    elif options ==2:
        print('Optimising rho and reff')
        params_nfw = [None]
        Reff = False
        optimise_scaling = True
        optimise_reff = True
        log_opt = False

    elif options ==3:
        print('Optimising reff')
        params_nfw = [None]
        Reff = False
        optimise_scaling = False
        optimise_reff = True
        log_opt = False

    elif options ==4:
        print('Optimising rho')
        params_nfw = [None]
        Reff = False
        optimise_scaling = True
        optimise_reff = False
        log_opt = False
    elif options ==5:
        params_nfw = [None]
        Reff = False
        optimise_scaling = False
        optimise_reff = False
        log_opt = True
    elif options ==6:
        params_nfw = [1]
        Reff = False
        optimise_scaling = False
        optimise_reff = True
        log_opt = True
    else:
        params_nfw = [1]
        Reff = False
        optimise_scaling = False
        optimise_reff = False
        log_opt = False


    likelihood = DMLikelihood('df.pkl', 'df1.pkl', params_nfw[0], optimise_scaling,  optimise_reff, Reff, name, name + '_local_params', data_dir=os.getcwd())   
    params, logl_lcdm_cc, Niter, Nconv, time_total = run_fit_single(likelihood, method, try_integration, log_opt, params_nfw[0], optimise_scaling, optimise_reff, fun)
   
    return fun, method, Niter, Nconv, options , logl_lcdm_cc, time_total


def fit_galaxy(name, comp):

    start = time.time()
    try_integration = False

    #OPTIONS
    #1. scale R by Reff and rho by a rho_0 from NFW
    #2. take reff and rho to be params to optimise
    #3. optimise reff
    #4 optimise rho
    #5. optimise in log scale
    #6. log space but optimise reff


    options = 7
    # method = "SBPLX"
    method = "BFGS"
    # fn = 'a0/(a1 - x)'
    # fn = 'a0/x'
    # fn = 'a0 + 1/x'
    # fn = 'a0 - Abs(a1)*x'
    # fn = 'a1 + exp(Abs(a0))'
    # fn = 'exp(pow(Abs(log(x)),x))'
    # fn = 'exp(- x)'
    # fn ='log(a0/(x + a1)'
    # fn = 'log(exp(a0)/(x + Abs(a1)))'
    # fn = 'exp(-1/log(Abs(a0)))'
    # fn = 'log(a0/(1 + (a1*x)**2))'
    fn = 'x + log(exp(a0))'
    # fn = 'log(a0/(a1*x*(a2 + a3*x)**3)) '
    # fn = 'pow(Abs(a0),(a1 + 1/x))'
    # fn = '1/(a0 + a1*x)'
    # fn = 'pow(Abs(a0),(Abs(a1) - x))'
    # fn = 'pow(Abs(a0 + x),a1)'
    # fn = 'pow(Abs(a0),(a1 - x))'
    # fn = 'a0-1'
    # fn = 'pow(Abs(a0),(pow(x,x)))'
    # fn = 'a0/(x/a1*(1+x/a1)^2)'

    print(method, options, fn, name)
    if options == 1:
        likelihood_NFW = DMLikelihood('df.pkl', 'df1.pkl', None, False, False, True, name, name + '_local_params', data_dir=os.getcwd(), fn_set = 'base_e_maths')
        params_nfw = run_fit_nfw(likelihood_NFW, try_integration)
        
        log_opt = False
        optimise_scaling = False
        optimise_reff = False
        Reff = True
    elif options ==2:
        print('Optimising rho and reff')
        params_nfw = [None]
        Reff = False
        optimise_scaling = True
        optimise_reff = True
        log_opt = False

    elif options ==3:
        print('Optimising reff')
        params_nfw = [None]
        Reff = False
        optimise_scaling = False
        optimise_reff = True
        log_opt = False

    elif options ==4:
        print('Optimising rho')
        params_nfw = [None]
        Reff = False
        optimise_scaling = True
        optimise_reff = False
        log_opt = False
    elif options ==5:
        params_nfw = [None]
        Reff = False
        optimise_scaling = False
        optimise_reff = False
        log_opt = True
    elif options ==6:
        params_nfw = [1]
        Reff = False
        optimise_scaling = False
        optimise_reff = True
        log_opt = True
    else:
        params_nfw = [1]
        Reff = False
        optimise_scaling = False
        optimise_reff = False
        log_opt = False


    likelihood = DMLikelihood('df.pkl', 'df1.pkl', params_nfw[0], optimise_scaling,  optimise_reff, Reff, name, name + '_local_params', data_dir=os.getcwd())   

    params, logl_lcdm_cc, Niter, Nconv, time_total = run_fit_single(likelihood, method, try_integration, log_opt, params_nfw[0], optimise_scaling, optimise_reff, fn)
   
    # esr.fitting.test_all.main(comp, likelihood,tmax=60, try_integration=try_integration, log_opt=log_opt, method=method)
    # end_opt = time.time()

    # start_fisher = time.time()
    # esr.fitting.test_all_Fisher.main(comp, likelihood, tmax=5, try_integration=try_integration)
    # end_fisher = time.time()
    # start_match = time.time()
    # esr.fitting.match.main(comp, likelihood, tmax=5, try_integration=try_integration)
    # end_match = time.time()
    # esr.fitting.combine_DL.main(comp, likelihood)

    # end = time.time()

    # if rank == 0:
    #     print('Time taken:', end - start)
    #     print('Optimisation took', end_opt - start)
    #     print('Fisher took', end_fisher - start_fisher)
    #     print('Match took', end_match - start_match)

    # esr.fitting.plot.main(comp, likelihood, tmax=5, try_integration=try_integration)

    # esr.plotting.plot.pareto_plot('fitting/output/output_' + name + '_local_params', 'pareto.png', do_DL=True, do_logL=True)


def compare_methods(funcs, methods, options, name):
    # Initialize results table
    results = []

    # Iterate over all combinations of method, option, and function
    for func, method, opt in itertools.product(funcs, methods, options):
        if opt == 5:
            log_opt = True
        else:
            log_opt = False
        # opt = 2
        print(func, method, opt)

        fun, method, Niter, Nconv, options , logl_lcdm_cc, time_total = fit_function(name, opt, func, method)

        convergence = Nconv/Niter

        N_needed =  -6/np.log10(1 - convergence)

        projected_time = time_total/Niter * N_needed
        
        # Store results in a dictionary
        result_dict = {
            'Function': func,
            'Method': method,
            'Log opt': log_opt,
            'Convergence': convergence,
            'Log Likelihood': logl_lcdm_cc,  # Example value, replace with your log likelihood
            'N_convergence': N_needed,
            'Projected time': projected_time
        }

        # Append result to the results table
        results.append(result_dict)

    # Write results to a file
    with open("optimization_results.txt", "w") as file:
        for result in results:
            file.write(str(result) + "\n")

    return results

def create_table(results):
    table_data = []
    current_function = results[0]['Function']

    # Define table headers
    headers = ['Function', 'Method', 'Log opt', 'Convergence', 'Log Likelihood', 'N_convergence', 'Projected time']

    for result in results:
        # Add horizontal line if function changes
        if result['Function'] != current_function:
            table_data.append(['-' * 30] * len(headers))
            current_function = result['Function']
        
        # Add result data
        row = [result['Function'], result['Method'], result['Log opt'],
            result['Convergence'], result['Log Likelihood'], result['N_convergence'], result['Projected time']]
        table_data.append(row)

    # Create a pretty table
    table = tabulate(table_data, headers=headers, tablefmt="pretty")

    # Print or save the table
    # print(table)
    return table


###########################################################################
                                 # MAIN #
###########################################################################
    
#------------------------------------------------------------
#run code for all galaxies
#------------------------------------------------------------
    
# with open("galaxy_names.txt", "r") as file:
# # with open("galaxies_left.txt", "r") as file:
#     galaxy_names = file.readlines()
#     galaxy_names = [x.strip() for x in galaxy_names]

# start_all = time.time()
# for i, name in enumerate(galaxy_names):
#     print(name, i, len(galaxy_names))
#     # name = 'NGC0100'
#     comp = 4
#     fit_galaxy(name, comp)
#     end_all = time.time()
# print('Total time taken:', end_all - start_all)

#------------------------------------------------------------
#run code for a single galaxy
#------------------------------------------------------------
    
# name = 'NGC0100' #
# name = 'UGC00128' ##
name = 'UGC07577' ##
# name = 'UGC02953' ##
# name = 'UGC00128'
# name = 'CamB'
# name = 'DDO154' ##
# name = 'DDO168'

comp = 5
start = time.time()
fit_galaxy(name, comp)
end = time.time()
print('Total time taken:', end - start)

#------------------------------------------------------------
#Compare methods with a table
#------------------------------------------------------------

# from math import inf
# # name = 'NGC0100' ##
# # name = 'CamB'
# # name = 'UGC07577'
# # name = 'DDO154'
# # name = 'UGC02953'
# name = 'UGC00128'
# methods = ["Nelder-Mead", "BFGS", "SBPLX"]

# with open("test_fun.txt", "r") as file:
#     funcs = [line.strip() for line in file.readlines()]
#     # funcs = [f'log({func})' for func in funcs]
#     # print(funcs)


# options = [2, 5]
# rerun = True

# if os.path.exists("optimization_results.txt") and not rerun:
#     with open("optimization_results.txt", "r") as file:
#         results = [eval(line) for line in file.readlines()]

# else:
#     results = compare_methods(funcs, methods, options, name)

# # Print results in a pretty table
# table = create_table(results)

# # Save table to a file
# with open('prettytable' + name + '_3.txt', 'w') as f:
#     f.write(table)


#Some other random things
# nparams_log = []
# n_convergence_log = []

# nparams = []
# n_convergence = []

# for result in results:
#     N_conv = result['N_convergence']
#     conv= result['Convergence']

#     fun = result['Function']
#     n_fun = simplifier.count_params([fun], 4)[0]
#     n_gal = 3
#     if result['Method'] == 'BFGS':
#         if result['Log opt']:
#             n_scale = 0
#             nparam_log = n_fun + n_gal + n_scale
#             nparams_log = np.append(nparams_log, nparam_log)
#             n_convergence_log = np.append(n_convergence_log, N_conv)
#         else:
#             n_scale = 2
#             nparam = n_fun + n_gal + n_scale
#             nparams = np.append(nparams, nparam)
#             n_convergence = np.append(n_convergence, N_conv)

# ------------------------------------------------------------
# Fit to all points
# ------------------------------------------------------------

# def f(x, A, B):
#     return A*x + B

# nparams = nparams[n_convergence != -np.inf]
# n_convergence = n_convergence[n_convergence!= -np.inf]
# nparams_log = nparams_log[n_convergence_log != -np.inf]
# n_convergence_log = n_convergence_log[n_convergence_log != -np.inf]

# popt, pcov = scipy.optimize.curve_fit(f, nparams, n_convergence)
# popt_log, pcov_log = scipy.optimize.curve_fit(f, nparams_log, n_convergence_log)

# plt.scatter(nparams, n_convergence)
# plt.scatter(nparams_log, n_convergence_log)
# # plt.plot(nparams, f(np.array(nparams), *popt), 'r-')
# # plt.plot(nparams_log, f(np.array(nparams_log), *popt_log), 'r-')
# plt.savefig('convergence_BFGS' + name + '.png')
# plt.show()


# ------------------------------------------------------------
# Take only the unique values of the maximum convergence values
# ------------------------------------------------------------


# Plot the scatter plot
# convergence_dict = {}
# convergence_dict_log = {}
# # Iterate over the (nparams, convergence) pairs and update the dictionary with maximum convergence values
# for nparam, conv in zip(nparams, n_convergence):
#     convergence_dict[nparam] = max(convergence_dict.get(nparam, 0), conv)

# # Iterate over the (nparams_log, convergence_log) pairs and update the dictionary with maximum convergence values
# for nparam_log, conv_log in zip(nparams_log, n_convergence_log):
#     convergence_dict_log[nparam_log] = max(convergence_dict_log.get(nparam_log, 0), conv_log)

# # Separate the unique nparams and nparams_log values along with their corresponding maximum convergence values
# unique_nparams, unique_convergence = zip(*sorted(convergence_dict.items()))

# # Separate the unique nparams and nparams_log values along with their corresponding maximum convergence values
# unique_nparams_log, unique_convergence_log = zip(*sorted(convergence_dict_log.items()))


# def f(x, A, B):
#     return A*x + B

# popt, pcov = scipy.optimize.curve_fit(f, unique_nparams[:-1], unique_convergence[:-1])
# # print(popt)

# popt_log, pcov_log = scipy.optimize.curve_fit(f, unique_nparams_log[:-1], unique_convergence_log[:-1])
# print(popt_log)

# # Plot the scatter plot using the unique values
# plt.scatter(unique_nparams, unique_convergence, label = 'rho0 and rscale')
# # plt.plot(unique_nparams, f(np.array(unique_nparams), *popt), 'r-')
# plt.scatter(unique_nparams_log, unique_convergence_log, label = 'log space')
# # plt.plot(unique_nparams_log, f(np.array(unique_nparams_log), *popt_log), 'r-')
# plt.legend()

# plt.show()
