import sys
import matplotlib.pyplot as plt
import csv
from mpi4py import MPI
import warnings
import numpy as np
import sympy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import os

from esr.fitting.sympy_symbols import *
import esr.generation.simplifier as simplifier
import jax.numpy as jnp

warnings.filterwarnings("ignore")

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def main(comp, likelihood, tmax=5, try_integration=False, xscale='linear', yscale='linear'):
    """Plot best 50 functions at given complexity against data and save plot to file
    
    Args:
        :comp (int): complexity of functions to consider
        :likelihood (fitting.likelihood object): object containing data, functions to convert SR expressions to variable of data and output path
        :tmax (float, default=5.): maximum time in seconds to run any one part of simplification procedure for a given function
        :try_integration (bool, default=False): when likelihood requires integral, whether to try to analytically integrate (True) or just numerically integrate (False)
        :xscale (str), default='linear'): Scaling for x-axis
        :yscale (str), default='linear'): Scaling for y-axis
        
    Returns:
        None
    
    """

    if rank != 0:
        return
    
    print('\nMaking plots', flush=True)

    vmin = 1e-3
    vmax = 1
    tmax = 5

    if comp>=8:
        sys.setrecursionlimit(2000 + 500 * (comp - 8))

    if not os.path.isdir(likelihood.fig_dir):
        print('Making:', likelihood.fig_dir)
        os.mkdir(likelihood.fig_dir)

    with open(likelihood.out_dir + '/final_'+str(comp)+'.dat', "r") as f:
        reader = csv.reader(f, delimiter=';')
        data = [row for row in reader]
    
    if len(data) == 0:
        print("No functions with finite DL found, so will not make figure")
        return
    
    max_param = len(data[0]) - 7
    print(data)
    fcn_list = [d[1] for d in data]
    params = np.array([d[-max_param:] for d in data], dtype=float)
    DL = np.array([d[2] for d in data], dtype=float)
    DL_min = np.amin(DL[np.isfinite(DL)])
    alpha = DL_min - DL
    alpha = np.exp(alpha)
    m = (alpha > vmin)
    fcn_list = [d for i, d in enumerate(fcn_list) if m[i]]
    params = params[m,:]
    alpha = alpha[m]

    fig  = plt.figure(figsize=(7,5))
    ax1  = fig.add_axes([0.10,0.10,0.70,0.85])
    cmap = cm.hot_r
    norm = mpl.colors.LogNorm(vmin=vmin,vmax=vmax)

    def get_params(a, optimise_scaling, optimise_reff):
        # a = a[0]
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
            rho = likelihood.rho0
            if likelihood.Lbul == 0:
                gamma_bul = 0
                Inc, D, gamma_disk, gamma_gas, Reff = a[-5:]
                params = a[:nparam]
            else:
                Inc, D, gamma_disk, gamma_gas, gamma_bul, Reff = a[-6:]
                params = a[:nparam]

        else:
            rho = 1
            Reff = 1
            if likelihood.Lbul == 0:
                gamma_bul = 0
                Inc, D, gamma_disk, gamma_gas = a[-4:]
                params = a[:nparam]
            else:
                Inc, D, gamma_disk, gamma_gas, gamma_bul = a[-5:]
                params = a[:nparam]
        return Inc, D, gamma_disk, gamma_gas, gamma_bul, Reff, rho, params

    for i in range(min(len(fcn_list),50)):

        fcn_i = fcn_list[i].replace('\'', '')
        
        k = simplifier.count_params([fcn_i], max_param)[0]

        print('%i of %i:'%(i+1,len(fcn_list)), fcn_i)

        measured = params[i,:]
        
        # try:
        fcn_i, eq, integrated = likelihood.run_sympify(fcn_i, tmax=tmax, try_integration=try_integration)
        
        if k == 0:
            eq_numpy = sympy.lambdify([x], eq, modules=["numpy"])
        elif k > 1:
            all_a = ' '.join([f'a{i}' for i in range(k)])
            all_a = list(sympy.symbols(all_a, real=True))
            eq_numpy = sympy.lambdify([x] + all_a, eq, modules=["numpy"])
        else:
            eq_numpy = sympy.lambdify([x, a0], eq, modules=["numpy"])
        Inc, D, gamma_disk, gamma_gas, gamma_bul, Reff, rho, a = get_params(measured, likelihood.optimise_scaling, likelihood.optimise_reff)
        r = likelihood.xvar/Reff*D/likelihood.distance_true
        ypred = likelihood.get_pred(r, np.atleast_1d(a), Reff, D, likelihood.distance_true, eq_numpy, integrated=integrated)*jnp.array(rho, dtype=jnp.float64)
        v = likelihood.get_vel(gamma_disk, gamma_bul, gamma_gas, Inc, D, ypred, r)
            # ypred = likelihood.get_pred(likelihood.xvar, measured, eq_numpy, integrated=integrated)

        # except:
        #     if try_integration:
        #         fcn_i, eq, integrated = likelihood.run_sympify(fcn_i, tmax=tmax, try_integration=False)
        #         if k > 0:
        #             all_a = ' '.join([f'a{i}' for i in range(k)])
        #             all_a = list(sympy.symbols(all_a, real=True))
        #             eq_numpy = sympy.lambdify([x] + all_a, eq, modules=["numpy"])
        #         else:
        #             eq_numpy = sympy.lambdify([x], eq, modules=["numpy"])
        #             Inc, D, gamma_disk, gamma_gas, gamma_bul, Reff, rho, a = get_params(measured, False, False)
        #             r = likelihood.xvar/Reff*D/likelihood.distance_true
        #             ypred = likelihood.get_pred(r, np.atleast_1d(a), eq_numpy, integrated=integrated)*rho
        #             v = likelihood.get_vel(gamma_disk, gamma_bul, gamma_gas, Inc, D, ypred, r)
        #             # ypred = likelihood.get_pred(likelihood.xvar, measured, eq_numpy, integrated=integrated)
        #     else:
        #         continue

        if np.isscalar(v):
            ax1.plot(likelihood.xvar, [v]*len(likelihood.xvar), color=cmap(norm(alpha[i])), zorder=len(fcn_list)-i)
        else:
            ax1.plot(likelihood.xvar, v, color=cmap(norm(alpha[i])), zorder=len(fcn_list)-i)
    # print(v)
    if hasattr(likelihood, 'yerr'):
        ax1.errorbar(likelihood.xvar, likelihood.yvar, yerr=likelihood.yerr, fmt='.', markersize=5, zorder=len(fcn_list)+1, capsize=1, elinewidth=1, color='k', alpha=1)
    else:
        ax1.plot(likelihood.xvar, likelihood.yvar, '.', color='k', ms=5, zorder=len(fcn_list)+1, alpha=1)
    ax1.set_xlabel(r'$x$')
    ax1.set_ylabel(likelihood.ylabel)
    ax1.set_xscale(xscale)
    ax1.set_yscale(yscale)
    if xscale != 'log':
        ax1.set_xlim(0, None)
    ax1.set_ylim(likelihood.yvar.min() * 0.9, likelihood.yvar.max() * 1.1)

    ax2  = fig.add_axes([0.85,0.10,0.05,0.85])
    cb1  = mpl.colorbar.ColorbarBase(ax2,cmap=cmap,norm=norm,orientation='vertical')
    cb1.set_label(r'$\exp \left( MDL - DL \right)$')
    fig.tight_layout()
    # plt.show()
    fig.savefig(likelihood.fig_dir + '/plot_%i.png'%comp)
    fig.clf()
    plt.close(fig)

    return