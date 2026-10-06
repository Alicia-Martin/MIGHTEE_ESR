import math
import numpy as np
from mpi4py import MPI
import os, sys
from prettytable import PrettyTable
import csv

import esr.fitting.test_all as test_all

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def _fmt_bounded(x, prec=2):
    """'%.{prec}f' for the common case, but falls back to scientific
    notation once |x| is large enough that fixed-decimal notation would
    blow up the column width -- a diverged/pathological fit (e.g. a
    negloglike that overflowed to ~1e178) formatted with plain '%.2f'
    prints out its FULL literal decimal expansion (~180 digits), and
    PrettyTable then stretches EVERY row's column to match that one
    worst-ranked row, making results_pretty_{N}.txt hundreds of
    characters wide and unreadable in a normal terminal even though only
    one row (usually near the bottom of the ranking, where nobody's
    looking anyway) was ever that extreme.
    """
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return str(x)
    if np.isfinite(xf) and abs(xf) >= 1e6:
        return f"%.{prec}e" % xf
    return f"%.{prec}f" % xf


def main(comp, likelihood, print_frequency=1000):
    """Combine the description lengths of all functions of a given complexity, sort by this and save to file.
    
    Args:
        :comp (int): complexity of functions to consider
        :likelihood (fitting.likelihood object): object containing data, likelihood functions and file paths
        :print_frequency (int, default=1000): the status of the fits will be printed every ``print_frequency`` number of iterations
    
    Returns:
        None
    
    """
    if likelihood.is_mse:
        raise ValueError('Cannot use MSE with description length')
    
    if rank == 0:
        print('\nComputing description lengths', flush=True)

    unifn_file = likelihood.fn_dir + "/compl_%i/unique_equations_%i.txt"%(comp,comp)
    allfn_file = likelihood.fn_dir + "/compl_%i/all_equations_%i.txt"%(comp,comp)
    aifeyn_file = likelihood.fn_dir + "/compl_%i/%s%i.txt"%(comp,likelihood.fnprior_prefix,comp)

    use_deriv = False

    with open(unifn_file, "r") as f:         # All
        fcn_list = f.read().splitlines()

    with open(allfn_file, "r") as f:         # All
        fcn_list_all = f.read().splitlines()

    # match.py's codelen_matches_comp{comp}.dat column layout:
    # [negloglike, codelen, index, params(max_param_total), Deltas(max_param_total),
    #  Inc, D, Nconv, Niter, times]
    # Computed explicitly (matching test_all.py/test_all_Fisher.py's own formula)
    # rather than inferred from file width via a fragile [:,3:-3] slice --
    # that slice previously (silently, pre-dating this round's Inc/D work)
    # lumped the Deltas block into "params" too, since match.py's params+Deltas
    # blocks were always both present; the table's header only ever had 4
    # hardcoded "a{i}" names regardless of the real column count, so this was
    # already a latent header/data mismatch, not something introduced now.
    max_param_shape = int(max(4, np.floor((comp - 1) / 2)))
    n_extra = 2 if getattr(likelihood, "use_physical_scale", False) else 0
    max_param_total = max_param_shape + n_extra
    num_galaxy_params = getattr(likelihood, "num_galaxy_params", 0)

    data = np.genfromtxt(likelihood.out_dir + "/codelen_matches_comp"+str(comp)+".dat") # All
    data = np.atleast_2d(data)
    negloglike = data[:,0]
    codelen = data[:,1]
    index = data[:,2]
    params = data[:, 3:3 + max_param_total]
    inc_col = data[:, 3 + 2 * max_param_total]
    d_col = data[:, 3 + 2 * max_param_total + 1]
    Nconv = data[:,-3]
    Niter = data[:,-2]
    time = data[:, -1]
    # print('neg', negloglike[Niter==24.00])

    aifeyn = np.genfromtxt(aifeyn_file) # All
    codelen = np.atleast_1d(codelen)
    index = np.atleast_1d(index)
    aifeyn = np.atleast_1d(aifeyn)
    Nconv = np.atleast_1d(Nconv)
    Niter = np.atleast_1d(Niter)
    time = np.atleast_1d(time)

    fcn_list_proc, data_start, data_end = test_all.get_functions(comp, likelihood)
    # print(fcn_list_proc)

    DL_min = np.zeros(len(fcn_list_proc))
    params_min = np.zeros((len(fcn_list_proc), params.shape[1]))  # These are all now specific to the proc
    inc_min = np.full(len(fcn_list_proc), np.nan)
    d_min = np.full(len(fcn_list_proc), np.nan)

    fcn_min = [None] * len(fcn_list_proc)
    negloglike_min = np.zeros(len(fcn_list_proc))
    codelen_min = np.zeros(len(fcn_list_proc))
    aifeyn_min = np.zeros(len(fcn_list_proc))
    Nconv_min = np.zeros(len(fcn_list_proc))
    Niter_min = np.zeros(len(fcn_list_proc))
    time_min = np.zeros(len(fcn_list_proc))

    xarr = np.linspace(0, len(fcn_list)-1, len(fcn_list)).astype(int)           # Indices of all the unique fcns, which are what we're looping over
    xarr_proc = xarr[data_start:data_end]        # Which unique function indices this proc will look at

    for i in range(len(fcn_list_proc)):          # Loop over all unique fcns to find variant with min codelength
        if rank==0 and i%print_frequency==0:
            print(f'{i+1} of {len(fcn_list_proc)}', flush=True)
        # print(fcn_list_proc[i])
        negloglike_i, codelen_i, aifeyn_i = negloglike[index==xarr_proc[i]], codelen[index==xarr_proc[i]], aifeyn[index==xarr_proc[i]]           # Arrays of all variants for this unique fcn
        # print(fcn_list_proc[i], negloglike_i, codelen_i, aifeyn_i)
        Nconv_i = Nconv[index==xarr_proc[i]]
        Niter_i = Niter[index==xarr_proc[i]]
        time_i = time[index==xarr_proc[i]]
        m = (index==xarr_proc[i])
        fcn_list_all_i = [fcn_list_all[j] for j in range(len(m)) if m[j]]
        # print(fcn_list_all_i)
        params_i = params[index==xarr_proc[i], :]
        inc_i = inc_col[index==xarr_proc[i]]
        d_i = d_col[index==xarr_proc[i]]
        DL = negloglike_i + codelen_i + aifeyn_i
        # print(DL)

        if np.sum(~np.isnan(DL))==0:
            DL_min[i] = np.nan
            aifeyn_min[i] = aifeyn_i[np.nanargmin(aifeyn_i)]
            # print(fcn_list_proc[i], aifeyn_min[i])
            # print(fcn_list_proc[i], 'has no finite DL', flush=True)
            continue

        DL_min[i] = np.nanmin(DL)
        params_min[i,:] = params_i[np.nanargmin(DL),:]
        inc_min[i] = inc_i[np.nanargmin(DL)]
        d_min[i] = d_i[np.nanargmin(DL)]
        fcn_min[i] = fcn_list_all_i[np.nanargmin(DL)]
        
        negloglike_min[i] = negloglike_i[np.nanargmin(DL)]
        codelen_min[i] = codelen_i[np.nanargmin(DL)]
        aifeyn_min[i] = aifeyn_i[np.nanargmin(DL)]
        Nconv_min[i] = Nconv_i[np.nanargmin(DL)]
        Niter_min[i] = Niter_i[np.nanargmin(DL)]
        time_min[i] = time_i[np.nanargmin(DL)]

        print(fcn_min[i], negloglike_min[i], codelen_min[i])

    # [DL_min, params_min(max_param_total), Inc, D, negloglike_min, codelen_min,
    #  aifeyn_min, Nconv_min, Niter_min, time_min] -- Inc/D inserted right after
    # params_min; the trailing 6-column block's relative order/composition is
    # unchanged from before, so the negative-index reload below still works.
    out_arr = np.transpose(np.vstack(
        [DL_min] + [params_min[:,i] for i in range(params_min.shape[1])]
        + [inc_min, d_min]
        + [negloglike_min, codelen_min, aifeyn_min] + [Nconv_min, Niter_min, time_min]
    ))
    prefix = likelihood.combineDL_prefix

    np.savetxt(likelihood.temp_dir + '/'+prefix+str(comp)+'_'+str(rank)+'.dat', out_arr, fmt='%.16e')        # Save the data for this proc in Partial
    np.savetxt(likelihood.temp_dir + '/'+prefix+'fcn_'+str(comp)+'_'+str(rank)+'.dat', fcn_min, fmt="%s")
    # One per unique eqn, but I save the form in "all" that gives the lowest DL

    comm.Barrier()

    if rank == 0:
        string = 'cat `find ' + likelihood.temp_dir + '/ -name "'+prefix+str(comp)+'_*.dat" | sort -V` > ' + likelihood.out_dir + '/'+prefix+'comp'+str(comp)+'.dat'
        os.system(string)
        string = 'rm ' + likelihood.temp_dir + '/'+prefix+str(comp)+'_*.dat'
        os.system(string)
        
        string = 'cat `find ' + likelihood.temp_dir + '/ -name "'+prefix+'fcn_'+str(comp)+'_*.dat" | sort -V` > ' + likelihood.out_dir + '/'+prefix+'fcn_comp'+str(comp)+'.dat'
        os.system(string)
        string = 'rm ' + likelihood.temp_dir + '/'+prefix+'fcn_'+str(comp)+'_*.dat'
        os.system(string)
        
    if rank==0:         # The rest is done by just one proc
        data = np.genfromtxt(likelihood.out_dir + '/'+prefix+'comp'+str(comp)+'.dat')            # This is the combined results from all procs, and the rest should be as before
        data = np.atleast_2d(data)
        DL_min = data[:,0]
        params_min = data[:,1:1+params.shape[1]]
        inc_min = data[:, 1 + params.shape[1]]
        d_min = data[:, 1 + params.shape[1] + 1]
        negloglike_min = data[:,-6]
        codelen_min = data[:,-5]
        aifeyn_min = data[:,-4]
        Nconv_min = data[:,-3]
        Niter_min = data[:,-2]
        time_min = data[:,-1]

        DL_min = np.atleast_1d(DL_min)
        params_min = np.atleast_2d(params_min)
        inc_min = np.atleast_1d(inc_min)
        d_min = np.atleast_1d(d_min)
        negloglike_min = np.atleast_1d(negloglike_min)
        codelen_min = np.atleast_1d(codelen_min)
        aifeyn_min = np.atleast_1d(aifeyn_min)
        Nconv_min = np.atleast_1d(Nconv_min)
        Niter_min = np.atleast_1d(Niter_min)
        time_min = np.atleast_1d(time_min)



        with open(likelihood.out_dir + '/'+prefix+'fcn_comp'+str(comp)+'.dat', "r") as f:         # All
            fcn_min = f.read().splitlines()

        mask = ~np.isnan(DL_min)

        xarr = np.linspace(0, len(fcn_list)-1, len(fcn_list)).astype(int)           # fcn_list should be as it was read in at the top

        DL_min = DL_min[mask]
        xarr = xarr[mask]

        if len(DL_min) > 0:
            arr_sort = np.transpose( sorted( np.transpose(np.vstack([DL_min, xarr])), key = lambda x: x[0] ) )     # Sort by DL but keep track of array indices
            DL_sort = arr_sort[0,:]
            indices_sort = arr_sort[1,:].astype(int)

            params_sort = params_min[indices_sort,:]
            inc_sort = inc_min[indices_sort]
            d_sort = d_min[indices_sort]
            fcn_min_sort = [fcn_min[i] for i in indices_sort]

            negloglike_sort = negloglike_min[indices_sort]
            codelen_sort = codelen_min[indices_sort]
            aifeyn_sort = aifeyn_min[indices_sort]
            Nconv_sort = Nconv_min[indices_sort]
            Niter_sort = Niter_min[indices_sort]
            time_sort = time_min[indices_sort]

        else:
            negloglike_sort = []
            codelen_sort = []
            aifeyn_sort = []
            DL_sort = []
            Nconv_sort = []
            Niter_sort = []
            time_sort = []
            inc_sort = []
            d_sort = []


        if os.path.exists(likelihood.out_dir + '/'+likelihood.final_prefix+str(comp)+'.dat'):           # Start this file from scratch here
            os.remove(likelihood.out_dir + '/'+likelihood.final_prefix+str(comp)+'.dat')

        # Nfuncs = 10
        Nfuncs = len(fcn_list)

        Prel_DL = np.zeros(len(negloglike_sort))+np.inf
        negloglike_list = []                    # Store all unique negloglikes
        for i in range(len(negloglike_sort)):
            if negloglike_sort[i] in negloglike_list:       # Never happens for 0th fcn bc negloglike would have to be nan
                continue                                        # Prel_DL stays at inf for this duplicate function, so Prel -> 0
            negloglike_list += [negloglike_sort[i]]
            Prel_DL[i] = DL_sort[i] - DL_sort[0]                # Always gives 0 for the 0th function, so this gets the highest Prel

        Prel = np.exp(-Prel_DL)             # Don't want to use every fcn here bc they could be inf or nan, but the best 1000 should be fine
        Prel /= np.sum(Prel)                # Relative probability of fcn, normalised over the top 1000 functions just of this complexity

        ptab = PrettyTable()
        # a{i} header count matches max_param_total (shape params, padded to
        # max_param_shape, plus rho0/rs if use_physical_scale) exactly --
        # previously hardcoded to 4 regardless of the real column count.
        names = ["Rank", "Function", "L(D)", "Prel", "-logL", "Codelen", "AIFeyn"] + [f"a{i}" for i in range(max_param_total)]

        # Galaxy nuisance parameters (Inc, D) -- MIGHTEELikelihood has no
        # bulge/Reff-scaling/rho0-scaling concept (those were CLASH_SPARC-
        # specific attributes that don't exist here; removed).
        if num_galaxy_params != 0:
            names += ["Inc", "D"]

        #Time and other things
        names += ["Nconv", "Niter", "Time"]

        ptab.field_names = names

        for i in range(len(DL_sort)):
            
            # Only happens for non-duplicates; all Prels should be non-zero
            if i < Nfuncs:
                # Combine all data into a single list
                row_data = [i+1, fcn_min_sort[i], _fmt_bounded(DL_sort[i]), '%.2e'%Prel[i], _fmt_bounded(negloglike_sort[i]), _fmt_bounded(codelen_sort[i]), '%.2e'%aifeyn_sort[i]]
                row_data += ['%.2e'%params_sort[i,j] for j in range(params.shape[1])]
                if num_galaxy_params != 0:
                    row_data += [_fmt_bounded(inc_sort[i]), _fmt_bounded(d_sort[i])]
                row_data += ['%.2f'%Nconv_sort[i], '%.2f'%Niter_sort[i], '%.2f'%time_sort[i]]

                # Add the row to the table
                ptab.add_row(row_data)

            csv_row = [i, fcn_min_sort[i], DL_sort[i], Prel[i], negloglike_sort[i], codelen_sort[i], aifeyn_sort[i]] + [params_sort[i,j] for j in range(params.shape[1])]
            if num_galaxy_params != 0:
                csv_row += [inc_sort[i], d_sort[i]]
            with open(likelihood.out_dir + '/'+likelihood.final_prefix+str(comp)+'.dat', 'a') as f:
                writer = csv.writer(f, delimiter=';')
                writer.writerow(csv_row)
        
        if len(DL_sort) == 0:
            os.system("touch " + likelihood.out_dir + '/'+likelihood.final_prefix+str(comp)+'.dat')
        
        print(ptab)
        
        with open(likelihood.out_dir + '/results_pretty_'+str(comp)+'.txt', 'w') as f:
            print(ptab, file=f)
    
    comm.Barrier()
        
    return
    
