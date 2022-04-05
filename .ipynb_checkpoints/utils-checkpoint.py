import os
import sys
import tskit
import msprime
import numpy as np
import pandas as pd
import seaborn as sns
from copy import deepcopy
import matplotlib.pyplot as plt
from IPython.display import SVG
from os.path import basename, abspath
from tqdm import tqdm
import pickle
import time


def run_bash_script(script):
    import subprocess
    return subprocess.check_output(script, shell=True, text=True, executable='/bin/bash')


def sleepy(
    n_simulations: int = 1,
    num_generations: int = 100000000,
    N: int = 500,
    m: int = 100,
    b: float = 1,
    gc: int = 50,
    r: float =5e-5, 
    L: int = 10000, 
    s: str = "1",
    d: str = "0.5", 
    selection_position: str = "5000", 
    selection_activation_generation: int = 100000,
    stop_after_mrca: bool = False, 
    output_name: str = "run", 
    generations_post_fixation_threshold: int = 0,
    add_mutations_after_fixation: bool = True,
    output_directory: str = "./", # make sure to end path with "/"
    n_parallel: int = 10, 
    continue_from: int = 0, 
    
    print_cmd: bool = False, # don't run program, but print bash script 
    slurm = False,
    slurm_start = 0, 
    slurm_divide = 2, 
    slurm_limit = 200
) -> None:
    
    """ Python-wrapper for sleepy simulator bash script generation. Also, capable of
    directly running simulations locally if print_cmd is false and slurm is false.
    
    Arg types: 
        * **n_simulations** *(int)* - Number of simulations if script is directly to be run locally.
        * **num_generations** *(int)* - Maximum number of generations (multiplied by gc).
        * **N** *(int)* - Diploid population size N=500 -> 1000 Haplotypes.
        * **m** *(int)* - Number of generations from which seeds can resusicate.
        * **b** *(float)* - Germination rate.
        * **gc** *(int)* - Garbage collection interval.
        * **r** *(int)* - Recombination rate.
        * **L** *(int)* - Mapping length.
        * **s** *(str)* - Selection coefficient.
        * **d** *(str)* - Dominance coefficient.
        * **selection_position** *(str)* - Selection position.
        * **selection_activation_generation** *(int)* - Generation from which to start introducing selective alleles.
        * **stop_after_mrca** *(bool)* - Stop simulation after MRCA is reached.
        * **output_name** *(str)* - Output name.
        * **generations_post_fixation_threshold** *(int)* - Number of generations to add after fixation event.
        * **add_mutations_after_fixation** *(bool)* - Adding mutation after first fixation event, necessary for recovery.
        * **output_directory** *(int)* - Output directory.
        * **n_parallel** *(int)* - Number of parallel simulation when running locally.
        * **continue_from** *(int)* - Continue simulations from run n if running locally.
        * **print_cmd** *(bool)* - Number of simulations if script is directly to be run locally.
        * **slurm** *(bool)* - Create part of slurm script for easy parallelization.
        * **slurm_start** *(int)* - Start slurm simulatons from.
        * **slurm_divide** *(int)* - Divid slurm_limit by.
        * **slurm_limit** *(int)* - Number of slurm simulations.
    """
    
    
    if output_directory[-1] != "/": output_directory += "/"
    
    debug_print=False
    kwargs = { 
        "num_generations" : num_generations,
        "N":N, 
        "m":m, 
        "b":b, 
        "gc":gc, 
        "r":r, 
        "L":L, 
        "selection_coefficient":s,
        "dominance_coefficient":d, 
        "selection_position":selection_position, 
        "selection_activation_generation":selection_activation_generation,
        "stop_after_mrca":stop_after_mrca, 
        "debug_print":debug_print, 
        "output_name":output_name, 
        "output_directory":output_directory,
        "generations_post_fixation_threshold": generations_post_fixation_threshold,
        "add_mutations_after_fixation":add_mutations_after_fixation 
    }
    
    cmd_arg_part = ""
    for k,v in zip(kwargs.keys(), kwargs.values()):
        cmd_arg_part += "--" + str(k) + " " + str(v) 
        if k == "output_name":
            if slurm:
                cmd_arg_part += str("_$(($i+$j))")
            else:
                cmd_arg_part += str("_$i")
        cmd_arg_part += " "
    
    if slurm:
        
        loop = "" 
        increment = 0 + slurm_start
        for i in range(slurm_divide):

            loop += str(int(increment)) + " "
            increment += slurm_limit / slurm_divide


        script = """for j in """ + loop + """; do 
        sleepy  """ + cmd_arg_part + """
        done"""
        
    else:    
        script = """

            n_simulations=""" + str(n_simulations) + """
            n_prog=""" + str(n_parallel) + """
            n_prog_count=0

            i=""" + str(continue_from) + """
            while [[ $i -lt $n_simulations ]]; do

            sleepy """ + cmd_arg_part + """ &
                 (( n_prog_count+=1 ))  
                 [[ $((n_prog_count%n_prog)) -eq 0 ]] && wait
                 (( i+=1 ))
            done
            """
    
    if print_cmd: print(script)
    else: return run_bash_script(script)



    



def subfig_diversity(models, ax, y_label=False, font_scale=0.8, legend_pos=-0.05):
    diversity_frame = pd.concat(models)
    sns.set(style= "white",palette="colorblind", font_scale=font_scale)  
    sns.lineplot(data=diversity_frame, x="positions", y="value", hue="model", ax=ax)
    if y_label: ax.set(xlabel='positions', ylabel='genetic diversity\n (tajima\'s pi)')
    else: ax.set(xlabel='positions', ylabel='')
        
        
    #ax.legend(loc='lower right')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])

    # Put a legend below current axis
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, legend_pos),
              fancybox=True, shadow=True)
    
    ax.grid(alpha=0.5)
    

def get_diversity(directory ,model, span_normalise=True, use=None, mode="branch", mutate=False, mu=5e-5):
    trees_path, _ = get_path_trees_recorders(directory)
    diversities = []
    
    for i in range(len(trees_path)):
        ts = tskit.load(trees_path[i])
        if mutate:
            ts = msprime.sim_mutations(ts, mu,discrete_genome=False, keep=False)
        windows = np.linspace(0, ts.sequence_length, 50)
        diversity = ts.diversity(windows=windows, span_normalise=span_normalise,   mode=mode )
        diversities.append(diversity)
    diversities = pd.DataFrame(diversities).fillna(0)
    diversities = diversities.T
    diversities['positions'] = windows[:-1]
    diversities = diversities.melt("positions")
    diversities['model'] = model
    return diversities

def get_path_trees_recorders(directory):
    files = np.array(os.listdir(directory))
    recorder_mask = [True if "csv" in i else False for i in files]
    trees_mask = [False if "csv" in i else True for i in files]
    recorders = files[recorder_mask].tolist()
    trees = files[trees_mask].tolist()
    trees.sort()
    recorders.sort()
    trees = [directory + i for i in trees]
    recorders = [directory + i for i in recorders]
    return trees, recorders


"""
def get_fixation_time(directory, model_name):
    csvs = get_path_trees_recorders(directory)[1]
    fixation_times = []
    for file in csvs:
        df = pd.read_csv(file)
        if df['active_mutations'].unique().shape[0] > 1:
            print(file, df['active_mutations'].unique())
            continue
        fixed_position = np.float64(df[df.abs_population_freq == df.abs_population_freq.max()].position)
        fixation_time = df[df['position'] == fixed_position].shape[0]
        fixation_times.append(fixation_time)
    fixation_times = pd.DataFrame([fixation_times, [model_name]*len(fixation_times)]).T
    fixation_times.columns = ["fixation_time", "model"]
    return fixation_times


def probability_fixation(directory, model_name):
    csvs = get_path_trees_recorders(directory)[1]
    num_mutations = []
    first_mutation_fixed = 0
    for file in csvs:
        df = pd.read_csv(file)
        num_mutation = len(df.position.unique())
        num_mutations.append(num_mutation)
    for i in num_mutations:
        if i == 1:
            first_mutation_fixed += 1
    first_mutation_fixed / len(num_mutations)
    probability_fixation = first_mutation_fixed / len(num_mutations)     
    probability_fixation = pd.DataFrame([probability_fixation, model_name]).T
    probability_fixation.columns = ["probability_fixation", "model"]
    return probability_fixation

def get_num_mutations(directory, model_name):
    files = get_path_trees_recorders(directory)[0]
    mutations = []
    for file in files:
        ts = tskit.load(file)
        genotype_matrix = ts.genotype_matrix()
        num_mutations = genotype_matrix[genotype_matrix.sum(1) != 1000].shape[0]
        mutations.append(num_mutations)
    mutations = pd.DataFrame([mutations, [model_name]*len(mutations)]).T
    mutations.columns = ["num_mutations", "model"]
    return mutations

"""

"""

###

import pandas as pd
import numpy as np
from collections import Counter



def get_fixed_and_all_mutations(file):
    
    df = pd.read_csv(file).sort_values("origin_generation").reset_index(drop=True)
    
    #last_mutation = df.position.iloc[-1]
    #df = df[df.position != last_mutation]
    
    fixed_candidates = df[df.abs_population_freq == 1000].position.tolist()
    fixed_candidates = pd.DataFrame.from_dict(dict(Counter(fixed_candidates)), orient="index").reset_index()
    fixed_candidates.columns = ["position", "num_fixed"]
    fixed_mutations = fixed_candidates[fixed_candidates.num_fixed >= 50].position.unique().tolist()
    
    all_mutations = df.position.unique().tolist()
    notfixed_mutations = []
    for m in all_mutations:
        if m not in fixed_mutations:
            notfixed_mutations.append(m)
            
    return fixed_mutations, all_mutations

def get_probability_fixation(file):
    fixed_mutations, all_mutations = get_fixed_and_all_mutations(file)
    return len(fixed_mutations) / len(all_mutations)

def get_fixation_times(file):
    
    fixed_mutations, _ = get_fixed_and_all_mutations(file)
    df = pd.read_csv(file).sort_values("origin_generation").reset_index(drop=True)
    last_mutation = df.position.iloc[-1]
    df = df[df.position != last_mutation]
    
    mask = [True if p in fixed_mutations else False for p in df.position]
    df_fixed = df[mask]
    
    fixation_times = []
    for m in fixed_mutations:
        fixation_times.append(df_fixed[df_fixed.position == m].shape[0])
        
    return fixation_times

def get_probability_fixation_table(file, germination_rate, selection_coefficient):
    prob = get_probability_fixation(file)
    table = pd.DataFrame([[prob, germination_rate, selection_coefficient]])
    table.columns = ["fixation probability", "germination rate", "selection coefficient"]
    return table

def get_time_fixation_table(file, b, c):
    df = pd.DataFrame(get_fixation_times(file))
    df['selection coefficient'] = c
    df['germination rate'] = b
    df.rename({0: "fixation time"}, axis=1, inplace=True)
    return df

"""

###


import pandas as pd
import numpy as np
from collections import Counter



def get_fixed_and_all_mutations(file):
    
    df = pd.read_csv(file).sort_values("origin_generation").reset_index(drop=True)
    
    #last_mutation = df.position.iloc[-1]
    #df = df[df.position != last_mutation]
    
    fixed_candidates = df[df.abs_population_freq == 1000].position.tolist()
    fixed_candidates = pd.DataFrame.from_dict(dict(Counter(fixed_candidates)), orient="index").reset_index()
    fixed_candidates.columns = ["position", "num_fixed"]
    fixed_mutations = fixed_candidates[fixed_candidates.num_fixed >= 50].position.unique().tolist()
    
    all_mutations = df.position.unique().tolist()
    notfixed_mutations = []
    for m in all_mutations:
        if m not in fixed_mutations:
            notfixed_mutations.append(m)
            
    return fixed_mutations, all_mutations

def get_probability_fixation(file):
    fixed_mutations, all_mutations = get_fixed_and_all_mutations(file)
    return len(fixed_mutations) / len(all_mutations)

def get_fixation_times(file):
    
    fixed_mutations, _ = get_fixed_and_all_mutations(file)
    
    df = pd.read_csv(file).sort_values("origin_generation").reset_index(drop=True)
    last_mutation = df.position.iloc[-1]
    #df = df[df.position != last_mutation]
    
    mask = [True if p in fixed_mutations else False for p in df.position]
    df_fixed = df[mask]
    
    fixation_times = []
    for m in fixed_mutations:
        fixation_times.append(df_fixed[df_fixed.position == m].shape[0])
        
    return fixation_times

def get_probability_fixation_table(file, germination_rate, selection_coefficient):
    prob = get_probability_fixation(file)
    table = pd.DataFrame([[prob, germination_rate, selection_coefficient]])
    table.columns = ["fixation probability", "germination rate", "selection coefficient"]
    return table




# omega plus

def get_likelihood_sweeps(trees, grid = 100, mu = 5e-5, threshold=5):
    num_sweeps = 0
    num_files = 0
    likelihoods = []
    for _, tree in enumerate(tqdm(trees)):
        ts = tskit.load(tree)
        ts = msprime.sim_mutations(ts, mu, discrete_genome=False, keep=False)
        random_float = np.random.rand(1).item()
        with open('out_' + str(random_float) + '.ms', 'w') as ms_file:
            tskit.write_ms(ts, ms_file, precision=16)
        script = "OmegaPlus -name out_" + str(random_float) + " -input out_" + str(random_float) +".ms -grid " + str(grid) + " -minwin 5 -maxwin 20 -length 10000"
        out = run_bash_script(script)
        sweed_out = pd.read_csv("OmegaPlus_Report.out_" + str(random_float), comment="/", sep="\t", header=None)
        likelihoods.append(sweed_out)
    return likelihoods



def omegaplus_helper(i):  
    ts = tskit.load(trees[i])
    ts = msprime.sim_mutations(ts, mu, discrete_genome=False, keep=False)
    random_float = np.random.rand(1).item()
    random_float = i
    with open('out_' + str(random_float) + '.ms', 'w') as ms_file:
        tskit.write_ms(ts, ms_file, precision=16)
    script = "OmegaPlus -name out_" + str(random_float) + " -input out_" + str(random_float) +".ms -grid " + str(grid) + " -minwin " + str(minwin) + " -maxwin " + str(maxwin) + " -length 100000"
    out = run_bash_script(script)
    omegaplus = pd.read_csv("OmegaPlus_Report.out_" + str(random_float), comment="/", sep="\t", header=None)
    omegaplus['run'] = i
    return omegaplus

def multiprocessed_omegaplus(num_process=5):
    p = multiprocessing.Pool(num_process)
    result = p.map(omegaplus_helper, range(0, len(trees)))
    p.close()
    p.join()
    return result

def get_sweeps_by_threshold(likelihood, model, threshold=np.arange(0, 1, 0.05)):
    sweeps = []
    for t in threshold:
    #for threshold in np.arange(0, 2, 0.1):
        sweep = likelihood[likelihood.Likelihood > t].run.unique().shape[0]
        sweeps.append(sweep)

    sweeps = pd.DataFrame(sweeps)
    sweeps.columns = [model]
    sweeps = sweeps.melt(value_vars=[model])
    sweeps.columns = ["model", "sweeps"]
    sweeps['threshold'] = threshold
    return sweeps

def get_scaling_factor(path1, path2, plot=True, mu=5e-5):
    mutations_b1 = get_mutations(path1, "b1.0", mu=mu)
    mutations_b05 = get_mutations(path2, "b0.5", mu=mu)
    scaling_factor = mutations_b1.mutations.mean() / mutations_b05.mutations.mean()
    mutations_b05 = get_mutations(path2, "b0.5", mu=mu*scaling_factor)
    mutation_model = pd.concat([mutations_b1, mutations_b05])
    if plot:
        sns.boxplot(data=mutation_model, x="model", y="mutations")
    return scaling_factor, mutation_model

def get_mutations(path, model, mu=5e-5):
    trees, _ = get_path_trees_recorders(path)
    num_mutations = []
    for tree in trees:
        ts = tskit.load(tree)
        ts = msprime.sim_mutations(ts, mu, discrete_genome=False, keep=False)
        num_mutations.append(ts.num_mutations)

    mutations = pd.DataFrame(num_mutations)
    mutations['model'] = model
    mutations.columns = ['mutations', 'model']
    return mutations

def get_scaling_factor(path1, path2, plot=True, mu=5e-5):
    mutations_b1 = get_mutations(path1, "b1.0", mu=mu)
    mutations_b05 = get_mutations(path2, "b0.5", mu=mu)
    scaling_factor = mutations_b1.mutations.mean() / mutations_b05.mutations.mean()
    mutations_b05 = get_mutations(path2, "b0.5", mu=mu*scaling_factor)
    mutation_model = pd.concat([mutations_b1, mutations_b05])
    if plot:
        sns.boxplot(data=mutation_model, x="model", y="mutations")
    return scaling_factor, mutation_model

def clean_up_OmegaPlus():
    os.system("rm -rf ./OmegaPlus_*")
    os.system("rm -rf ./out_*")
    
def get_sweeps(models, likelihoods, threshold=np.arange(0, 3000, 250)):
    sweeps = []
    for i, m in enumerate(models):
        l = likelihoods[likelihoods['model'] == m]
        sweep = get_sweeps_by_threshold(l, models[i], threshold)
        sweeps.append(sweep)
    sweeps = pd.concat(sweeps)
    return sweeps

def get_likelihoods(paths, files):
    likelihoods = []
    for i, path in enumerate(tqdm(paths)):
        likelihood = pickle.load(open(files[i], 'rb'))
        likelihood['model'] = models[i]
        likelihoods.append(likelihood)

    likelihoods = pd.concat(likelihoods)
    return likelihoods


def get_relative_likelihoods(paths, files):
    likelihoods = []
    for i, path in enumerate(tqdm(paths)):
        likelihood = pickle.load(open(files[i], 'rb'))
        #print(likelihood.head())
        #length = likelihood.shape[0]
        likelihood['model'] = models[i]
        #print(length)
        #likelihood['Likelihood'] = likelihood['Likelihood'] / length
        likelihoods.append(likelihood)
    likelihoods = pd.concat(likelihoods)
    return likelihoods