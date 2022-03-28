import os
import tskit
import msprime
import subprocess
import numpy as np
import pandas as pd
import seaborn as sns
import multiprocessing
import matplotlib.pyplot as plt

def ld_matrix(ts, save_name=None):
    """Plotting linkage linkage desquilibrium and dormancy from tskit tree_sequence ts"""
    ld_calc = tskit.LdCalculator(ts)
    A = ld_calc.r2_matrix()
    # Now plot this matrix.
    x = A.shape[0] / plt.rcParams["figure.dpi"]
    x = max(x, plt.rcParams["figure.figsize"][0])
    fig, ax = plt.subplots(figsize=(x, x))
    fig.tight_layout(pad=0)
    im = ax.imshow(A, interpolation="none", vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in "top", "bottom", "left", "right":
        ax.spines[s].set_visible(False)
    plt.gcf().colorbar(im, shrink=0.5, pad=0)
    if save_name:
        plt.savefig(save_name + ".png")

def ld_matrix(ts, save_name=None):
    """Plotting linkage linkage desquilibrium and dormancy from tskit tree_sequence ts"""
    ld_calc = tskit.LdCalculator(ts)
    A = ld_calc.r2_matrix()
    # Now plot this matrix.
    x = A.shape[0] / plt.rcParams["figure.dpi"]
    x = max(x, plt.rcParams["figure.figsize"][0])
    fig, ax = plt.subplots(figsize=(x, x))
    fig.tight_layout(pad=0)
    im = ax.imshow(A, interpolation="none", vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in "top", "bottom", "left", "right":
        ax.spines[s].set_visible(False)
    plt.gcf().colorbar(im, shrink=0.5, pad=0)
    if save_name:
        plt.savefig(save_name + ".png")


# input formatting helper functions

#def GetVariantPositions(ts): return np.array([v.site.position for v in ts.variants()])
def GetVariantPositions(ts, mask, sample_idxs): 
    positions = np.array([v.site.position for v in ts.variants()])[mask]
    return positions[sample_idxs]

def Vec2Mat(vec): return vec.reshape(-1, 1)

def GetUpperMatrixTri(mat): return mat[np.triu_indices(mat.shape[0], k = 1)]

def GetPairwiseDistanceValues(ts, mask, sample_idxs, length=50000, sample_size=200, sample_mutation_size=1000):
    from scipy.spatial import distance_matrix
    positions = GetVariantPositions(ts, mask, sample_idxs)
    positions = Vec2Mat(positions)
    dist_mat = distance_matrix(positions, positions)
    dist_mat_upper_tri = GetUpperMatrixTri(dist_mat)
    return dist_mat_upper_tri

def GetPairwiseLDValues(ts, length=50000, sample_size=200, sample_mutation_size=1000):
    
    #print("sample_mutation_size ", sample_mutation_size)
    #print("sample_size ", sample_size)

    
    #ld_calc = tskit.LdCalculator(ts)
    #matrix = ld_calc.r2_matrix()
    
    import allel
    genotypes = ts.genotype_matrix()
    
    
    not_fixed_mask = genotypes.sum(1) != genotypes.shape[1]
    genotypes = genotypes[not_fixed_mask]
    
    import random
    sample_idxs = random.sample(range(0, genotypes.shape[0]),  int(genotypes.shape[0] / sample_mutation_size))
    genotypes = genotypes[sample_idxs]
        
    
    h = allel.HaplotypeArray(genotypes)
    g = h.to_genotypes(ploidy=2)
    gn = g.to_n_alt(fill=-1)
    r = allel.rogers_huff_r(gn)
    from scipy.spatial.distance import squareform
    matrix = squareform(r ** 2)
    matrix = np.nan_to_num(matrix)
    n = matrix.shape[0]
    LDs = matrix[np.triu_indices(n, k = 1)]
    return LDs, not_fixed_mask, sample_idxs

def GetLDDistribution(ts,remove_first_intervall=False, normalize=True, length=10000, sample_size=200, sample_mutation_size=1000):
    lds = GetPairwiseLDValues(ts, length, sample_size, sample_mutation_size)
    hist, bins = np.histogram(lds, bins=np.linspace(0,length,41))
    bin_counts = zip(bins, bins[0:], hist)  # [(bin_start, bin_end, count), ... ]
    #if normalize:
    #    hist = hist / hist.sum()
    lds = pd.DataFrame([bins, hist]).T
    lds.columns = ["intervall", "proportion"]
    lds['intervall'] = np.round(lds['intervall'], 1)
    lds.drop(lds.tail(1).index,inplace=True)
    if remove_first_intervall: return lds.iloc[1:]
    return lds

def FilterLowLDSites(lds, distances, threshold=0.1):
    mask = [True if ld >= threshold else False for ld in lds]
    lds = lds[mask]
    distances = distances[mask]
    return lds, distances

def CombineLDDistanceList(lds, distances):
    ld_dist_df = pd.DataFrame([lds, distances]).T
    ld_dist_df.columns = ["ld", "distance"]
    ld_dist_df = ld_dist_df.dropna()
    return ld_dist_df

def GetDistanceBin(distance): 
    spans = np.linspace(0,10000,41)
    return int(np.argmax(spans[spans - distance < 0]))

def GetMeanPerDistanceInterval(ld_distance: pd.DataFrame):
    ld_distance['distance_bin'] =  ld_distance['distance'].apply(GetDistanceBin)
    ld_distance = ld_distance.drop(columns="distance")
    ld_distance = ld_distance.groupby(by="distance_bin").mean("ld")
    ld_distance.reset_index(inplace=True)
    return ld_distance

def GetLDPerBin(ts, length=10000, sample_size=200, sample_mutation_size=1000):
    lds, not_fixed_mask, sample_idxs = GetPairwiseLDValues(ts, length, sample_size, sample_mutation_size)
    distances = GetPairwiseDistanceValues(ts, not_fixed_mask, sample_idxs, length, sample_size, sample_mutation_size)
    
    lds, distances = FilterLowLDSites(lds, distances, 0.1)
    ld_dist_df = CombineLDDistanceList(lds, distances)
    ld_dist_df = GetMeanPerDistanceInterval(ld_dist_df)
    return ld_dist_df


"""
def GetLDDistributionFromDirectory(path, remove_first_intervall=True,  first_n=None, normalize=True, length=10000, sample_size=200, sample_mutation_size=1000):
    files = os.listdir(path)
    tree_files = []
    for file in files: 
        if ".trees" in file: tree_files.append(file)

    print(f"Tree files: {len(tree_files)}", end="\r")

    ld = []
    for i, file in enumerate(tree_files[:first_n]):  
        print(f"{i}/{first_n}", end="\r")
        ts = tskit.load(path + file)
        
        ld_rep = GetLDDistribution(ts, remove_first_intervall, normalize, length, sample_size, sample_mutation_size)
        ld_rep["rep"] = i
        ld.append(ld_rep)

    ld = pd.concat(ld)
    return ld
    
"""

"""
def GetLDDistanceFromDirectory(path, first_n=None, length=10000, sample_size=200, sample_mutation_size=1000):
    files = os.listdir(path)
    tree_files = []
    for file in files: 
        if ".trees" in file: tree_files.append(file)

    print(f"Tree files: {len(tree_files)}")
    print("LENGTH is hardcoded to be 10000!")
    print(f"MUTATIONS are {sample_mutation_size}x downsampled.")
    print()

    ld = []
    for i, file in enumerate(tree_files[:first_n]):  
        print(f"{i}/{first_n}", end="\r")
        ts = tskit.load(path + file)
        ts = msprime.sim_mutations(ts, 5e-5,discrete_genome=False, keep=False)
        
        if i==0:
            genotypes = ts.genotype_matrix()
            not_fixed_mask = genotypes.sum(1) != genotypes.shape[1]
            genotypes = genotypes[not_fixed_mask]
            import random
            sample_idxs = random.sample(range(0, genotypes.shape[0]),  int(genotypes.shape[0] / sample_mutation_size))
            print(f"Mutations in first sample: {len(sample_idxs)}")
            
        
        ld_rep = GetLDPerBin(ts, length, sample_size, sample_mutation_size)
        ld_rep["rep"] = i
        ld.append(ld_rep)

    ld = pd.concat(ld)
    return ld
    
"""


####


def GetLDDistanceFromDirectory(path, j, length=10000, sample_size=200, sample_mutation_size=10):
    files = os.listdir(path)
    tree_files = []
    for file in files: 
        if ".trees" in file: 
            tree_files.append(file)

    tree_files = [tree_files[j]]        
    ld = []
    for i, file in enumerate(tree_files):  
        ts = tskit.load(path + file)
        ts = msprime.sim_mutations(ts, 5e-5,discrete_genome=False, keep=False)
        ld_rep = GetLDPerBin(ts, length, sample_size, sample_mutation_size)
        ld_rep["rep"] = j
        ld.append(ld_rep)
    ld = pd.concat(ld)
    return ld


def linkage_helper(i):  
    ld = GetLDDistanceFromDirectory(path, i)
    return ld


def multiprocessed_linkage(num_process=7):
    p = multiprocessing.Pool(num_process)
    result = p.map(linkage_helper, range(0, 90))
    p.close()
    p.join()
    return pd.concat(result)