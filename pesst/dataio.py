import os
import shutil
import json
import datetime
import warnings
import random
import pkg_resources
from operator import itemgetter
from collections import namedtuple

import numpy as np
import scipy as sp
import scipy.stats
import pandas as pd
import imageio

from .utilities import compact_protein


Generation = namedtuple('Generation', ['population', 'stabilities'])

def load_LG_matrix(full_file_name=None):
    """Load amino acid transition probabilities matrix.

    Load .csv file defining aa substitution probabilities calculated from R
    matrix multiplied by PI matrix, with normalised rows and diagonals set to
    zero to force mutation then converted to event rates p(lambda) where:
    lambda = sum Qx
    and
    p(lambda) x = Qxy / lambda.
    """
    # TODO: Check: p(lambda) x = Qxy / lambda
    if full_file_name is None:
        resource = os.path.join("data", "LGaa.csv")
        full_file_name = pkg_resources.resource_filename("pesst", resource)
        # full_file_name = pkg_resources.resource_filename("pesst.data", "LGaa.csv")
        # pwd, this_filename = os.path.split(__file__)
        # full_file_name = os.path.join(pwd, "data", "LGaa.csv")
    LG_matrix = pd.read_csv(full_file_name, index_col="Original")
    return LG_matrix


def write_stability_table(stability_table, out_paths):
    """Write matrix of stability changes."""
    stability_file_name = os.path.join(out_paths["initial"], "stability_table.csv")
    stability_table.to_csv(stability_file_name, index_label="Position")


def write_initial_protein(initial_protein, out_paths):
    protein_full_name = os.path.join(out_paths["initial"], "firstprotein.fas")
    with open(protein_full_name, "w") as ipf:  # open file
        ipf.write('>firstprotein\n')
        ipf.write(compact_protein(initial_protein))


def write_roots(root_keys, out_paths):
    """Write the list of roots indicies."""
    with open(os.path.join(out_paths["initial"], "roots.txt"), "w") as rootsfile:  # open file
        rootsfile.write("Roots:")
        for k in root_keys:
            rootsfile.write(f"\nClone {str(k+1)}")


def write_tree(generation, tree, out_paths):
    """Write the roots and branches created by bifurcations."""
    with open(os.path.join(out_paths["tree"], "tree.txt"), "a") as bf:
        bf.write(f"Generation {generation}\n")
        bf.write(f"Roots: {tree['roots']}\n")
        for b, branch in enumerate(tree['branches']):
            bf.write(f"Branch {b}: {branch}\n")


def append_ks_statistics(stats_full_name, distribution_stability, initial_stability):
    with open(stats_full_name, "a") as stats_file:  # Append to file
        # Kolmogorov-Smirnov test of similarity to original distributuion
        ksdata = sp.stats.ks_2samp(distribution_stability, initial_stability)
        stats_file.write("\n\n\n2-sided Kolmogorov-Smirnov test of similarity "
                         "between the stability space and evolving protein\n"
                         "-----------------------------------------------"
                         "-----------------------------------------------\n\n")
        stats_file.write("The Kolmogorov-Smirnov test between the stability "
                         "space and the evolving protein gives a p-value of: "
                         f"{ksdata.pvalue}. \n")

        if ksdata.pvalue < 0.05:
            stats_file.write("Therefore, as the p-value is smaller than 0.05 "
                             "we can reject the hypothesis that the stability "
                             "space distribution and the evolving sequence "
                             "distribution are the same.")
        else:
            stats_file.write("Therefore, as the p-value is larger than 0.05 "
                             "we cannot reject the hypothesis that the stability "
                             "space distribution and the evolving sequence "
                             "distribution are the same.")


def write_histogram_statistics(stats_full_name, aa_variant_stabilities):
    """Write the results of 5 statistical tests on the global stability space."""

    stats_file = open(stats_full_name, "w")  # open file

    stats_file.write("Tests for normality on the amino acid stabilities\n"
                     "=================================================\n\n\n")

    # TODO: Check that the ordering is correct (default: row major)
    stabilities = aa_variant_stabilities.ravel()

    # Skewness
    stats_file.write("Skewness\n"
                     "--------\n\n"
                     "The skewness of the data is: "
                     f"{sp.stats.skew(stabilities)}.\n\n\n")

    # Kurtosis
    stats_file.write("Kurtosis\n"
                     "--------\n\n"
                     "The kurtosis of the data is: "
                     f"{sp.stats.kurtosis(stabilities)}.\n\n\n")

    # Normality (Shapiro-Wilk)
    stats_file.write("Shapiro-Wilk test of non-normality\n"
                     "----------------------------------\n\n")
    with warnings.catch_warnings():
        # warnings.simplefilter("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", message="p-value may not be accurate for N > 5000.")
        W_shapiro, p_shapiro = sp.stats.shapiro(stabilities)
    stats_file.write("The Shapiro-Wilk test of non-normality for the entire "
                     f"dataset gives p = {p_shapiro}.\n")
    if p_shapiro >= 0.05:
        shapiro = 'not '
    else:
        shapiro = ''
    stats_file.write("Therefore the Shapiro-Wilk test suggests that the whole "
                     f"dataset is {shapiro}confidently non-normal.\n")
    if len(stabilities) > 5000:
        stats_file.write("Warning: There are more than 5,000 datapoints "
                         f"({len(stabilities)}) so the p-value may be inaccurate.\n\n")
    # stats_file.write("However if there are more than 5000 datapoints this "
    #                  "test is inaccurate. This test uses {} datapoints.\n\n"
    #                  .format(len(stabilities)))
    else:
        stats_file.write("\n\n")
    passpercentcalc = []
    for aa in aa_variant_stabilities:
        with warnings.catch_warnings():
            # warnings.simplefilter("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", message="p-value may not be accurate for N > 5000.")
            (_, p_value) = sp.stats.shapiro(aa)
        if p_value >= 0.05:
            passpercentcalc.append(1)
        else:
            passpercentcalc.append(0)
    stats_file.write("According to the Shapiro-Wilk test, the proportion of "
                     "individual positions that are not confidently "
                     f"non-normal is: {sum(passpercentcalc) / len(passpercentcalc):.2%}.\n\n\n")

    # Normality (Anderson-Darling)
    # Significance levels (percentages) for normal distributions
    significance_levels = (15, 10, 5, 2.5, 1)
    stats_file.write("Anderson-Darling test of normality\n"
                     "----------------------------------\n\n")
    anderson_results = sp.stats.anderson(stabilities)
    stats_file.write("The Anderson-Darling test of normality for the entire "
                     f"dataset gives a test statistic of {anderson_results.statistic} "
                     f"and critical values of {anderson_results.critical_values}.\n")
    if anderson_results.statistic > anderson_results.critical_values[-1]:
        stats_file.write("Therefore according to the Anderson-Darling test, "
                         "the hypothesis of normality is rejected for the "
                         "entire dataset.\n\n")
    else:
        level_index = np.searchsorted(anderson_results.critical_values,
                                      anderson_results.statistic, side="left")
        stats_file.write("Therefore according to the Anderson-Darling test, "
                         "the hypothesis of normality is not rejected at the "
                         f"{significance_levels[level_index]}% significance "
                         "level for the entire dataset.\n\n")

    # Set up output for significance levels - final bin represents "reject"
    hypothesis_tally = np.zeros(len(significance_levels) + 1)
    for aa in aa_variant_stabilities:
        result = sp.stats.anderson(aa)
        level_index = np.searchsorted(anderson_results.critical_values,
                                      result.statistic, side="left")
        hypothesis_tally[level_index] += 1
    hypothesis_tally /= aa_variant_stabilities.shape[0]  # Normalise
    stats_file.write("According to the Anderson-Darling test, "
                     "the hypothesis of normality is not rejected for each "
                     "position in the dataset for: \n")
    for proportion, level in zip(hypothesis_tally, significance_levels):
        stats_file.write(f"{proportion:.2%} of positions at the "
                         f"{level}% significance level\n")
    stats_file.write(f"and {hypothesis_tally[-1]:.2%} of positions are rejected.\n\n\n")

    # Normality (Skewness-Kurtosis)
    stats_file.write("Skewness-kurtosis all test of difference from normality\n"
                     "-------------------------------------------------------\n\n")
    skewkurtall = sp.stats.normaltest(stabilities)

    stats_file.write("According to the skewness-kurtosis all test, the whole "
                     f"dataset gives p = {skewkurtall.pvalue}.")
    if skewkurtall.pvalue >= 0.05:
        stats_file.write("\nTherefore the dataset does not differ significantly "
                         "from a normal distribution.\n\n")
    else:
        stats_file.write("\nTherefore the dataset differs significantly from "
                         "a normal distribution.\n\n")

    skewkurtpass = []
    for aa in aa_variant_stabilities:
        distskewkurt = sp.stats.normaltest(aa)
        if distskewkurt.pvalue >= 0.05:
            skewkurtpass.append(1)
        else:
            skewkurtpass.append(0)
    stats_file.write("According to the skewness-kurtosis all test, "
                     f"{sum(skewkurtpass) / len(skewkurtpass):.2%} of sites "
                     "do not differ significantly from a normal distribution.")

    stats_file.close()


def write_fasta_alignment(generation, population, out_paths):
    """Write fasta alignment from sequences provided."""
    fastafilename = f"clones_G{generation}.fasta"
    fullname = os.path.join(out_paths["fastas"], fastafilename)
    with open(fullname, "w") as fastafile:  # open file
        # Write fasta header followed by residue in generation string
        for p, protein in population.items():
            fastafile.write(f">clone_{p+1}\n")
            fastafile.write(compact_protein(protein)+"\n")


def write_final_fasta(population, tree, out_paths):
    """Write a random selection of proteins from each branch with a random
    root.
    """
    tree_size = sum([len(branch) for branch in tree["branches"]])
    average_branch_size = tree_size / len(tree["branches"])
    n_clones_to_take = int((average_branch_size-1)/2)  # if 5, gives 2, if 4 gives 1, if 3 gives 1.
    # Choose a random selection of proteins from each branch
    selection = []
    for branch in tree["branches"]:
        selection.extend(random.sample(set(branch), n_clones_to_take))

    full_name = os.path.join(out_paths["treefastas"], "selected_fastas.fasta")
    with open(full_name, "w") as treefastafile:  # open file
        # Write fasta header followed by residue in generation string
        for pi in selection:
            treefastafile.write(f">clone_{pi+1}\n")
            treefastafile.write(compact_protein(population[pi]))
            treefastafile.write('\n')
        # Choose a random root to write
        root = population[random.choice(tree["roots"])]
        treefastafile.write(">root\n")
        treefastafile.write(compact_protein(root))


def create_output_folders(output_dir=None, write_to_disk=True):
    """Create output directory structure.

    If no output directory is passed, each run will be saved in a time-stamped
    folder within the run path.
    """

    paths = ['initial', 'data', 'figures', 'animations',
             'tree', 'fastas', 'treefastas', 'statistics']

    if output_dir is None:
        output_dir = datetime.datetime.now().strftime("%y-%m-%d-%H-%M")
    run_path = os.path.join("results", output_dir)
    if write_to_disk:
        if os.path.isdir(run_path):
            response = input(f"Warning: the output directory '{run_path}' already exists! Delete? [Y/n]: ")
            if response.lower() == "y" or response.lower() == "":
                shutil.rmtree(run_path)
            else:
                exit()
        os.makedirs(run_path)
        full_run_path = os.path.join(os.getcwd(), run_path)
        if os.path.isdir(run_path):
            print(f"Results directory successfully created: {full_run_path}")
        else:
            warnings.warn(f"Results directory not created: {full_run_path}")

        for path in paths:
            os.makedirs(os.path.join(run_path, path))

    # innerpaths = ['statistics', 'histograms']
    # for innerpath in innerpaths:
    #     os.makedirs(os.path.join(run_path, "stabilitydistribution", innerpath))

    out_paths = {path: os.path.join(run_path, path) for path in paths}
    out_paths["results"] = run_path
    return out_paths


def save_settings(settings, out_paths):
    settingsfullname = os.path.join(out_paths["initial"], "settings.json")
    with open(settingsfullname, "w") as sf:  # open file
        json.dump(settings, sf, indent=4)


def load_settings(out_paths):
    if isinstance(out_paths, dict):
        settingsfullname = os.path.join(out_paths["initial"], "settings.json")
    else:  # Assume path is passed as string
        settingsfullname = out_paths
    with open(settingsfullname, "r") as sf:  # open file
        settings = json.load(sf)
    return settings


def create_gif(filenames, out_paths, duration=0.5):
    """Create an animated gif from a list of static figure filenames."""
    filename = os.path.basename(filenames[0])
    basename, ext = os.path.splitext(filename)
    output_basename, _ = basename.rsplit('_', 1)  # Trim generation number
    output_file = os.path.join(out_paths["animations"], f"{output_basename}.gif")
    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave(output_file, images, duration=duration)

    # Streaming approach for longer animations
    # with imageio.get_writer('/path/to/movie.gif', mode='I') as writer:
    # for filename in filenames:
    #     image = imageio.imread(filename)
    #     writer.append_data(image)


def save_history(generation, history, out_paths):
    """Save the last generation to disk."""

    stabilities_file = os.path.join(out_paths["data"], f"stabilities_G{generation}.csv")
    np.savetxt(stabilities_file, history[generation].stabilities, delimiter=',')

    # Save protein sequences
    clones_file = os.path.join(out_paths["data"], f"clones_G{generation}.json")
    population = {key: compact_protein(protein)
                  for key, protein in history[generation].population.items()}
    with open(clones_file, "w") as gf:
        json.dump(population, gf, indent=4, sort_keys=True)


def load_history(data_dir):
    """Load data from a previous simulation."""
    # NOTE: Work in progress!
    stability_files = glob.glob(os.path.join("results", data_dir, "stabilities_*.csv"))
    generations = []
    for file in stability_files:
        (root, ext) = os.path.splitext(os.path.basename(file))
        generations.append(root.split('_')[-1])

    population_files = glob.glob(os.path.join("results", data_dir, "clones_*.json"))
    sorted_files = sorted(zip(generations, population_files, stability_files),
                          key=itemgetter(0))
    history = {}
    generations = []
    for (generation, population_file, stability_file) in sorted_files:
        # TODO: Write a JSON object_hook to expand sequence strings
        with open(population_file, "r") as pf:
            population = json.load(pf)
        with open(stability_file, "r") as sf:
        # history.append({"population": population, "stabilities": stabilities})
        history[generation] = Generation(population=population, stabilities=stabilities)
        generations.append(generation)
    return (history, generations)


def load_stability_table(out_paths):
    return pd.read_csv(os.path.join(out_paths["initial"],
                                    'stability_table.csv'),
                       index_label="Position")
