import os
import json
import datetime
import warnings
import random

import numpy as np
import scipy as sp
from scipy import stats
# from scipy.stats import binned_statistic
# from scipy.stats import anderson, normaltest, skew, skewtest, kurtosistest, shapiro, kurtosis, ks_2samp
import pandas as pd

# from .evolution import build_generation_fitness_table
# from .plotting import plot_threshold_fitness, plot_histogram_of_fitness


def load_LG_matrix(full_file_name=None):
    """Load amino acid substitution probabilities matrix.

    Load .csv file defining aa substitution probabilities calculated from R
    matrix multiplied by PI matrix, with diagonals forced to zero as mutation
    has to happen then converted to event rates p(lambda) where lambda = sum Qx
    and p(lambda)x=Qxy/lambda.
    """
    if full_file_name is None:
        full_file_name = os.path.join("data", "LGaa.csv")
    LG_matrix = pd.read_csv(full_file_name, index_col="Original")
    return LG_matrix


def write_protein_fitness(run_path, directory, fitness_table):

    fitness_file_name = os.path.join(run_path, directory, "fitnesslibrary.csv")
    fitness_table.to_csv(fitness_file_name, index_label="Position")


def write_initial_protein(initial_protein, run_path):
    protein_full_name = os.path.join(run_path, "start", "firstprotein.fas")
    with open(protein_full_name, "w") as ipf:  # open file
        ipf.write('>firstprotein\n')
        ipf.write(''.join(initial_protein))


def append_ks_statistics(stats_full_name, distribution_fitness, initial_fitness):
    with open(stats_full_name, "a") as stats_file:  # Append to file
        # Kolmogorov-Smirnov test of similarity to original distributuion
        ksdata = sp.stats.ks_2samp(distribution_fitness, initial_fitness)
        stats_file.write("\n\n\n2-sided Kolmogorov-Smirnov test of similarity "
                         "between the fitness space and evolving protein\n"
                         "----------------------------------------------"
                         "----------------------------------------------\n\n")
        stats_file.write("The Kolmogorov-Smirnov test between the fitness "
                         "space and the evolving protein gives a p-value of: "
                         "{}\n".format(ksdata.pvalue))

        if ksdata.pvalue < 0.05:
            stats_file.write("Therefore, as the p-value is smaller than 0.05 "
                             "we can reject the hypothesis that the fitness "
                             "space distribution and the evolving sequence "
                             "distribution are the same.")
        else:
            stats_file.write("Therefore, as the p-value is larger than 0.05 "
                             "we cannot reject the hypothesis that the fitness "
                             "space distribution and the evolving sequence "
                             "distribution are the same.")


def write_histogram_statistics(stats_full_name, aa_variant_fitnesses, record):
    """Write the results of 5 statistical tests on the global fitness space."""

    stats_file = open(stats_full_name, "w")  # open file

    stats_file.write("Tests for normality on the amino acid fitnesses\n"
                     "===============================================\n\n\n")

    # TODO: Check that the ordering is correct (default: row major)
    fitnesses = aa_variant_fitnesses.ravel()

    # Skewness
    stats_file.write("Skewness\n"
                     "--------\n\n"
                     "The skewness of the data is: "
                     "{}\n\n\n".format(sp.stats.skew(fitnesses)))

    # Kurtosis
    stats_file.write("Kurtosis\n"
                     "--------\n\n"
                     "The kurtosis of the data is: "
                     "{}\n\n\n".format(sp.stats.kurtosis(fitnesses)))

    # Normality (Shapiro-Wilk)
    stats_file.write("Shapiro-Wilk test of non-normality\n"
                     "----------------------------------\n\n")
    W_shapiro, p_shapiro = sp.stats.shapiro(fitnesses)
    stats_file.write("The Shapiro-Wilk test of non-normality for the entire "
                     "dataset gives p = {}\n".format(p_shapiro))
    if p_shapiro >= 0.05:
        shapiro = 'not '
    else:
        shapiro = ''
    stats_file.write("Therefore the Shapiro-Wilk test suggests that the whole "
                     "dataset is {}confidently non-normal\n".format(shapiro))
    stats_file.write("However if there are more than 5000 datapoints this "
                     "test is inaccurate. This test uses {} datapoints.\n\n"
                     .format(len(fitnesses)))
    passpercentcalc = []
    for aa in aa_variant_fitnesses:
        (_, p_value) = sp.stats.shapiro(aa)
        if p_value >= 0.05:
            passpercentcalc.append(1)
        else:
            passpercentcalc.append(0)
    stats_file.write("According to Shapiro-Wilk test, the proportion of "
                     "individual positions that are not confidently "
                     "non-normal is: {:.2%}\n\n\n"
                     .format(sum(passpercentcalc) / len(passpercentcalc)))

    # Normality (Anderson-Darling)
    # Significance levels (percentages) for normal distributions
    significance_levels = (15, 10, 5, 2.5, 1)
    stats_file.write("Anderson-Darling test of normality\n"
                     "----------------------------------\n\n")
    anderson_results = sp.stats.anderson(fitnesses)
    stats_file.write("The Anderson-Darling test of normality for the entire "
                     "dataset gives a test statistic of {} "
                     "and critical values of {}\n"
                     .format(anderson_results.statistic,
                             anderson_results.critical_values))
    if anderson_results.statistic > anderson_results.critical_values[-1]:
        stats_file.write("Therefore according to the Anderson-Darling test, "
                         "the hypothesis of normality is rejected for the "
                         "entire dataset.\n\n")
    else:
        level_index = np.searchsorted(anderson_results.critical_values,
                                      anderson_results.statistic, side="left")
        stats_file.write("Therefore according to the Anderson-Darling test, "
                         "the hypothesis of normality is not rejected at the "
                         "{}% significance level for the entire dataset.\n\n"
                         .format(significance_levels[level_index]))

    # Set up output for significance levels - final bin represents "reject"
    hypothesis_tally = np.zeros(len(significance_levels) + 1)
    for aa in aa_variant_fitnesses:
        result = sp.stats.anderson(aa)
        level_index = np.searchsorted(anderson_results.critical_values,
                                      result.statistic, side="left")
        hypothesis_tally[level_index] += 1
    hypothesis_tally /= aa_variant_fitnesses.shape[0]  # Normalise
    stats_file.write("According to the Anderson-Darling test, "
                     "the hypothesis of normality is not rejected for each "
                     "position in the dataset for: \n")
    for proportion, level in zip(hypothesis_tally, significance_levels):
        stats_file.write("{:.2%} of positions at the "
                         "{}% significance level\n".format(proportion, level))
    stats_file.write("and {:.2%} of positions are rejected.\n\n\n"
                     .format(hypothesis_tally[-1]))

    # Normality (Skewness-Kurtosis)
    stats_file.write("Skewness-kurtosis all test of difference from normality\n"
                     "-------------------------------------------------------\n\n")
    skewkurtall = sp.stats.normaltest(fitnesses)

    stats_file.write("According to the skewness-kurtosis all test, the whole "
                     "dataset gives p = {}.".format(skewkurtall.pvalue))
    if skewkurtall.pvalue >= 0.05:
        stats_file.write("\nTherefore the dataset does not differ significantly "
                         "from a normal distribution.\n\n")
    else:
        stats_file.write("\nTherefore the dataset differs significantly from "
                         "a normal distribution.\n\n")

    skewkurtpass = []
    for aa in aa_variant_fitnesses:
        distskewkurt = sp.stats.normaltest(aa)
        if distskewkurt.pvalue >= 0.05:
            skewkurtpass.append(1)
        else:
            skewkurtpass.append(0)
    stats_file.write("According to the skewness-kurtosis all test, "
                     "{:.2%} of sites do not differ significantly from a "
                     "normal distribution."
                     .format(sum(skewkurtpass) / len(skewkurtpass)))

    stats_file.close()


def write_fasta_alignment(population, generation, run_path):
    """Write fasta alignment from sequences provided."""
    fastafilepath = os.path.join(run_path, "fastas")
    fastafilename = "generation_{}.fasta".format(generation)
    fullname = os.path.join(fastafilepath, fastafilename)
    with open(fullname, "w") as fastafile:  # open file
        # Write fasta header followed by residue in generation string
        # TODO: This should be an ordered dict or list to preserve the order...
        for p, protein in list(population.items()):
            fastafile.write("\n>clone_%s\n" % (p+1))
            fastafile.write(''.join(protein))


def write_final_fasta(population, tree, run_path):
    bifsize = 0
    for bifs in tree["branches"]:
        bifsize += len(bifs)
    bifursize = bifsize/len(tree["branches"])
    n_clones_to_take = int((bifursize-1)/2)  # if 5, gives 2, if 4 gives 2, if 3 gives 1.
    generation_numbers = []
    for branch in tree["branches"]:
        # Sample from unique clones
        clone_selection = random.sample(set(branch), n_clones_to_take)
        for c in clone_selection:
            generation_numbers.append(c)

    full_name = os.path.join(run_path, "treefastas", "selected_fastas.fasta")
    with open(full_name, "w") as treefastafile:  # open file
        # Write fasta header followed by residue in generation string
        for p in generation_numbers:
            treefastafile.write(">clone_%s\n" % (p+1))
            treefastafile.write(''.join(population[p]))
            treefastafile.write('\n')
        # Choose a random root to write
        # BUG: This was originally choosing from n_roots
        # root = population[random.choice(n_roots)]
        root = population[random.choice(tree["roots"])]
        treefastafile.write(">root\n")
        treefastafile.write(''.join(root))


def create_output_folders(output_directory=""):
    """Create output directory structure.

    Each run will be saved in a time-stamped folder within the run path.
    """
    paths = ['runsettings', 'start', 'fastas', 'fitnessgraph',
             'fitnessdotmatrix', 'fitnessdistribution', 'treefastas']

    date_time = "run{}".format(datetime.datetime.now().strftime("%y-%m-%d-%H-%M"))
    run_path = os.path.join(output_directory, "results", date_time)
    os.makedirs(run_path)
    if os.path.isdir(run_path):
        print("Results directory successfully created: {}".format(os.path.join(os.getcwd(), run_path)))
    else:
        warnings.warn("Results directory not created: {}".format(os.path.join(os.getcwd(), run_path)))

    for path in paths:
        os.makedirs(os.path.join(run_path, path))

    innerpaths = ['statistics', 'histograms']
    for innerpath in innerpaths:
        os.makedirs(os.path.join(run_path, "fitnessdistribution", innerpath))
    return run_path


def write_settings_file(run_path, settings):
    settingsfullname = os.path.join(run_path, "runsettings", "settings.json")
    with open(settingsfullname, "w") as sf:  # open file
        json.dump(settings, sf, indent=4)
