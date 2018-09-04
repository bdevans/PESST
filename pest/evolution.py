import copy
import os  # .path
import csv
import datetime
from textwrap import wrap
from collections import namedtuple
import warnings
import random
# from random import sample, choice, shuffle  # TODO: Consolidate with numpy
import numpy as np
# from numpy.random import normal, uniform, gamma
import scipy as sp
from scipy.stats import binned_statistic
# from scipy.stats import anderson, normaltest, skew, skewtest, kurtosistest, shapiro, kurtosis, ks_2samp
# import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
from tqdm import trange


# amino acids - every fitness value string references residues string
# RESIDUES = ("R", "H", "K", "D", "E", "S", "T", "N", "Q", "C",
#             "G", "P", "A", "V", "I", "L", "M", "F", "Y", "W")
# TODO: Make the ordering match LG_matrix?
RESIDUES = "RHKDESTNQCGPAVILMFYW"  # Strings are immutable
RESIDUES_INDEX = {aa: ai for ai, aa in enumerate(RESIDUES)}  # Faster than calling .index()


def generate_protein(n_amino_acids, start_amino_acid="M"):
    """Generate an original starting protein n_amino_acids long with a start
    amino acid set to methionine.
    """
    protein = random.choices(RESIDUES, k=n_amino_acids-1)
    # protein = [random.choice(RESIDUES) for _ in range(n_amino_acids)]  # ver < 3.6
    protein.insert(0, start_amino_acid)  # Start with methionine
    return protein


# NOTE: unused
def test_normal_distribution():
    """Plot a distribution to test normalality."""
    s = np.random.normal(mu, sigma, 2000)  # generate distribution
    count, bins, ignored = plt.hist(s, 30, density=True)  # plot distribuiton
    plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
                   np.exp(- (bins - mu)**2 / (2 * sigma**2)),
             linewidth=2, color='r')
    plt.show()
    return


# def fit_module(mu, sigma):
#     """Generates an array of fitness values for each amino acid in residues."""
#     return np.random.normal(mu, sigma, len(RESIDUES))


# def get_protein_fitness(protein):  # x=protein;
def get_protein_fitness(n_amino_acids, n_variants=len(RESIDUES)):
    """Generate a dictionary describing list of fitness values at each position
    of the generated protein.
    """
    # Create dictionary containing position in the protein as keys and the array of fitness values for each amino acid as values
    # fitnesslib = {i: fit_module(mu, sigma) for i in range(len(protein))}
    fitness_table = np.random.normal(mu, sigma, size=(n_amino_acids, n_variants))
    # fitness_table = [{aa: np.random.normal(mu, sigma, size=(len(RESIDUES)))
    #                   for aa in RESIDUES}
    #                  for ai in range(n_amino_acids)]
    # print(mu, sigma, n_amino_acids, len(RESIDUES))
    # TODO:
    # data = np.random.normal(mu, sigma, size=(n_amino_acids, len(RESIDUES)))
    # fitness_table = pd.DataFrame(data=data, columns=RESIDUES)
    return fitness_table


def write_protein_fitness(run_path, directory, fitness_table):

    fitness_file_name = os.path.join(run_path, directory, "fitnesslibrary.csv")
    # with open(fitness_file_name, "w") as aminofile:  # open file
    #     # Write header
    #     aminofile.write("aminoposition"),
    #     for aa in RESIDUES:
    #         aminofile.write(",%s" % aa)
    #     # Write fitness values
    #     for ai in range(n_amino_acids):
    #         aminofile.write('\n%s' % ai),
    #         # for f in fitness_table[i]:
    #         #     aminofile.write(',%s' % f)
    #         for r in RESIDUES:
    #             aminofile.write(',%s' % RESIDUES[r])
    # NOTE: This will not include the index column
    header = ",".join(RESIDUES)
    np.savetxt(fitness_file_name, fitness_table, delimiter=",", header=header)
    # fitness_table.to_csv(fitness_file_name)


def clone_protein(protein, n_clones):
    """Generate a dictionary containing n_clones of generated protein
    - this contains the evolving dataset.
    """
    return {l: protein for l in range(n_clones)}


def get_allowed_sites(n_amino_acids, n_anchors):
    """Select invariant sites in the initially generated protein and return
    allowed values.
    """
    # TODO: Methionine is always the first anchor, so change to range(n_amino_acids) and add one to n_anchors definition
    allowed_values = list(range(1, n_amino_acids))  # keys for mutable sites
    # Randomly define invariant sites (without replacement)
    anchored_sequences = random.sample(allowed_values, n_anchors)
    anchored_sequences.insert(0, 0)  # First aa is always anchored (Methionine)
    # Remove the invariant sites from allowed values
    for a in anchored_sequences[1:]:
        allowed_values.remove(a)
    # anchored_sequences = [allowed_values.pop(random.randrange(len(allowed_values))) for r in range(n_anchors)]
    # NOTE: Should this be appended (as it may mean indexing out of range)?
    # allowed_values.append(n_amino_acids)
    return allowed_values


def gamma_ray(n_amino_acids, gamma):  # kappa, theta, n_iterations=100, n_samples=10000):
# def gammaray(a, b, c, d, e):  # a = iterations to run gamma sampling, b = number of gamma samples per iteration, c = gamma shape (kappa), d = gamma scale (theta), e = amount of aminos
    """Generate a set of gamma rate categories.

    Does so by sampling many times from a gamma distribution.
    Tests of the trade-off between computing time and variance led me to set
    this to 10000 samples from the distribution.
    Computes quartiles from the data with equal likelihood by finding bounds
    for quartiles and collecting values between bounds.
    Finds discrete rate values by taking the median of sorted collected values.
    Then iterates for a predefined set of runs, recording the median values.
    Tests of the tradeoff between computing time and variance led me to set
    this to 100 independent runs (1 million total samples from distribution).
    """
    # medians = []
    # medianlower = []
    # medianlowermid = []
    # medianuppermid = []
    # medianupper = []
    # bottomquarts = []
    # bottommidquarts = []
    # topmidquarts = []
    # topquarts = []

    # quartiles = [[] * 4]
    kappa, theta, n_iterations, n_samples = gamma["shape"], gamma["scale"], gamma["iterations"], gamma["samples"]
    # TODO:
    # kappa, theta, n_iterations, n_samples = shape, scale, iterations, samples
    medians = np.zeros(shape=(n_iterations, 4))

    for i in range(n_iterations):
        # Draw n_samples from the gamma distribution
        samples = np.random.gamma(kappa, theta, n_samples)
        # Define quartiles in that data with equal probability
        quartiles = np.percentile(samples, (0, 25, 50, 75, 100),
                                  interpolation='midpoint')
        # Find the median of each quartile
        medians[i, :], _, _ = binned_statistic(samples, samples,
                                               statistic='median',
                                               bins=quartiles)

    # Calculate average of medians across iterations
    average_medians = np.mean(medians, axis=0)

    #     # define quartiles in that data with equal probability
    #     for q in range(4):
    #         quartiles[q].append(np.percentile(samples, [q*25, (q+1)*25],
    #                                           interpolation='midpoint'))
    #
    #     quartiles = np.percentile(samples, [0, 25, 50, 75, 100], interpolation='midpoint')
    #     bin_medians, bin_edges, binnumber = sp.stats.binned_statistic(samples, samples, statistic='median', bins=quartiles)
    #     # samples[np.where(quartiles[q][0] <= samples < quartiles[q][-1])]
    #
    #     # bottomquart = np.percentile(samples, [0, 25], interpolation='midpoint')
    #     # bottomquarts.append(bottomquart)
    #     # bottommidquart = np.percentile(samples, [25, 50], interpolation='midpoint')
    #     # bottommidquarts.append(bottommidquart)
    #     # topmidquart = np.percentile(samples, [50, 75], interpolation='midpoint')
    #     # topmidquarts.append(topmidquart)
    #     # topquart = np.percentile(samples, [75, 100], interpolation='midpoint')
    #     # topquarts.append(topquart)
    #
    #     # generate space for the values within each quartile, sort them, find the median, record the median.
    #     bottomlist = []
    #     bottommidlist = []
    #     topmidlist = []
    #     toplist = []
    #     for s in samples:
    #         if bottomquart[0] <= s < bottomquart[-1]:
    #             bottomlist.append(s)
    #         elif bottommidquart[0] <= s < bottommidquart[-1]:
    #             bottommidlist.append(s)
    #         elif topmidquart[0] <= s < topmidquart[-1]:
    #             topmidlist.append(s)
    #         else:
    #             toplist.append(s)
    #     bottomlist.sort()
    #     bottommidlist.sort()
    #     topmidlist.sort()
    #     toplist.sort()
    #     ratecategoriesquartile = [np.median(bottomlist), np.median(bottommidlist), np.median(topmidlist), np.median(toplist)]
    #     medians.append(ratecategoriesquartile)
    #
    #     # print ratecategoriesquartile
    #
    # # calculate average of medians from each iteration
    # for k in medians:
    #     medianlower.append(k[0])
    #     medianlowermid.append(k[1])
    #     medianuppermid.append(k[2])
    #     medianupper.append(k[3])
    #
    # finalmedians = [np.mean(medianlower), np.mean(medianlowermid), np.mean(medianuppermid), np.mean(medianupper)]

    # This section will display the gamma distribution if desired.
    # bottomquartlowerbounds = []
    # bottomquartupperbounds = []
    # bottommidquartupperbounds = []
    # topmidquartupperbounds = []
    # topquartupperbounds = []

    # for i in bottomquarts:
        # bottomquartlowerbounds.append(i[0])
        # bottomquartupperbounds.append(i[-1])

    # for i in bottommidquarts:
        # bottommidquartupperbounds.append(i[-1])

    # for i in topmidquarts:
        # topmidquartupperbounds.append(i[-1])

    # for i in topquarts:
        # topquartupperbounds.append(i[-1])

    # plot the distribution as well as the quartiles and medians
    # xtoplot = [np.mean(bottomquartlowerbounds), np.mean(bottomquartupperbounds), np.mean(bottommidquartupperbounds),
               # np.mean(topmidquartupperbounds), np.mean(topquartupperbounds)]
    # x = np.linspace(0, 6, 1000)
    # y = x ** (gamma_shape - 1) * (np.exp(-x / gamma_scale) / (sps.gamma(gamma_shape) * gamma_scale ** gamma_shape))
    # plt.plot(x, y, linewidth=2, color='k', alpha=0)
    # plt.fill_between(x, y, where=x > xtoplot[0], color='#4c4cff')
    # plt.fill_between(x, y, where=x > xtoplot[1], color='#7f7fff')
    # plt.fill_between(x, y, where=x > xtoplot[2], color='#b2b2ff')
    # plt.fill_between(x, y, where=x > xtoplot[3], color='#e5e5ff')
    # plt.axvline(x=finalmedians[0], color="#404040", linestyle=":")
    # plt.axvline(x=finalmedians[1], color="#404040", linestyle=":")
    # plt.axvline(x=finalmedians[2], color="#404040", linestyle=":")
    # plt.axvline(x=finalmedians[3], color="#404040", linestyle=":")
    # plt.title("\n".join(wrap('gamma rate categories calculated as the the average of %s median values of 4 equally likely quartiles of %s randomly sampled vaules' % (gamma_iterations, gamma_samples), 60)), fontweight='bold', fontsize=10)
    # plt.text(5, 0.6, "$\kappa$ = %s\n$\\theta$ = $\\frac{1}{\kappa}$" % (gamma_shape))
    # plt.show()

    # gammaaminos = [random.choice(average_medians) for aa in range(n_amino_acids)]

    # return [random.choice(average_medians) for aa in range(n_amino_acids)]
    return random.choices(average_medians, k=n_amino_acids)


def mutate_amino_acid(amino_acid, LG_matrix, LG_residues, LG_indicies):  # b = matrix, a = current amino acid
    """Mutate a residue to another residue based on the LG matrix."""
    # Get the order of the aminos corresponding to the values in the array
    # aminolist = LG_matrix[:, 0].ravel().tolist()
    # Build cumulative sum of row of probabilities corresponding to amino_acid
    # old_aa_index = aminolist.index(amino_acid)
    old_aa_index = LG_indicies[amino_acid]
    aa_cumsum = np.cumsum(np.asarray(LG_matrix[old_aa_index, 1:], dtype=float))
    # Return new_aa_index where random variable <= aa_cumsum[new_aa_index]
    new_aa_index = np.searchsorted(aa_cumsum, np.random.uniform(0, 1))
    # return aminolist[new_aa_index]
    return LG_residues[new_aa_index]


def calculate_fitness(protein, fitness_table):
    """Calculate fitness of a protein given the sequence and fitness values."""
    # protein_fitness = []  # where fitness values will be added
    # for ai, amino_acid in enumerate(protein):
    #     # Append fitness values corresponding to amino_acid at position ai
    #     protein_fitness.append(fitness_table[ai][RESIDUES.index(amino_acid)])
    # protein_fitness = [fitness_table[ai][RESIDUES.index(amino_acid)]
    #                    for ai, amino_acid in enumerate(protein)]
    # protein_fitness = [fitness_table[ai][amino_acid]
    #                    for ai, amino_acid in enumerate(protein)]
    # 2d numpy array
    # protein_fitness = [fitness_table[ai, RESIDUES.index(amino_acid)]
    #                    for ai, amino_acid in enumerate(protein)]
    protein_fitness = [fitness_table[ai, RESIDUES_INDEX[amino_acid]]
                       for ai, amino_acid in enumerate(protein)]
    # DataFrame
    # protein_fitness = [fitness_table.loc[ai, amino_acid]
    #                    for ai, amino_acid in enumerate(protein)]
    return sum(protein_fitness)


def superfit(fitness_table, variant_sites, initial_protein, fitness_level):
    """Generate a protein of a specified fitness.

    Make either a superfit protein, a superunfit protein or a 'medium
    fitness' protein with fitness just higher than the fitness threshold.

    This function currently locks in invariant sites before finding the fittest
    sites, meaning the invariant sites are simply sampled from the normal
    distribution, and not from the superfit distribution.

    Generating the protein in this manner avoids bias towards increased fitness
    that could be generated by the invariant sites.
    """
    if fitness_level != 'medium':
        if fitness_level == 'low':  # generate unfit protein
            sequence = [0, 1, 2]  # Three lowest

        elif fitness_level == 'high':  # generate superfit protein
            sequence = [-1, -2, -3]  # Three highest

        aminos = ["M"] * 3
        for ai in range(1, len(fitness_table)):  # range(n_amino_acids):
            if ai not in variant_sites:
                aminos.append([initial_protein[ai]] * 3)
            else:
                i_sorted = np.argsort(fitness_table[ai])
                aminos.append([RESIDUES[fitness_table[ai][i_sorted[start]]]
                               for start in sequence])
                # sorted_aa = sorted(fitness_table[ai], key=lambda k: fitness_table[ai][k])
                # DataFrame
                # sorted_aa = RESIDUES[fitness_table.loc[ai].argsort()]
                # aminos.append(sorted_aa[rank] for rank in sequence)
        afitprotein = []
        # Generate a superunffit protein by randomly picking one of the 3 most fit amino acids at each position
        for candidates in aminos:
            afitprotein.append(random.choice(candidates))

    # Generate medium fitness protein. This module is a little buggy. It takes
    # the starting protein sequence, mutates 5 residues until the protein is
    # fitter, then chooses 5 new residues and continues.
    # If it cannot make a fitter protein with the 5 residues its mutating it
    # reverts back to the previous state and picks 5 new residues.
    elif fitness_level == 'medium':
        n_variants = 5
        start_protein = initial_protein  # Copies the external initial_protein
        start_fitness = calculate_fitness(start_protein, fitness_table)
        # NOTE: This chooses from anchored_sequences whereas the other conditions exclude them FIXED
        new_protein = start_protein  # copy.deepcopy(start_protein)

        while start_fitness < fitness_threshold+10 or start_fitness > fitness_threshold+20:
            # Mutate the new protein (sample without replacement)
            chosen_variants = random.sample(variant_sites, n_variants)
            for aa in chosen_variants:
                new_protein[aa] = random.choice(RESIDUES)
            new_fitness = calculate_fitness(new_protein, fitness_table)
            counter = 0

            if start_fitness < fitness_threshold+5:  # setting lower bounds of medium fitness
                while new_fitness < start_fitness and counter <= 100:
                    # Continue to mutate until it is better than the start_protein
                    new_protein = start_protein  # copy.deepcopy(start_protein)
                    for aa in chosen_variants:
                        new_protein[aa] = random.choice(RESIDUES)
                    new_fitness = calculate_fitness(new_protein, fitness_table)
                    counter += 1

            elif start_fitness > fitness_threshold+10:  # set upper bounds of medium fitness
                while new_fitness > start_fitness and counter <= 100:
                    # Continue to mutate until it is better than the start_protein
                    new_protein = start_protein  # copy.deepcopy(start_protein)
                    for aa in chosen_variants:
                        new_protein[aa] = random.choice(RESIDUES)
                    new_fitness = calculate_fitness(new_protein, fitness_table)
                    counter += 1

            start_protein = new_protein
            start_fitness = calculate_fitness(start_protein, fitness_table)
        afitprotein = start_protein

    return afitprotein

    # if fitness_level == 'low':  # generate unfit protein
    #     unfittestaminos = []
    #     for i in range(len(fitness_table)):
    #         if i == 0:
    #             unfittestaminos.append(["M"] * 3)
    #         elif i not in variant_aminos and not 0:
    #             toappend = initial_protein[i]
    #             unfittestaminos.append([toappend] * 3)  # add invariant sites if an anchor position is defined
    #         else:  # find the indexes of the 3 least fit amino acids in RESIDUES and record then as lists for each position
    #             # unfitaminos = []
    #             # amin = fitness_table[i]
    #             # aminsort = sorted(amin)
    #             # unfittestaminoposition = amin.index(aminsort[0])
    #             # secondunfittestaminoposition = amin.index(aminsort[1])
    #             # thirdunfittestaminoposition = amin.index(aminsort[2])
    #             # unfittestamino = RESIDUES[unfittestaminoposition]
    #             # secondunfittestamino = RESIDUES[secondunfittestaminoposition]
    #             # thirdunfittestamino = RESIDUES[thirdunfittestaminoposition]
    #             # unfitaminos.append(unfittestamino)
    #             # unfitaminos.append(secondunfittestamino)
    #             # unfitaminos.append(thirdunfittestamino)
    #             # unfittestaminos.append(unfitaminos)
    #
    #             i_sorted = np.argsort(fitness_table[i])
    #             unfittestaminos.append([RESIDUES[fitness_table[i][i_sorted[start]]] for start in [0, 1, 2]])
    #
    #     afitprotein = []
    #     # for j in range(len(unfittestaminos)):  # generate a superunffit protein by randomly picking one of the 3 most fit amino acids at each position
    #     #     randombin = random.randint(0, 2)
    #     #     possibleaminos = unfittestaminos[j]
    #     #     afitprotein.append(possibleaminos[randombin])
    #
    #     # Generate a superunffit protein by randomly picking one of the 3 most fit amino acids at each position
    #     for candidates in unfittestaminos:
    #         afitprotein.append(random.choice(candidates))
    #
    # if fitness_level == 'high':  # generate superfit protein
    #     fittestaminos = []
    #     for i in range(len(fitness_table)):
    #         if i == 0:
    #             fittestaminos.append(["M", "M", "M"])
    #         elif i not in variant_aminos and not 0:
    #             toappend = initial_protein[i]
    #             fittestaminos.append([toappend, toappend, toappend])  # add invariant sites if an anchor position is defined
    #         else:  # find the indexes of the 3 fittest amino acids in RESIDUES and record then as lists for each position
    #             fitaminos = []
    #             amin = fitness_table[i]
    #             aminsort = sorted(amin)
    #             fittestaminoposition = amin.index(max(amin))
    #             secondfittestaminoposition = amin.index(aminsort[-2])
    #             thirdfittestaminoposition = amin.index(aminsort[-3])
    #             fittestamino = RESIDUES[fittestaminoposition]
    #             secondfittestamino = RESIDUES[secondfittestaminoposition]
    #             thirdfittestamino = RESIDUES[thirdfittestaminoposition]
    #             fitaminos.append(fittestamino)
    #             fitaminos.append(secondfittestamino)
    #             fitaminos.append(thirdfittestamino)
    #             fittestaminos.append(fitaminos)
    #     afitprotein = []
    #
    #     for j in range(len(fittestaminos)):  # generate a superfit protein by randomly picking one of the 3 most fit amino acids at each position
    #         randombin = random.randint(0, 2)
    #         possibleaminos = fittestaminos[j]
    #         afitprotein.append(possibleaminos[randombin])
    # # generate medium fitness protein. This module is a little buggy. It takes the starting protein sequence, mutates 5 residues until the protein is fitter, then chooses 5 new residues and continues.
    # # If it cannot make a fitter protein with the 5 residues its mutating it reverts back to the previous state and picks 5 new residues.
    # if fitness_level == 'medium':
    #     startprotein = initial_protein
    #     startproteinfitness = calculate_fitness(startprotein, fitness_table)
    #     variantstochoosefrom = variant_aminos
    #     secondprotein = startprotein
    #
    #     while startproteinfitness < fitness_threshold+30:
    #         choiceofvariants = random.sample(variantstochoosefrom, 5)
    #         secondprotein[choiceofvariants[0]] = random.choice(RESIDUES)
    #         secondprotein[choiceofvariants[1]] = random.choice(RESIDUES)
    #         secondprotein[choiceofvariants[2]] = random.choice(RESIDUES)
    #         secondprotein[choiceofvariants[3]] = random.choice(RESIDUES)
    #         secondprotein[choiceofvariants[4]] = random.choice(RESIDUES)
    #         secondproteinfitness = calculate_fitness(secondprotein, fitness_table)
    #         counting = 0
    #
    #         while secondproteinfitness < startproteinfitness:
    #             secondprotein = startprotein
    #             secondproteinfitness = calculate_fitness(secondprotein, fitness_table)
    #             secondprotein[choiceofvariants[0]] = random.choice(RESIDUES)
    #             secondprotein[choiceofvariants[1]] = random.choice(RESIDUES)
    #             secondprotein[choiceofvariants[2]] = random.choice(RESIDUES)
    #             secondprotein[choiceofvariants[3]] = random.choice(RESIDUES)
    #             secondprotein[choiceofvariants[4]] = random.choice(RESIDUES)
    #             secondproteinfitness = calculate_fitness(secondprotein, fitness_table)
    #             counting += 1
    #
    #             if counting > 99:
    #                 choiceofvariants = random.sample(variantstochoosefrom, 5)
    #                 counting -= 100
    #                 break
    #
    #         startprotein = secondprotein
    #         startproteinfitness = calculate_fitness(startprotein, fitness_table)
    #     afitprotein = startprotein
    #
    # return afitprotein


# NOTE: Not used
def plot_fitness_histogram(n_proteins, n_amino_acids, fitness_table):
    """Generate and plot fitness values for f proteins."""
    fitnesses = [calculate_fitness(generate_protein(n_amino_acids), fitness_table)
                 for p in range(n_proteins)]
    plt.hist(fitnesses, density=True)  # plot fitnesses as histogram
    plt.show()
    return


def mutate(current_generation, n_mutations_per_gen, variant_sites,
           gamma_categories, LG_matrix, LG_residues, LG_indicies):
    """Mutate a given sequence based on the LG+I+G model of amino acid
    substitution.
    """
    # NOTE: This could be removed for speed after checking it is not used later
    # next_generation = copy.deepcopy(current_generation)  # make a deep copy of the library so changing the library in the function doesn't change the library outside the function
    next_generation = current_generation

    # Sum gammas to make a probability distribution to randomly select from.
    cumulative_gamma = np.cumsum(gamma_categories)  # n_amino_acids long

    for q in range(n_mutations_per_gen):  # impliment gamma
        # Pick random key, clone to make a random generation
        mutant_key, mutant_clone = random.choice(list(next_generation.items()))
        mutation_target = copy.deepcopy(mutant_clone)  # make a deep copy of the libaries value as to not change it in the library until we want to
        # NOTE: There is no way to revert to the original so copying is unnecessary?!
        # mutation_target = mutant_clone

        # mutated_residues = []
        # residue_index = [0]
        # while residue_index[0] not in variant_sites:  # always initiates as residue_index set to 0 and residue zero 0 should always be disallowed (start methionine locked). Also ensures only mutates at variant sites
        #     mutant_residue_area = np.random.uniform(0, cumulative_gamma[-1])  # [0, highest_gamma_sum)
        #     # Find the first bin with gamma > mutant_residue_area
        #     for gi, gamma in enumerate(cumulative_gamma):
        #         if mutant_residue_area < gamma:
        #             mutated_residues.append(gi)
        #             residue_index[0] = gi
        #             break
        #         else:
        #             continue
        # # residue_index = random.choice(c) # pick a random residue in the selected mutant to mutate that isnt the start M or an anchor (old)
        # target_residue = mutation_target[residue_index[0]]
        # newresidue = mutate_amino_acid(target_residue, LG_matrix)  # implement LG
        # mutation_target[residue_index[0]] = newresidue  # mutate the copy with the randomly chosen residue

        # TODO: Could gamma_categories be the length of anchor_sites?
        residue_index = 0
        # cumulative_gamma[anchor_sites] = -np.inf  # Would this work instead of while loop?
        # while residue_index in anchor_sites:
        while residue_index not in variant_sites:
            mutant_residue_area = np.random.uniform(0, cumulative_gamma[-1])  # [0, highest_gamma_sum)
            residue_index = np.searchsorted(cumulative_gamma, mutant_residue_area)  # Return index where mutant_residue_area <= cumulative_gamma[i]

        mutation_target[residue_index] = mutate_amino_acid(mutation_target[residue_index], LG_matrix, LG_residues, LG_indicies)  # mutate the copy with the randomly chosen residue
        next_generation[mutant_key] = mutation_target  # update with new sequence

    return next_generation


def calculate_generation_fitness(population, fitness_table):
    """Calculate the fitness of every protein in a population."""
    # Record calculated fitness for each protein in dictionary
    return {pi: calculate_fitness(protein, fitness_table)
            for pi, protein in list(population.items())}
            # for pi, protein in enumerate(population)}
    # TODO: Replace with list once population is a list (or OrderedDict)
    # fitnesslist = [calculate_fitness(protein, fitness_table)
    #                for pi, protein in list(population.items())]


def build_generation_fitness_table(population, variant_sites, fitness_table):
    """Build a fitness table for given generation's population.

    The array has one row for each protein in the population and the fitness
    value for each amino acid in its position.
    """
    dist_clone_fitness = []
    # Find and plot all fitness values in the current generation
    for pi, protein in list(population.items()):

        if record["invariants"]:
            protein_fitness = [fitness_table[ai, RESIDUES_INDEX[amino_acid]]
                               for ai, amino_acid in enumerate(protein)]
        else:
            protein_fitness = [fitness_table[ai, RESIDUES_INDEX[amino_acid]]
                               for ai, amino_acid in enumerate(protein)
                               if ai in variant_sites]

        dist_clone_fitness.append(protein_fitness)  # Becomes a new row
    return np.asarray(dist_clone_fitness)


def plot_threshold_fitness(generation, population, variant_sites, fitness_table, fitfullname):
    # Store fitness values for each amino in the dataset for the left side of the figure
    # (n_amino_acids, n_variants) = fitness_table.shape
    mean_initial_fitness = np.mean(fitness_table)  # Average across flattened array

    plt.figure()
    plt.subplot(121)
    # Plot each column of fitness_table as a separate dataseries against 0..N-1
    plt.plot(fitness_table, ".", color='k')  # np.arange(n_amino_acids)+1,
    plt.plot([0, n_amino_acids-1], [mean_initial_fitness, mean_initial_fitness], 'r--', lw=3)
    plt.ylim(((-4 * sigma) - 1), ((4 * sigma) + 1))
    plt.xticks([])  # n_variants
    plt.text(0, 3.5*sigma, "\n".join([r"$\mu_1$ = {:.3}".format(mean_initial_fitness),
                                    "threshold = {}".format(fitness_threshold)]), size=6.5)
    plt.title(r"Fitness distribution of $\Delta T_m$ matrix", size=8)

    # Find and plot all fitness values in the current generation
    generation_fitneses = build_generation_fitness_table(population, variant_sites, fitness_table)
    # TODO: Check this is the intended average value to check. Previously it was np.mean(Y2fitness) i.e. the mean of the last protein in the loop
    mean_generation_fitness = np.mean(generation_fitneses)
    plt.subplot(122)
    plt.plot(np.arange(len(population)), generation_fitneses, "o", markersize=2)  # plot y using x as index array 0..N-1
    plt.plot([0, len(population)-1], [mean_generation_fitness, mean_generation_fitness], 'r--', lw=3)
    plt.ylim(((-4 * sigma) - 1), ((4 * sigma) + 1))
    plt.xticks([])  # len(population)
    plt.text(0, 3.5*sigma, "\n".join([r"$\mu_2$ = {:.3}".format(mean_generation_fitness),
                                    "threshold = {}".format(fitness_threshold)]), size=6.5)
    plt.title("\n".join(wrap("Fitness distribution of every sequence in the evolving dataset", 40)), size=8)

    plt.subplots_adjust(top=0.85)
    plt.suptitle(('Generation %s' % generation), fontweight='bold')
    plt.savefig(fitfullname)
    plt.close()


def append_ks_statistics(stats_full_name, distribution_fitness, initial_fitness):
    with open(stats_full_name, "a") as stats_file:  # Append to file
        # Kolmogorov-Smirnov test of similarity to original distributuion
        ksdata = sp.stats.ks_2samp(distribution_fitness, initial_fitness)
        stats_file.write("\n\n\n2-sided Kolmogorov-Smirnov test of similarity "
                         "between the fitness space and evolving protein:\n\n")
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


def write_histogram_statistics(stats_full_name, aa_variant_fitnesses):
    """record["hist_fitness_stats"] == True"""
    # This section writes a file describing 5 statistical tests on the global fitness space.

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
    # for i in distshapirolist:
    for aa in aa_variant_fitnesses:
        (_, p_value) = sp.stats.shapiro(aa)
        if p_value >= 0.05:
            passpercentcalc.append(1)
        else:
            passpercentcalc.append(0)
    # passpercent = (sum(passpercentcalc) / len(passpercentcalc)) * 100
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
                         "entire dataset.")
    else:
        level_index = np.searchsorted(anderson_results.critical_values,
                                      anderson_results.statistic, side="left")
        stats_file.write("Therefore according to the Anderson-Darling test, "
                         "the hypothesis of normality is not rejected at the "
                         "{}% significance level for the entire dataset.\n\n"
                         .format(significance_levels[level_index]))

    # Set up output for significance levels - final bin represents "reject"
    # hypothesis_tally = [0 for result in range(len(significance_levels) + 1)]
    hypothesis_tally = np.zeros(len(significance_levels) + 1)
    # for result in distandersonlist:
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
    # for i in distskewkurtalllist:
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


def plot_histogram_of_fitness(disthistfullname, distributions, initial):
    plt.figure()
    plt.axis([-10, 8, 0, 0.5])  # generate attractive figure

    mu1distspace = sum(initial) / len(initial)  # plot normal distribution of the original fitness space
    plt.hist(initial, 50, density=True, color='k', alpha=0.4)
    plt.title("\n".join(wrap('Fitness distribution of the total fitness space', 60)), fontweight='bold')
    plt.axvline(x=mu1distspace, color="#404040", linestyle=":")

    mu2distspace = sum(distributions) / len(distributions)
    plt.hist(distributions, 50, density=True, color='r', alpha=0.4)
    plt.title("\n".join(wrap('Fitness distribution of the total fitness space vs. changing fitness distribution across every evolving clone', 60)), fontweight='bold')
    plt.axvline(x=mu2distspace, color="#404040", linestyle=":")
    plt.text(4.1, 0.42, "\n".join([r"$\mu_1$ = {:.3}".format(mu1distspace),
                                   r"$\mu_2$ = {:.3}".format(mu2distspace),
                                   "threshold = {}".format(fitness_threshold)]))

    plt.savefig(disthistfullname)
    plt.close()


# NOTE: variant_sites is passed in as invariant_sites!
def record_generation_fitness(generation, population, variant_sites,
                              fitness_table, fitness_threshold, record, run_path):
    """Record the fitness of every protein in the generation and store them in
    dictionary. Optionally generate data and figures about fitness.
    """

    if not (generation == 0 or generation % record["rate"] == 0):
        return

    if record["dot_fitness"]:
        # # TODO: There will be a bug in plotting the mean fitness at the wrong x point (as before)
        fitfilename = "generation_{}.png".format(generation)  # define dynamic filename
        fitfullname = os.path.join(run_path, "fitnessdotmatrix", fitfilename)
        plot_threshold_fitness(generation, population, variant_sites, fitness_table, fitfullname)

    # Build distribution of fitness values existing in evolving protein
    generation_fitneses = build_generation_fitness_table(population, variant_sites, fitness_table)

    if record["hist_fitness_stats"]:
        # Write a file describing 5 statistical tests on the protein fitness space
        if generation == 0:
            stats_file_name = "normal_distribution_statistics_fitness_space.txt"
            distributions = fitness_table
        else:
            stats_file_name = "normal_distribution_statistics_generation{}.txt".format(generation)
            distributions = generation_fitneses
        stats_full_name = os.path.join(run_path, "fitnessdistribution",
                                       "statistics", stats_file_name)
        write_histogram_statistics(stats_full_name, distributions)
        if generation > 0:
            append_ks_statistics(stats_full_name, distributions.ravel(),
                                 fitness_table.ravel())

    if record["hist_fitness"]:

        disthistfilename = "generation_{}.png".format(generation)  # define dynamic filename
        disthistfullname = os.path.join(run_path, "fitnessdistribution",
                                        "histograms", disthistfilename)

        plot_histogram_of_fitness(disthistfullname, generation_fitneses.ravel(),
                                  fitness_table.ravel())


def write_fasta_alignment(population, generation, run_path):  # x = current generation of sequence, y = generation number
    """Write fasta alignment from sequences provided."""
    fastafilepath = os.path.join(run_path, "fastas")
    fastafilename = "generation_{}.fasta".format(generation)  # define dynamic filename
    fullname = os.path.join(fastafilepath, fastafilename)
    with open(fullname, "w") as fastafile:  # open file
        # Write fasta header followed by residue in generation string
        # TODO: This should be an ordered dict or list to preserve the order...
        for p, protein in list(population.items()):
            fastafile.write("\n>clone_%s\n" % (p+1))
            for residue in protein:
                fastafile.write(residue)


def write_final_fasta(population, bifurcations, n_roots, run_path):
    bifsize = 0
    for bifs in bifurcations:
        bifsize += len(bifs)
    bifursize = bifsize/len(bifurcations)
    n_clones_to_take = int((bifursize-1)/2)  # if 5, gives 2, if 4 gives 2, if 3 gives 1.
    generation_numbers = []
    for i in bifurcations:
        clone_selection = random.sample(set(i), n_clones_to_take)  # NOTE: Only sample from unique clones? YES
        for c in clone_selection:
            generation_numbers.append(c)

    full_name = os.path.join(run_path, "treefastas", "selected_fastas.fasta")
    with open(full_name, "w") as treefastafile:  # open file
        # Write fasta header followed by residue in generation string
        for p in generation_numbers:
            treefastafile.write(">clone_%s\n" % (p+1))
            for residue in population[p]:
                treefastafile.write(residue)
            treefastafile.write('\n')
        # Choose a random root to write
        root = population[random.choice(n_roots)]
        treefastafile.write(">root\n")
        for m in root:
            treefastafile.write(m)


def replace_protein(protein, candidates, fitnesses, fitness_threshold):
    # Filter out original protein and those below the fitness threshold
    pool = [c for c in candidates
            if c != protein and fitnesses[c] >= fitness_threshold]
    if len(pool) > 0:
        new_protein = random.choice(pool)
    else:
        # raise Exception("No suitable candidates on branch!")
        warnings.warn("No suitable candidates on branch!")
        new_protein = np.nan
    return new_protein


def evolve(n_generations, initial_population, fitness_table, fitness_threshold,
           variant_sites, gamma_categories, n_mutations_per_gen, fasta_rate,
           LG_matrix, LG_residues, LG_indicies, run_path):
    """Generation generator - mutate a protein for a defined number of
    generations according to an LG matrix and gamma distribution.
    """

    # Generate list of clone keys for bifurication
    clonelist = list(range(n_clones))
    # Randomly sample without replacement n_roots items from clonelist
    rootlist = random.sample(clonelist, n_roots)
    for r in rootlist:
        clonelist.remove(r)
    # rootlist = [clonelist.pop(random.randrange(len(clonelist))) for r in range(n_roots)]

    rootsfullname = os.path.join(run_path, "start", "Roots.txt")
    with open(rootsfullname, "w") as rootsfile:  # open file
        rootsfile.write('Roots:')
        for k in rootlist:
            rootsfile.write('\nClone %s' % str(k+1))

    # Calculate number of bifurications per generation.
    bifuraction_start = n_clones - n_roots
    bifurlist = [1]
    for m in bifurlist:
        bifurlist.append(1)
        bifuraction_start /= 2
        if bifuraction_start < 6:  # stop when there are 3, 4, 5 or 6 leaves per branch.
            break
    bifurgeneration = int(n_generations/len(bifurlist))  # number of generations per bifurication.

    clonelistlist = []  # place to store bifurcations (list of lists of clone keys)
    clonelistlist.append(clonelist)  # store all clones that are not root to start
    population = copy.deepcopy(initial_population)  # current generation
    fitnesses = calculate_generation_fitness(population, fitness_table)
    # Record initial population
    record_generation_fitness(0, population, variant_sites,
                              fitness_table, fitness_threshold,
                              record, run_path)
    write_fasta_alignment(population, 0, run_path)

    # Store each generation along with its fitness
    Generation = namedtuple('Generation', ['population', 'fitness'])
    # Create a list of generations and add initial population and fitness
    evolution = [Generation(population=population, fitness=fitnesses)]

    for gen in trange(n_generations):  # run evolution for n_generations
        # Bifuricationmaker. Bifuricates in even generation numbers so every branch on tree has 3 leaves that have been evolving by the last generation
        if gen % bifurgeneration == 0 and gen > 0 and len(clonelistlist[0]) > 3:
            lists = []  # space to store bifurcations before adding them to clonelistlist
            for clones in clonelistlist:  # bifuricate each set of leaves
                random.shuffle(clones)
                midpoint = int(len(clones)/2)
                lists.append(clones[midpoint:])
                lists.append(clones[:midpoint])
            del clonelistlist[:]
            for k in lists:  # append bifurcations to a cleared clonelistlist
                clonelistlist.append(k)

        # TODO: If no suitable clones are available, re-mutate the generation and start again
        # Mutate population
        population = mutate(population, n_mutations_per_gen,
                            variant_sites, gamma_categories,
                            LG_matrix, LG_residues, LG_indicies)
        # TODO: Split out writing to file until the end so that all branches are valid
        # Re-calculate fitness
        if gen == 0 or gen % record["rate"] == 0:  # TODO: (gen+1) see write_fasta_alignment
            fitnesses = calculate_generation_fitness(population, fitness_table)

        for pi in range(len(fitnesses)):  # if there are, start loop on fitnesses
            if fitnesses[pi] < fitness_threshold:  # if fitness is less than threshold clone a random sequence in its place.
                if pi in rootlist:
                    clonekey = replace_protein(pi, rootlist, fitnesses,
                                               fitness_threshold)
                else:  # Protein is in one of the branches
                    for branch in clonelistlist:
                        if pi in branch:
                            clonekey = replace_protein(pi, branch, fitnesses,
                                                       fitness_threshold)
                if np.isnan(clonekey):
                    # Could not find fit enough candidate
                    warnings.warn("clone %s is unfit with a value of %s, it will be replaced by:" % (pi, fitnesses[pi]))
                    warnings.warn("clone %s with a fitness of %s" % (clonekey, fitnesses[clonekey]))
                    warnings.warn('Clonekey fitness is too low or mutation rate is too high')  # Bug in this section that causes infinite loop if mutation rate is too high. Happens when a bifurication has a small number of clones to be replaced by, and the high mutation rate causes all clones to dip below the threshold in one generation.
                    print(fitnesses)
                    print(clonelistlist)
                else:
                    population[pi] = population[clonekey]  # swap out unfit clone for fit clone

        # Allow sequences to die and be replacecd at a predefined rate
        if deaths_per_generation > 0 and gen % deaths_per_generation == 0:
            # NOTE: Originally this was sampled with replacement: FIXED
            mortals = random.sample(range(n_clones), int(n_clones * death_ratio))
            for pi in mortals:
                if pi in rootlist:
                    clonekey = replace_protein(pi, rootlist, fitnesses,
                                               fitness_threshold)
                else:  # Protein is in one of the branches
                    for branch in clonelistlist:
                        if pi in branch:
                            clonekey = replace_protein(pi, branch, fitnesses,
                                                       fitness_threshold)
                if np.isnan(clonekey):
                    warnings.warn("Unable to kill protein {} - no suitable replacements!".format(pi))
                else:
                    population[pi] = population[clonekey]  # Replace dead protein

        # Record population details at the end of processing
        record_generation_fitness(gen, population, variant_sites,
                                  fitness_table, fitness_threshold,
                                  record, run_path)
        evolution.append(Generation(population=population, fitness=fitnesses))
        # NOTE: Should this be at the end of each timestep? FIXED
        if ((gen+1) % fasta_rate) == 0:  # write fasta every record["fasta_rate"] generations
            write_fasta_alignment(population, gen+1, run_path)
    write_final_fasta(population, clonelistlist, rootlist, run_path)

    return evolution


def fitbit(evolution, n_generations, n_clones, initial_protein, fitness_table, run_path):
    """Plot fitness against generation for all clones."""
    # NOTE: Final element previously excluded - for i in range(len(evolution)-1):  FIXED
    # Create array of fitness values with shape (n_generations, n_clones)
    # fitnesses = np.array([[evolution[g][-1][c] for c in range(n_clones)]
    #                       for g in range(n_generations)])
    fitnesses = np.array([[evolution[g].fitness[c] for c in range(n_clones)]
                          for g in range(n_generations+1)])

    initial_fitness = calculate_fitness(initial_protein, fitness_table)
    generation_numbers = np.arange(n_generations+1)  # Skip initial generation

    plt.figure()
    plt.plot(generation_numbers, fitnesses)
    plt.plot([0, n_generations], [fitness_threshold, fitness_threshold], 'k-', lw=2)
    # NOTE: Final element previously excluded - for c in range(n_clones-1):
    plt.plot(generation_numbers, np.mean(fitnesses, axis=1), "k--", lw=2)  # Average across clones
    # plt.ylim([fitness_threshold-25, initial_fitness+10])  # not suitable for "low or med" graphs
    # plt.ylim([fitness_threshold-5, ((n_amino_acids+1)*mu)+80]) # for low graphs
    plt.ylim([fitness_threshold-25, initial_fitness+100])  # suitable for med graphs
    plt.xlim([0, n_generations])
    plt.xlabel("Generations", fontweight='bold')
    plt.ylabel("$T_m$", fontweight='bold')
    plt.title("\n".join(wrap('Fitness change for %s randomly generated "superfit" clones of %s amino acids, mutated over %s generations' % (n_clones, (n_amino_acids), n_generations), 60)), fontweight='bold')
    plt.text(n_generations+15, fitness_threshold-3, r"$\Omega$")
    plt.text(n_generations-1000, initial_fitness+50,
             "\n".join([r"$\mu$ = {}".format(mu),
                        r"$\sigma$ = {}".format(sigma),
                        r"$\delta$ = {}".format(mutation_rate)]))

    # Define dynamic filename
    fitgraphfilename = "fitness_change_over{}generations.png".format(n_generations)
    fitgraphfullname = os.path.join(run_path, "fitnessgraph", fitgraphfilename)
    plt.savefig(fitgraphfullname)
    return


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


def write_settings_file(run_path, **kwargs):
    settingsfullname = os.path.join(run_path, "runsettings", "runsettings.txt")
    # TODO: Change to logger
    with open(settingsfullname, "w") as sf:  # open file
        sf.write("Random number generator seed: {}\n".format(seed))
        sf.write("Protein length: %s" % (n_amino_acids))
        sf.write("\nAmount of mutations per generation: %s" % n_mutations_per_gen)
        sf.write("\nAmount of clones in the population: %s" % n_clones)
        sf.write("\nAmount of generations simulation is run for: %s" % n_generations)
        sf.write("\nFitness threshold: %s" % fitness_threshold)
        sf.write("\n\nNormal distribution properties: mu = %s, sigma = %s" % (mu, sigma))
        sf.write("\nGamma distribution properties: kappa = %s, theta = %s" % (gamma["shape"], gamma["scale"]))
        sf.write("\n\nWrite rate for FASTA: every %s generations" % record["fasta_rate"])
        sf.write("\n\nTrack rate for graphing and statistics: every %s generations" % record["rate"])
        sf.write("\nTracking state: Fitness dot matrix = %s; Fitness histogram = %s; Fitness normality statistics = %s" % (record["dot_fitness"], record["hist_fitness"], record["hist_fitness_stats"]))


def get_LG_matrix(full_file_name=None):
    """Get .csv file defining aa substitution probabilities calculated from R
    matrix multiplied by PI matrix, with diagonals forced to zero as mutation
    has to happen then conferted to event rates p(lambda) where lambda = sum Qx
    and p(lambda)x=Qxy/lambda
    """
    if full_file_name is None:
        full_file_name = os.path.join("data", "LGaa.csv")
    with open(full_file_name) as matrix_file:  # Open in read-only mode
        LG_matrix_list = list(csv.reader(matrix_file, delimiter=","))
    LG_matrix = np.array(LG_matrix_list)  # load matrix into a numpy array
    LG_residues = LG_matrix[0, 1:]  # Get first row skipping first element ('0')
    LG_indicies = {aa: ai for ai, aa in enumerate(LG_residues)}
    LG_matrix = np.delete(LG_matrix, 0, axis=0)  # trim first line of the array as it's not useful
    return (LG_matrix, LG_residues, LG_indicies)


def write_initial_protein(run_path, initial_protein):
    protein_full_name = os.path.join(run_path, "start", "firstprotein.fas")
    with open(protein_full_name, "w") as ipf:  # open file
        ipf.write('>firstprotein\n')
        for prot in initial_protein:
            ipf.write(prot)


def pest(n_generations, fitness_start, fitness_threshold, mu, sigma,
         n_clones=52, n_amino_acids=80, mutation_rate=0.001,
         n_mutations_per_gen=None, n_anchors=None,
         deaths_per_generation=5, death_ratio=0.05, seed=None,
         n_roots=4, gamma=None, record=None):

    if n_mutations_per_gen is None:
        n_mutations_per_gen = int(n_clones*(n_amino_acids)*mutation_rate)
    if n_anchors is None:
        n_anchors = int((n_amino_acids)/10)
    # TODO: switch from random to np.random for proper seeding
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    # TODO: Put run_path (and subdirs) in record dict
    # create folder and subfolders
    # PWD = os.path.dirname(__file__)
    run_path = create_output_folders()
    # record run settings
    write_settings_file(run_path)
    # TODO: Refactor to use the same ordering as RESIDUES
    (LG_matrix, LG_residues, LG_indicies) = get_LG_matrix()  # Load LG matrix

    initial_protein = generate_protein(n_amino_acids)  # make first protein
    fitness_table = get_protein_fitness(n_amino_acids)  # make first fitness dictionary
    variant_sites = get_allowed_sites(n_amino_acids, n_anchors)  # generate invariant sites
    initial_protein = superfit(fitness_table, variant_sites, initial_protein, fitness_start)  # generate a superfit protein taking into account the invariant sites created (calling variables in this order stops the evolutionary process being biased by superfit invariant sites.)
    gamma_categories = gamma_ray(n_amino_acids, gamma)  # generate gamma categories for every site
    write_initial_protein(run_path, initial_protein)  # Record initial protein
    initial_population = clone_protein(initial_protein, n_clones)  # make some clones to seed evolution
    evolution = evolve(n_generations, initial_population, fitness_table,
                       fitness_threshold, variant_sites, gamma_categories,
                       n_mutations_per_gen, record["fasta_rate"],
                       LG_matrix, LG_residues, LG_indicies, run_path)
    fitbit(evolution, n_generations, n_clones, initial_protein, fitness_table, run_path)


if __name__ == '__main__':
    # NOTE: __path__ is initialized to be a list containing the name of the directory holding the package’s __init__.py

    # from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    # from matplotlib.figure import Figure
    # matplotlib.use('TkAgg')
    mpl.rc('savefig', dpi=300)

    # TODO: Give these default values

    # parameters of protein evolution
    n_generations = 2000  # amount of generations the protein evolves for
    fitness_start = 'medium'  # high, medium or low; must be lower case. If selecting low, fitness threshold needs to be significantly smaller (i.e. 4x) than #positions*mu
    # TODO: Create parameters for the numeric literals which define teh medium boundaries
    fitness_threshold = 0  # arbitrary number for fitness threshold
    # parameters for normal distribution used to select fitness values
    mu = -1.2
    sigma = 2.5

    # TODO: These could possibly gi into their own dictionary too
    n_clones = 52  # amount of clones that will be generated in the first generation #5 10 20 40 80
    # TODO: Change to the more logical n+1!
    n_amino_acids = 80  # number of amino acids in the protein after the start methionine
    mutation_rate = 0.001  # should be small!
    # TODO: Allow user to pass a number but defau;t to None and calculate as follows
    n_mutations_per_gen = int(n_clones*(n_amino_acids)*mutation_rate)  # number of mutations per generation
    n_anchors = int((n_amino_acids)/10)  # amount of invariant sites in a generation (not including root)
    deaths_per_generation = 5  # Set to 0 to turn off protein deaths
    death_ratio = 0.05
    seed = 42

    # TODO: Place bifurcation parameters into kwargs dict with a flag for bifurcations
    n_roots = 4

    # TODO: Put into dictionary
    # parameters for forming discrete gamma distribution used for evolution of protein
    gamma = {"shape": 1.9,  # Most phylogenetic systems that use gamma only let you set kappa (often called shape alpha) and calculate theta as 1/kappa giving mean of 1
             "scale": 1/1.9,  # NOTE: 1/gamma_shape. Set as default in func?
             "iterations": 50,
             "samples": 10000}
    # gamma_iterations = 100
    # gamma_samples = 10000
    # gamma_shape = 1.9  # Most phylogenetic systems that use gamma only let you set kappa (often called shape alpha) and calculate theta as 1/kappa giving mean of 1
    # gamma_scale = 1/gamma_shape

    # Set what to record
    record = {"rate": 50,           # write a new fasta file every x generations
              "fasta_rate": 50,     # write a new fasta file every x generations
              "dot_fitness": False,
              "hist_fitness_stats": False,
              "hist_fitness": False,
              "invariants": False}

    pest(n_generations, fitness_start, fitness_threshold, mu, sigma, n_clones,
         n_amino_acids, mutation_rate, n_mutations_per_gen, n_anchors,
         deaths_per_generation, death_ratio, seed, n_roots, gamma, record)
