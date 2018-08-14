import copy
import os  # .path
import csv
import datetime
import random
# from random import randint, sample, choice, shuffle  # TODO: Consolidate with numpy
from textwrap import wrap
# import shutil
# import json
from collections import namedtuple
import warnings

import numpy as np
# from numpy import median
# from numpy.random import normal, uniform, gamma
import scipy as sp
# from scipy.stats import anderson, normaltest, skew, skewtest, kurtosistest
# from scipy.stats import shapiro as shp
# from scipy.stats import kurtosis as kurt
# from scipy.stats import ks_2samp as kosmo
# import scipy.special as sps
import matplotlib as mpl
from matplotlib import pyplot as plt

from tqdm import trange


# define starting variables
# amino acids - every fitness value string references residues string
RESIDUES = ("R", "H", "K", "D", "E", "S", "T", "N", "Q", "C",
            "G", "P", "A", "V", "I", "L", "M", "F", "Y", "W")

# parameters for normal distribution used to select fitness values
mu = -1.2
sigma = 2.5

# parameters for forming discrete gamma distribution used for evolution of protein
gamma_iterations = 100
gamma_samples = 10000
gamma_shape = 1.9  # Most phylogenetic systems that use gamma only let you set kappa (often called shape alpha) and calculate theta as 1/kappa giving mean of 1
gamma_scale = 1/gamma_shape

# parameters of protein evolution
n_clones = 52  # amount of clones that will be generated in the first generation #5 10 20 40 80
n_generations = 2000  # amount of generations the protein evolves for
fitness_start = 'medium'  # high, medium or low; must be lower case. If selecting low, fitness threshold needs to be significantly smaller (i.e. 4x) than #positions*mu
fitness_threshold = 0  # arbitrary number for fitness threshold
n_amino_acids = 79  # number of amino acids in the protein after the start methionine
mutation_rate = 0.001  # should be small!
n_mutations_per_generation = int(n_clones*(n_amino_acids+1)*mutation_rate)  # number of mutations per generation
n_anchors = int((n_amino_acids+1)/10)  # amount of invariant sites in a generation (not including root)
n_roots = 4

# set what to record
write_rate = 50  # write a new fasta file every x generations
track_rate = 50  # track every x generations
track_dot_fitness = False  # True or False.
track_hist_fitness_stats = False  # True or False.
track_hist_fitness = False  # True or False.
track_invariants = False  # if True, invariants are tracked in the histogram analysis. If false, invariants are ignored.


def generate_protein(n_amino_acids, start_amino_acid="M"):
    """Generate an original starting protein 20aa long with a start methionine.
    """
    protein = random.choices(RESIDUES, k=n_amino_acids)
    # protein = [random.choice(RESIDUES) for _ in range(n_amino_acids)]  # ver < 3.6
    protein.insert(0, start_amino_acid)  # Start with methionine
    return protein


# NOTE: unused
def test_normal_distribution():
    """Plot a distribution to test normalality."""
    s = np.random.normal(mu, sigma, 2000)  # generate distribution
    count, bins, ignored = plt.hist(s, 30, density=True)  # plot distribuiton
    plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(- (bins - mu)**2 / (2 * sigma**2)), linewidth=2, color='r')
    return plt.show()

# test_normal_distribution()


def fit_module(mu, sigma):
    """Generates an array of fitness values for each amino acid in residues."""
    return np.random.normal(mu, sigma, len(RESIDUES))


def get_protein_fitness(protein):  # x=protein;
    """Generate a dictionary describing list of fitness values at each position
    of the generated protein.
    """
    # Create dictionary containing position in the protein as keys and the array of fitness values for each amino acid as values
    fitnesslib = {i: fit_module(mu, sigma) for i in range(len(protein))}

    aminofitsavepath = '%s/start' % runpath
    aminofitfilename = "fitnesslibrary"
    aminofitfullname = os.path.join(aminofitsavepath, aminofitfilename+".csv")
    aminofile = open(aminofitfullname, "w+")  # open file
    # Write header
    aminofile.write("aminoposition"),
    for aa in RESIDUES:
        aminofile.write(",%s" % aa)
    # Write fitness values
    for i in range(n_amino_acids+1):
        aminofile.write('\n%s' % i),
        for f in fitnesslib[i]:
            aminofile.write(',%s' % f)

    return fitnesslib


def clones(n_clones, protein):
    """Generate a dictionary containing n_clones of generated protein
    - this contains the evolving dataset.
    """
    return {l: protein for l in range(n_clones)}


def get_allowed_sites(n_amino_acids, n_anchors):
    """Select invariant sites in the initially generated protein and return
    allowed values.
    """
    allowed_values = list(range(1, n_amino_acids))  # keys for mutable sites
    # Randomly define invariant sites (without replacement)
    anchored_sequences = random.sample(allowed_values, n_anchors)
    # Remove the invariant sites from allowed values
    for a in anchored_sequences:
        allowed_values.remove(a)

    allowed_values.append(n_amino_acids)
    return allowed_values


def gammaray(a, b, c, d, e):  # a = iterations to run gamma sampling, b = number of gamma samples per iteration, c = gamma shape (kappa), d = gamma scale (theta), e = amount of aminos
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
    medians = []
    medianlower = []
    medianlowermid = []
    medianuppermid = []
    medianupper = []
    bottomquarts = []
    bottommidquarts = []
    topmidquarts = []
    topquarts = []

    for i in (list(range(a))):
        # sample gamma i times
        samples = np.random.gamma(c, d, b)

        # define quartiles in that data with equal probability
        bottomquart = np.percentile(samples, [0, 25], interpolation='midpoint')
        bottomquarts.append(bottomquart)
        bottommidquart = np.percentile(samples, [25, 50], interpolation='midpoint')
        bottommidquarts.append(bottommidquart)
        topmidquart = np.percentile(samples, [50, 75], interpolation='midpoint')
        topmidquarts.append(topmidquart)
        topquart = np.percentile(samples, [75, 100], interpolation='midpoint')
        topquarts.append(topquart)
        # print bottomquart, bottommidquart, topmidquart, topquart

        # generate space for the values within each quartile, sort them, find the median, record the median.
        bottomlist = []
        bottommidlist = []
        topmidlist = []
        toplist = []
        for j in samples:
            if bottomquart[0] <= j < bottomquart[-1]:
                bottomlist.append(j)
            elif bottommidquart[0] <= j < bottommidquart[-1]:
                bottommidlist.append(j)
            elif topmidquart[0] <= j < topmidquart[-1]:
                topmidlist.append(j)
            else:
                toplist.append(j)
        bottomlist.sort()
        bottommidlist.sort()
        topmidlist.sort()
        toplist.sort()
        ratecategoriesquartile = [np.median(bottomlist), np.median(bottommidlist), np.median(topmidlist), np.median(toplist)]
        medians.append(ratecategoriesquartile)

        # print ratecategoriesquartile

    # calculate average of medians from each iteration
    for k in medians:
        medianlower.append(k[0])
        medianlowermid.append(k[1])
        medianuppermid.append(k[2])
        medianupper.append(k[3])

    finalmedians = [np.mean(medianlower), np.mean(medianlowermid), np.mean(medianuppermid), np.mean(medianupper)]

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

    gammaaminos = []
    for i in range(e):
        gammacategorychoice = random.choice(finalmedians)
        gammaaminos.append(gammacategorychoice)

    #print "discrete gamma categories: %s" %(finalmedians)
    #print gammaaminos
    return gammaaminos


def mutate_matrix(a, b):  # a = matrix, b = current amino acid
    """Mutate a residue to another residue based on the LG matrix."""
    aminolist = []  # space for the order of the aminos corresponding to the values in the dictionaries this code makes from the numpy matrix
    for i in a:
        aminolist.append(i[0])

    aminodict = {}  # space to make a dictionary of current possible amino acids and that amino acid's event probability matrix
    for j in a:  # makes the probability dictionary
        aminodictkey = 0
        valuelist = []
        matrixline = j
        for k in range(len(matrixline)):
            if k == 0:
                aminodictkey = matrixline[k]
            else:
                valuelist.append(matrixline[k])
        key = aminodictkey[0]
        aminodict[key] = valuelist

    aminosumdict = {}  # space to make a dictionary of cumulative probability for changing one amino acid to another
    for l in aminodict:  # makes the cumulative probability dictionary
        aminosum = 0
        sumlist = []
        lgforamino = aminodict[l]
        for m in lgforamino:
            aminosum = aminosum + float(m)
            sumlist.append(aminosum)
        aminosumdict[l] = sumlist

    randomgrab = np.random.uniform(0, 1)  # pick a random number in the cumulative probability distribution
    mutationselector = aminosumdict[b]  # pull out the probabilities corresponding to the current amino acid

    newresidue = 0  # space to store the new residue
    for n in mutationselector:  # find the new residue corresponding to the random number by finding the first residue with a cumulative probability bigger than the number selected.
        if randomgrab < n:
            newresidue = aminolist[mutationselector.index(n)]
            break

    return newresidue


def calculate_fitness(z):  # z=protein input. calculates fitness of a protein given the fitness values and the sequence.
    protein = z
    aminofitnesses = []  # where fitness values will be added
    for m in range(len(protein)):
        amino = protein[m]  # find ith amino acid
        fitindex = RESIDUES.index(amino)  # find index of first amino acid in RESIDUES list
        fitstring = proteinfitness[m]  # find fitness values for ith amino acid position
        fitvalue = fitstring[fitindex]  # find fitness value corresponding to amino acid at position
        aminofitnesses.append(fitvalue)  # append these to string
    fitness = 0
    for i in range(len(aminofitnesses)):
        fitness = fitness+aminofitnesses[i]  # compute fitness
    return fitness


def superfit(n, o, p, q):  # n=proteinfitness, o=anchored sequences, p=initial_protein, q = fitness_start; Generates a protein with high, low or medium fitness, with anchored sequences from the initial generation
    """Make either a superfit protein, a superunfit protein or a 'medium
    fitness' protein with fitness just higher than the fitness threshold.

    This function currently locks in invariant sites before finding the fittest
    sites, meaning the invariant sites are simply sampled from the normal
    distribution, and not from the superfit distribution.

    Generating the protien in this manner avoids bias towards increased fitness
    that could be generated by the invariant sites.
    """
    if q == 'low':  # generate unfit protein
        unfittestaminos = []
        for i in range(len(n)):
            if i == 0:
                unfittestaminos.append(["M", "M", "M"])
            elif i not in o and not 0:
                toappend = p[i]
                unfittestaminos.append([toappend, toappend, toappend])  # add invariant sites if an anchor position is defined
            else:  # find the indexes of the 3 least fit amino acids in RESIDUES and record then as lists for each position
                unfitaminos = []
                amin = n[i]
                aminsort = sorted(amin)
                unfittestaminoposition = amin.index(aminsort[0])
                secondunfittestaminoposition = amin.index(aminsort[1])
                thirdunfittestaminoposition = amin.index(aminsort[2])
                unfittestamino = RESIDUES[unfittestaminoposition]
                secondunfittestamino = RESIDUES[secondunfittestaminoposition]
                thirdunfittestamino = RESIDUES[thirdunfittestaminoposition]
                unfitaminos.append(unfittestamino)
                unfitaminos.append(secondunfittestamino)
                unfitaminos.append(thirdunfittestamino)
                unfittestaminos.append(unfitaminos)
        afitprotein = []
        for j in range(len(unfittestaminos)):  # generate a superunffit protein by randomly picking one of the 3 most fit amino acids at each position
            randombin = random.randint(0, 2)
            possibleaminos = unfittestaminos[j]
            afitprotein.append(possibleaminos[randombin])

    if q == 'high':  # generate superfit protein
        fittestaminos = []
        for i in range(len(n)):
            if i == 0:
                fittestaminos.append(["M", "M", "M"])
            elif i not in o and not 0:
                toappend = p[i]
                fittestaminos.append([toappend, toappend, toappend])  # add invariant sites if an anchor position is defined
            else:  # find the indexes of the 3 fittest amino acids in RESIDUES and record then as lists for each position
                fitaminos = []
                amin = n[i]
                aminsort = sorted(amin)
                fittestaminoposition = amin.index(max(amin))
                secondfittestaminoposition = amin.index(aminsort[-2])
                thirdfittestaminoposition = amin.index(aminsort[-3])
                fittestamino = RESIDUES[fittestaminoposition]
                secondfittestamino = RESIDUES[secondfittestaminoposition]
                thirdfittestamino = RESIDUES[thirdfittestaminoposition]
                fitaminos.append(fittestamino)
                fitaminos.append(secondfittestamino)
                fitaminos.append(thirdfittestamino)
                fittestaminos.append(fitaminos)
        afitprotein = []

        for j in range(len(fittestaminos)):  # generate a superfit protein by randomly picking one of the 3 most fit amino acids at each position
            randombin = random.randint(0, 2)
            possibleaminos = fittestaminos[j]
            afitprotein.append(possibleaminos[randombin])
    # generate medium fitness protein. This module is a little buggy. It takes the starting protein sequence, mutates 5 residues until the protein is fitter, then chooses 5 new residues and continues.
    # If it cannot make a fitter protein with the 5 residues its mutating it reverts back to the previous state and picks 5 new residues.
    if q == 'medium':
        startprotein = p
        startproteinfitness = calculate_fitness(startprotein)
        variantstochoosefrom = o
        secondprotein = startprotein

        while startproteinfitness < fitness_threshold+30:
            choiceofvariants = random.sample(variantstochoosefrom, 5)
            secondprotein[choiceofvariants[0]] = random.choice(RESIDUES)
            secondprotein[choiceofvariants[1]] = random.choice(RESIDUES)
            secondprotein[choiceofvariants[2]] = random.choice(RESIDUES)
            secondprotein[choiceofvariants[3]] = random.choice(RESIDUES)
            secondprotein[choiceofvariants[4]] = random.choice(RESIDUES)
            secondproteinfitness = calculate_fitness(secondprotein)
            counting = 0

            while secondproteinfitness < startproteinfitness:
                secondprotein = startprotein
                secondproteinfitness = calculate_fitness(secondprotein)
                secondprotein[choiceofvariants[0]] = random.choice(RESIDUES)
                secondprotein[choiceofvariants[1]] = random.choice(RESIDUES)
                secondprotein[choiceofvariants[2]] = random.choice(RESIDUES)
                secondprotein[choiceofvariants[3]] = random.choice(RESIDUES)
                secondprotein[choiceofvariants[4]] = random.choice(RESIDUES)
                secondproteinfitness = calculate_fitness(secondprotein)
                counting += 1

                if counting > 99:
                    choiceofvariants = random.sample(variantstochoosefrom, 5)
                    counting -= 100
                    break

            startprotein = secondprotein
            startproteinfitness = calculate_fitness(startprotein)
        afitprotein = startprotein

    return afitprotein


# NOTE: Not used
def histfitness(n_proteins):
    """Generate and plot fitness values for f proteins."""
    fitnesses = [calculate_fitness(generate_protein(n_amino_acids))
                 for p in range(n_proteins)]
    plt.hist(fitnesses, density=True)  # plot fitnesses as histogram
    return plt.show()


def get_thresholded_protein(b, c):  # b=fitness_start, c=initial_protein;
    """Make a protein with a fitness value above a defined threshold.

    Note: This takes a long time if thresh is too high.
    """
    f = calculate_fitness(c)
    while f < b:  # keep making proteins until the protein's fitness satisfies the fitness threshold.
        proteintoevolve = generate_protein(n_amino_acids)
        f = calculate_fitness(proteintoevolve)
    return f


def mutate(current_generation, n_mutations_per_generation, variant_sites,
           gamma_categories, LG_matrix):
    """Mutate a given sequence based on the LG+I+G model of amino acid
    substitution.
    """
    next_generation = copy.deepcopy(current_generation)  # make a deep copy of the library so changing the library in the function doesn't change the library outside the function

    # Sum gammas to make a probability distribution to randomly select from.
    cumulative_gamma = np.cumsum(gamma_categories)  # n_amino_acids long

    for q in range(n_mutations_per_generation):  # impliment gamma
        # Pick random key, clone to make a random generation
        mutant_key, mutant_clone = random.choice(list(next_generation.items()))
        mutation_target = copy.deepcopy(mutant_clone)  # make a deep copy of the libaries value as to not change it in the library until we want to

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
        # newresidue = mutate_matrix(LG_matrix, target_residue)  # implement LG
        # mutation_target[residue_index[0]] = newresidue  # mutate the copy with the randomly chosen residue

        # TODO: Could gamma_categories be the length of anchor_sites?
        residue_index = 0
        # cumulative_gamma[anchor_sites] = -np.inf  # Would this work instead of while loop?
        # while residue_index in anchor_sites:
        while residue_index not in variant_sites:
            mutant_residue_area = np.random.uniform(0, cumulative_gamma[-1])  # [0, highest_gamma_sum)
            residue_index = np.searchsorted(cumulative_gamma, mutant_residue_area)  # Return index where mutant_residue_area <= cumulative_gamma[i]
        mutation_target[residue_index] = mutate_matrix(LG_matrix, mutation_target[residue_index])  # mutate the copy with the randomly chosen residue

        next_generation[mutant_key] = mutation_target  # update with new sequence

    return next_generation


def record_generation_fitness(c, d, e, f, g, h, m, n, o):  # c=protein generation; d=track_dot_fitness, e=generationtotrack, f=generationnumber, g=proteinfitness, h=track_hist_fitness_stats, m=track_hist_fitness, n=track invariants, o= invariant sites
    """Record the fitness of every protein in the generation and store them in
    dictionary. Optionally generate data and figures about fitness.
    """
    keycounter = -1  # loop over each key - unelegent but it works
    fitnessdict = {}
    for r in range(len(c)):  # record calculated fitness for each protein in dictionary
        keycounter += 1
        fitnessofprotein = calculate_fitness(c[keycounter])
        fitnessdict.update({keycounter: fitnessofprotein})

    if (d is True) and (f % e == 0 or f == 0):  # if the switch is on, and record fitness on the first generation and every x generation thereafter
        plt.figure()  # make individual figure in pyplot

        fittrackeryaxis = []  # store values for the left side of the figure
        for i in range(n_amino_acids+1):
            fittrackeryaxis.append(g[i])  # for each amino in the dataset append its fitness space to list
        additionleft = 0  # calculate average fitness of dataset (messy but I can keep track of it)
        pointsleft = 0
        for i in fittrackeryaxis:
            for j in i:
                pointsleft += 1
                additionleft += j
        avgtoplotleft = additionleft / pointsleft

        for i in range(len(fittrackeryaxis)):
            yaxistoplot = fittrackeryaxis[i]
            plt.plot(len(yaxistoplot) * [i], yaxistoplot, ".", color='k')  # plot data

        transform = n_amino_acids + (n_amino_acids / 10)  # generate values for right side of the figure. Transform function sets offet to make figure look nice.
        Keycounter = -1
        additionright = 0
        pointsright = 0
        for j in range(len(c)):  # find and plot all fitness values in the current generation
            Keycounter += 1
            Y2fitness = []  # space for values to plot
            if n is True:  # check if invariants need to be ignored
                Y2aminos = c[Keycounter]  # load fitness of each gen (a keys are numbers so easy to iterate)
            else:
                Y2aminos = []  # ignore variant sites
                invariantY2checkaminos = c[Keycounter]
                for i in range(len(invariantY2checkaminos)):
                    if i in o:
                        Y2aminos.append(invariantY2checkaminos[i])
                    else:
                        Y2aminos.append('X')
            for k in range(len(Y2aminos)):  # generate values from generation x to plot
                amino = Y2aminos[k]  # find ith amino acid
                if amino is not 'X':
                    pointsright += 1
                    fitindex = RESIDUES.index(amino)  # find index of first amino acid in RESIDUES list
                    fitstring = proteinfitness[k]  # find fitness values for ith amino acid position
                    fitvalue = fitstring[fitindex]  # find fitness value corresponding to amino acid at position
                    additionright += fitvalue
                    Y2fitness.append(fitvalue)  # append these to list
            #print Y2fitness
            plt.plot(len(Y2fitness) * [j + transform], Y2fitness, "o", markersize=1)  # plot right hand side with small markers
            plt.title("\n".join(wrap('Fitness of every amino acid in the fitness matrix vs fitness of every amino acid in generation %s' % f, 60)), fontweight='bold')
        avgtoplotright = additionright / pointsright  # calculate right side average
        plt.axis([-10, (n_amino_acids * 2) + (n_amino_acids / 10)+10, -11, 11])  # generate attractive figure
        plt.plot([0, len(fittrackeryaxis) + 1], [avgtoplotleft, avgtoplotleft], 'r--', lw=3)
        plt.plot([0 + transform, len(c) + 1 + transform], [avgtoplotright, avgtoplotright], 'r--', lw=3)
        muleftdistdp = "%.3f" % avgtoplotleft
        murightdistdp = "%.3f" % avgtoplotright
        plt.text(0, 8.7, "$\mu$1 = %s\n$\mu$2 = %s\nthreshold = %s" % (muleftdistdp, murightdistdp, fitness_threshold), size = 6.5)

        plt.xticks([])  # remove x axis ticks
        fitfilename = "generation_%s" % f  # define dynamic filename
        fitsavepath = '%s/fitnessdotmatrix' % runpath
        fitfullname = os.path.join(fitsavepath, fitfilename+".png")
        plt.savefig(fitfullname)
        plt.close()  # close plot (so you dont generate 100 individual figures)

    disttrackerlist = []  # build fitness space numbers
    disttrackeryaxis = []
    for i in range(n_amino_acids+1):
        disttrackerlist.append(proteinfitness[i])
    for i in disttrackerlist:
        for j in i:
            disttrackeryaxis.append(j)

    if f % e == 0 or f == 0:
        keycounterdist = -1  # build distribution of fitness values existing in evolving protein
        additiondist = 0
        pointsdist = 0
        distclonefitnesslist = []
        disttotalfitness = []  # space for values to plot
        for j in range(len(c)):  # find and plot all fitness values in the current generation
            keycounterdist += 1
            if n is True:
                disttotalaminos = c[keycounterdist]  # load fitness of each clone (as keys are numbers so easy to iterate)
            else:
                disttotalaminos = []  # ignore variant sites
                invariantcheckaminos = c[keycounterdist]
                for i in range(len(invariantcheckaminos)):
                    if i in o:
                        disttotalaminos.append(invariantcheckaminos[i])
                    else:
                        disttotalaminos.append('X')
            clonefitnesslist = []
            for k in range(len(disttotalaminos)):  # generate values from generation x to plot
                amino = disttotalaminos[k]  # find ith amino acid
                if amino is not 'X':
                    pointsdist += 1
                    distindex = RESIDUES.index(amino)  # find index of first amino acid in RESIDUES list
                    diststring = proteinfitness[k]  # find fitness values for ith amino acid position
                    distvalue = diststring[distindex]  # find fitness value corresponding to amino acid at position
                    additiondist += distvalue
                    clonefitnesslist.append(distvalue)
                    disttotalfitness.append(distvalue)  # append these to list
            distclonefitnesslist.append(clonefitnesslist)

    if (h is True) and (f % e == 0 or f == 0):  # if the switch is on, and record fitness on the first generation and every x generation thereafter
        # print 'lets'
        # This section writes a file describing 5 statistical tests on the global fitness space.

        distclonetrackfilepath = '%s/statistics' % innerrunpath
        distclonetrackfilename = "normal_distribution_statistics_generation %s" % f  # define evo filename
        distclonetrackfullname = os.path.join(distclonetrackfilepath, distclonetrackfilename+".txt")
        distclonetrackfile = open(distclonetrackfullname, "w+")  # open file

        distclonetrackfile.write('Tests for normality on the amino acid fitness of each clone: \n\n\n')

        distclonesshapirolist = []
        distclonesandersonlist = []
        distclonesskewkurtalllist = []

        distclonetrackfile.write('Skewness: \n')

        for i in distclonefitnesslist:
            distclonesshapirolist.append(sp.stats.shapiro(i))
            distclonesandersonlist.append(sp.stats.anderson(i))
            distclonesskewkurtalllist.append(sp.stats.normaltest(i))

        skewness = sp.stats.skew(np.asarray(disttotalfitness))
        # skewstats = sp.stats.normaltest(np.asarray(disttotalfitness))
        distclonetrackfile.write("\nThe skewness of the data is %s\n\n\n" % skewness)
        distclonetrackfile.write("Kurtosis: \n")

        clonekurtosis = sp.stats.kurtosis(disttotalfitness)
        # kurtclonestats = sp.stats.kurtosistest(disttotalfitness)
        distclonetrackfile.write("\nThe kurtosis of the data is %s\n\n\n" % clonekurtosis)
        distclonetrackfile.write('Shapiro-Wilk test of non-normality: \n')

        totalcloneshapiro = sp.stats.shapiro(disttotalfitness)

        # shapiro-wilk tests
        distclonetrackfile.write("\nThe Shapiro-Wilk test of non-normality for the entire dataset gives p = %s" % totalcloneshapiro[-1])
        if totalcloneshapiro[-1] >= 0.05:
            shapiro = 'is not confidently non-normal'
        else:
            shapiro = 'is confidently non-normal'
        distclonetrackfile.write("\nTherefore the Shapiro-Wilk test suggests whole dataset %s" % shapiro)
        distclonetrackfile.write("\nHowever if there are more than 5000 datapoints this test is inaccurate. This test uses %s datapoints" % len(disttotalfitness))
        clonepasspercentcalc = []
        for i in distclonesshapirolist:
            if i[-1] >= 0.05:
                clonepasspercentcalc.append(1)
            else:
                clonepasspercentcalc.append(0)
        clonepasspercent = (sum(clonepasspercentcalc) / len(clonepasspercentcalc)) * 100
        distclonetrackfile.write("\n\nAccording to Shapiro-Wilk test, the proportion of individual positions that are not confidently non-normal is: %s%%" % clonepasspercent)

        # anderson-darling tests
        distclonetrackfile.write('\n\n\nAnderson-Darling test of normality: \n')
        # x = np.random.rand(10000)
        totalcloneanderson = sp.stats.anderson(disttotalfitness)
        distclonetrackfile.write("\nThe Anderson-Darling test of normality for the entire dataset gives a test statistic of %s " % totalcloneanderson.statistic)
        distclonetrackfile.write("and critical values of %s\nTherefore " % totalcloneanderson.critical_values)
        if totalcloneanderson.statistic < totalcloneanderson.critical_values[0]:
            distclonetrackfile.write("according to the Anderson-Darling test, the hypothesis of normality not rejected at 15% significance level for the entire dataset.")
        elif totalcloneanderson.critical_values[0] < totalcloneanderson.statistic < totalcloneanderson.critical_values[1]:
            distclonetrackfile.write("according to the Anderson-Darling test, the hypothesis of normality not rejected at 10% significance level for the entire dataset.")
        elif totalcloneanderson.critical_values[1] < totalcloneanderson.statistic < totalcloneanderson.critical_values[2]:
            distclonetrackfile.write("according to the Anderson-Darling test, the hypothesis of normality not rejected at 5% significance level for the entire dataset.")
        elif totalcloneanderson.critical_values[2] < totalcloneanderson.statistic < totalcloneanderson.critical_values[3]:
            distclonetrackfile.write("according to the Anderson-Darling test, the hypothesis of normality not rejected at 2.5% significance level for the entire dataset.")
        elif totalcloneanderson.critical_values[3] < totalcloneanderson.statistic < totalcloneanderson.critical_values[4]:
            distclonetrackfile.write("according to the Anderson-Darling test, the hypothesis of normality not rejected at 1% significance level for the entire dataset.")
        else:
            distclonetrackfile.write("according to the Anderson-Darling test, the hypothesis of normality is rejected for the entire dataset.")

        cloneconffifteen = []
        cloneconften = []
        cloneconffive = []
        cloneconftpf = []
        cloneconfone = []
        clonereject = []
        for i in distclonesandersonlist:
            if i.statistic < i.critical_values[0]:
                cloneconffifteen.append(1)
            elif i.critical_values[0] < i.statistic < i.critical_values[1]:
                cloneconften.append(1)
            elif i.critical_values[1] < i.statistic < i.critical_values[2]:
                cloneconffive.append(1)
            elif i.critical_values[2] < i.statistic < i.critical_values[3]:
                cloneconftpf.append(1)
            elif i.critical_values[3] < i.statistic < i.critical_values[4]:
                cloneconfone.append(1)
            else:
                clonereject.append(1)
        clonepercentfifteen = (len(cloneconffifteen) / len(distclonesandersonlist)) * 100
        clonepercentten = (len(cloneconften) / len(distclonesandersonlist)) * 100
        clonepercentfive = (len(cloneconffive) / len(distclonesandersonlist)) * 100
        clonepercenttpf = (len(cloneconftpf) / len(distclonesandersonlist)) * 100
        clonepercentone = (len(cloneconfone) / len(distclonesandersonlist)) * 100
        clonepercentreject = (len(clonereject) / len(distclonesandersonlist)) * 100
        distclonetrackfile.write("\n\nAccording to the Anderson-Darling test the hypothesis of normality is not rejected for each position in the dataset for:")
        distclonetrackfile.write("\n%s%% of positions at the 15%% significance level" % clonepercentfifteen)
        distclonetrackfile.write("\n%s%% of positions at the 10%% significance level" % clonepercentten)
        distclonetrackfile.write("\n%s%% of positions at the 5%% significance level" % clonepercentfive)
        distclonetrackfile.write("\n%s%% of positions at the 2.5%% significance level" % clonepercenttpf)
        distclonetrackfile.write("\n%s%% of positions at the 1%% significance level" % clonepercentone)
        distclonetrackfile.write("\nand %s%% of positions are rejected" % clonepercentreject)

        # skewness-kurtosis all tests
        distclonetrackfile.write('\n\n\nSkewness-kurtosis all test of difference from normality: \n')
        clonetotalskewkurtall = sp.stats.normaltest(disttotalfitness)

        distclonetrackfile.write("\nAccording to the skewness-kurtosis all test, the whole dataset gives p = %s," % clonetotalskewkurtall.pvalue)
        if clonetotalskewkurtall.pvalue >= 0.05:
            distclonetrackfile.write("\nTherefore the dataset does not differ significantly from a normal distribution")
        else:
            distclonetrackfile.write("\nTherefore the dataset differs significantly from a normal distribution")

        cloneskewkurtpass = []
        for i in distclonesskewkurtalllist:
            if i.pvalue >= 0.05:
                cloneskewkurtpass.append(1)
            else:
                cloneskewkurtpass.append(0)
        cloneskewkurtpercent = (sum(cloneskewkurtpass) / len(cloneskewkurtpass)) * 100
        distclonetrackfile.write("\n\nAccording to the skewness-kurtosis all test, %s%% of sites do not differ significantly from a normal distribution" % cloneskewkurtpercent)

        # Kolmogorov-Smirnov test of similarity to original distributuion
        ksdata = sp.stats.ks_2samp(np.asarray(disttotalfitness), np.asarray(disttrackeryaxis))
        ksp = ksdata.pvalue
        distclonetrackfile.write("\n\n\n2-sided Kolmogorov-Smirnov test of similarity between the fitness space and evolving protein")
        distclonetrackfile.write("\n\nThe Kolmogorov-Smirnov test between the fitness space an the evolving protein gives a p-value of: %s" % ksp)

        if ksdata.pvalue < 0.05:
            distclonetrackfile.write("\nTherefore, as the pvalue is smaller than 0.05 we can reject the hypothesis that the fitness space distribution and the evolving sequence distribution are the same")
        else:
            distclonetrackfile.write("\nTherefore, as the pvalue is larger than 0.05 we canot reject the hypothesis that the fitness space distribution and the evolving sequence distribution are the same")

        if f == 0:
            disttrackfilepath = '%s/statistics' % innerrunpath
            disttrackfilename = "normal_distribution_statistics_fitness_space"  # define evo filename
            disttrackfullname = os.path.join(disttrackfilepath, disttrackfilename+".txt")
            disttrackfile = open(disttrackfullname, "w+")  # open file

            disttrackfile.write('Tests for normality on the global amino acid fitness space at each position: \n\n\n')
            disttrackfile.write('Skewness: \n')

            distshapirolist = []
            distandersonlist = []
            distskewkurtalllist = []
            for i in disttrackerlist:
                distshapirolist.append(sp.stats.shapiro(i))
                distandersonlist.append(sp.stats.anderson(i))
                distskewkurtalllist.append(sp.stats.normaltest(i))

            skewness = sp.stats.skew(disttrackeryaxis)
            skewstats = sp.stats.normaltest(disttrackeryaxis)
            disttrackfile.write("\nThe skewness of the data is %s\n\n\n" % skewness)

            disttrackfile.write("Kurtosis: \n")

            kurtosis = sp.stats.kurtosis(disttrackeryaxis)
            kurtstats = sp.stats.kurtosistest(disttrackeryaxis)
            disttrackfile.write("\nThe kurtosis of the data is %s\n\n\n" % kurtosis)

            disttrackfile.write('Shapiro-Wilk test of non-normality: \n')

            totalshapiro = sp.stats.shapiro(disttrackeryaxis)

            disttrackfile.write(
                "\nThe Shapiro-Wilk test of non-normality for the entire dataset gives p = %s" % totalshapiro[-1])
            if totalshapiro[-1] >= 0.05:
                shapiro = 'is not confidently non-normal'
            else:
                shapiro = 'is confidently non-normal'
            disttrackfile.write("\nTherefore the Shapiro-Wilk test suggests whole dataset %s" % shapiro)
            passpercentcalc = []
            for i in distshapirolist:
                if i[-1] >= 0.05:
                    passpercentcalc.append(1)
                else:
                    passpercentcalc.append(0)
            passpercent = (sum(passpercentcalc) / len(passpercentcalc)) * 100
            disttrackfile.write("\n\nAccording to Shapiro-Wilk test, the proportion of individual positions that are not confidently non-normal is: %s%%" % passpercent)

            disttrackfile.write('\n\n\nAnderson-Darling test of normality: \n')
            # x = np.random.rand(10000)
            totalanderson = sp.stats.anderson(disttrackeryaxis)
            disttrackfile.write("\nThe Anderson-Darling test of normality for the entire dataset gives a test statistic of %s " % totalanderson.statistic)
            disttrackfile.write("and critical values of %s\nTherefore " % totalanderson.critical_values)
            if totalanderson.statistic < totalanderson.critical_values[0]:
                disttrackfile.write("according to the Anderson-Darling test, the hypothesis of normality not rejected at 15% significance level for the entire dataset.")
            elif totalanderson.critical_values[0] < totalanderson.statistic < totalanderson.critical_values[1]:
                disttrackfile.write("according to the Anderson-Darling test, the hypothesis of normality not rejected at 10% significance level for the entire dataset.")
            elif totalanderson.critical_values[1] < totalanderson.statistic < totalanderson.critical_values[2]:
                disttrackfile.write("according to the Anderson-Darling test, the hypothesis of normality not rejected at 5% significance level for the entire dataset.")
            elif totalanderson.critical_values[2] < totalanderson.statistic < totalanderson.critical_values[3]:
                disttrackfile.write("according to the Anderson-Darling test, the hypothesis of normality not rejected at 2.5% significance level for the entire dataset.")
            elif totalanderson.critical_values[3] < totalanderson.statistic < totalanderson.critical_values[4]:
                disttrackfile.write("according to the Anderson-Darling test, the hypothesis of normality not rejected at 1% significance level for the entire dataset.")
            else:
                disttrackfile.write("according to the Anderson-Darling test, the hypothesis of normality is rejected for the entire dataset.")
            # set up output for significance levels
            conffifteen = []
            conften = []
            conffive = []
            conftpf = []
            confone = []
            reject = []
            for i in distandersonlist:
                if i.statistic < i.critical_values[0]:
                    conffifteen.append(1)
                elif i.critical_values[0] < i.statistic < i.critical_values[1]:
                    conften.append(1)
                elif i.critical_values[1] < i.statistic < i.critical_values[2]:
                    conffive.append(1)
                elif i.critical_values[2] < i.statistic < i.critical_values[3]:
                    conftpf.append(1)
                elif i.critical_values[3] < i.statistic < i.critical_values[4]:
                    confone.append(1)
                else:
                    reject.append(1)
            percentfifteen = (len(conffifteen) / len(distandersonlist)) * 100
            percentten = (len(conften) / len(distandersonlist)) * 100
            percentfive = (len(conffive) / len(distandersonlist)) * 100
            percenttpf = (len(conftpf) / len(distandersonlist)) * 100
            percentone = (len(confone) / len(distandersonlist)) * 100
            percentreject = (len(reject) / len(distandersonlist)) * 100
            disttrackfile.write(
                "\n\nAccording to the Anderson-Darling test the hypothesis of normality is not rejected for each position in the dataset for:")
            disttrackfile.write("\n%s%% of positions at the 15%% significance level" % percentfifteen)
            disttrackfile.write("\n%s%% of positions at the 10%% significance level" % percentten)
            disttrackfile.write("\n%s%% of positions at the 5%% significance level" % percentfive)
            disttrackfile.write("\n%s%% of positions at the 2.5%% significance level" % percenttpf)
            disttrackfile.write("\n%s%% of positions at the 1%% significance level" % percentone)
            disttrackfile.write("\nand %s%% of positions are rejected" % percentreject)

            disttrackfile.write('\n\n\nSkewness-kurtosis all test of difference from normality: \n')
            totalskewkurtall = sp.stats.normaltest(disttrackeryaxis)

            disttrackfile.write("\nAccording to the skewness-kurtosis all test, the whole dataset gives p = %s," % totalskewkurtall.pvalue)
            if totalskewkurtall.pvalue >= 0.05:
                disttrackfile.write("\nTherefore the dataset does not differ significantly from a normal distribution")
            else:
                disttrackfile.write("\nTherefore the dataset differs significantly from a normal distribution")

            skewkurtpass = []
            for i in distskewkurtalllist:
                if i.pvalue >= 0.05:
                    skewkurtpass.append(1)
                else:
                    skewkurtpass.append(0)
            skewkurtpercent = (sum(skewkurtpass) / len(skewkurtpass)) * 100
            disttrackfile.write("\n\nAccording to the skewness-kurtosis all test, %s%% of sites do not differ significantly from a normal distribution" % skewkurtpercent)

    if (m is True) and (f % e == 0 or f == 0):  # if the switch is on, and record fitness histograms on the first generation and every x generation thereafter
        # print 'go'
        plt.figure()
        plt.axis([-10, 8, 0, 0.5])  # generate attractive figure

        mudistspace = sum(disttrackeryaxis) / len(disttrackeryaxis)  # plot normal distribution of the original fitness space
        plt.hist(disttrackeryaxis, 50, density=True, color='k', alpha=0.4)
        plt.title("\n".join(wrap('Fitness distribution of the total fitness space', 60)), fontweight='bold')
        plt.axvline(x=mudistspace, color="#404040", linestyle=":")

        mudistspace2 = additiondist / pointsdist
        plt.hist(disttotalfitness, 50, density=True, color='r', alpha=0.4)
        plt.title("\n".join(wrap('Fitness distribution of the total fitness space vs changing fitness distribution across every evolving clone', 60)), fontweight='bold')
        plt.axvline(x=mudistspace2, color="#404040", linestyle=":")
        mu1distdp = "%.3f" % mudistspace
        mu2distdp = "%.3f" % mudistspace2
        plt.text(4.1, 0.42, "$\mu$1 = %s\n$\mu$2 = %s\nthreshold = %s" % (mu1distdp, mu2distdp, fitness_threshold))

        disthisttrackfilepath = '%s/histograms' % innerrunpath
        disthistfilename = "generation_%s" % f  # define dynamic filename
        disthistfullname = os.path.join(disthisttrackfilepath, disthistfilename+".png")
        plt.savefig(disthistfullname)
        plt.close()
    return fitnessdict


def write_fasta_alignment(population, generation):  # x = current generation of sequence, y = generation number
    """Write fasta alignment from sequences provided."""
    fastafilepath = '%s/fastas' % runpath
    fastafilename = "generation_%s" % generation  # define dynamic filename
    fullname = os.path.join(fastafilepath, fastafilename+".fasta")
    fastafile = open(fullname, "w+")  # open file
    # Write fasta header followed by residue in generation string
    for p, protein in list(population.items()):
        fastafile.write("\n>clone_%s\n" % (p+1))
        for residue in protein:
            fastafile.write(residue)


def write_final_fasta(population, bifurcations, n_roots):  # x = current generation, y = bifurication state, z = n_roots
    treefastafilepath = '%s/treefastas' % runpath
    treefastafilename = "selected_fastas"
    fullname = os.path.join(treefastafilepath, treefastafilename+".fasta")
    treefastafile = open(fullname, "w+")  # open file
    bifsize = 0
    for bifs in bifurcations:
        bifsize += len(bifs)
    bifursize = bifsize/len(bifurcations)
    n_clones_to_take = int((bifursize-1)/2)  # if 5, gives 2, if 4 gives 2, if 3 gives 1.
    generation_numbers = []
    for i in bifurcations:
        clone_selection = random.sample(set(i), n_clones_to_take)  # NOTE: Only sample from unique clones?
        for c in clone_selection:
            generation_numbers.append(c)
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


def generationator(n_generations, initial_population, fitness_threshold,
                   n_mutations_per_generation, write_rate, LGmatrix):
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

    rootssavepath = "%s/start" % runpath
    rootsfilename = "Roots"
    rootsfullname = os.path.join(rootssavepath, rootsfilename+".txt")
    rootsfile = open(rootsfullname, "w+")  # open file
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
    # Record initial population
    fitnesses = record_generation_fitness(population, track_dot_fitness,
                                          track_rate, 0, proteinfitness,
                                          track_hist_fitness_stats,
                                          track_hist_fitness, track_invariants,
                                          variantaminos)  # current generation fitness
    write_fasta_alignment(population, 0)

    # Store each generation along with its fitness
    Generation = namedtuple('Generation', ['population', 'fitness'])
    # Create a list of generations and add initial population and fitness
    evolution = [Generation(population=population, fitness=fitnesses)]

    for gen in trange(n_generations):  # run evolution for n_generations
        # Bifuricationmaker. Bifuricates in even generation numbers so every branch on tree has 3 leaves that have been evolving by the last generation
        if gen % bifurgeneration == 0 and gen > 0 and len(clonelistlist[0]) > 3:
            lists = []  # space to store bifurcations before adding them to clonelistlist
            for j in clonelistlist:  # bifuricate each set of leaves
                random.shuffle(j)
                midpoint = int(len(j)/2)
                lists.append(j[midpoint:])
                lists.append(j[:midpoint])
            del clonelistlist[:]
            for k in lists:  # append bifurcations to a cleared clonelistlist
                clonelistlist.append(k)

        # Mutate population
        population = mutate(population, n_mutations_per_generation,
                            variantaminos, gammacategories, LGmatrix)
        # Re-calculate fitness
        fitnesses = record_generation_fitness(population, track_dot_fitness,
                                              track_rate, gen, proteinfitness,
                                              track_hist_fitness_stats,
                                              track_hist_fitness, track_invariants,
                                              variantaminos)

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

        evolution.append(Generation(population=population, fitness=fitnesses))
        # NOTE: Should this be at the end of each timestep?
        if ((gen+1) % write_rate) == 0:  # write fasta every write_rate generations
            write_fasta_alignment(population, gen+1)
    write_final_fasta(population, clonelistlist, rootlist)

    return evolution


def fitbit(evolution, n_generations, n_clones, initial_protein):
    """Plot fitness against generation for all clones."""
    # NOTE: Final element previously excluded - for i in range(len(evolution)-1):
    # Create array of fitness values with shape (n_generations, n_clones)
    # fitnesses = np.array([[evolution[g][-1][c] for c in range(n_clones)]
    #                       for g in range(n_generations)])
    fitnesses = np.array([[evolution[g].fitness[c] for c in range(n_clones)]
                          for g in range(n_generations+1)])

    initial_fitness = calculate_fitness(initial_protein)
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
    plt.ylabel("Fitness", fontweight='bold')
    plt.title("\n".join(wrap('Fitness change for %s randomly generated "superfit" clones of %s amino acids, mutated over %s generations' % (n_clones, (n_amino_acids+1), n_generations), 60)), fontweight='bold')
    plt.text(n_generations-1000, initial_fitness+50,
             r"$\mu$ = {}".format(mu) + "\n" + \
             r"$\sigma$ = {}".format(sigma) + "\n" + \
             r"$\delta$ = {}".format(mutation_rate))
             # r"$\mu$ = %s\n$\sigma$ = %s\n$\delta$ = %s" % (mu, sigma, mutation_rate))

    # Define dynamic filename
    fitgraphfilepath = '%s/fitnessgraph' % runpath
    fitgraphfilename = "fitness_change_over %s generations" % n_generations
    fitgraphfullname = os.path.join(fitgraphfilepath, fitgraphfilename+".png")

    plt.savefig(fitgraphfullname)
    return


if __name__ == '__main__':

    np.random.seed(42)
    # from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    # from matplotlib.figure import Figure
    # matplotlib.use('TkAgg')
    mpl.rc('savefig', dpi=300)

    # create folder and subfolders
    paths = ['runsettings', 'start', 'fastas', 'fitnessgraph',
             'fitnessdotmatrix', 'fitnessdistribution', 'treefastas']

    runpath = "results/run%s" % datetime.datetime.now().strftime("%y-%m-%d-%H-%M")
    for path in paths:
        os.makedirs(os.path.join(runpath, path))

    innerpaths = ['statistics', 'histograms']
    innerrunpath = '%s/fitnessdistribution' % runpath
    for innerpath in innerpaths:
        os.makedirs(os.path.join(innerrunpath, innerpath))

    # record run settings
    settingsfilepath = '%s/runsettings' % runpath
    settingsfilename = "runsettings"  # define evo filename
    settingsfullname = os.path.join(settingsfilepath, settingsfilename+".txt")
    settingsfile = open(settingsfullname, "w+")  # open file

    # TODO: Change to logger
    settingsfile.write("Protein length: %s" % (n_amino_acids+1))
    settingsfile.write("\nAmount of mutations per generation: %s" % n_mutations_per_generation)
    settingsfile.write("\nAmount of clones in the population: %s" % n_clones)
    settingsfile.write("\nAmount of generations simulation is run for: %s" % n_generations)
    settingsfile.write("\nFitness threshold: %s" % fitness_threshold)
    settingsfile.write("\n\nNormal distribution properties: mu = %s, sigma = %s" % (mu, sigma))
    settingsfile.write("\nGamma distribution properties: kappa = %s, theta = %s" % (gamma_shape, gamma_scale))
    settingsfile.write("\n\nWrite rate for FASTA: every %s generations" % write_rate)
    settingsfile.write("\n\nTrack rate for graphing and statistics: every %s generations" % track_rate)
    settingsfile.write("\nTracking state: Fitness dot matrix = %s; Fitness histrogram = %s; Fitness normality statistics = %s" % (track_dot_fitness, track_hist_fitness, track_hist_fitness_stats))

    # load matrix
    aamatrix = "data/LGaa.csv"  # .csv file defining aa substitution probabilities calculated from R matrix multiplied by PI matrix, with diagonals forced to zero as mutation has to happen then conferted to event rates p(lambda) where lambda = sum Qx and p(lambda)x=Qxy/lambda
    LGmatrixreader = csv.reader(open(aamatrix), delimiter=",")
    LGmatrixlist = list(LGmatrixreader)
    LGmatrix = np.array(LGmatrixlist)  # load matrix into a numpy array
    LGmatrix = np.delete(LGmatrix, 0, 0)  # trim first line of the array as its not useful

    initial_protein = generate_protein(n_amino_acids)  # make first protein
    proteinfitness = get_protein_fitness(initial_protein)  # make first fitness dictionary

    variantaminos = get_allowed_sites(n_amino_acids, n_anchors)  # generate invariant sites
    initial_protein = superfit(proteinfitness, variantaminos, initial_protein, fitness_start)  # generate a superfit protein taking into account the invariant sites created (calling variables in this order stops the evolutionary process being biased by superfit invariant sites.)
    gammacategories = gammaray(gamma_iterations, gamma_samples, gamma_shape, gamma_scale, n_amino_acids)  # generate gamma categories for every site

    firstproteinsavepath = '%s/start' % runpath
    firstproteinfilename = "firstprotein"
    firstproteinfullname = os.path.join(firstproteinsavepath, firstproteinfilename+".fas")
    firstproteinfile = open(firstproteinfullname, "w+")  # open file
    firstproteinfile.write('>firstprotein\n')
    for prot in initial_protein:
        firstproteinfile.write(prot)
    # print 'first superfit protein:', initial_protein
    # print 'fitness of the first protein:', calculate_fitness(initial_protein)

    initial_population = clones(n_clones, initial_protein)  # make some clones to seed evolution

    evolution = generationator(n_generations, initial_population,
                               fitness_threshold, n_mutations_per_generation,
                               write_rate, LGmatrix)

    fitbit(evolution, n_generations, n_clones, initial_protein)
