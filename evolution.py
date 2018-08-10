import copy
import os.path
import csv
import datetime
from random import randint, sample, choice, shuffle  # TODO: Consolidate with numpy
from textwrap import wrap
# import shutil
# import json

import numpy as np
from numpy import median
from numpy.random import gamma
from numpy.random import normal, uniform
from scipy.stats import anderson, normaltest, skew, skewtest, kurtosistest
from scipy.stats import shapiro as shp
from scipy.stats import kurtosis as kurt
from scipy.stats import ks_2samp as kosmo
# import scipy.special as sps
# import matplotlib
from matplotlib import pyplot as plt

from tqdm import tqdm


# define starting variables
# amino acids - every fitness value string references residues string
residues = ["R", "H", "K", "D", "E", "S", "T", "N", "Q", "C",
            "G", "P", "A", "V", "I", "L", "M", "F", "Y", "W"]

# parameters for normal distribution used to select fitness values
mu = -1.2
sigma = 2.5

# parameters for forming discrete gamma distribution used for evolution of protein
gamma_iterations = 100
gamma_samples = 10000
gamma_shape = 1.9  # Most phylogenetic systems that use gamma only let you set kappa (often called shape alpha) and calculate theta as 1/kappa giving mean of 1
gamma_scale = 1/gamma_shape

# parameters of protein evolution
amountofclones = 52  # amount of clones that will be generated in the first generation #5 10 20 40 80
amountofgenerations = 2000  # amount of generations the protein evolves for
startingfitness = 'medium'  # high, medium or low; must be lower case. If selecting low, fitness threshold needs to be significantly smaller (i.e. 4x) than #positions*mu
fitnessthreshold = 0  # arbitrary number for fitness threshold
amountofaminos = 79  # number of amino acids in the protein after the start methionine
mutationrate = 0.001  # should be small!
amountofmutations = int(amountofclones*(amountofaminos+1)*mutationrate)  # number of mutations per generation
writerate = 50  # write a new fasta file every x generations
amountofanchors = int((amountofaminos+1)/10)  # amount of invariant sites in a generation (not including root)
trackrate = 50  # track every x generations
roots = 4

# set what to record
trackdotfitness = False  # True or False.
trackhistfitnessstats = False  # True or False.
trackhistfitness = False  # True or False.
trackinvariants = False  # if True, invariants are tracked in the histogram analysis. If false, invariants are ignored.


def generate_protein(a):
    """Generate an original starting protein 20aa long with a start methionine.
    """
    amino = a
    firstresidue = "M"
    original_protein = []
    original_protein.append(firstresidue)
    for i in range(amino):
        x = randint(0, len(residues)-1)
        original_protein.append(residues[x])
    return original_protein


# NOTE: unused
def test_normal_distribution():
    """Plot a distribution to test normalality."""
    s = normal(mu, sigma, 2000)  # generate distribution
    count, bins, ignored = plt.hist(s, 30, density=True)  # plot distribuiton
    plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(- (bins - mu)**2 / (2 * sigma**2)), linewidth=2, color='r')
    return plt.show()

# test_normal_distribution()


def fit_module():
    """Generates a string of fitness values for each amino acid in residues."""
    fitness = []
    for j in range(len(residues)):
        fitnessvalue = normal(mu, sigma)
        fitness.append(fitnessvalue)
    return fitness


def get_protein_fitness(x):  # x=protein;
    """Generate a dictionary describing list of fitness values at each position
    of the generated protein.
    """
    fitnesslib = {}
    for k in range(len(x)):
        fitvalues = fit_module()
        fitnesslib.update({k: fitvalues}) # dictionary contains position in the protein as keys and the string of fitness values for each amino acids as variable
        # extra section to remove fitness value for start methionine, hash out line above and unhash lines below if to be used.
        # if k = 0:
            # mposition = residues.index("M")
            # fitvalues[mposition] = 0
            # fitnesslib.update({k: fitvalues}) #dictionary contains position in the protein as keys and the string of fitness values for each amino acids as variable
        # else
            # fitnesslib.update({k: fitvalues})

    aminofitsavepath = '%s/start' % runpath
    aminofitfilename = "fitnesslibrary"
    aminofitfullname = os.path.join(aminofitsavepath, aminofitfilename + ".csv")
    aminofile = open(aminofitfullname, "w+")  # open file
    aminofile.write("aminoposition"),
    for i in residues:
        aminofile.write(",%s" % i)
    for j in range(amountofaminos+1):
        keytowrite = j
        fitnesses = fitnesslib[j]
        aminofile.write('\n%s' % keytowrite),
        for m in fitnesses:
            aminofile.write(',%s' % m)

    return fitnesslib


def clones(x, y):  # x= number of clones to generate, y = protein
    """Generate a dictionary containing X clones of generated protein
    - this contains the evolving dataset.
    """
    cloneslib = {}
    for l in range(x):
        cloneslib.update({l: y})
    return cloneslib


def get_allowed_sites(a, b): # a = amountofaminos, b = amountofanchors; this module defines the invariant sites within the protein.
    """Select invariant sites in the initially generated protein and return
    allowed values.
    """
    anchoredsequences = sample(list(range(1, a)), b)  # randomly define invariant sites
    allowedvalues = list(range(1, a+1))  # keys for sites that can be modified by mutation
    for i in anchoredsequences:
        invariant = i
        if invariant in allowedvalues:
            allowedvalues.remove(invariant)  # remove the invariant sites from allowed values
    return allowedvalues


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
        samples = gamma(c, d, b)

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
        ratecategoriesquartile = [median(bottomlist), median(bottommidlist), median(topmidlist), median(toplist)]
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
        gammacategorychoice = choice(finalmedians)
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

    randomgrab = uniform(0, 1)  # pick a random number in the cumulative probability distribution
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
        fitindex = residues.index(amino)  # find index of first amino acid in residues list
        fitstring = proteinfitness[m]  # find fitness values for ith amino acid position
        fitvalue = fitstring[fitindex]  # find fitness value corresponding to amino acid at position
        aminofitnesses.append(fitvalue)  # append these to string
    fitness = 0
    for i in range(len(aminofitnesses)):
        fitness = fitness+aminofitnesses[i]  # compute fitness
    return fitness


def superfit(n, o, p, q):  # n=proteinfitness, o=anchored sequences, p=firstprotein, q = startingfitness; Generates a protein with high, low or medium fitness, with anchored sequences from the initial generation
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
            else:  # find the indexes of the 3 least fit amino acids in residues and record then as lists for each position
                unfitaminos = []
                amin = n[i]
                aminsort = sorted(amin)
                unfittestaminoposition = amin.index(aminsort[0])
                secondunfittestaminoposition = amin.index(aminsort[1])
                thirdunfittestaminoposition = amin.index(aminsort[2])
                unfittestamino = residues[unfittestaminoposition]
                secondunfittestamino = residues[secondunfittestaminoposition]
                thirdunfittestamino = residues[thirdunfittestaminoposition]
                unfitaminos.append(unfittestamino)
                unfitaminos.append(secondunfittestamino)
                unfitaminos.append(thirdunfittestamino)
                unfittestaminos.append(unfitaminos)
        afitprotein = []
        for j in range(len(unfittestaminos)):  # generate a superunffit protein by randomly picking one of the 3 most fit amino acids at each position
            randombin = randint(0, 2)
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
            else:  # find the indexes of the 3 fittest amino acids in residues and record then as lists for each position
                fitaminos = []
                amin = n[i]
                aminsort = sorted(amin)
                fittestaminoposition = amin.index(max(amin))
                secondfittestaminoposition = amin.index(aminsort[-2])
                thirdfittestaminoposition = amin.index(aminsort[-3])
                fittestamino = residues[fittestaminoposition]
                secondfittestamino = residues[secondfittestaminoposition]
                thirdfittestamino = residues[thirdfittestaminoposition]
                fitaminos.append(fittestamino)
                fitaminos.append(secondfittestamino)
                fitaminos.append(thirdfittestamino)
                fittestaminos.append(fitaminos)
        afitprotein = []

        for j in range(len(fittestaminos)):  # generate a superfit protein by randomly picking one of the 3 most fit amino acids at each position
            randombin = randint(0, 2)
            possibleaminos = fittestaminos[j]
            afitprotein.append(possibleaminos[randombin])
    # generate medium fitness protein. This module is a little buggy. It takes the starting protein sequence, mutates 5 residues until the protein is fitter, then chooses 5 new residues and continues.
    # If it cannot make a fitter protein with the 5 residues its mutating it reverts back to the previous state and picks 5 new residues.
    if q == 'medium':
        startprotein = p
        startproteinfitness = calculate_fitness(startprotein)
        variantstochoosefrom = o
        secondprotein = startprotein

        while startproteinfitness < fitnessthreshold+30:
            choiceofvariants = sample(variantstochoosefrom, 5)
            secondprotein[choiceofvariants[0]] = choice(residues)
            secondprotein[choiceofvariants[1]] = choice(residues)
            secondprotein[choiceofvariants[2]] = choice(residues)
            secondprotein[choiceofvariants[3]] = choice(residues)
            secondprotein[choiceofvariants[4]] = choice(residues)
            secondproteinfitness = calculate_fitness(secondprotein)
            counting = 0

            while secondproteinfitness < startproteinfitness:
                secondprotein = startprotein
                secondproteinfitness = calculate_fitness(secondprotein)
                secondprotein[choiceofvariants[0]] = choice(residues)
                secondprotein[choiceofvariants[1]] = choice(residues)
                secondprotein[choiceofvariants[2]] = choice(residues)
                secondprotein[choiceofvariants[3]] = choice(residues)
                secondprotein[choiceofvariants[4]] = choice(residues)
                secondproteinfitness = calculate_fitness(secondprotein)
                counting += 1

                if counting > 99:
                    choiceofvariants = sample(variantstochoosefrom, 5)
                    counting -= 100
                    break

            startprotein = secondprotein
            startproteinfitness = calculate_fitness(startprotein)
        afitprotein = startprotein

    return afitprotein


# NOTE: Not used
def histfitness(f):
    """Generate and plot fitness values for f proteins."""
    graphnumbers = []  # numbers to plot
    for p in range(f):  # generate the fitness values
        aprotein = generate_protein()
        toplot = calculate_fitness(aprotein)
        graphnumbers.append(toplot)
    plt.hist(graphnumbers, density=True)  # plot fitnesses as histogram
    return plt.show()


def get_thresholded_protein(b, c):  # b=startingfitness, c=firstprotein;
    """Make a protein with a fitness value above a defined threshold.

    Note: This takes a long time if thresh is too high.
    """
    f = calculate_fitness(c)
    while f < b:  # keep making proteins until the protein's fitness satisfies the fitness threshold.
        proteintoevolve = generate_protein()
        f = calculate_fitness(proteintoevolve)
    return f


def mutate(a, b, c, d, e):  # a = number of mutations in the generation; b = protein generation, c = allowed sites, d = assignedgammas, e = matrix
    """Mutate a given sequence based on the LG+I+G model of amino acid
    substitution.
    """
    currentgeneration = copy.deepcopy(b)  # make a deep copy of the library so changing the library in the function doesnt change the library outside the function
    summedgammas = []
    gammasum = 0

    for i in d:  # sum up the gammas to make a probability distribution to randomly select from.
        gammasum = gammasum + i
        summedgammas.append(gammasum)

    highestgammasum = summedgammas[-1]

    for q in range(a):  # impliment gamma

        clonetomutatekey = randint(0, len(b)-1)  # pick random key to make a random generation
        clonetomutate = currentgeneration[clonetomutatekey]  # select the clone corresponding to the random key
        mutatedresidues = []
        residuetomutate = [0]

        while residuetomutate[0] not in c:  # always initiates as residuetomutate set to 0 and residue zero 0 should always be disallowed (start methionine locked). Also ensures only mutates at variant sites
            residueareatomutate = uniform(0, highestgammasum)
            for j in summedgammas:
                if residueareatomutate < j:
                    mutatedresidues.append(summedgammas.index(j))
                    residuetomutate[0] = summedgammas.index(j)
                    break
                else:
                    continue

        mutationtarget = copy.deepcopy(clonetomutate)  # make a deep copy of the libaries value as to not change it in the library until we want to
        # residuetomutate = choice(c) # pick a random residue in the selected mutant to mutate that isnt the start M or an anchor (old)
        targetresidue = mutationtarget[residuetomutate[0]]
        newresidue = mutate_matrix(e, targetresidue)  # implement LG

        # old way of selecting random residues to mutate to below
        # x = randint(0, len(residues)-1)
        # newresidue = residues[x]  # pick a random residue to replace the mutable residue
        # while newresidue == clonetomutate[residuetomutate[0]]:  # ensure the new residue is different to the current residue
        #    x = randint(0, len(residues)-1)
        #    newresidue = residues[x]

        # below are print functions to check the above part of the fucntion is working correctly
        # print "generation", i+1
        # print "new residue to be added:", newresidue
        # print "position of residue to be mutated:", residuetomutate
        # print "position of clone to be mutated", clonetomutatekey
        # print "sequence of clone to be mutated:", clonetomutate
        # print "old residue that will be changed", clonetomutate[residuetomutate]

        mutationtarget[residuetomutate[0]] = newresidue  # mutate the copy with the randomly chosen residue
        currentgeneration[clonetomutatekey] = mutationtarget  # update with new sequence

        # below are print functions to check the above part of the fucntion is working correctly
        # print "sequence of clone after mutation", mutationtarget
        # print "sequence of clonetomutate function", clonetomutate
        # print "generation of clones with new mutation", currentgeneration
        # print ""
    return currentgeneration


def record_generation_fitness(c, d, e, f, g, h, m, n, o):  # c=protein generation; d=trackdotfitness, e=generationtotrack, f=generationnumber, g=proteinfitness, h=trackhistfitnessstats, m=trackhistfitness, n=track invariants, o= invariant sites
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
        for i in range(amountofaminos+1):
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

        transform = amountofaminos + (amountofaminos / 10)  # generate values for right side of the figure. Transform function sets offet to make figure look nice.
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
                    fitindex = residues.index(amino)  # find index of first amino acid in residues list
                    fitstring = proteinfitness[k]  # find fitness values for ith amino acid position
                    fitvalue = fitstring[fitindex]  # find fitness value corresponding to amino acid at position
                    additionright += fitvalue
                    Y2fitness.append(fitvalue)  # append these to list
            #print Y2fitness
            plt.plot(len(Y2fitness) * [j + transform], Y2fitness, "o", markersize=1)  # plot right hand side with small markers
            plt.title("\n".join(wrap('Fitness of every amino acid in the fitness matrix vs fitness of every amino acid in generation %s' % f, 60)), fontweight='bold')
        avgtoplotright = additionright / pointsright  # calculate right side average
        plt.axis([-10, (amountofaminos * 2) + (amountofaminos / 10)+10, -11, 11])  # generate attractive figure
        plt.plot([0, len(fittrackeryaxis) + 1], [avgtoplotleft, avgtoplotleft], 'r--', lw=3)
        plt.plot([0 + transform, len(c) + 1 + transform], [avgtoplotright, avgtoplotright], 'r--', lw=3)
        muleftdistdp = "%.3f" % avgtoplotleft
        murightdistdp = "%.3f" % avgtoplotright
        plt.text(0, 8.7, "$\mu$1 = %s\n$\mu$2 = %s\nthreshold = %s" % (muleftdistdp, murightdistdp, fitnessthreshold), size = 6.5)

        plt.xticks([])  # remove x axis ticks
        fitfilename = "generation_%s" % f  # define dynamic filename
        fitsavepath = '%s/fitnessdotmatrix' % runpath
        fitfullname = os.path.join(fitsavepath, fitfilename + ".png")
        plt.savefig(fitfullname)
        plt.close()  # close plot (so you dont generate 100 individual figures)

    disttrackerlist = []  # build fitness space numbers
    disttrackeryaxis = []
    for i in range(amountofaminos+1):
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
                    distindex = residues.index(amino)  # find index of first amino acid in residues list
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
        distclonetrackfullname = os.path.join(distclonetrackfilepath, distclonetrackfilename + ".txt")
        distclonetrackfile = open(distclonetrackfullname, "w+")  # open file

        distclonetrackfile.write('Tests for normality on the amino acid fitness of each clone: \n\n\n')

        distclonesshapirolist = []
        distclonesandersonlist = []
        distclonesskewkurtalllist = []

        distclonetrackfile.write('Skewness: \n')

        for i in distclonefitnesslist:
            distclonesshapirolist.append(shp(i))
            distclonesandersonlist.append(anderson(i))
            distclonesskewkurtalllist.append(normaltest(i))

        skewness = skew(np.asarray(disttotalfitness))
        # skewstats = skewtest(np.asarray(disttotalfitness))
        distclonetrackfile.write("\nThe skewness of the data is %s\n\n\n" % skewness)
        distclonetrackfile.write("Kurtosis: \n")

        clonekurtosis = kurt(disttotalfitness)
        # kurtclonestats = kurtosistest(disttotalfitness)
        distclonetrackfile.write("\nThe kurtosis of the data is %s\n\n\n" % clonekurtosis)
        distclonetrackfile.write('Shapiro-Wilk test of non-normality: \n')

        totalcloneshapiro = shp(disttotalfitness)

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
        totalcloneanderson = anderson(disttotalfitness)
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
        clonetotalskewkurtall = normaltest(disttotalfitness)

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
        ksdata = kosmo(np.asarray(disttotalfitness), np.asarray(disttrackeryaxis))
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
            disttrackfullname = os.path.join(disttrackfilepath, disttrackfilename + ".txt")
            disttrackfile = open(disttrackfullname, "w+")  # open file

            disttrackfile.write('Tests for normality on the global amino acid fitness space at each position: \n\n\n')
            disttrackfile.write('Skewness: \n')

            distshapirolist = []
            distandersonlist = []
            distskewkurtalllist = []
            for i in disttrackerlist:
                distshapirolist.append(shp(i))
                distandersonlist.append(anderson(i))
                distskewkurtalllist.append(normaltest(i))

            skewness = skew(disttrackeryaxis)
            skewstats = skewtest(disttrackeryaxis)
            disttrackfile.write("\nThe skewness of the data is %s\n\n\n" % skewness)

            disttrackfile.write("Kurtosis: \n")

            kurtosis = kurt(disttrackeryaxis)
            kurtstats = kurtosistest(disttrackeryaxis)
            disttrackfile.write("\nThe kurtosis of the data is %s\n\n\n" % kurtosis)

            disttrackfile.write('Shapiro-Wilk test of non-normality: \n')

            totalshapiro = shp(disttrackeryaxis)

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
            totalanderson = anderson(disttrackeryaxis)
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
            totalskewkurtall = normaltest(disttrackeryaxis)

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
        plt.text(4.1, 0.42, "$\mu$1 = %s\n$\mu$2 = %s\nthreshold = %s" % (mu1distdp, mu2distdp, fitnessthreshold))

        disthisttrackfilepath = '%s/histograms' % innerrunpath
        disthistfilename = "generation_%s" % f  # define dynamic filename
        disthistfullname = os.path.join(disthisttrackfilepath, disthistfilename + ".png")
        plt.savefig(disthistfullname)
        plt.close()
    return fitnessdict


def write_fasta_alignment(x, y):  # x = current generation of sequence, y = generation number
    """Write fasta alignment from sequences provided."""

    fastafilepath = '%s/fastas' % runpath
    fastafilename = "generation_%s" % y  # define dynamic filename
    fullname = os.path.join(fastafilepath, fastafilename+".fasta")
    fastafile = open(fullname, "w+")  # open file
    for i in range(len(x)):  # write fasta header followed by residue in generation string
        listtowrite = x[i]
        fastafile.write("\n>clone_%s\n" % (i+1))
        for j in listtowrite:
            fastafile.write(j)


def finalfastawriter(x, y, z):  # x = current generation, y = bifurication state, z = roots
    treefastafilepath = '%s/treefastas' % runpath
    treefastafilename = "selected_fastas"
    fullname = os.path.join(treefastafilepath, treefastafilename+".fasta")
    treefastafile = open(fullname, "w+")  # open file
    bifsize = 0
    for bifs in y:
        bifsize += len(bifs)
    bifursize = bifsize/len(y)
    amountofclonestotake = int((bifursize-1)/2)  # if 5, gives 2, if 4 gives 2, if 3 gives 1.
    generationnumbers = []
    for i in y:
        cloneselection = sample(set(i), amountofclonestotake)
        for j in cloneselection:
            generationnumbers.append(j)
    for k in generationnumbers:  # write fasta header followed by residue in generation string
        listtowrite = x[k]
        treefastafile.write(">clone_%s\n" % (k+1))
        for l in listtowrite:
            treefastafile.write(l)
        treefastafile.write('\n')
    rootselection = choice(z)
    roottowrite = x[rootselection]
    treefastafile.write(">root\n")
    for m in roottowrite:
        treefastafile.write(m)


def generationator(d, e, f, g, h, z):  # d = number of generations to run; e = protein generation to start; f = fitnessthreshold; g = amount of mutations per generation, h = writerate, z = matrix
    """Generation generator - mutate a protein for a defined number of
    generations according to an LG matrix and gamma distribution.
    """

    clonelist = []  # generate list of clone keys for bifurication
    for n in range(amountofclones):
        clonelist.append(n)

    rootlist = []  # define set number of random roots from list of clones
    for j in range(roots):
        integertosplit = randint(0, len(clonelist) - 1)
        while integertosplit in rootlist:
            integertosplit = randint(0, len(clonelist) - 1)
        clonelist.remove(integertosplit)
        rootlist.append(integertosplit)

    rootssavepath = "%s/start" % runpath
    rootsfilename = "Roots"
    rootsfullname = os.path.join(rootssavepath, rootsfilename + ".txt")
    rootsfile = open(rootsfullname, "w+")  # open file
    rootsfile.write('Roots:')
    for k in rootlist:
        rootsfile.write('\nClone %s' % str(k+1))

    bifurstart = amountofclones - roots  # do sums to calculate amount of bifurications per generation.
    bifurlist = [1]
    for m in bifurlist:
        bifurlist.append(1)
        bifurstart = bifurstart / 2
        if bifurstart < 6:  # stop when there are 3, 4, 5 or 6 leaves per branch.
            break
    amountofbifurs = len(bifurlist)
    bifurgeneration = int(amountofgenerations/amountofbifurs)  # amount of generations per bifurication.

    clonelistlist = []  # place to store bifurcations (list of lists of clone keys)
    clonelistlist.append(clonelist)  # store all clones that are not root to start
    generationdict = {}  # where all of the generations will be stored
    generationfitdict = {}  # where the fitness of each generation will be stored
    genfitdict = {}  # where the final output dictionary of format {generationkey:{[dictionary of mutatants],[dictionary of fitness]}}
    generation = copy.deepcopy(e)  # current generation
    generationcounter = 0  # count generation for filenames
    generationfitness = record_generation_fitness(generation, trackdotfitness, trackrate, generationcounter, proteinfitness, trackhistfitnessstats, trackhistfitness, trackinvariants, variantaminos)  # current generation fitness
    generationdict.update({0: generation})  # append fitness of starting generation
    generationfitdict.update({0: generationfitness})  # append fitness of starting generation
    fitnessthresh = f

    for i in tqdm(list(range(d))):  # run evolution for d generations
        if i == 0 or (i % h) == 0:  # record fasta every x generations
            write_fasta_alignment(generation, generationcounter)
        if i % bifurgeneration == 0 and i != 0 and len(clonelistlist[0]) > 3:  # Bifuricationmaker. Bifuricates in even generation numbers so every branch on tree has 3 leaves that have been evolving by the last generation
            lists = []  # space to store bifurcations before adding them to clonelistlist
            for j in clonelistlist:  # bifuricate each set of leaves
                shuffle(j)
                half = int(len(j)/2)
                half1 = j[half:]
                half2 = j[:half]
                lists.append(half1)
                lists.append(half2)
            del clonelistlist[:]
            for k in lists:  # append bifurcations to a cleared clonelistlist
                clonelistlist.append(k)
        generation = mutate(g, generation, variantaminos, gammacategories, z)  # mutate generation
        generationcounter += 1
        generationfitness = record_generation_fitness(generation, trackdotfitness, trackrate, generationcounter, proteinfitness, trackhistfitnessstats, trackhistfitness, trackinvariants, variantaminos)  # re-calculate fitness
        duplicationcounter = 0  # need to fix the counter
        if any(j < fitnessthresh for j in list(generationfitness.values())):  # check if any of the current generations fitness values are below the threshold
            for k in range(len(generationfitness)):  # if there are, start loop on generationfitness
                if generationfitness[k] < fitnessthreshold:  # if fitness is less than threshold clone a random sequence in its place.

                    duplicationcounter = duplicationcounter+1
                    unfitkey = k
                    clonelistlistcount = 0
                    for m in clonelistlist:
                        if k not in m:  # check bifurications
                            clonelistlistcount += 1  # Root searching - counts every bifurcation that does not contain the unfit clone.
                        if k in m:
                            clonekey = choice(m)
                            while clonekey == k or generationfitness[clonekey] < fitnessthreshold:  # ensure you don't clone the target unfit generation or another clone that is also of fitness below the threshold
                                clonekey = choice(m)  # choose another random member in the bifurcation.
                    if clonelistlistcount == len(clonelistlist):  # if every bifurcation does not contain the unfit clone it belongs to the root
                        clonekey = choice(rootlist)  # choose random root to replace unfit sequence
                        while clonekey == k or generationfitness[clonekey] < fitnessthreshold:
                            clonekey = choice(rootlist)  # ensure you don't clone the target unfit generation or another clone that is also of fitness below the threshold
                    if generationfitness[clonekey] < fitnessthreshold:  # make warning messages if code breaks
                        print("clone %s is unfit with a value of %s, it will be replaced by:" % (k, generationfitness[k]))
                        print("clone %s with a fitness of %s" % (clonekey, generationfitness[clonekey]))
                        print('WARNING: clonekey fitness is too low or mutation rate is too high')  # Bug in this section that causes infinite loop if mutation rate is too high. Happens when a bifurication has a small number of clones to be replaced by, and the high mutation rate causes all clones to dip below the threshold in one generation.
                        print(generationfitness)
                        print(clonelistlist)
                    generation[unfitkey] = generation[clonekey]   # swap out unfit clone for fit clone

        generationdict.update({i + 1: generation})  # add next generation to dictionary
        generationfitdict.update({i + 1: generationfitness})  # add next generation fitness to dictionary
    write_fasta_alignment(generation, generationcounter)
    finalfastawriter(generation, clonelistlist, rootlist)
    for l in range(len(generationdict)):  # combine fitness and generation dictionaries into one.
        dictlist = [generationdict[l], generationfitdict[l]]
        genfitdict.update({l: dictlist})
    # print genfitdict[d]
    return genfitdict


def fitbit(a, b, c):  # a=evolution dictionary; b=amount of generations; c=amountofclones; plots fitness against generation for all clones
    plt.figure()
    fitnessarray = np.zeros(shape=(b, c))  # build an array in computer memory the size of the final arrayed dataset. This is the mose efficient data structure
    # print fitnessarray
    for i in range(len(a)-1):  # generates a matrix of the fitness values
        dictaccess = a[i]  # access generation i
        accessfitness = dictaccess[-1]  # access fitness of clones in generation i
        fitnesslist = []  # store fitnesses in a list
        for j in range(len(accessfitness)):
            fitnesslist.append(accessfitness[j])
        fitnessarray[i] = fitnesslist  # append fitnesses to array. Array form: x = clones, y = generations
    for k in range(c-1):  # record how each clone's fitness changes over each generation
        clonefitness = []  # list describing how a clone changes over each generation (y axis to plot)
        gen = -1  # generation counter
        genlist = []  # list containing each generation (x axis to plot)
        for l in range(len(fitnessarray)):  # for each generation
            gen += 1  # advance generation counter
            genlist.append(gen)  # append to x axis
            fitnessofclone = fitnessarray[l]  # access lth generation list
            fitnessofgeneration = fitnessofclone[k]  # access lth generation list's kth clone fitness
            clonefitness.append(fitnessofgeneration)  # append it y axis
        plt.plot(genlist, clonefitness)
    averagefitness = []
    for m in range(len(fitnessarray)):  # record average fitness of all clones over each generation
        fitnessofclone2 = fitnessarray[m]
        sumfitness = sum(fitnessofclone2)
        avgfitness = float(sumfitness)/float(amountofclones)
        averagefitness.append(avgfitness)
    plt.plot([0, b], [fitnessthreshold, fitnessthreshold], 'k-', lw=2)
    plt.plot(averagefitness, "k--", lw=2)
    #plt.ylim([fitnessthreshold-25, calculate_fitness(firstprotein)+10])  # not suitable for "low or med" graphs
    #plt.ylim([fitnessthreshold-5, ((amountofaminos+1)*mu)+80]) # for low graphs
    plt.ylim([fitnessthreshold-25, calculate_fitness(firstprotein)+100])  # suitable for med graphs
    plt.xlabel("Generations", fontweight='bold')
    plt.ylabel("Fitness", fontweight='bold')
    plt.title("\n".join(wrap('Fitness change for %s randomly generated "superfit" clones of %s amino acids, mutated over %s generations' % (amountofclones, (amountofaminos+1), amountofgenerations), 60)), fontweight='bold')
    plt.text(amountofgenerations-1000, calculate_fitness(firstprotein)+50, "$\mu$ = %s\n$\sigma$ = %s\n$\delta$ = %s" % (mu, sigma, mutationrate))

    fitgraphfilepath = '%s/fitnessgraph' % runpath
    fitgraphfilename = "fitness_change_over %s generations" % b # define dynamic filename
    fitgraphfullname = os.path.join(fitgraphfilepath, fitgraphfilename + ".png")

    return plt.savefig(fitgraphfullname)


if __name__ == '__main__':

    # from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    # from matplotlib.figure import Figure
    # matplotlib.use('TkAgg')

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
    settingsfullname = os.path.join(settingsfilepath, settingsfilename + ".txt")
    settingsfile = open(settingsfullname, "w+")  # open file

    settingsfile.write("Protein length: %s" % (amountofaminos+1))
    settingsfile.write("\nAmount of mutations per generation: %s" % amountofmutations)
    settingsfile.write("\nAmount of clones in the population: %s" % amountofclones)
    settingsfile.write("\nAmount of generations simulation is run for: %s" % amountofgenerations)
    settingsfile.write("\nFitness threshold: %s" % fitnessthreshold)
    settingsfile.write("\n\nNormal distribution properties: mu = %s, sigma = %s" % (mu, sigma))
    settingsfile.write("\nGamma distribution properties: kappa = %s, theta = %s" % (gamma_shape, gamma_scale))
    settingsfile.write("\n\nWrite rate for FASTA: every %s generations" % writerate)
    settingsfile.write("\n\nTrack rate for graphing and statistics: every %s generations" % trackrate)
    settingsfile.write("\nTracking state: Fitness dot matrix = %s; Fitness histrogram = %s; Fitness normality statistics = %s" % (trackdotfitness, trackhistfitness, trackhistfitnessstats))

    # load matrix
    aamatrix = "data/LGaa.csv"  # .csv file defining aa substitution probabilities calculated from R matrix multiplied by PI matrix, with diagonals forced to zero as mutation has to happen then conferted to event rates p(lambda) where lambda = sum Qx and p(lambda)x=Qxy/lambda
    LGmatrixreader = csv.reader(open(aamatrix), delimiter=",")
    LGmatrixlist = list(LGmatrixreader)
    LGmatrix = np.array(LGmatrixlist)  # load matrix into a numpy array
    LGmatrix = np.delete(LGmatrix, 0, 0)  # trim first line of the array as its not useful

    firstprotein = generate_protein(amountofaminos)  # make first protein
    proteinfitness = get_protein_fitness(firstprotein)  # make first fitness dictionary

    variantaminos = get_allowed_sites(amountofaminos, amountofanchors)  # generate invariant sites
    firstprotein = superfit(proteinfitness, variantaminos, firstprotein, startingfitness)  # generate a superfit protein taking into account the invariant sites created (calling variables in this order stops the evolutionary process being biased by superfit invariant sites.)
    gammacategories = gammaray(gamma_iterations, gamma_samples, gamma_shape, gamma_scale, amountofaminos)  # generate gamma categories for every site

    firstproteinsavepath = '%s/start' % runpath
    firstproteinfilename = "firstprotein"
    firstproteinfullname = os.path.join(firstproteinsavepath, firstproteinfilename + ".fas")
    firstproteinfile = open(firstproteinfullname, "w+")  # open file
    firstproteinfile.write('>firstprotein\n')
    for prot in firstprotein:
        firstproteinfile.write(prot)
    # print 'first superfit protein:', firstprotein
    # print 'fitness of the first protein:', calculate_fitness(firstprotein)

    somestartingclones = clones(amountofclones, firstprotein)  # make some clones to seed evolution

    evolution = generationator(amountofgenerations, somestartingclones,
                               fitnessthreshold, amountofmutations, writerate,
                               LGmatrix)

    fitbit(evolution, amountofgenerations, amountofclones)
