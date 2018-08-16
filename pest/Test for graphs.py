
import numpy as np
from numpy import median
from numpy.random import gamma
import scipy.special as sps
import matplotlib
import copy
import os.path
import shutil
import csv
from scipy.stats import shapiro, anderson, normaltest, skew, skewtest, kurtosis, kurtosistest
import json
from random import randint
from random import sample
from random import choice
from numpy.random import normal
from numpy.random import uniform
from matplotlib import pyplot as plt
from textwrap import wrap
from tqdm import tqdm

matplotlib.use('TkAgg')

# define starting variables
# amino acids - every fitness value string references residues string
residues = ["R", "H", "K", "D", "E", "S", "T", "N", "Q", "C", "G", "P", "A", "V", "I", "L", "M", "F", "Y", "W"]

# parameters for normal distribution used to select fitness values
mu = -1
sigma = 2

# parameters for forming discrete gamma distribution used for evolution of protein
gammaiterations = 100
gammasamples = 10000
gammashape = 1.9  # Most phylogenetic systems that use gamma only let you set kappa (often called shape alpha) and calculate theta as 1/kappa giving mean of 1
gammascale = 1 / gammashape

# parameters of protein evolution
amountofclones = 50  # amount of clones that will be generated in the first generation
amountofgenerations = 100  # amount of generations the protein evolves for
startingfitness = 5  # fitness cutoff - only used if generating a protein above a set fitness
fitnessthreshold = 10  # arbitrary number for fitness threshold
amountofaminos = 200  # number of amino acids in the protein
mutationrate = 0.00075  # should be small!
amountofmutations = int((amountofclones * amountofaminos) * mutationrate)  # number of mutations per generation
writerate = 100  # write a new fasta file every x generations
amountofanchors = int(amountofaminos / 10)  # amount of invariant sites in a generation

# create folder for fastas to be stored in, overwriting any file of same name
pathforfastas = "/Users/Adam/Documents/PhD/PhDyear2/Modelling/fastasfromrun"
if not os.path.exists(pathforfastas):
    os.makedirs(pathforfastas)
else:
    shutil.rmtree(pathforfastas)
    os.makedirs(pathforfastas)

# load matrix
aamatrix = "/Users/Adam/Documents/PhD/PhDyear2/Modelling/LGaa.csv"  # .csv file defining aa substitution probabilities calculated from R matrix multiplied by PI matrix, with diagonals forced to zero as mutation has to happen then conferted to event rates p(lambda) where lambda = sum Qx and p(lambda)x=Qxy/lambda
LGmatrixreader = csv.reader(open(aamatrix), delimiter=",")
LGmatrixlist = list(LGmatrixreader)
LGmatrix = np.array(LGmatrixlist)  # load matrix into a numpy array
LGmatrix = np.delete(LGmatrix, 0, 0)  # trim first line of the array as its not useful


# This module generates an original starting protein 20aa long, with a start methionine
def proteingenerator():
    amino = amountofaminos
    firstresidue = "M"
    original_protein = []
    original_protein.append(firstresidue)
    for i in range(amino):
        x = randint(0, len(residues) - 1)
        original_protein.append(residues[x])

    return original_protein


# This module is unused, but generates a graph to test normal distribution shape if required
def normaldistributiontester():
    # generate distribution
    s = normal(mu, sigma, 2000)
    # plot distribuiton
    count, bins, ignored = plt.hist(s, 30, density=True)
    plt.plot(bins, 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (bins - mu) ** 2 / (2 * sigma ** 2)), linewidth=2,
             color='r')
    return plt.show()


# normaldistributiontester()


# This module generates a string of fitness values for each amino acid in residues argument
def fitmodule():
    fitness = []
    for j in range(len(residues)):
        fitnessvalue = normal(mu, sigma)
        fitness.append(fitnessvalue)
    return fitness

# This module generates a dictionary describing list of fitness values at each position of the generated protein
def proteinfitnessdefiner(x):  # x=protein,
    fitnesslib = {}
    for k in range(len(x)):
        fitvalues = fitmodule()
        fitnesslib.update({
                              k: fitvalues})  # dictionary contains position in the protein as keys and the string of fitness values for each amino acids as variable
        # extra section to remove fitness value for start methionine, hash out line above and unhash lines below if to be used.
        # if k = 0:
        # mposition = residues.index("M")
        # fitvalues[mposition] = 0
        # fitnesslib.update({k: fitvalues}) #dictionary contains position in the protein as keys and the string of fitness values for each amino acids as variable
        # else
        # fitnesslib.update({k: fitvalues})
    return fitnesslib


# This module generates a dictionary containing X clones of generated protein - this contains the evolving dataset.
def clones(x, y):  # x= number of clones to generate, y = protein
    cloneslib = {}
    for l in range(x):
        cloneslib.update({l: y})
    return cloneslib


# normaldistributiontester()


firstprotein = proteingenerator()  # make first protein
proteinfitness = proteinfitnessdefiner(firstprotein)  # make first fitness dictionary

trackfitness = "dist"  # point, dist, both or off.
f=10
e=2

if trackfitness == ("point" or "both") and (f%e==0 or f==0):
    print("hi")
    pathforpointfittrack = "/Users/Adam/Documents/PhD/PhDyear2/Modelling/pointfittrackfromrun"
    if not os.path.exists(pathforpointfittrack):
        os.makedirs(pathforpointfittrack)
    else:
        shutil.rmtree(pathforpointfittrack)
        os.makedirs(pathforpointfittrack)

    fittrackerYaxis = []
    for i in range(amountofaminos):
        fittrackerYaxis.append(proteinfitness[i])
    addition = 0
    points = 0
    for i in fittrackerYaxis:
        for j in i:
            points += 1
            addition += j
    avgtoplot = addition/points
    transform = amountofaminos+(amountofaminos/10)
    for i in range(len(fittrackerYaxis)):
        Yaxistoplot = fittrackerYaxis[i]
        plt.plot(len(Yaxistoplot) * [i], Yaxistoplot, ".", color='k')
    for j in range(len(fittrackerYaxis)):
        Yaxistoplot = fittrackerYaxis[j]
        plt.plot(len(Yaxistoplot) * [j+transform], Yaxistoplot, ".", color='k')
    plt.axis([-10, (amountofaminos*2)+(amountofaminos/10)+10, -15, 15])
    plt.plot([0, len(fittrackerYaxis)+1], [avgtoplot, avgtoplot], 'r--', lw=3)
    plt.plot([0+transform, len(fittrackerYaxis)+1+transform], [avgtoplot, avgtoplot], 'r--', lw=3)
    plt.xticks([])
    fitfilename = "generation_%s" % 0  # define dynamic filename
    fitsavepath = pathforpointfittrack
    fitfullname = os.path.join(fitsavepath, fitfilename + ".png")
    plt.savefig(fitfullname)


if trackfitness == ("dist" or "both"):
    pathfordistfittrack = "/Users/Adam/Documents/PhD/PhDyear2/Modelling/distfittrackfromrun"
    if not os.path.exists(pathfordistfittrack):
        os.makedirs(pathfordistfittrack)
    else:
        shutil.rmtree(pathfordistfittrack)
        os.makedirs(pathfordistfittrack)

    disttrackfilename = "normal_distribution_statistics"  # define evo filename
    disttrackfullname = os.path.join(pathfordistfittrack, disttrackfilename + ".txt")
    disttrackfile = open(disttrackfullname, "w+")  # open file

    disttrackfile.write('Tests for normality on the global amino acid fitness space at each position: \n\n\n')
    disttrackfile.write('Skewness: \n')

    disttrackerlist = []
    disttrackeryaxis = []
    distshapirolist = []
    distandersonlist = []
    distskewkurtalllist = []
    for i in range(amountofaminos):
        disttrackerlist.append(proteinfitness[i])
    distaddition = 0
    distpoints = 0

    for i in disttrackerlist:
        distshapirolist.append(shapiro(i))
        distandersonlist.append(anderson(i))
        distskewkurtalllist.append(normaltest(i))
        for j in i:
            disttrackeryaxis.append(j)

    skewness = skew(disttrackeryaxis)
    skewstats = skewtest(disttrackeryaxis)
    disttrackfile.write("\nThe skewness of the data is %s" % skewness)
    disttrackfile.write("\nAccording to the test of skewness, the hypothesis that the skew matches that from a normal distribution is ")
    if skewstats >= 0.05:
        disttrackfile.write("True\n\n\n")
    else:
        disttrackfile.write("False\n\n\n")

    disttrackfile.write("Kurtosis: \n")

    kurtosis = kurtosis(disttrackeryaxis)
    kurtstats = kurtosistest(disttrackeryaxis)
    disttrackfile.write("\nThe kurtosis of the data is %s" % kurtosis)
    disttrackfile.write("\nAccording to the test of kurtosis, the hypothesis that the kurtosis matches that from a normal distribution is ")
    if kurtstats >= 0.05:
        disttrackfile.write("True\n\n\n")
    else:
        disttrackfile.write("False\n\n\n")

    disttrackfile.write('Shapiro-Wilk test of non-normality: \n')

    totalshapiro = shapiro(disttrackeryaxis)

    disttrackfile.write("\nThe Shapiro-Wilk test of non-normality for the entire dataset gives p = %s" % totalshapiro[-1])
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
    passpercent = (sum(passpercentcalc)/len(passpercentcalc))*100
    disttrackfile.write("\n\nAccording to Shapiro-Wilk test, the proportion of individual positions that are not confidently non-normal is: %s%%" % passpercent)

    disttrackfile.write('\n\n\nAnderson-Darling test of normality: \n')
    x = np.random.rand(10000)
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

    conffifteen = []
    conften = []
    conffive = []
    conftpf = []
    confone = []
    reject = []
    for i in distandersonlist:
        print(i)
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
    percentfifteen = (len(conffifteen)/len(distandersonlist))*100
    percentten = (len(conften) / len(distandersonlist)) * 100
    percentfive = (len(conffive) / len(distandersonlist)) * 100
    percenttpf = (len(conftpf) / len(distandersonlist)) * 100
    percentone = (len(confone) / len(distandersonlist)) * 100
    percentreject = (len(reject) / len(distandersonlist)) * 100
    disttrackfile.write("\n\nAccording to the Anderson-Darling test the hypothesis of normality is not rejected for each position in the dataset for:")
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
    skewkurtpercent = (sum(skewkurtpass)/len(skewkurtpass))*100
    disttrackfile.write("\n\nAccording to the skewness-kurtosis all test, %s%% of sites do not differ significantly from a normal distribution" % skewkurtpercent)
    plt.figure()
    plt.axis([-10, 8, 0, 0.5])  # generate attractive figure
    mudist = sum(disttrackeryaxis)/len(disttrackeryaxis)
    plt.hist(disttrackeryaxis, 20, density=True, color='k', alpha=0.4)
    plt.title("\n".join(wrap('Fitness distribution of the total fitness space', 60)), fontweight='bold')
    plt.axvline(x=mudist, color="#404040", linestyle=":")
    mudistdp = "%.3f" % mudist
    plt.text(4.3, 0.46, "$\mu$1 = %s" % mudistdp)
    disthistfilename = "generation_%s" % 0  # define dynamic filename
    disthistsavepath = pathfordistfittrack
    disthistfullname = os.path.join(disthistsavepath, disthistfilename + ".png")
    plt.savefig(disthistfullname)
    plt.close()
elif trackfitness == "off":
    pass


