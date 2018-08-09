import numpy as np
import csv
from numpy.random import uniform
import matplotlib.pyplot as plt
from textwrap import wrap

aminoacid = "A"  # amino acid to change
generations = 1000  # number of generations for bar graph

aamatrix = "/Users/Adam/Documents/PhD/PhDyear2/Modelling/LGaa.csv"  # csv file defining aa substitution probabilities calculated from R matrix multiplied by PI matrix, with diagonals forced to zero as mutation has to happen then conferted to event rates p(lambda) where lambda = sum Qx and p(lambda)x=Qxy/lambda
LGmatrixreader = csv.reader(open(aamatrix), delimiter=",")
LGmatrixlist = list(LGmatrixreader)
LGmatrix = np.array(LGmatrixlist)  # load matrix into a numpy array
LGmatrix = np.delete(LGmatrix, 0, 0)  # trim first line of the array as its not useful


def matrixmutator(a, b):  # a = matrix, b = current amino acid
    aminolist = []  # space for the order of the aminos correspondong to the values in the dictionaries this code makes from the numpy matrix
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


print 'Mutation: %s -> %s' % (aminoacid, matrixmutator(LGmatrix, aminoacid))


# This produces a bar chart of selections from X mutation events of a pre-defined amino acid.
choice = []
for p in range(generations):
    aminochoice = matrixmutator(LGmatrix, aminoacid)
    choice.append(aminochoice)

x = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']

counted = []
for q in x:
    if q in choice:
        counted.append(choice.count(q))
    else:
        counted.append(0)

bar = plt.bar(x, counted, align='center')
for rect in bar:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width()/2.0, height, '%d' % int(height), ha='center', va='bottom')
plt.title("\n".join(wrap('%s' % aminoacid, 60)), fontweight='bold', fontsize=10)
plt.text(0, 250, "Generations: %s" % generations)
plt.show()
