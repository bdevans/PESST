import csv
import numpy as np
from Bio import SeqIO as seq
import os.path
import shutil


#fitnessmatrix = '/Users/Adam/Documents/PhD/PhDyear2/Modelling/testlibrary.csv'
#fasta = '/Users/Adam/Documents/PhD/PhDyear2/Modelling/test.fasta'

# /Users/Adam/Documents/PhD/PhDyear2/Modelling/XXXXXXXX/start/fitnesslibrary.csv
# /Users/Adam/Documents/PhD/PhDyear2/Modelling/XXXXXXXX/sequences_of_the_marginal_reconstruction_without_reconstruction_of_indels.fas

# /Users/Adam/Documents/PhD/PhDyear2/Modelling/mildfitness_ancescon/mildfitnessrun1/start/fitnesslibrary.csv
# /Users/Adam/Documents/PhD/PhDyear2/Modelling/mildfitness_ancescon/mildfitnessrun1/FastML_full_dataset_reconstruction.fas

fitnessmatrix = input("Fitness matrix location (must be CSV from previous evolution run): ")
fasta = input("input fasta file location (must be a sequence or list of sequences in fasta format): ")


fitnessmatrixreader = csv.reader(open(fitnessmatrix), delimiter=",")
seqlib = {}

fileext = os.path.basename(fasta)
filename = os.path.splitext(fileext)[0]

pathforcalcs="/Users/Adam/Documents/PhD/PhDyear2/Modelling/fitcalc_%s" %filename
if not os.path.exists(pathforcalcs):
    os.makedirs(pathforcalcs)
else:
    shutil.rmtree(pathforcalcs)
    os.makedirs(pathforcalcs)


calcfilename = "%s" % filename  # define dynamic filename
fullname = os.path.join(pathforcalcs, calcfilename+".csv")
calcfile = open(fullname, "w+")  # open file


for sequence in seq.parse(fasta, "fasta"):
    seqid=sequence.id
    seqlib[seqid] = sequence.seq


fitnessmatrixlist = list(fitnessmatrixreader)
fitnessmatrix = np.array(fitnessmatrixlist)  # load matrix into a numpy array
firstline = fitnessmatrix[0]

dictoffitness = {}

for key, value in seqlib.items():
    #print key
    #print value
    seqlist = []
    dictlist = []
    dictlist.append(value)
    for j in value:
        seqlist.append(j)
    count = 0
    fitnesssum = 0
    for i in seqlist:
        count += 1
        index = np.where(firstline == i)
        fitnessrow = fitnessmatrix[count]
        fitnessvalue = fitnessrow[index]
        #print float(fitnessvalue[0])
        fitnesssum += float(fitnessvalue[0])
    dictlist.append(fitnesssum)
    dictoffitness[key] = dictlist

print(dictoffitness)

calcfile.write("clone,fitness,sequence\n")
for key, value in dictoffitness.items():
    calcfile.write("%s,%s,%s\n" % (key, value[1], value[0]))


#LGmatrix = np.delete(LGmatrix, 0, 0)  # trim first line of the array as its not useful

#print LGmatrixlist
#print LGmatrix

#run18-06-22-12-41