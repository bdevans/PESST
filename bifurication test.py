from random import randint
from random import shuffle

generations = 5000
clones = 100
roots = clones/25

bifurstart = clones - roots
bifurlist = [1]
for i in bifurlist:
    bifurlist.append(1)
    bifurstart = bifurstart/2
    if bifurstart <= 3:
        break

amountofbifurs = len(bifurlist)

bifurgeneration = generations/amountofbifurs+1
print bifurgeneration

clonelist = []
for i in range(clones):
    clonelist.append(i)


rootlist = []
for i in range(roots):
    integertosplit = randint(0, len(clonelist)-1)
    while integertosplit in rootlist:
            integertosplit = randint(0, len(clonelist) - 1)
    clonelist.remove(integertosplit)
    rootlist.append(integertosplit)

print clonelist
print rootlist

clonelistlist = []
clonelistlist.append(clonelist)

for j in range(len(bifurlist)-1):
    lists = []
    for i in clonelistlist:
        shuffle(i)
        half = len(i)/2
        half1 = i[half:]
        half2 = i[:half]
        lists.append(half1)
        lists.append(half2)
    del clonelistlist[:]
    for k in lists:
        clonelistlist.append(k)
    print clonelistlist
