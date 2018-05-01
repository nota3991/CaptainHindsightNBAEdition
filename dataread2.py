import numpy as np
import csv
import reader1
import matplotlib.pyplot as plt 

data_dir = 'C:/Users/Dongjie/Desktop/4502 project/edit1.csv'
data = reader1.read_Data(data_dir)

print(data.shape)
n1 = np.zeros(229)
ptsdict = {}
for i in range(data.shape[0]):
    if data[i][5] in ptsdict.keys():
        n1[data[i][5]] = n1[data[i][5]] + 1
        ptsdict[data[i][5]] = (ptsdict[data[i][5]] + data[i][11])
        
    else:
        ptsdict[data[i][5]] = data[i][11]
        n1[data[i][5]] = 1
        
for key in ptsdict.keys():
    ptsdict[key] = ptsdict[key]/n1[key]
        

n2 = np.zeros(229)   
trbdict = {}
for i in range(data.shape[0]):
    if data[i][5] in trbdict.keys():
        n2[data[i][5]] = n2[data[i][5]] + 1
        trbdict[data[i][5]] = (trbdict[data[i][5]] + data[i][12])
    else:
        trbdict[data[i][5]] = data[i][12]
        n2[data[i][5]] = 1
        
for key in trbdict.keys():
    trbdict[key] = trbdict[key]/n2[key]
 
n3 = np.zeros(229) 
astdict = {}
for i in range(data.shape[0]):
    if data[i][5] in astdict.keys():
        n3[data[i][5]] = n3[data[i][5]] + 1
        astdict[data[i][5]] = astdict[data[i][5]] + data[i][13]
    else:
        astdict[data[i][5]] = data[i][13]
        n3[data[i][5]] = 1

for key in astdict.keys():
    astdict[key] = astdict[key]/n3[key]    

    

for key in range(1,61):
    print('For pick:',key)
    print(ptsdict[key])
    print(trbdict[key])
    print(astdict[key])

a1 = np.zeros(61)
b1 = np.zeros(61)

for i in range(1,61):
    a1[i] = ptsdict[i]
    b1[i] = i

plt.bar(b1,a1,width=0.8,facecolor="#9999ff",edgecolor="white")
plt.title('The average pts for every pick')
plt.show()

a2 = np.zeros(61)
b2 = np.zeros(61)

for i in range(1,61):
    a2[i] = trbdict[i]
    b2[i] = i

plt.bar(b2,a2,width=0.8,facecolor="#9999ff",edgecolor="white")
plt.title('The average trb for every pick')
plt.show()

a3 = np.zeros(61)
b3 = np.zeros(61)

for i in range(1,61):
    a3[i] = astdict[i]
    b3[i] = i
    
plt.bar(b3,a3,width=0.8,facecolor="#9999ff",edgecolor="white")
plt.title('The average ast for every pick')
plt.show()



