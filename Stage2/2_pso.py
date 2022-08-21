import numpy as np
import matplotlib.pyplot as plt

# TOGU CENG 
# Dr. Mahir Kaya

def fun(x):
    return sum(np.power(x,2))

alt_sinir =-5
ust_sinir = 5
problem_boyutu = 5
populasyon_boyutu = 10
w  = 0.8
c1 = 2
c2 = 2

populasyon = np.random.ranf([populasyon_boyutu,problem_boyutu]) * (ust_sinir-alt_sinir) + alt_sinir

obj = np.zeros(populasyon_boyutu)

for i in range(populasyon_boyutu):
    obj[i]=fun(populasyon[i,:])
        
    
velocity = np.zeros([populasyon_boyutu,problem_boyutu])

pBestPos = populasyon
pBestVal = obj

gBestVal = min(obj)
idx = np.where(obj==gBestVal)
gBestPos = populasyon[idx,:]

objit = list()
objit.append(gBestVal)

for k in range(100):
    for i in range(populasyon_boyutu):
        velocity[i,:] = w*velocity[i,:] + \
                        c1*np.random.ranf()*(pBestPos[i,:]-populasyon[i,:]) + \
                        c2*np.random.ranf()*(gBestPos-populasyon[i,:])

    vmax = (ust_sinir - alt_sinir) / 2
    for i in range(populasyon_boyutu):
        for j in range(problem_boyutu):
            if velocity[i,j]>vmax:
                velocity[i,j]=vmax
            elif velocity[i,j]<-vmax:
                velocity[i,j]=-vmax

    populasyon = populasyon + velocity

    for i in range(populasyon_boyutu):
        for j in range(problem_boyutu):
            if populasyon[i,j]>ust_sinir:
                populasyon[i,j]=ust_sinir
            elif populasyon[i,j]<alt_sinir:
                populasyon[i,j]=alt_sinir

    for i in range(populasyon_boyutu):
        obj[i]=fun(populasyon[i,:])

    for i in range(populasyon_boyutu):
        if obj[i]<pBestVal[i]:
            pBestVal[i,:]=populasyon[i,:]
            pBestVal[i]=obj[i]

    if min(obj)<gBestVal:
        gBestVal=min(obj)
        idx = np.where(obj==gBestVal)
        gBestPos = populasyon[idx,:]
    
    objit.append(gBestVal)
    
    print("iterasyon :{}, obj :{}".format(k,gBestVal))

    
plt.plot(objit)
plt.xlabel("iterasyon")
plt.show()

gBestPos = gBestPos[0][0]

print("{:.2f}^2 + {:.2f}^2 + {:.2f}^2 + {:.2f}^2 + {:.2f}^2 = {:.2f}".
      format(gBestPos[0],gBestPos[1],
             gBestPos[2],gBestPos[3],
             gBestPos[4],gBestVal))