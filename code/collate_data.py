import scipy.io as sio
import numpy as np
from constants import *

X = [] # real/imaginary power demand for all nodes 
for i in range(1,10001):
    file = "../data/mpc_%d.mat" % i
    mpc = sio.loadmat(file)
    version, baseMVA, bus, gen, branch, gencost = mpc['mpc'][0][0]
    pd = bus[:,PD] / baseMVA
    qd = bus[:,QD] / baseMVA
    X.append(np.append(pd,qd))
X = np.array(X)
print("Input:", X.shape)
np.save("X",X)

y = [] # real power injection/voltage magnitude on generator nodes 
for i in range(1,10001):
    file = "../data/result_%d.mat" % i
    results= sio.loadmat(file)
    objs = results['results'][0][0]
    gen = objs[3]
    pg = gen[:,PG]
    vg = gen[:,VG]
    y.append(np.append(pg,vg))
y = np.array(y)
print("Output:", y.shape)
np.save("y",y)