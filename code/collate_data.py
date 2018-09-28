import scipy.io as sio
import numpy as np

# Our X is composed of


X = []

# Remember, matlab is 1-index
for i in range(1,10001):
    file = "../data/mpc_%d.mat" % i
    mpc = sio.loadmat(file)
    version, baseMVA, bus, gen, branch, gencost = mpc['mpc'][0][0]
    pd = bus[:,2]
    qd = bus[:,3]
    X.append(np.append(pd,qd))
X = np.array(X)
print("Input:", X.shape)

np.save("X",X)
y = []
for i in range(1,10001):
    file = "../data/result_%d.mat" % i
    results= sio.loadmat(file)
    objs = results['results'][0][0]
    bus = objs[2]
    gen = objs[3]
    pg = gen[:,1]
    qg = gen[:,2]
    y.append(np.append(pg,qg))
y = np.array(y)
print("Output:", y.shape)
np.save("y",y)