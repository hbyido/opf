from pypower.api import case9, ppoption, runpf, printpf
from constants import *
import numpy as np
# See http://rwl.github.io/PYPOWER/api/ for description of variables
'''
Slack Bus: At the slack bus, the voltage magnitude and angle are fixed and the power
    injections are free. There is only one slack bus in a power system.
Load Bus: At a load bus, or PQ bus, the power injections are fixed while the voltage
    magnitude and angle are free. There are M PQ buses in the system.
Voltage-Controlled Bus: At a voltage controlled bus, or PV bus, the real power
    injection and voltage magnitude are fixed while the reactive power injection
    and the voltage angle are free. (This corresponds to allowing a local source
    of reactive power to regulate the voltage to a desired setpoint.) There are
    N − M − 1 PV buses in the system.
'''
ppc = case9()
ppopt = ppoption(PF_ALG=2, VERBOSE=0, OUT_ALL=0)

r1, succeed = runpf(ppc, ppopt)
print("Success:", succeed)
ppc['gen'][2, PG] = 100000.0
r2, succeed = runpf(ppc, ppopt)
print("Success:", succeed)