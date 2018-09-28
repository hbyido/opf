from pypower.api import case9, ppoption, runpf, printpf

# See http://rwl.github.io/PYPOWER/api/ for description of variables
ppc = case9()
ppopt = ppoption(PF_ALG=2)
r = runpf(ppc, ppopt)
#printpf(r[0])