
# Data format taken from: http://www.pserc.cornell.edu/matpower/manual.pdf (page 122)

PQ = 1
PV = 2
SLACK = 3

# BUS CONSTANTS - 0-indexing for python, add 1 to all get 1-indexing in matlab
BUS_I = 0 # bus number (positive integer)
BUS_TYPE = 1 #  bus type (1 = PQ (load), 2 = PV (generator), 3 = ref (slack), 4 = isolated)
PD = 2 # real power demand (MW)
QD = 3 # reactive power demand (MVar)
GS = 4 # shunt conductance (MW demand at V = 1.0 pu)
BS = 5  # shunt susceptance (MVAr injected at V = 1.0 pu)
BUS_AREA = 6 # area number (positive integer)
VM = 7 # voltage magnitude (p.u.)
VA = 8 # voltage angle (degrees)
baseKV = 9 # base voltage (kV)
zone = 10 # Loss zone (positive integer)
Vmax = 11 # maximum voltage magnitude (p.u.) 
Vmin = 12 # minimum voltage magnitude (p.u)

# Generator Constants
GEN_BUS = 0  # bus number
PG = 1 # real power output (MW)
QG = 2 # reactive power output (MVAr)
QMAX = 3 # maximum reactive power output (MVAr)
QMIN = 4 # minimum reactive power output (MVAr)
VG = 5 # voltage magnitude setpoint (p.u.)
MBASE = 6 # total MVA base of machine, defaults to baseMVA
STATUS = 7 # machine status, > 0 = machine in-service, ≤ 0 = machine out-of-service
PMAX = 8 # maximum real power output (MW)
PMIN = 9 # minimum real power output (MW)
PC1 = 10 # lower real power output of PQ capability curve (MW)
PC2 = 11 # upper real power output of PQ capability curve (MW)
QC1MIN =  12 # minimum reactive power output at PC1 (MVAr)
QC1MAX = 13 # maximum reactive power output at PC1 (MVAr)
QC2MIN = 14 # minimum reactive power output at PC2 (MVAr)
QC2MAX = 15 # maximum reactive power output at PC2 (MVAr)
RAMP_AGC =  16 # ramp rate for load following/AGC (MW/min)
RAMP_10 =  17 # ramp rate for 10 minute reserves (MW)
RAMP_30 = 18 # ramp rate for 30 minute reserves (MW)
RAMP_Q =  19 # ramp rate for reactive power (2 sec timescale) (MVAr/min)
APF = 20 # area participation factor

# Branch Constants:
FBUS = 0 # "from" bus number 
TBUS = 1 # "to" bus number
BR_R = 2 # Resistance (p.u)
BR_X = 3 # reactance (p.u)
BR_B = 4 # total line charging susceptance (p.u)
RATEA = 5 # MVA rating A    
RATEB = 6 # MVA rating B
RATEC = 7 # MVA rating C
TAP = 8   # transformer off nominal turns ratio
SHIFT = 9 # transformer phase shift angles (degrees)
BR_STATUS = 10 # initial branch status 1 = in-service, 0 = out of service 
ANGMIN = 11 # minimum angle difference 
ANGMAX = 12 # maximum angle difference 

# Generator Cost Data: 
MODEl = 0 # cost model, 1 = piecewise linear, 2 = polynomial 
STARTUP = 1 # startup cost 
SHUTDOWN = 2 # shutdown cost 
NCOST = 3 # nnumber of cost coefficients for polynomial cost function, or number of datapoints for piecewise linear 
''' 
parameters defining total cost function f(p) begin in this column,
units of f and p are $/hr and MW (or MVAr), respectively
(MODEL = 1) ⇒ p0, f0, p1, f1, . . . , pn, fn
where p0 < p1 < · · · < pn and the cost f(p) is defined by
the coordinates (p0, f0), (p1, f1), . . . , (pn, fn)
of the end/break-points of the piecewise linear cost
(MODEL = 2) ⇒ cn, . . . , c1, c0
n + 1 coefficients of n-th order polynomial cost, starting with
highest order, where cost is f(p) = cnp
n + · · · + c1p + c0
''' 
COST = 4