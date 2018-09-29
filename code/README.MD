The best way to cast this as a ML problem is to have the output of a model F correspond to 
inputs for conventional power flow problem. In a conventional PF problem, there is 1 slack bus, 
M load buses, and N total buses.  

1. Slack Bus: At the slack bus, the voltage magnitude and angle are fixed and the power
    injections are free. There is only one slack bus in a power system.
2. Load Bus (PQ): At a load bus, or PQ bus, the power injections are fixed while the voltage
    magnitude and angle are free. There are M PQ buses in the system.
3. Voltage-Controlled Bus (PV): At a voltage controlled bus, or PV bus, the real power
    injection and voltage magnitude are fixed while the reactive power injection
    and the voltage angle are free. (This corresponds to allowing a local source
    of reactive power to regulate the voltage to a desired setpoint.) There are
    N − M − 1 PV buses in the system.

Hence, for each bus type, F should output:

1. Slack bus: Nothing, as voltage is fixed to 1.0 pu with angle 0.0 degrees. 
2. PQ Buses: Nothing, as the real/imaginary power demand constitute the input to F.
3. PV Buses: Real power injection (P) and voltage magnitude. 

Hence, F accepts as input a vector of length M*2 corresponding to real and imaginary power loads
and outputs a vector of length (N − M − 1)*2 corresponding to real power generation and voltage magnitude. 
