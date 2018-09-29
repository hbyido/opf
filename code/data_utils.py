import scipy.io as sio
from constants import *
import numpy as np
import cmath
import math 
from collections import defaultdict

class Grid:

    def __init__(self, filename):
        mpc = sio.loadmat(filename)
        key = list(mpc.keys())[3]
        self.version = mpc[key][0][0][0] 
        self.baseMVA = mpc[key][0][0][1][0][0] 
        self.bus_data = mpc[key][0][0][2] 
        self.gen = mpc[key][0][0][3] 
        self.branch = mpc[key][0][0][4] 
        self.gencost = mpc[key][0][0][5]
        self.num_bus = len(self.bus_data)
        self.num_branch = len(self.branch)

    def to_dict(self):
        """ Return MPC object in dict format (for pypower) """
        
        return {
            'version': self.version, 
            'baseMVA': self.baseMVA,
            'bus': self.bus_data[:,:13],
            'gen': self.gen[:,:21], 
            'branch': self.branch[:,:13], 
            'areas': np.array([[1, 5]]), 
            'gencost': self.gencost
        }   
    
    def get_pv_voltage(self):
        """" Return voltage for PV nodes """ 
        gen_indices = self.gen[:,BUS_I].astype(int) - 1 # indices of generators 
        return self.bus_data[gen_indices, VM]

    def get_voltage_magnitude(self):
        """ return voltage magnitude """
        return self.bus_data[:,VM].reshape(self.num_bus)
    
    def get_voltage_angle(self):
        """ return voltage angles """ 
        return self.bus_data[:, VA].reshape(self.num_bus)
    
    def get_pd(self):
        """ Return real power demand """ 
        return self.bus_data[:,PD]
    
    def get_qd(self):
        """ Return real power demand """ 
        return self.bus_data[:,QD]

    def get_pg(self, just_gens = True):
        """ 
        Return the real power injections.
        just_gens: if true (default), return just the generators. Otherwise return for all buses, 
        with 0 for non-generators
        """
        if just_gens:
            return self.gen[:, PG]
        else:
            pg = np.zeros(self.num_bus)
            gen_nums = self.gen[:,BUS_I].astype(int) - 1 # indices of generators 
            pg[gen_nums] = self.gen[:, PG] 
        return pg
    
    def get_qg(self, just_gens = True):
        """ 
        Return the reactive power injections.
        just_gens: if true (default), return just the generators. Otherwise return for all buses, 
        with 0 for non-generators
        """
        if just_gens:
            return self.gen[:, QG]
        else:
            qg = np.zeros(self.num_bus)
            gen_nums = self.gen[:,BUS_I].astype(int) - 1 # indices of generators 
            qg[gen_nums] = self.gen[:, QG] 
        return qg
    
    def check_voltages(self, voltages):
        """ Return True if passed voltages are within min/max voltage constraints """
        for bus, voltage in enumerate(voltages):
            if voltage < self.bus_data[bus, Vmin] or voltage > self.bus_data[bus, Vmax]:
                return False
        return True
    
    def get_cost(self):
        total_cost = 0
        for i in range(len(self.gen)):
            pg = self.gen[i, PG]
            if self.gencost[i, MODEl] != 2:
                raise(NotImplementedError)
            terms = self.gencost[i, NCOST+1:][::-1] # reverse order so coefficients correspond to increasing exponents
            #total_cost += self.gencost[i, STARTUP] Not included in matpower calculation
            for i in range(len(terms)): 
                #print("%f*(%f)^%d" % (terms[i], pg, i))
                total_cost += terms[i]*math.pow(pg, i)
        return total_cost

    def set_powers(self, pg):
        self.gen[:, PG] = pg

    def set_voltage(self, voltage):
        self.gen[:, VG] = voltage        
    
    def check_voltage(self):
        for i in range(self.num_bus):
            if self.bus_data[i, VM] < self.bus_data[i, Vmin] or self.bus_data[i, VM] > self.bus_data[i, Vmax]:
                return False
        return True

    def check_p(self):
        ''' Return true if active power falls within limits ''' 
        for bus, p in enumerate(self.gen[:, PG]):
            if p < self.gen[bus, PMIN] or p > self.gen[bus, PMAX]:
                return False
        return True
    
    def check_q(self):
        ''' Return true if reactive power falls within limits ''' 
        for bus, q in enumerate(self.gen[:, QG]):
            if q < self.gen[bus, QMIN] or q > self.gen[bus, QMAX]:
                return False
        return True
    
    def check_angle_diffs(self):
        ''' Check that the angle differences between buses fall within limits. ''' 
        for branch in range(len(self.branch)):
            fbus = int(self.branch[branch, FBUS]) - 1 # Buses are one-indexed
            tbus = int(self.branch[branch, TBUS]) - 1 # Buses are one-indexed 
            angle_diff = self.bus_data[fbus, VA] - self.bus_data[tbus, VA]
            if angle_diff < self.branch[branch, ANGMIN] or angle_diff > self.branch[branch, ANGMAX]:
                return False
        return True  
    
    def check_all_constraints(self):
        return self.check_p() and self.check_q() and self.check_angle_diffs() and self.check_voltage()

    def construct_admittance_matrix(self):
        # The sparse admittance matrix for the grid. This matrix has only N + 2L entries. If there is no connection 
        # between two buses i and k, Y_ik = 0. We decompose this into a real and imaginary part. 
        # A full description of this construction can be found at: https://pdfs.semanticscholar.org/b2d3/ec1e810b3ee55489647a86f314316bcba3a1.pdf
        # on page 13
        Y = np.zeros((self.num_bus, self.num_bus), dtype=complex) 

        # We begin by calculating the off-diagonal entries for branches from bus i to bus k
        for row in range(self.num_branch):
            i = int(self.branch[row, FBUS]) - 1 # buses are one-indexed
            k = int(self.branch[row, TBUS]) - 1 # buses are one-indexed
            r_ik, x_ik = self.branch[row, BR_R], self.branch[row, BR_X]
            y_ik = complex(r_ik / (r_ik**2 + x_ik**2), -x_ik / (r_ik**2 + x_ik**2))
            rho_ik = math.radians(self.branch[row, SHIFT])
            T_ik = self.branch[row, TAP]
            # When the transformer has a nominal turns ratio (1:1 voltage ration in 
            # per-unit), it is identical to a branch.
            if T_ik == 0:
                T_ik = 1
            alpha_ik = T_ik*cmath.exp(complex(0, rho_ik))
            Y[i,k] += (-1.0 / alpha_ik.conjugate())*y_ik #- (1.0 /alpha_ik)*y_ik
            Y[k, i] += (-1.0 / alpha_ik)*y_ik #- (1.0 /alpha_ik)*y_ik
            

            ySh_ik = complex(0, self.branch[row, BR_B])
            Y[i, i] += 1 / (alpha_ik*alpha_ik.conjugate())*(y_ik + 0.5*ySh_ik)
            Y[k, k] += y_ik + 0.5*ySh_ik
        
        # Fill in diagonal entries 
        for i in range(self.num_bus):
            Y[i,i] +=  complex(self.bus_data[i, GS], self.bus_data[i, BS])
        print(np.array2string(np.round(Y, 2), max_line_width=np.inf))
        return Y

    def get_edges(self):
        self.edges = defaultdict(list)
        for i,k in zip(self.branch[:, FBUS], self.branch[:, TBUS]):
            self.edges[int(i-1)].append(int(k-1))
            self.edges[int(k-1)].append(int(i-1))
        
        for i in range(self.num_bus):
            self.edges[int(i)].append(int(i))
                
    def check_pv(self, i, Y, cp, cq, voltages, angles):
        Vi = voltages[i]
        # loop through branches: 
        for k in self.edges[i]:
            Vk = voltages[k]
            Gik, Bik = Y[i, k].real, Y[i, k].imag
            dT = math.radians(angles[i]) - math.radians(angles[k])
            cp -= Vi * Vk *(Gik*math.cos(dT) + Bik*math.sin(dT))
            cq -= Vi * Vk *(Gik*math.sin(dT) - Bik*math.cos(dT))
        return abs(np.round(cp, 2)) < 0.1 and abs(np.round(cq, 2)) < 0.1
        
    def check_powers(self, reals, reactives, voltages, angles):
        """ 
        Check equality constraints on real and reactive power. Specifically, that: 
            1) P_i(V, \delta) - P_G + P_D = 0
            2) Q_i(V, \delta) - Q_G + Q_D = 0
        where: 
            dT = \delta_i - \delta_k
            P_I(V, \delta) = V_i \Sum^n_{k=1} V_k*(G_ik*cos(dT) + B_ik*sin(dT))
            Q_I(V, \delta) = V_i \Sum^n_{k=1} V_k*(G_ik*sin(dT) - B_ik*cos(dT))

        Taken from page 18 of https://pdfs.semanticscholar.org/b2d3/ec1e810b3ee55489647a86f314316bcba3a1.pdf
        """
        # TODO: Assert that v
        Y = self.construct_admittance_matrix()
        self.get_edges()
        buses =range(self.num_bus)
        #buses = [4]
        for i in buses:
            bt = self.bus_data[i, BUS_TYPE]
            pd = self.bus_data[i, PD] / self.baseMVA
            qd = self.bus_data[i, QD] / self.baseMVA
            if bt == PV or bt == SLACK:
                condition = self.check_pv(i, Y, reals[i] - pd, reactives[i] - qd, voltages, angles)
                print("Bus %d, Succeed: %d " % (i, condition))
            elif bt == PQ:
                condition = self.check_pv(i, Y, 0.0-pd, 0.0-qd, voltages, angles)
                print("Bus %d, Succeed: %d " % (i, condition))
        
def set_test_grid():
    filename = "../data/result_1.mat"
    grid = Grid(filename)

    #TODO: Check solution against Page 30 example
    branches = np.zeros((6, 13))
    branches[:, FBUS] = [1, 1, 2, 3, 3, 4]
    branches[:, TBUS] = [2, 3, 4, 4, 5, 5]
    branches[:, BR_R] = [0.000, 0.023, 0.006, 0.020, 0.000, 0.000]
    branches[:, BR_X] = [0.300, 0.145, 0.032, 0.260, 0.320, 0.500]
    branches[:, BR_B] = [0.000, 0.040, 0.010, 0.000, 0.000, 0.000]
    branches[:,  TAP] = [0.0, 0.0, 0.0, 0.0, 0.95, 0.0]
    branches[:, SHIFT] = [0.0, 0.0, 0.0, 12.38, 0.0, 0.0]
    grid.branch = branches
    
    bus_data = np.zeros((5, 13))
    bus_data[:, BUS_TYPE] = [3, 1, 2, 2, 1]
    bus_data[1, BS] = 0.3
    bus_data[2,GS] = 0.05
    bus_data[3, PD] = 0.900
    bus_data[4, PD] = 0.239
    bus_data[3, QD] = 0.400
    bus_data[4, QD] = 0.129 
    bus_data[:, Vmin] = [1.00, 0.95, 0.95, 0.95, 0.95]
    bus_data[:, Vmax] = [1.00, 1.05, 1.05, 1.05, 1.05]
    grid.bus_data = bus_data
    voltages = [1.0, 0.981, 0.957, 0.968, 0.959]
    deltas = [0.0, -12.59, -1.67, -13.86, -9.13]
    pg = [0.947, np.nan, 0.192, 0.053, np.nan]
    qg = [0.387, np.nan, -0.127, 0.2, np.nan]
    grid.num_branch = 6
    grid.num_bus = 5
    return grid, voltages, deltas, pg, qg

def main():
    #grid, voltages, deltas, pg, qg = set_test_grid()
    filename = "../data/result_1.mat"
    grid = Grid(filename)
    pg = grid.get_pg()
    qg = grid.get_qg()
    voltages = grid.get_voltage_magnitude()
    deltas = grid.get_voltage_angle()
    grid.check_powers(pg, qg, voltages, deltas)
    print(grid.get_cost())
    print(grid.check_all_constraints())
    

if __name__ == "__main__":
    main()

