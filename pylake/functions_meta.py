import numpy as np
from .functions import *

def getSchmidt(temperature, gas):
    range_t = [4,35]
    gases = {"He":[368,-16.75,0.374,-0.0036],
	            "O2":[1568,-86.04,2.142,-0.0216],
	            "CO2":[1742,-91.24,2.208,-0.0219],
	            "CH4":[1824,-98.12,2.413,-0.0241],
	            "SF6":[3255,-217.13,6.837,-0.0861],
	            "N2O":[2105,-130.08,3.486,-0.0365],
	            "Ar":[1799,-106.96,2.797,-0.0289],
	            "N2":[1615,-92.15,2.349,-0.0240]}
    A = gases[gas][0]
    B = gases[gas][1]
    C = gases[gas][2]
    D = gases[gas][3]

    Sc = A+B*temperature+C*temperature**2+D*temperature**3
    return Sc

def thermalExpFromTemp(Ts):
    
    V = 1       
    dT = 0.001 
    T1 = dens0(t=Ts, s=0.2)
    T2 = dens0(t=Ts+dT, s=0.2)
    V2 = T1/T2
    dv_dT = (V2-V)/dT
    alpha = dv_dT
    return alpha

def getKinematicVis(Ts):
    # from Mays 2005, Water Resources Engineering
    tempTable = np.arange(0,101,5)
    # table in m2/s E-6
    visTable = np.array([1.792,1.519,1.308,1.141,1.007,0.897,
        0.804,0.727,0.661,0.605,0.556,0.513,0.477,0.444,
        0.415,0.39,0.367,0.347,0.328,0.311,0.296])
    v = np.interp(Ts, visTable, tempTable)
    v = v*1e-6
    return(v)