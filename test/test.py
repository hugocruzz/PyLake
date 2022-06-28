import pylake
import numpy as np

temp = np.array([14.3,14,12.1,10,9.7,9.5])
depth = np.array([1,2,3,4,5,6])
hypolimnion, epilimnion = pylake.metalimnion(temp, depth)