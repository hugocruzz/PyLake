import puchu as chu
import numpy as np

temp = np.array([14.3,14,12.1,10,9.7,9.5])
depth = np.array([1,2,3,4,5,6])
meta_depth = chu.meta_depths(temp, depth, thermocline_output=True)