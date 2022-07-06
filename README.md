# PyLake

This work present methods used to compute meaningful physical properties in aquatic sciences.

Multi-dimensional array (time and depth) are compatible.

Algorithms and documentation are inspired by LakeAnalyzer in R (https://github.com/GLEON/rLakeAnalyzer)

Implemented methods:
* Thermocline
* Mixed layer
* Metalimnion extent (top metalimnion and bottom metalimnion)
* Wedderburn Number
* Schmidt stability
* internal energy
* Seiche periode
* Lake Number
* Brunt-Vaisala frequency
* Average layer temperature
* Monin-Obhukov 


## Installation

`pip install pylake`

## Usage
```python
import pylake
import numpy as np

temp = np.array([14.3,14,12.1,10,9.7,9.5])
depth = np.array([1,2,3,4,5,6])
hypolimnion, epilimnion = pylake.metalimnion(temp, depth)
```