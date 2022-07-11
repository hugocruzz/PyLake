# PyLake

This work present methods used to compute meaningful physical properties in aquatic sciences.

The methods are based on Xarray. 
Multi-dimensional large time-series array are compatible if an xarray is passed as input.

Algorithms and documentation are sometimes inspired by LakeAnalyzer in R (https://github.com/GLEON/rLakeAnalyzer)

Implemented methods:
* Thermocline
* Mixed layer
* Metalimnion extent (epilimnion and hypolimnion depth)
* Wedderburn Number
* Schmidt stability
* Heat content
* Seiche periode
* Lake Number
* Brunt-Vaisala frequency
* Average layer temperature
* Monin-Obhukov 

## Installation

`pip install pylake`

## Usage

Pylake use Dask which require a python version >=3.8

Have a look in the notebooks, an example is provided

```python
import pylake
import numpy as np

Temp = np.array([14.3,14,12.1,10,9.7,9.5,6,5])
depth = np.array([1,2,3,4,5,6,7,8])
epilimnion, hypolimnion = pylake.metalimnion(temp, depth)
```

 ## Work in progress
 Warning messages
 Lake metabolizer is being implemented. 