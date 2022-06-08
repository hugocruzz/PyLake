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

Future updates:
* Data check and comparison with other sources
* Xarray based algorithms for spatial data compatibility
* Thermocline uses a smoothing algorithm (savgol filter) to correct the variability in vertical resolution. This method is temporary and need to be replaced.
* Mixed layer interpolation need to be optimized, set as parameter for now

## Installation

`pip install puchu`

## Usage
```python
import puchu as chu
import numpy as np

temp = np.array([14.3,14,12.1,10,9.7,9.5])
depth = np.array([1,2,3,4,5,6])
meta_depth = chu.meta_depths(temp, depth, thermocline_output=True)
```
