import numpy as np
def format_Temp(depths, Temp):
    if Temp.ndim==2:
        if Temp.shape[0]==depths.shape[0]:
            Temp = Temp.T
    elif Temp.ndim==1:
        Temp = Temp.reshape(-1,1).T
    return Temp
    
def find_nearest_index(old_depths,SthermoD):
    depth_index = np.argmin(np.abs(SthermoD-old_depths.reshape(-1,1)), axis=0)
    return depth_index

def find_nearest(old_depths,SthermoD):
    depth_index = find_nearest_index(old_depths,SthermoD)
    nearest_depth = old_depths[depth_index]
    nearest_depth = set_nan(SthermoD,nearest_depth)
    return nearest_depth

def set_nan(vec1, vec2):
    #If vec1 has NaN, set to NaN the values of vec2
    NaN = np.isnan(vec1)
    if any(NaN):
        if len(NaN)==1:
            vec2=np.array([np.nan])
        else:
            vec2[NaN] = np.nan
    return vec2

def round_up_to_odd(f):
    return int(np.ceil(f) // 2 * 2 + 1)

# returns the index of the first occurrence of the element
def find(array_1,element):
    for i in range(len(array_1)):
        if array_1[i]==element:
            return(i)
    return(False)

#returns the index of the first occurrence of nan values
def find_nan(array_1):
    for i in range(len(array_1)):
        if np.isnan(array_1[i]):
            return(i)
    return(False)

#returns the index of the first occurrence of not nan values
def find_not_nan(array_1):
    for i in range(len(array_1)):
        if np.isnan(array_1[i])==False:
            return(i)
    return(False)

#returns the index of the first occurrence element which higher than the input number
def find_sup(array_1,element):
    for i in range(len(array_1)):
        if array_1[i]>element:
            return(i)
    return(False)

def find_inf(array_1,element):
    for i in range(len(array_1)):
        if array_1[i]<element:
            return(i)
    return(False)
