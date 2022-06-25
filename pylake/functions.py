import numpy as np

def smooth_temp(Temp, depths, smooth):
    from scipy.signal import savgol_filter
    if type(smooth)==dict:
        window_size = smooth.get("window_size",round_up_to_odd(len(depths)/10))
        mode = smooth.get("method",'nearest')
        order = smooth.get("order",3)
    else:
        window_size= round_up_to_odd(len(depths)/10)
        mode = 'nearest'
        order = 3
    new_Temp = savgol_filter(Temp, window_size, order, mode=mode)
    return new_Temp

def refine_scale(depths, drho_dz, thermoInd, thermoD):
    '''
    Estimate where the thermocline lies even between two temperature 
    measurement depths, giving a potentially finer-scale estimate 
    than usual techniques.

    Parameters
    -----------
    depths: array_like
        depth array
    drho_dz: array_like 
        density gradient 
    thermoInd: array_like
        thermocline index corresponding to the depths 
    thermoD: array_like
        thermocline depth in which the thermocline is changed.
    
    Returns
    -----------
    thermoD: array_like
        the adjusted thermocline depth.
    '''
    depths, drho_dz, thermoInd, thermoD = list(map(np.asanyarray, (depths, drho_dz, thermoInd, thermoD)))
    mask_updown=(thermoD>1)&(thermoInd<len(depths)-1)
    #if thermoInd is maximum, remove it (the mask will not calculate this point anyway)
    remove_bot = np.where((thermoInd==depths.size-1))[0]
    thermoInd[remove_bot] = thermoInd[remove_bot]-1

    D = depths[thermoInd]
    dnD = depths[thermoInd+1]
    upD = depths[thermoInd-1]

    thermoD_drho_dz = drho_dz[np.arange(drho_dz.shape[0]), thermoInd]
    down_thermoD_drho_dz = drho_dz[np.arange(drho_dz.shape[0]), thermoInd+1]
    up_thermoD_drho_dz = drho_dz[np.arange(drho_dz.shape[0]), thermoInd-1]

    Sdn = -(dnD-D)/(down_thermoD_drho_dz-thermoD_drho_dz)
    Sup = (D-upD)/(thermoD_drho_dz-up_thermoD_drho_dz)

    mask_inf = (~np.isinf(Sup)) & (~np.isinf(Sdn))
    mask = mask_inf & mask_updown

    new_thermoD = dnD*(Sdn/(Sdn+Sup))+D*(Sup/(Sdn+Sup))
    thermoD[mask] = new_thermoD[mask]
    return thermoD

def weighted_method(depths, rho, z_idx):
    '''
    Estimate where the thermocline lies even between two temperature 
    measurement depths, giving a potentially finer-scale estimate 
    than usual techniques.

    Parameters
    -----------
    depths: array_like
        depth array
    rho: array_like 
        density array 
    z_idx: array_like
        thermocline index corresponding to the depths 

    Returns
    -----------
    weighted_thermoD: array_like
        the adjusted weighted thermocline depth.
    '''
    depths, rho, z_idx = list(map(np.asanyarray, (depths, rho, z_idx)))

    drho_dz = np.diff(rho)/np.diff(depths)

    #Mask boundarie values
    mask_up = z_idx==0
    mask_down = z_idx==(len(depths)-1)

    #Change values for boundaries to avoid any bug on the indexation
    z_masked = z_idx.copy()
    z_masked[mask_up] = z_masked[mask_up]+1
    z_masked[mask_down] = z_masked[mask_down]-1

    #Weighted method 
    hplus = (depths[z_masked]   - depths[z_masked+2])/2
    hminu = (depths[z_masked-1] - depths[z_masked+1])/2

    time = np.arange(drho_dz.shape[0])
    drho = drho_dz[time, z_masked]
    drho_plus = drho_dz[time, z_masked+1]
    drho_minu = drho_dz[time, z_masked-1]

    Dplus = hplus/(drho-drho_plus)
    Dminu = hminu/(drho-drho_minu)

    weighted_thermoD = depths[z_masked+1]*(Dplus/(Dminu+Dplus)) + depths[z_masked]*(Dminu/(Dminu+Dplus))
    
    mask_inf = (np.isinf(Dplus/(Dminu+Dplus))) & (np.isinf(Dminu/(Dminu+Dplus)))

    #Restablish correct values for boundaries
    weighted_thermoD[mask_up] = (depths[0]+depths[1])/2
    weighted_thermoD[mask_down] = (depths[-1]+depths[-2])/2
    weighted_thermoD[mask_inf] = np.nan

    return weighted_thermoD

def T68conv(T90):
    return T90 * 1.00024

def dens0(s,t):
    b = (8.24493e-1, -4.0899e-3, 7.6438e-5, -8.2467e-7, 5.3875e-9)
    c = (-5.72466e-3, 1.0227e-4, -1.6546e-6)
    d = 4.8314e-4
    T68 = T68conv(t)
    out =  (smow(t) + (b[0] + (b[1] + (b[2] + (b[3] + b[4] * T68) * T68) *
            T68) * T68) * s + (c[0] + (c[1] + c[2] * T68) * T68) * s *
            s ** 0.5 + d * s ** 2)
    return out

def smow(t):
    a = (999.842594, 6.793952e-2, -9.095290e-3, 1.001685e-4, -1.120083e-6, 6.536332e-9)
    T68 = T68conv(t)
    out= (a[0] + (a[1] + (a[2] + (a[3] + (a[4] + a[5] * T68) * T68) * T68) * T68) * T68)
    return out    

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
