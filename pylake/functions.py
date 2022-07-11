import numpy as np
import xarray as xr
import warnings 

def control(Temp, depths):
    #Too chronophage and not so usefull
    #if np.isnan(Temp).all:
    #    warnings.warn("Could not deduce thermocline with only nan vector")
    #    return np.nan
    if Temp.shape[1]<3:
        warnings.warn("Could not deduce thermocline with less than 3 measurements")
        return np.nan
    elif len(depths)!=len(np.unique(depths)):
        warnings.warn("depths must be unique")
        return np.nan
    else:
        return 1

def to_xarray(Temp, depths, time=None):
    if (type(Temp)==np.ndarray)or(type(Temp)==list):
        Temp = format_Temp(depths, Temp)
        if time is not None:
            coords = {'time':time,'depth':depths}
        else:
            coords = {'time':list(range(0,Temp.shape[0])),'depth':depths}

        Temp = xr.DataArray(Temp, coords)
    else:
        depths = Temp["depth"].to_numpy()
        Temp.load()
    return Temp,depths
    
def smooth_1D(Temp, smooth):
    from scipy.signal import savgol_filter
    if type(smooth)==dict:
        window_size = smooth.get("window_size",len(Temp)/10)
        mode = smooth.get("method",'nearest')
        order = smooth.get("order",3)
    else:
        window_size= round_up_to_odd(len(Temp)/10)
        mode = 'nearest'
        order = 3
    new_Temp = savgol_filter(Temp, window_size, order, mode=mode)
    return new_Temp

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


def weighted_method(depths, rho, z_idx):
    '''
    Estimate where the thermocline lies even between two temperature 
    measurement depths, giving a potentially finer-scale estimate 
    than usual techniques.

    Parameters
    -----------
    depths: array_like
        depth array
    drho_dz: array_like 
        density array 
    z_idx: array_like
        thermocline index corresponding to the depths 

    Returns
    -----------
    weighted_thermoD: array_like
        the adjusted weighted thermocline depth.
    '''
    if type(rho)==np.ndarray:
        depths, rho, z_idx = list(map(np.asanyarray, (depths, rho, z_idx)))
        rho = format_Temp(depths, rho)
        coords = {'time':list(range(0,rho.shape[0])),'depth':depths}
        rho = xr.DataArray(rho, coords)
    else:
        depths = rho.depth
        rho.load()

    drho_dz = rho.diff('depth')/rho.depth.diff('depth')

    #Mask boundarie values
    mask_up = z_idx==0
    mask_down = (z_idx>=len(depths)-2)

    #Change values for boundaries to avoid any bug on the indexation
    z_masked = z_idx.copy()
    z_masked = z_masked.where(~mask_up, z_masked+1)
    z_masked = z_masked.where(~mask_down, z_masked-1)

    #Weighted method 
    hplus = (depths.isel(depth=z_masked) - depths.isel(depth=z_masked+2))/2
    hminu = (depths.isel(depth=z_masked-1) - depths.isel(depth=z_masked+1))/2

    drho = drho_dz.isel(depth=z_masked)
    drho_plus = drho_dz.isel(depth=z_masked+1)
    drho_minu = drho_dz.isel(depth=z_masked-1)

    Dplus = hplus/(drho-drho_plus)
    Dminu = hminu/(drho-drho_minu)

    weighted_thermoD = depths[z_masked+1]*(Dplus/(Dminu+Dplus)) + depths[z_masked]*(Dminu/(Dminu+Dplus))
    
    mask_inf = (np.isinf(Dplus/(Dminu+Dplus))) & (np.isinf(Dminu/(Dminu+Dplus)))

    #Restablish correct values for boundaries
    weighted_thermoD = weighted_thermoD.where(~mask_up, (depths[0]+depths[1])/2 )
    weighted_thermoD =  weighted_thermoD.where(~mask_down, (depths[-1]+depths[-2])/2)
    weighted_thermoD = weighted_thermoD.where(~mask_inf, np.nan)

    return weighted_thermoD

def check_bathy(Temp, bthA, bthD, depth):
    #Check this with xarray
    numD = Temp.shape[1]-1
    if max(bthD) > depth[numD]:
        Temp  = np.append(Temp,Temp[:,numD])
        depth = np.append(depth,max(bthD))
    elif max(bthD)<depth[numD]:
        bthD = np.append(bthD,depth[numD])
        bthA = np.append(bthA, 0)
    if min(bthD)<depth[0]:
        Temp = np.hstack((Temp[:,0].reshape(-1,1),Temp))
        depth = np.append(np.min(bthD), depth)
    return Temp,bthA,bthD,depth

def T68conv(T90):
    return T90 * 1.00024

def dens0(t,s=0.2):
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
