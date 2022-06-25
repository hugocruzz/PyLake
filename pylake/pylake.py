import numpy as np
from .functions import *
import seawater as sw
from scipy.interpolate import interp1d
import warnings
from scipy.signal import find_peaks,savgol_filter
import xarray as xr


    
def thermocline(Temp, depths, Smin=0.1, seasonal=True, mixed_cutoff=1, s=0.2, smooth=False, index=False, gridded=False, seasonal_smoothed=False):
    '''
    Calculate depth of the thermocline from a temperature profile.

    Method
    ----------
    Thermocline
    The diurnal thermocline is calculated using the maximum gradient of density.
    If the temperature profile have a variability in depth resolution, it is recommended 
    to smooth the profile to avoid resolution influence on the algorithm.
    Once the maximum gradient of density is found, a special technique to refine the 
    thermocline depth is used and presented in (Read et al., 2011).

    Seasonal thermocline
    The seasonal thermocline uses the find_peaks from scipy to find the first 
    local maximum (higher than a certain threshold, based on Smin) starting 
    from the bottom of the profile.
    By default, if no peak is found, the seasonal thermocline is set equal to 
    the diurnal thermocline. 
    The same refining technique is used.
    A savgol filter (special moving averaged filter) is used to smooth the data 
    and discard extremums.
    The seasonal is set to be more or equal to the diurnal thermocline.
    Keep in mind that if the seasonal thermocline is often assimilated 
    with the diurnal, with the exception that the seasonal is artificially smoothed.

    Parameters
    ----------
    Temp :  array_like
        a numeric vector of water temperature in degrees C
    depths : array_like
        a numeric vector corresponding to the depths (in m) of the Temp
    Smin : float, default: 0.1 °C/m
        Optional parameter defining minimum density gradient for
        thermocline. 
    seasonal : bool, default: True
        A logical value indicating whether the seasonal thermocline
        should be returned. This is fed to thermo.depth, which is used as the
        starting point.  The seasonal thermocline is defined as the deepest density
        gradient found in the profile. If FALSE, the depth of the maximum
        density gradient is used as the starting point. Defaults to True
    index : bool, default: False
        value indicated if index of the thermocline depth,
        instead of the depth value, should be returned. Defaults to False
        - cutoff (float) A cutoff (deg C) where below this threshold,
        thermo.depth and meta.depths are not calculated (NaN is returned). Defaults
        to 1 deg C.
    mixed_cutoff : scalar, default: 1
        A cutoff (deg C) where below this threshold,
        thermo.depth and meta.depths are not calculated (NaN is returned). Defaults
        to 1 deg C.
    smooth : bool, default: False
        Smooth the curve following the scipy savgol filter (window size: 1/10 of the 
        depths length, order:3, method=nearest)
        Smoothing is recommended when the thermocline is located at a lower resolution 
        sensors (sensors are more spaced at the thermocline)
    s : array_like, default : 0.2
        Salinity of the water column in PSU
    gridded : bool, default: False
        The calculated thermocline depth can be located between two depths.
        If gridded=True, a simple algorithm would compare the thermocline depth with the 
        depth vector given as input, and return the closest gridded depth.

    Returns 
    ----------
    thermoD: array_like
        thermocline depth (m)
    thermoInd: array_like
        thermocline index corresponding to the thermocline depth
    
    Examples
    ----------
    >>>     import pylake
    ...     temp = np.array([14.3,14,12.1,10,9.7,9.5,5,4.5,4.4,4.3])
    ...     depth = np.array([1,2,3,4,5,6,7,8,9,10])
    ...     thermo, thermoInd = pylake.thermocline(temp,depth)
    ...     print(f"thermocline depth: {thermo}\n")
    ...     print(f"thermocline depth index: {thermoInd}\n")
    ...     thermocline depth: 2.878790569183019
    ...     thermocline depth index: 2
    '''
    Temp = format_Temp(depths, Temp)

    if np.isnan(Temp).all():
        warnings.warn("Could not deduce thermocline with only nan vector")
        return depths*np.nan
    if Temp.shape[1]<3:
        warnings.warn("Could not deduce thermocline with less than 3 measurements")
        return depths*np.nan
    if len(depths)!=len(np.unique(depths)):
        warnings.warn("depths must be unique")
        return np.nan

    is_not_significant = np.nanmax(Temp,axis=1)-np.nanmin(Temp,axis=1)<mixed_cutoff
    
    cutoff_time_index =  np.where(is_not_significant)[0]
    if is_not_significant.any():
        cutoff_nb = len(cutoff_time_index)
        warnings.warn(f"Temperature difference within the profile is too low to detect any thermocline for {cutoff_nb} profiles")

    if smooth:
        Temp = smooth_temp(Temp, depths, smooth)
    
    rhoVar = sw.dens0(s=s,t=Temp)
    drho_dz=np.diff(rhoVar)/np.diff(depths)
    
    thermoInd = np.nanargmax(drho_dz, axis=1)

    thermoD = weighted_method(depths, rhoVar, thermoInd)
    thermoInd = find_nearest_index(depths, thermoD)

    thermoD[cutoff_time_index] = np.nan
    if len(thermoD)==1:
        thermoD = np.asscalar(thermoD)
        thermoInd = np.asscalar(thermoInd)
    return thermoD, thermoInd

def seasonal_thermocline(Temp, depths, s=0.2, Smin=0.1, seasonal_smoothed=True, smooth=False, mixed_cutoff=1):
    '''
    Calculate depth of the thermocline from a temperature profile.
    
    Method
    ---------
    The seasonal thermocline uses the find_peaks from scipy to find the first 
    local maximum (higher than a certain threshold, based on Smin) starting 
    from the bottom of the profile.
    By default, if no peak is found, the seasonal thermocline is set equal to 
    the diurnal thermocline. 
    The same refining technique is used.
    A savgol filter (special moving averaged filter) is used to smooth the data 
    and discard extremums.
    The seasonal is set to be more or equal to the diurnal thermocline.
    Keep in mind that if the seasonal thermocline is often assimilated 
    with the diurnal, with the exception that the seasonal is artificially smoothed.

    Parameters
    ----------
    Temp :  array_like
        a numeric vector of water temperature in degrees C
    depths : array_like
        a numeric vector corresponding to the depths (in m) of the Temp
    s : array_like, default : 0.2
        Salinity of the water column in PSU
    Smin : float, default: 0.1 °C/m
        Optional parameter defining minimum density gradient for
        thermocline.
    smooth : bool, default: False
        Smooth the curve following the scipy savgol filter (window size: 1/10 of the 
        depths length, order:3, method=nearest)
        Smoothing is recommended when the thermocline is located at a lower resolution 
        sensors (sensors are more spaced at the thermocline)
    seasonal_smoothed: bool, default: True
        Smooth the seasonal thermocline on the entire time serie. The smooth depends on the time serie length.
    mixed_cutoff : scalar, default: 1 °C
        A cutoff (deg C) where below this threshold,
        thermocline are not calculated (NaN is returned).
    Returns 
    ----------
    thermoD: array_like
        thermocline depth (m)
    thermoInd: array_like
        thermocline index corresponding to the thermocline depth
    
    Examples
    ----------
    >>>     import pylake
    ...     temp = np.array([14.3,14,12.1,10,9.7,9.5,5,4.5,4.4,4.3])
    ...     depth = np.array([1,2,3,4,5,6,7,8,9,10])
    ...     Sthermo, SthermoInd = pylake.seasonal_thermocline(temp,depth)
    ...     print(f"Seasonal thermocline depth: {Sthermo}\n")
    ...     print(f"Seasonal thermocline depth index: {SthermoInd}\n")
    ...     Seasonal thermocline depth: 6.4880232728589355 
    ...     Seasonal thermocline depth index: 5
    '''
    Temp = format_Temp(depths, Temp)
    
    rhoVar = sw.dens0(s=s,t=Temp)
    drho_dz=np.diff(rhoVar)/np.diff(depths)
    mDrhoZ = np.nanmax(drho_dz,axis=1)
    dRhoCut = Smin*np.ones(Temp.shape[0])

    thermoD, thermoInd = thermocline(Temp, depths, smooth=smooth, mixed_cutoff=mixed_cutoff)
    #format if it's a scalar
    thermoD, thermoInd = list(map(np.asanyarray, (thermoD, thermoInd)))
    thermoD = thermoD.reshape(-1)
    thermoInd = thermoInd.reshape(-1)

    SthermoD = thermoD.copy()
    SthermoInd = thermoInd.copy()

    #Define seasonal thermocline based on the first peak starting from the bottom of the water column
    for i in range(0,drho_dz.shape[0]):
        locs, peaks = find_peaks(drho_dz[i,:],height=dRhoCut[i])
        pks = peaks["peak_heights"]
        if len(pks)!=0:
            mDrhoZ[i] = pks[-1]
            SthermoInd[i] = locs[-1]

    SthermoD = weighted_method(depths, rhoVar, SthermoInd)

    #Compare the seasonal and the diurnal thermocline, seasonal should be at higher depth than the diurnal, if not, both are set equal.
    idxthermoD = np.where(SthermoD<thermoD)[0]
    SthermoD[idxthermoD] = thermoD[idxthermoD]
    SthermoInd[idxthermoD] = thermoInd[idxthermoD]
    
    SthermoInd = find_nearest_index(depths,SthermoD)

    if len(thermoD)!=1:
        if seasonal_smoothed:
            SthermoD = savgol_filter(SthermoD, round_up_to_odd(len(SthermoD)/30), 3, mode='nearest')
    else:
        SthermoD = np.asscalar(SthermoD)
        SthermoInd = np.asscalar(SthermoInd)
    return SthermoD, SthermoInd


def meta_depths(Temp, depths, slope=0.1, seasonal=False, mixed_cutoff=1, smooth=False, s=0.2):
    '''
    Calculates the top and bottom depths of the metalimnion in a stratified
    lake. The metalimnion is defined as the water stratum in a stratified lake
    with the steepest thermal gradient and is demarcated by the bottom of the
    epilimnion and top of the hypolimnion.
    
    Method
    ----------
    
    Parameters
    ----------
    Temp :  array_like
        a numeric vector of water temperature in degrees C
    depths : array_like
        a numeric vector corresponding to the depths (in m) of the Temp
        measurements
    slope : scalar, str, default: 0.1
        a numeric vector corresponding to the minimum slope. Can be set to "relative", if it's the case,
        the threshold will be 10% of the max slope density gradient.
    seasonal : bool, default: False
        Calculates the metalimnion based on the seasonal thermocline if set to True.
    mixed_cutoff : scalar, default: 1
        A cutoff (deg C) where below this threshold, thermo.depth and meta.depths are 
        not calculated (NaN is returned). Defaults to 1 deg C.
    smooth : bool, default: False
        Smooth the curve following the scipy savgol filter (window size: 1/5 of the 
        depths length, order:3, method=nearest)
        Smoothing is recommended when the thermocline is located at a lower resolution 
        sensors (sensors are more spaced at the thermocline)
    s : array_like, default : 0.2
        Salinity of the water column in PSU
    thermocline_output : bool, default : False
        Return the calculated thermocline depth if set to True 

    Returns 
    ----------
    metatop : array_like
        A numeric vector of the top and bottom metalimnion depths in meters.
        Returns the bottom depth if no distinct metalimion top and bottom found.
    
    See also
    ----------
    lakewater.thermocline

    Examples
    ----------
    >>>     import pylake
    ...     temp = np.array([14.3,14,12.1,10,9.7,9.5])
    ...     depth = np.array([1,2,3,4,5,6])
    ...     pylake.meta_depths(temp, depth, thermocline_output=True)
    {'thermocline': array([3], dtype=int64), 'top_metalimnion': array([1.5]), 'bot_metalimnion': array([4.5])}

    >>> lw.meta_depths(temp,depth, thermocline_output=True, smooth={"window_size":5, "mode":"nearest", "order":3})
    {'thermocline': array([3], dtype=int64), 'top_metalimnion': array([1.5]), 'bot_metalimnion': array([4.5])}

    >>> lw.meta_depths(temp,depth, mixed_cutoff=10, thermocline_output=True)
    {'thermocline': array([nan]), 'top_metalimnion': array([nan]), 'bot_metalimnion': array([nan])}


    References
    ----------
    Wetzel, R. G. 2001. Limnology: Lake and River Ecosystems, 3rd ed. Academic Press.'''

    #Ensure that we are interpreting the lines as time and columns as depths
    Temp = format_Temp(depths, Temp)

    if np.isnan(Temp).all():
        warnings.warn("Could not deduce thermocline with only nan vector")
        return depths*np.nan
    if Temp.shape[1]<3:
        warnings.warn("Could not deduce thermocline with less than 3 measurements")
        return depths*np.nan
    if len(depths)!=len(np.unique(depths)):
        warnings.warn("depths must be unique")
        return np.nan

    if smooth:
        Temp = smooth_temp(Temp, depths, smooth)

    if seasonal:
        thermoD, thermoInd = seasonal_thermocline(Temp, depths, mixed_cutoff=mixed_cutoff, smooth=False)
    else:
        thermoD, thermoInd = thermocline(Temp, depths, mixed_cutoff=mixed_cutoff, smooth=False)

    thermoD, thermoInd = list(map(np.asanyarray, (thermoD, thermoInd)))
    thermoD = thermoD.reshape(-1)
    thermoInd = thermoInd.reshape(-1)

    if type(Temp)==np.ndarray:
        Temp = format_Temp(depths, Temp)
        coords = {'time':list(range(0,Temp.shape[0])),'depth':depths}
        Temp = xr.DataArray(Temp, coords)
        Temp["thermoInd"] = ('time', thermoInd)
        Temp["thermoD"] = ('time', thermoD)
    else:
        Temp["thermoInd"] = ('time', thermoInd)
        Temp["thermoD"] = ('time', thermoD)

    rhoVar = dens0(s=s,t=Temp)
    drho_dz = rhoVar.diff('depth')/rhoVar.depth.diff('depth')
    drho_dz["depth"] = [(a+b)/2 for a,b in zip(depths, depths[1:])]
    drho_dz["thermoInd"] = ('time', find_nearest_index(drho_dz["depth"].to_numpy(), drho_dz["thermoD"].to_numpy()))
    mark =  drho_dz["depth"]-drho_dz["thermoD"]

    if slope=="relative":
        slope = drho_dz.max('depth')/10
    
    cond = drho_dz<slope

    mark_value = mark.where(~cond)
    e_mark = mark_value.where(mark_value<0)
    h_mark = mark_value.where(mark_value>0)
    
    epilimnion_idx_no_nan = e_mark.fillna(-999).argmax(dim='depth')-1
    epilimnion_idx = epilimnion_idx_no_nan.where(epilimnion_idx_no_nan!=-1,drho_dz["thermoInd"]) #If not found, make it equal to the thermoInd
    hypolimnion_idx_no_nan = h_mark.fillna(999).argmin(dim='depth')+1
    hypolimnion_idx = hypolimnion_idx_no_nan.where(hypolimnion_idx_no_nan!=1,drho_dz["thermoInd"])

    hypo_depth = drho_dz["depth"].isel(depth=hypolimnion_idx)
    hypo_plus = drho_dz["depth"].isel(depth=hypolimnion_idx+1)
    epi_depth = drho_dz["depth"].isel(depth=epilimnion_idx)
    epi_minus = drho_dz["depth"].isel(depth=epilimnion_idx-1)

    averaged_epi = (epi_minus+epi_depth)/2
    averaged_hypo = (hypo_plus+hypo_depth)/2

    hypo_depth_filt = averaged_hypo.where(averaged_hypo>averaged_hypo["thermoD"],averaged_hypo["thermoD"])
    epi_depth_filt = averaged_epi.where(averaged_epi<averaged_epi["thermoD"],averaged_epi["thermoD"])

    return hypo_depth_filt.to_numpy(), epi_depth_filt.to_numpy()
    
def wedderburn(delta_rho, metaT, uSt, AvHyp_rho, Lo=False, Ao=False, g=9.81):
    ''' 
    Wedderburn Number (Wn) is a dimensionless parameter measuring the balance
    between wind stress and bouyancy force and is used to estimate the amount of
    upwelling occuring in a lake.  When Wn is much greater than 1, the buoyancy
    force is much greater than the wind stress and therefore there is a strong
    vertical stratification with little horizontal variation in the
    stratification. When Wn is much less than 1, the wind stress is much greater
    than the bouyancy force and upwelling is likely occuring at the upwind end
    of the lake. When Wn is near 1, the bouyance force and wind stress are
    nearly equal and horizontal mixing is considered important

    Parameters
    ----------
    delta_rho : array_like 
        density difference between the epilimnion and the hypolimnion (kg/m3).
    metaT : array_like:
        thickness of the surface layer (m)
    uSt : array_like
        water friction velocity due to wind stress (m/s)
    AvHyp_rho : array_like, scalar
        average water density of the hypolimnion layer (kg/m3)
    Lo : bool, scalar, default : False
        fetch length in the direction of the wind (m). If Lo=False, calculate it based on Ao
    Ao : bool, scalar, default : False
        Lake surface (m2). Used to calculate Lo (if not given), assume the lake as a perfect circle.
    g : scalar
        gravity acceleration (m/s2)

    Returns
    ----------
    Wedderburn Number
    
    Examples 
    ----------
    >>>    delta_rho = np.array([3.1,1.5])
    ...    metaT = np.array([5.5,2.4])
    ...    uSt = np.array([0.0028,0.0032])
    ...    Ao = np.array([80300,120000])
    ...    AvHyp_rho = np.array([999.31,999.1])
    ...    pylake.wedderburn(delta_rho, metaT, uSt, AvHyp_rho, Ao=Ao)
    array([367.22052925  21.19474286])

    Equation
    ----------   
    W = (g*delta_rho*(h**2))/(pHyp*(uSt**2)*Lo)
    where
    g = gravity acceleration
    delta_rho (kg/m3) = density difference beTempeen the epilimnion and the hypolimnion 
    metaT (m)= thickness of the surface layer
    uSt (m/s)= water friction velocity due to wind stress 
    Lo = fetch length in the direction of the wind.

    References
    ----------
    Read, J.S. et al., 2011. Derivation of lake mixing and stratification indices from high-resolution lake
    buoy data. Environ. Model. Software 26, 1325–1336. https://doi.org/10.1016/j.
    Imberger, J., Patterson, J.C., 1990. Physical limnology. Advances in Applied Mechanics 27, 353-370.
    '''
    #Must exist a better way to do this
    try:
        Lo_cond = len(Lo)
    except:
        Lo_cond = Lo
    try:
        Ao_cond = len(Ao)
    except:
        Ao_cond = Ao

    if not Lo_cond:
        if Ao_cond:
            Lo = 2 * np.sqrt(Ao/np.pi);      #Length at thermocline depth
        else:
            warnings.warnings("Please indicate either Lo or Ao")

    go = g*delta_rho/AvHyp_rho
    W = go*metaT**2/(uSt**2*Lo)
    return W

def schmidt_stability(Temp, depths, bthA, bthD, sal = 0.2, g=9.81, dz=0.1, NaN_interp=False):
    '''
    Schmidt stability, or the resistance to mechanical mixing due to the potential energy inherent in the stratification of the water column.

    Parameters
    -----------
    Temp: array_like
        water temperature in degrees C
    depths:  array_like
        depth of the Temp measurements (m)
    bthA: array_like: 
        cross sectional areas (m**2) corresponding to bthD depths
    bthD: array_like
        depths (m) which correspond to areal measures in bthA
    sal: scalar,array_like, default: 0.2
        Salinity in Practical Salinity Scale units
    g: scalar, defaults: 9.81
        gravity acceleration (m/s2)
    dz: scalar, default: 0.1
        depth resolution for the integral calculus
    NaN_interp: bool, defaults : False
        If NaN_interp=True, it will perform a linear interpolation to replace NaN values
        The Schmidt stability calculation perform poorly if NaN are present.
    
    Returns
    ----------
        - Schmidt stability (J/m**2)

    Example
    ----------
    >>>    bthA	=	np.array([1000,900,864,820,200,10])
    ...    bthD	=	np.array([0,2.3,2.5,4.2,5.8,7]) 
    ...    wtr	=	np.array([28,27,26.4,26,25.4,24,23.3])
    ...    depths	=np.array([0,1,2,3,4,5,6])  
    ...    pylake.schmidt_stability(wtr, depths, bthA, bthD, sal=.2, g=self.g)
    array([21.20261732])

    equation
    ----------
    g/A0 int(0,zmax, (zv-z)(rho_i-rho_v)A(z)dz)

    '''
    Temp = format_Temp(depths, Temp)
    if NaN_interp:
        if sum(sum(np.isnan(Temp))):
            nan_index_time = np.where(np.isnan(Temp))[0]
            for time in nan_index_time:
                mask = np.isnan(Temp[time,:])
                Temp[time,:][mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), Temp[time,:][~mask])

    z0 = np.min(depths)
    I0 = np.argmin(depths)
    A0 = bthA[I0]
    rhoL = sw.dens0(t=Temp,s=sal)
    
    layerD = np.arange(z0, np.max(depths),dz)
    f = interp1d(depths,rhoL, axis=1)
    layerP = f(layerD)
    layerA = np.interp(layerD, bthD, bthA)

    Zcv = np.matmul(layerD,layerA)/np.sum(layerA)
    St = np.matmul(layerP, ((layerD - Zcv) * layerA) * dz * g / A0)
    return St 

def internal_energy(Temp, depths, bthA, bthD, s=0.2):
    '''
    Calculates the internal energy of the water column with temperature and hypsography
    Internal energy is the thermal energy in the water column, which is
    calculated by multiplying the specific heat of water (J kg-1 K-1) by the
    temperature and mass of the water in the lake.

    Parameters
    -----------
    Temp: array_like: 
        water temperature in degrees C
    depths: array_like:
        depths (in m) of the Temp measurements
    bthA: array_like:
        cross sectional areas (m^2) corresponding to bthD depths
    bthD array_like:
        depths (m) which correspond to areal measures in bthA

    Returns
    ---------
        U: array_like
            internal energy in Joules m-2

    Example
    ---------
    >>>    bthA =np.array([1000,900,864,820,200,10])
    ...    bthD=np.array([0,2.3,2.5,4.2,5.8,7])
    ...    wtr	=np.array([28,27,26.4,26,25.4,24,23.3])
    ...    depths	= np.array([0,1,2,3,4,5,6])
    ...    lw.internal_energy(wtr, depths, bthA, bthD, s=0.2)
    ...    520423172.7994813
    '''
    dz = 0.1
    cw = 4186
    Temp = format_Temp(depths, Temp)
    numD = Temp.shape[1]-1
    if max(bthD) > depths[numD]:
        Temp  = np.append(Temp,Temp[:,numD])
        depths = np.append(depths,max(bthD))
    elif max(bthD)<depths[numD]:
        bthD = np.append(bthD,depths[numD])
        bthA = np.append(bthA, 0)
    if min(bthD)<depths[0]:
        Temp = np.hstack((Temp[:,0].reshape(-1,1),Temp))
        depths = np.append(np.min(bthD), depths)
    Zo = min(depths)
    Io = np.argmin(depths)
    Ao = bthA[Io]

    if Ao==0:
        print("surface area cannot be zero, check bathymetric file")
    rhoL = sw.dens0(s=s,t=Temp)
    layerD = np.arange(Zo, np.max(depths),dz)
    frhoL = interp1d(depths,rhoL)
    layerP = frhoL(layerD)
    fTemp = interp1d(depths,Temp)
    layerT = fTemp(layerD)
    layerA = np.interp(layerD, bthD, bthA)

    v_i = layerA*dz
    m_i = layerP*v_i
    u_i = layerT*m_i*cw
    if u_i.ndim==2:
        U = np.nansum(u_i,axis=1)/layerA[0]
    elif u_i.ndim==1:
        U = np.nansum(u_i)/layerA[0]
    else:
        print("Could not find the dimension")
    return U

def seiche_period_1(depth, Zt, Lt, delta_rho, AvHyp_rho, g= 9.81) :
    '''
    Estimation of the Seiche periode Monismith (1986).

    Parameters
    -----------
    depths: array_like:
        depths (in m) of the Temp measurements
    Zt: array_like, scalar
        Thermocline depth 
    At: scalar 
        Surface at the thermocline depth 
    delta_rho: scalar, array_like
        Density difference between the epilimion and the hypolimnion 
    AvHyp_rho: array_like, scalar
        Average density of the hypolimnion
    g: scalar, default: 9.81
        gravity acceleration (m/s2)

    Returns
    ---------
    T1: array_like, scalar
        Mode-1 vertical seiche period (s)

    example
    ---------
    >>>    bthA =np.array([1000,900,864,820,200,10])
    ...    bthD=np.array([0,2.3,2.5,4.2,5.8,7])
    ...    depths	= np.array([0,1,2,3,4,5,6])
    ...    Zt = 4.5
    ...    Lt = 4000
    ...    delta_rho = 0.5
    ...    AvHyp_rho = 997
    ...    lw.seiche_period_1(depths, Zt, Lt, delta_rho, AvHyp_rho, g= 9.81)
    ...    1445418 
    '''
    g_reduced = g*delta_rho/AvHyp_rho
    Zd = depth[-1]
    T1 = 2*Zd*Lt/(g_reduced*Zt*(Zd-Zt))
    return T1

def Lake_number(bthA, bthD, uStar, St, metaT, metaB, averageHypoDense, g=9.81):
    '''
    Description: The Lake Number, defined by Imberger and Patterson (1990), has been used to
    describe processes relevant to the internal mixing of lakes induced by wind
    forcings. Lower values of Lake Number represent a higher potential for
    increased diapycnal mixing, which increases the vertical flux of mass and
    energy across the metalimnion through the action of non-linear internal
    waves. Lake Number is a dimensionless index.

    Lake Number has been used, for example, to estimate the flux of oxygen
    across the thermocline in a small lake (Robertson and Imberger, 1994), and
    to explain the magnitude of the vertical flux of ammonium in a lake (Romero
    et al., 1998).
    In Imberger and Patterson (1990), Lake Number was defined as
    Ln = (g * St * (zm - zT)) / (rho_0 * u*^2 * A0^3/2 * (zm - zg))
    with all values referenced from the lake bottom, e.g.,
    zm being the height of the water level, zT the height of metalimnion
    and zg the height of center volume.
    Our calculations assume that the reference is at the lake surface, therefore:
    height of metalimnion becomes metalimnion depth (average of meta top and bot): 
    (zm - zT) --> (metaT + metaB)/2 
    height of center of volume depth becomes center of volume depth Zcv:
    (zm - zg) --> Zcv
    Further, we note that in that original work St was defined as
    St = int (z - zg) A(z) rho(z) dz
    and rLakeAnalyzer defines St as
    St = g/A0 int (z - zg) rho(z) dz
    Therefore, we calculate St_uC = St * Ao / g

    Parameters
    -----------
    bthA: array_like:
        a numeric vector of cross sectional areas (m2) corresponding to bthD depths, hypsographic areas
    bthD: array_like:
        a numeric vector of depths (m) which correspond to areal measures in bthA, hypsographic depths
    uStar: array_like:
        a numeric array of u* (m/s), water friction velocity due to wind stress
    St: array_like
        a numeric array of Schmidt stability (J/m2), as defined by Idso 1973
    metaT: array_like
        a numeric array of the top of the metalimnion depth (m from the surface)
    metaB: array_like
        a numeric array of the bottom of the metalimnion depth (m from the surface)
    averageHypoDense: array_like:
        a numeric array of the average density of the hypolimnion (kg/m3)
    g: scalar, default: 9.81
        gravity acceleration (m/s2)
    Returns
    -----------
        Ln: A numeric vector of Lake Number [dimensionless]

    Example
    -----------
    >>>    bthA =np.array([1000,900,864,820,200,10])
    ...    bthD=np.array([0,2.3,2.5,4.2,5.8,7])
    ...    uStar = np.array([0.0032,0.0024])
    ...    St = np.array([140,153])
    ...    metaT = np.array([1.34,1.54])
    ...    metaB = np.array([4.32,4.33])
    ...    averageHypoDense = np.array([999.3,999.32])
    ...    lw.Lake_number(bthA, bthD, uStar, St, metaT,metaB,averageHypoDense,self.g)
    ...    [472.30373072, 951.64555323]
    references
    -----------
    Imberger, J., Patterson, J.C., 1990. Advances in Applied Mechanics 27, 303-475.
    '''
    dz	=	0.1
    A0 = bthA[0]
    Z0 = bthD[0]

    layerD = np.arange(Z0, np.max(bthD),dz)
    layerA = np.interp(layerD, bthD, bthA)
    Zv = layerD*layerA*dz                    
    Zcv = sum(Zv)/sum(layerA)/dz
    St_uC = St*A0/g
    Ln = g*St_uC*(metaT+metaB)/(2*averageHypoDense*uStar**2*A0**(3/2)*Zcv)
    return Ln

def buoyancy_freq(Temp, depths, g=9.81):
    '''
    Description: Calculate the buoyancy frequency (Brunt-Vaisala frequency) for a temperature profile.

    Parameters
    ----------- 
    Temp: array_like
        A numeric vector of water temperature in degrees C
    depths: array_like
        a numeric vector corresponding to the depths (in m) of the Temp measurements
    g: scalar, default: 9.81
        gravity acceleration (m/s2)
    
    Returns
    ----------
    n2: array_like
        a vector of buoyancy frequency in units \code{sec^-2}.
    n2depth: array_like
        Return value has attribute "depths" which define buoyancy frequency depths (which differ from supplied depths).
    
    Example
    ----------
    >>>     wtr = np.array([22.51, 22.42, 22.4, 22.4, 22.4, 22.36, 22.3, 22.21, 22.11, 21.23, 16.42,15.15, 14.24, 13.35, 10.94, 10.43, 10.36, 9.94, 9.45, 9.1, 8.91, 8.58, 8.43])
    ...    depths = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
    ...    avg_depth, buoy_freq = lw.buoyancy_freq(wtr, depths, self.g)
    ...    plt.pcolormesh(buoy_freq)
    ... array([4.10503572e-04, 9.10051748e-05, 0.00000000e+00, 0.00000000e+00,
        9.08868567e-05, 1.36034054e-04, 2.03383759e-04, 2.25040545e-04,
        1.93749334e-03, 9.17368987e-03, 2.00052155e-03, 1.31951341e-03,
        1.19625956e-03, 2.75490678e-03, 4.89153665e-04, 6.45127824e-05,
        3.73658150e-04, 4.06586584e-04, 2.70840415e-04, 1.40129203e-04,
        2.31752785e-04, 1.00430572e-04])
    '''
    Temp = format_Temp(depths, Temp)
    if depths.ndim==1:
        depths_reshaped=depths.reshape(-1,1).T
    rho = sw.dens0(s=0.2, t=Temp)
    numDepths = len(depths)
    if Temp.ndim==2:
        rho_2=rho[:,:-1]
    else:
        rho_2=rho[:-1]

    n2 = g/rho_2*np.diff(rho, axis=1)/np.diff(depths_reshaped, axis=1)
    n2depth = [(a+b)/2 for a,b in zip(depths, depths[1:])]
    return n2depth, n2 

def Average_layer_temp(Temp, depths, depth_ref, top=False, bot=False):
    '''
    Perform the layer average temperature based on the thermocline depth 

    Parameters
    -----------
    Temp: array_like, xarray: 
        A dataset containing the temperature. Must be of the same dimensions than the thermocline depth.
    depths: array_like
        a numeric vector corresponding to the depths (in m) of the Temp measurements
    depth_ref: array_like:
        reference depth in which the averaged temperature above or under is calculated.

    Returns
    -----------
    mean_temp: array_like
        dataset with the mean layer temperature

    Examples
    -----------
    >>> wtr = np.array([22.51, 22.42, 22.4, 22.4, 22.4, 22.36, 22.3, 22.21, 22.11, 21.23, 16.42,15.15, 14.24, 13.35, 10.94, 10.43, 10.36, 9.94, 9.45, 9.1, 8.91, 8.58, 8.43])
    ... depths = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
    ... depth_ref = 9 
    ... lw.Average_layer_temp(wtr, depths, depth_ref, top=True)
    ... 21.70
    ... lw.Average_layer_temp(wtr, depths, depth_ref, top=True)
    ... 10.33
    '''
    if type(Temp)==np.ndarray:
        Temp = format_Temp(depths, Temp)
        coords = {'time':list(range(0,Temp.shape[0])),'depth':depths}
        Temp = xr.DataArray(Temp, coords)
    if top:
        mask = (Temp.depth.values.reshape(-1,1)<depth_ref).T
    elif bot:
        mask = (Temp.depth.values.reshape(-1,1)>depth_ref).T
    else:
        warnings.warnings("Temperature average of the whole water column")
        mean_temp = Temp.mean(dim="depth")
        return mean_temp
    masked_temp = Temp.where(mask)
    mean_temp = masked_temp.mean(dim="depth")
    return mean_temp.to_numpy()

def mixed_layer_depth(depths, Temp, threshold=0.2, method='interp'):
    '''
    Calculates the mixed layer depth by using the difference temperature method.
    The depth of the mixed layer is defined as the depth where the temperature difference with the temperature of the surface is greater than a threshold (default 0.1°C).
    The algorithm does the difference of temperature from the surface to the bottom, reaching lower depth until it reaches a temperature difference lower than the threshold.
    Surface Temperature might have NaNs. If it's the case, we take a deeper sensor (not more than 1m)

    Parameters
    -----------
    depth : array_like 
        depth vector (m)
    Temp : array_like
        Temperature matrix (degrees)
    Threshold : scalar
        threshold for the mixing layer detection
    method: str, bool, default: interp
        If set to interp, the dataset is rescaled with a smaller grid, and the data is therfore interpolated. 

    Returns
    -----------
    hML array_like
        Mixed layer depth (m)
    
    Example
    ----------
    >>>    wtr = np.array([22.51, 22.42, 22.4, 22.4, 22.4, 22.36, 22.3, 22.21, 22.11, 21.23, 16.42,15.15, 14.24, 13.35, 10.94, 10.43, 10.36, 9.94, 9.45, 9.1, 8.91, 8.58, 8.43])
    ...    depths = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
    ...    lw.mixed_layer_depth(depths, wtr, threshold=0.1)
    ...    1
    '''
    Temp = format_Temp(depths, Temp)
    T_surf = Temp[:,0]

    def nanTsurf(depths, Temp, T_surf, NaN_treshold=1):
        # Check if sensor surface is not NaN, if it is, take the one below
        # if one bellow deeper than NaN_treshold, warn that it's too deep
        depth_idx_1m = np.where(depths>=NaN_treshold)[0][0]
        idx_Tsurf_nan = np.where(np.isnan(T_surf))
        depth_idx = np.zeros(len(T_surf))
        for t in idx_Tsurf_nan[0]:
            for T in Temp[t,1:]:
                depth_idx[t] = depth_idx[t]+1
                if not np.isnan(T):
                    T_surf[t] = T
                    break
        exceed_threshold = np.where(depth_idx>=depth_idx_1m)[0]
        if len(exceed_threshold):
            warnings.warn(f"NaN values are present at the surface sensors for {len(exceed_threshold)} profiles")
        return T_surf, exceed_threshold
    
    T_surf, nan_mask = nanTsurf(depths, Temp, T_surf, NaN_treshold=1)
        
    T_diff = T_surf-Temp.T-threshold

    if method=='interp':
        dz=0.1
        T_diff = format_Temp(depths, T_diff)
        coords = {'time':list(range(0,T_diff.shape[0])),'depth':depths}
        ds = xr.DataArray(T_diff, coords)
        depth_interp = np.arange(ds.depth.min(), ds.depth.max(), dz)
        ds_interp = ds.interp(depth=depth_interp)
        hML = abs(ds_interp).idxmin("depth")
        hML[nan_mask] = np.nan
        return hML.to_numpy()
    else:
        hML_idx = np.nanargmin(np.abs(ds), axis=0)
        hML = depths[hML_idx]
        hML[nan_mask] = np.nan
        return hML
