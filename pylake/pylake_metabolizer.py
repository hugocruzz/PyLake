from logging import warning
from pickle import TRUE
import pandas as pd 
import xarray as xr
import numpy as np 
from .functions import *
from .functions_meta import *
import warnings

def metab_bookkeep(do_obs, do_sat, k_gas, z_mix, irr, Datetime):
    #Lake Metabolizer gives n number of points, 
    
    Datetime, do_obs, do_sat, k_gas, z_mix, irr = list(map(np.asanyarray, (Datetime, do_obs, do_sat, k_gas, z_mix, irr)))
    Datetime = pd.to_datetime(Datetime)
    #freq = (Datetime[1]-Datetime[0]).total_seconds()/60
    loc = locals()
    ds = xarray_from_input(metab_bookkeep, loc, coords = {'time':Datetime})
    freq = ds.time.size
    '''
    if(any(z_mix <= 0)):
        warnings.warn("z_mix must be greater than zero.")'''
    
    dayI = (ds["irr"]>0)
    nightI = (ds["irr"]<=0)

    delta_do = ds["do_obs"].diff('time')
    miss_delta = sum(np.isnan(delta_do)) # number of NA's

    #gas flux out is negative
    #normalized to z_mix, del_concentration/timestep (e.g., mg/L/10m)
    gas_flux = (ds["do_sat"] - ds["do_obs"]) * (ds["k_gas"]/freq) / ds["z_mix"].data

    #remove the component of delta.do that is due to gas flux
    delta_do_metab = delta_do - gas_flux

    #normalize units to per-day
    # delta.do.meta.daily = delta.do.metab * (60*60*24)/as.numeric(delta.times, 'secs')

    nep_day = delta_do_metab.where(dayI).resample(time='1D')
    nep_night = delta_do_metab.where(nightI).resample(time='1D')

    R = nep_night.mean('time') * freq # should be negative
    NEP = delta_do_metab.resample(time='1D').mean('time') * freq # can be positive or negative
    GPP = nep_day.mean('time') * dayI.mean('time') - nep_night.mean('time') * dayI.mean('time') # should be positive
    metab = {"GPP":GPP, "R":R, "NEP":NEP}
    return metab

def xarray_from_input(func, locals, coords):
    ds = xr.Dataset(coords=coords)
    for var in func.__code__.co_varnames:
        try:
            ds[var]=('time', locals[var])
        except:
            try:
                ds[var]=locals[var]
            except:pass
    return ds

def metab_ols(wtr, do_obs, do_sat, k_gas, z_mix, irr, freq):
    from sklearn.linear_model import LinearRegression
    #Format and get rid of NaNs
    freq, do_obs, do_sat, k_gas, z_mix, irr = list(map(np.asanyarray, (freq, do_obs, do_sat, k_gas, z_mix, irr)))
    mask_nan = ~np.isnan(wtr)&~np.isnan(do_obs)&~np.isnan(do_sat)&~np.isnan(k_gas)&~np.isnan(z_mix)&~np.isnan(irr)
    do_obs, do_sat, k_gas, z_mix, irr, wtr = do_obs[mask_nan], do_sat[mask_nan], k_gas[mask_nan], z_mix[mask_nan], irr[mask_nan], wtr[mask_nan]

    nobs = len(do_obs)
    do_diff = np.diff(do_obs)

    inst_flux = (k_gas/freq) * (do_sat - do_obs)  # positive is into the lake

    # flux = (inst_flux[1:(n.obs-1)] + inst_flux[-1])/2
    flux = inst_flux[:-1]

    noflux_do_diff = do_diff - flux/z_mix[:-1]

    lntemp = np.log(wtr)

    regr = LinearRegression()
    df = pd.DataFrame({"noflux_do_diff":noflux_do_diff, "irr": irr[:-1], "lntemp":lntemp[:-1]})
    X= df[['irr', 'lntemp']]
    y = df["noflux_do_diff"] 
    reg = regr.fit(X,y)
    iota = reg.coef_[0]
    rho = reg.coef_[1]

    gpp = np.nanmean(iota*irr[:-1])*freq
    resp = np.nanmean(rho*lntemp[:-1])*freq
    nep = gpp + resp

    results = {"GPP":gpp, "R":resp, "NEP":nep}
    return(results)
'''
def metab_mle(do_obs, do_sat, k_gas, z_mix, irr, wtr, freq, error_type="OE"):
    freq, do_obs, do_sat, k_gas, z_mix, irr = list(map(np.asanyarray, (freq, do_obs, do_sat, k_gas, z_mix, irr)))
    mask_nan = ~np.isnan(wtr)&~np.isnan(do_obs)&~np.isnan(do_sat)&~np.isnan(k_gas)&~np.isnan(z_mix)&~np.isnan(irr)
    do_obs, do_sat, k_gas, z_mix, irr, wtr = do_obs[mask_nan], do_sat[mask_nan], k_gas[mask_nan], z_mix[mask_nan], irr[mask_nan], wtr[mask_nan]
    nobs = len(do_obs)

    Q0 = ((np.diff(range(do_obs)) - np.mean(do_obs))^2 / len(do_obs))

    guesses = np.array([1E-4, 1E-4, np.log(Q0)])

    #We have a different number of fitted parameters depending on error type of the model
    if(error_type=='OE'):
        guesses = c(guesses, do_obs[1]) 

        fit = optim(guesses, fn=mleNllOE, do_obs=do_obs, do_sat=do_sat, k_gas=(k_gas/freq), z_mix=z_mix, irr=irr, wtr=wtr)
        
        pars0 = fit$par
        pars = c("gppCoeff"=pars0[1], "rCoeff"=pars0[2], "Q"=exp(pars0[3]), "nll"=fit$value, "doInit"=pars0[4])
        
    else if(error_type=='PE'):
        guesses = c(guesses) 
        
        fit = optim(guesses, fn=mleNllPE, do_obs=do_obs, do_sat=do_sat, k_gas=(k_gas/freq), z_mix=z_mix, irr=irr, wtr=wtr)
        
        pars0 = fit$par
        pars = c("gppCoeff"=pars0[1], "rCoeff"=pars0[2], "Q"=exp(pars0[3]), "nll"=fit$value)
        
    else:
        stop("error.type must be either 'OE' or 'PE', Observation Error or Process Error respectively.")

    # ====================================
    # = Use fits to calculate metabolism =
    # ====================================
    GPP = mean(pars[1]*irr, na.rm=TRUE) * freq
    R = mean(pars[2]*np.log(wtr), na.rm=TRUE) * freq

    return(list("params"=pars, "metab"=c("GPP"=GPP,"R"=R,"NEP"=GPP+R)))
    '''
def o2_at_sat(temp, baro=None, altitude=0, salinity=.2, model='garcia-benson'):
    #temp, baro, altitude, salinity = list(map(np.asanyarray, (temp, baro, altitude, salinity)))
    # Conversion from mL/L (the usual output of the garcia, weiss, etc. equations)
    # to mg/L per USGS memo 2011.03
    mgL_mlL = 1.42905

    # Correction for air pressure; incorportes effects of altitude & vapor pressure of water
    mmHg_mb = 0.750061683 # conversion from mm Hg to millibars
    if baro is None:
        mmHg_inHg = 25.3970886 # conversion from inches Hg to mm Hg
        standard_pressure_sea_level = 29.92126 # Pb, inches Hg
        standard_temperature_sea_level = 15 + 273.15 # Tb, 15 C = 288.15 K
        gravitational_acceleration = 9.80665 # g0, m/s**2
        air_molar_mass = 0.0289644 # M, molar mass of Earth's air (kg/mol)
        universal_gas_constant = 8.31447 #8.31432 # R*, N*m/(mol*K)

        # estimate pressure by the barometric formula
        baro = (1/mmHg_mb) * mmHg_inHg * standard_pressure_sea_level * \
            np.exp( (-gravitational_acceleration * air_molar_mass * altitude) / (universal_gas_constant * standard_temperature_sea_level) )
        baro = np.ones(len(temp))*baro
    
    # pressure correction per USGS memos 81.11 and 81.15. calculate u by Antoine equation.
    u = 10 ** (8.10765 - 1750.286 / (235 + temp)) # u is vapor pressure of water; water temp is used as an approximation for water & air temp at the air-water boundary
    press_corr = (baro*mmHg_mb - u) / (760 - u) # pressure correction is ratio of current to standard pressure after correcting for vapor pressure of water. 0.750061683 mmHg/mb

    # Estimate O2 at saturation in mL/L by several models
    if(model == 'garcia'):

        Ts = np.log((298.15 - temp)/(273.15 + temp))

        lnC = 2.00856 + 3.22400 *Ts + 3.99063*Ts**2 + 4.80299*Ts**3 + 9.78188e-1*Ts**4 + \
        1.71069*Ts**5 - salinity*(6.24097e-3 + 6.93498e-3*Ts + 6.90358e-3*Ts**2 + 4.29155e-3*Ts**3) - 3.1168e-7*salinity**2

        o2_sat = np.exp(lnC)

    elif((model) == 'garcia-benson'):
        
        Ts = np.log((298.15 - temp)/(273.15 + temp))
        
        lnC = 2.00907 + 3.22014*Ts + 4.05010*Ts**2 + 4.94457*Ts**3 + -2.56847e-1*Ts**4 + \
        3.88767*Ts**5 - salinity*(6.24523e-3 + 7.37614e-3*Ts + 1.03410e-2*Ts**2 + 8.17083e-3*Ts**3) - 4.88682e-7*salinity**2
        
        o2_sat = np.exp(lnC)
        
    elif((model) == 'weiss'):
        tempk = temp + 273.15

        lnC = -173.4292 + 249.6339 * (100 / tempk) + 143.3483 * \
        np.log(tempk / 100) - 21.8492 * (tempk / 100) + \
        salinity * (-0.033096 + 0.014259 * (tempk / 100) - 0.0017000 * (tempk / 100)**2)
            
        o2_sat = np.exp(lnC)


    o2_sat = o2_sat * mgL_mlL * press_corr

    return(o2_sat)

def k_cole(wnd):
    k600 = 2.07 + (0.215*(wnd**(1.7)))
    k600 = k600*24/100
    return k600

def k_crusius(wnd, method='power'):
    # -- References 
    # CRUSIUS, JOHN, AND RIK WANNINKHOF. 2003
    # Gas transfer velocities measured at low wind speed over a lake.
    # Limnology and Oceanography. 48(3): 1010:1017.
    U10 = wnd
    if method=='constant':
        mask = U10<3.7
        k600 = 0.72*U10
        k600[~mask] = 14*U10-17.9
    elif method=='bilinear':
        mask = U10<3.7
        k600 = 0.72*U10
        k600[~mask] = 4.33*U10-13.3
    elif method =='power':
        k600 = 0.228*U10**2.2+0.168 # units in cm h-1

    k600 = k600*24/100
    return k600

def k_read(wnd_z, Qsen, Qlat, Cd, Kd, lat, A0, air_press, dateTime, Ts, hML, airT, wnd, RH, sw, lwnet, lwnet_mode=1, s=0.2):
    #'@param wnd Numeric value of wind speed, (Units:m/s)
    #'@param method Only for \link{k.crusius.base}. String of valid method . Either "constant", "bilinear", or "power"
    #'@param wnd_z Height of wind measurement, (Units: m)
    #'@param Kd Light attenuation coefficient (Units: m**-1)
    #'@param lat Latitude, degrees north
    #'@param A0 Lake area, m**2
    #'@param air_press Atmospheric pressure, (Units: millibar)
    #'@param dateTime datetime (Y-\%m-\%d \%H:\%M), (Format: \code{\link{POSIXct}})
    #'@param Ts Numeric vector of surface water temperature, (Units(deg C)
    #'@param hML Numeric vector of actively mixed layer depths. Must be the same length as the Ts parameter
    #'@param airT Numeric value of air temperature, Units(deg C)
    #'@param RH Numeric value of relative humidity, \%
    #'@param sw Numeric value of short wave radiation, W m**-2
    #'@param lwnet Numeric value net long wave radiation, W m**-2, 
    # define constants used in function
    wnd, Qsen, Qlat, Cd, Kd, air_press, dateTime, Ts, hML, airT, wnd, RH, sw, lwnet = list(map(np.asanyarray, (wnd, \
        Qsen, Qlat, Cd, Kd, air_press, dateTime, Ts, hML, airT, wnd, RH, sw, lwnet)))
    #if no net, convert it to net
    if not lwnet_mode:
        Tk = Ts+Kelvin # water temperature in Kelvin
        LWo = S_B*emiss*Tk**4 # long wave out
        lwnet = lwnet-LWo

    Kelvin = 273.15 # temp mod for deg K   
    emiss = 0.972 # emissivity;
    S_B = 5.67E-8 # Stefan-Boltzman constant (?K is used)
    vonK = 0.41 # von Karman  constant
    dT = 0.5   # change in temp for mixed layer depth
    C1 = 114.278 # from Soloviev et al. 2007
    nu = 0.29 # proportionality constant from Zappa et al. 2007, lower bounds
    KeCrit = 0.18     # constant for wave age = 20 (Soloviev et al. 2007)
    albedo_SW = 0.07
    swRat = 0.46 # percentage of SW radiation that penetrates the water column
    g = 9.81 # gravity
    C_w = 4186 # J kg-1 ?C-1 (Lenters et al. 2005)
    mnWnd = 0.2 # minimum wind speed

    # impose limit on wind speed
    rpcI = wnd < mnWnd
    wnd[rpcI] = mnWnd

    # calculate sensible and latent heat fluxes
    C_D = Cd # drag coefficient for momentum
    E = Qlat # latent heat flux
    H = Qsen # sensible heat flux

    # calculate total heat flux
    dUdt = sw*0.93 - E - H + lwnet
    Qo = sw*(1-albedo_SW)*swRat

    # calculate water density
    rho_w = dens0(t=Ts, s=0.2)

    # calculate u*
    if (wnd_z != 10):
        e1 = np.sqrt(C_D)
        wnd = wnd/(1-e1/vonK*np.log(10/wnd_z))
        
    rhoAir = 1.2 #  air density
    tau = C_D*wnd**2*rhoAir
    uSt = np.sqrt(tau/rho_w)

    # calculate the effective heat flux
    q1 = 2-2*np.exp(hML*-Kd)
    q2 = hML*Kd
    q3 = np.exp(hML*-Kd)
    H_star = dUdt-Qo*(q1/q2-q3) # Kim 1976

    # calculate the thermal expansion coefficient 
    tExp = thermalExpFromTemp(Ts)

    # calculate buoyancy flux and w*
    B1 = H_star*tExp*g
    B2 = rho_w*C_w
    Bflx = B1/B2
    ltI = Bflx>0
    if type(B1)==np.float64:
        B1 = np.array(B1)
    B1[ltI] = 0
    divi = 1/3
    w1 = -B1*hML
    wSt = w1**divi

    # calculate kinematic viscosiy
    kinV = getKinematicVis(Ts)

    KeDe = (kinV*g)
    KeNm = uSt**3
    Ke = KeNm/KeDe
    tau = tau    # estimate of total tau (includes wave stress)
    euPw = (1+Ke/KeCrit)  # tau_shear = tau/(1+Ke/Kecr)
    tau_t = tau/euPw      # tau_shear, Soloviev
    uTanS = tau_t/rho_w   
    uTanS = uTanS**0.5

    # calculate viscous sublayer
    Sv = C1*kinV/uTanS
    eu_N = uTanS**3      # e_u(0) = (tau_t/rho)**1.5/(vonK*Sv)
    eu_D = vonK*Sv       # denominator
    eu_0 = eu_N/eu_D    # in m2/s3
    ew_0 = -1.0*B1       # buoyancy flux, but only when outward
    e_0 = ew_0+eu_0     # e(0) from Soloviev (w/o wave effects)
    K1 = e_0*kinV       # in units of m4/s4, want cm4/hr4
    K2 = ew_0*kinV      # convective component (m4/s4)
    K1 = K1*100**4*3600**4 # now in cm4/hr4  (Total)
    K2 = K2*100**4*3600**4 # now in cm4/hr4  (Convective)
    K600 = nu*600**(-0.5)*K1**(1/4)   # in cm/hr (Total)

    #k600 = np.numeric(K600)
    k600 = K600*24/100 #now in units in m d-1
    return(k600)

def k_heiskanen(wnd_z, Cd, Qlat, Qsen, Kd, air_press, dateTime, Ts, hML, airT, wnd, RH, sw, lwnet):

    #Constants
    S_B = 5.67E-8 # Stefan-Boltzman constant (K is used)
    emiss = 0.972 # emissivity
    Kelvin = 273.15 #conversion from C to Kelvin
    albedo_SW = 0.07
    vonK = 0.41 #von Karman constant
    swRat = 0.46 # percentage of SW radiation that penetrates the water column
    mnWnd = 0.2 # minimum wind speed
    g = 9.81 # gravity
    C_w = 4186 # J kg-1 ?C-1 (Lenters et al. 2005)

    # impose limit on wind speed
    rpcI = wnd < mnWnd
    if type(wnd)==int:
        wnd=np.array(wnd)
    wnd[rpcI] = mnWnd


    # calculate sensible and latent heat fluxes
    #mm = calc.zeng(dateTime,Ts,airT,wnd,RH,air_press,wnd_z)
    C_D = Cd # drag coefficient for momentum
    E = Qlat # latent heat flux
    H = Qsen # sensible heat flux

    # calculate total heat flux
    dUdt = sw*0.93 - E - H + lwnet
    Qo = sw*(1-albedo_SW)*swRat

    # calculate water density
    rho_w = dens0(t=Ts, s=0.2)

    # calculate u*
    if (wnd_z != 10):
        e1 = np.sqrt(C_D)
        u10 = wnd/(1-e1/vonK*np.log(10/wnd_z))
    else:
        u10 = wnd


    rhoAir = 1.2 #  air density
    vonK = 0.41 # von Karman  constant
    tau = C_D*u10**2*rhoAir
    uSt = np.sqrt(tau/rho_w)

    # calculate the effective heat flux
    q1 = 2-2*np.exp(hML*-Kd)
    q2 = hML*Kd
    q3 = np.exp(hML*-Kd)
    H_star = dUdt-Qo*(q1/q2-q3) # Kim 1976

    # calculate the thermal expansion coefficient 
    tExp = thermalExpFromTemp(Ts)

    B1 = H_star*tExp*g #Imberger 1985: Effective heat flux * thermal expansion of water * gravity
    B2 = rho_w*C_w # mean density of the water column * specific heat of water at constant pressure
    Bflx = B1/B2

    if Bflx<0:
        wstar = (-Bflx*hML)**(1/3)#penetrative convective velocity Heiskanen 2014 (Imberger 1985)
    else:
        wstar = 0
    Hk   = np.sqrt((0.00015*u10)**2 + (0.07*wstar)**2) 
    Hk   = Hk*100*3600 # Heiskanen's K in cm/hr
    Hk600 = Hk*600**(-0.5)
    k600 = Hk600*24/100 #units in m d-1
    return(k600)


def k_macIntyre(wnd_z, Cd, Qlat, Qsen, Kd, air_press, dateTime, Ts, hML, airT, wnd, RH, sw, lwnet,
                                params=np.array([1.2,0.4872,1.4784])):

    #Constants
    S_B = 5.67E-8 # Stefan-Boltzman constant (K is used)
    emiss = 0.972 # emissivity;
    Kelvin = 273.15 #conversion from C to Kelvin
    albedo_SW = 0.07
    vonK = 0.41 #von Karman constant
    swRat = 0.46 # percentage of SW radiation that penetrates the water column
    mnWnd = 0.2 # minimum wind speed
    g = 9.81 # gravity
    C_w = 4186 # J kg-1 ?C-1 (Lenters et al. 2005)

    # impose limit on wind speed
    rpcI = wnd < mnWnd
    if type(wnd)==int:
        wnd=np.array(wnd)
    wnd[rpcI] = mnWnd


    # calculate sensible and latent heat fluxes
    C_D = Cd # drag coefficient for momentum
    E = Qlat # latent heat flux
    H = Qsen # sensible heat flux

    # calculate total heat flux
    dUdt = sw*0.93 - E - H + lwnet
    Qo = sw*(1-albedo_SW)*swRat

    # calculate water density
    rho_w = dens0(t=Ts, s=0.2)

    # calculate u*
    if (wnd_z != 10):
        e1 = np.sqrt(C_D)
        u10 = wnd/(1-e1/vonK*np.log(10/wnd_z))
    else:
        u10 = wnd
    

    rhoAir = 1.2 #  air density
    vonK = 0.41 # von Karman  constant
    tau = C_D*u10**2*rhoAir
    uSt = np.sqrt(tau/rho_w)

    # calculate the effective heat flux
    q1 = 2-2*np.exp(hML*-Kd)
    q2 = hML*Kd
    q3 = np.exp(hML*-Kd)
    H_star = dUdt-Qo*(q1/q2-q3) # Kim 1976


    # calculate the thermal expansion coefficient
    tExp = thermalExpFromTemp(Ts)

    B1 = H_star*tExp*g
    B2 = rho_w*C_w
    Bflx = B1/B2


    # calculate kinematic viscosiy
    kinV = getKinematicVis(Ts)
    KeNm = uSt**3

    #SmE   = 0.84*(-0.58*Bflx+1.76*KeNm/(vonK*hML))
    SmE = params[0]*-Bflx+params[1]*KeNm/(vonK*hML) #change to two coefficients
    if type(SmE)==np.float64:
        SmE=np.array(SmE)
    SmE[SmE<0] = 0    # set negative to 0
    Sk   = SmE*kinV
    Sk   = Sk*100**4*3600**4 # Sally's K now in cm4/h4
    Sk600 = params[2]*600**(-0.5)*Sk**(1/4) # in cm/hr (Total)

    k600 = Sk600 # why is this not already numeric?
    k600 = k600*24/100 #units in m d-1
    return k600

def k_read_soloviev(wnd_z, Cd, Qlat, Qsen, Kd, lat, A0, air_press, dateTime, Ts, hML, airT, wnd, RH, sw, lwnet):
    wnd_z, Cd, Qlat, Qsen, Kd, lat, A0, air_press, dateTime, Ts, hML, airT, wnd, RH, sw, lwnet = list(map(np.asanyarray, (wnd_z,\
         Cd, Qlat, Qsen, Kd, lat, A0, air_press, dateTime, Ts, hML, airT, wnd, RH, sw, lwnet)))
    # define constants used in function
    Kelvin = 273.15 # temp mod for deg K
    emiss = 0.972 # emissivity;
    S_B = 5.67E-8 # Stefan-Boltzman constant (?K is used)
    vonK = 0.41 # von Karman  constant
    dT = 0.5   # change in temp for mixed layer depth
    C1 = 114.278 # from Soloviev et al. 2007
    nu = 0.29 # proportionality constant from Zappa et al. 2007, lower bounds
    KeCrit = 0.18     # constant for wave age = 20 (Soloviev et al. 2007)
    albedo_SW = 0.07
    swRat = 0.46 # percentage of SW radiation that penetrates the water column
    g = 9.81 # gravity
    C_w = 4186 # J kg-1 ?C-1 (Lenters et al. 2005)
    mnWnd = 0.2 # minimum wind speed

    # impose limit on wind speed
    rpcI = wnd < mnWnd
    wnd[rpcI] = mnWnd

    # calculate sensible and latent heat fluxes
    C_D = Cd # drag coefficient for momentum
    E = Qlat # latent heat flux
    H = Qsen # sensible heat flux

    # calculate total heat flux
    dUdt = sw*0.93 - E - H + lwnet
    Qo = sw*(1-albedo_SW)*swRat #PAR

    # calculate the effective heat flux
    q1 = 2-2*np.exp(hML*-Kd)
    q2 = hML*Kd
    q3 = np.exp(hML*-Kd)
    H_star = dUdt-Qo*(q1/q2-q3) #Effective surface heat flux Kim 1976

    # calculate water density
    rho_w = dens0(t=Ts, s=0.2)

    # calculate u*
    if (wnd_z != 10):
        e1 = np.sqrt(C_D)
        wnd = wnd/(1-e1/vonK*np.log(10/wnd_z))
    
    rhoAir = 1.2 #  air density
    tau = C_D*wnd**2*rhoAir
    uSt = np.sqrt(tau/rho_w)
    uSta = np.sqrt(tau/rhoAir)  #friction velocity in air

    # calculate the thermal expansion coefficient
    tExp = thermalExpFromTemp(Ts)

    # calculate buoyancy flux and w*
    B1 = H_star*tExp*g #Hstar * coefficient of thermal expansion * gravity
    B2 = rho_w*C_w
    Bflx = B1/B2
    
    if type(Bflx)==np.float64:
        Bflx = np.array(Bflx)
    Bflx[Bflx>0] = 0

    wSt = (-Bflx*hML)**1/3

    # calculate kinematic viscosiy
    kinV = getKinematicVis(Ts)
    kinVa = getKinematicVis(airT)

    KeDe = (kinV*g)
    KeNm = uSt**3
    Ke = KeNm/KeDe
    tau = tau    # estimate of total tau (includes wave stress)
    euPw = (1+Ke/KeCrit)  # tau_shear = tau/(1+Ke/Kecr) Ke is the Keulegan number
    # Could calculate KeCrit (critical Keulegan number) from wave age
    #KeCrit = (kinVa/kinV)*((rhoAir/rho_w)**1.5)*(Rbcr/Aw) # Eq1.16-Soloviev et al(2007)

    tau_t = tau/euPw      # tau_t = tangential wind stress, tau = total wind stress
    uTanS = tau_t/rho_w
    uTanS = uTanS**0.5

    # calculate viscous sublayer
    Sv = C1*kinV/uTanS  # effective thickness of the aqueous viscous sublayer
    eu_N = uTanS**3      # e_u(0) = (tau_t/rho)**1.5/(vonK*Sv)
    eu_D = vonK*Sv      # denominator
    eu_0 = eu_N/eu_D    # in m2/s3
    ec_0 = -1.0*Bflx       # buoyancy flux, but only when outward

    #ewave_0 turbulence due to wave breaking
    A0 = A0/1e6 # convert surface area to km
    Fetch = 2*np.sqrt(A0/np.pi) # fetch in km (assuming a conical lake)
    Hs = 0.0163*(Fetch**0.5)*wnd # significant wave height - Woolf (2005)
    Aw = (1/(2*np.pi))*(( (g*Hs*rhoAir)/(0.062*rho_w*
            uSt**2))**(2/3)) # wave age - eqn 1.11 Soloviev et al. (2007)

    W = 3.8e-6*wnd**3.4 # simplified whitecap fraction (Fariall et al. 2000)


    Ap = 2.45*W*((1/(W**0.25))-1)
    alphaW = 100 # p. 185 - Soloviev et al. (2007)
    B = 16.6 # p. 185 - Soloviev et al. (2007)
    Sq = 0.2 # p. 185 - Soloviev et al. (2007)
    cT = 0.6 # p. 188 - Soloviev et al. (2007)
    ewave_0 = ((Ap**4)*alphaW)*((3/(B*Sq))**0.5) * \
                    (((Ke/KeCrit)**1.5)/((1+Ke/KeCrit)**1.5))* \
                    (uSt*g*kinV)/(0.062*vonK*cT*((2*np.pi*Aw)**1.5)) * \
                    (rhoAir/rho_w)

    #------------------------------------
    e_0 = ec_0+eu_0+ewave_0    # e(0) from Soloviev (w/o wave effects)
    Kc = ec_0*kinV*100**4*3600**4      # convective component now in cm4/hr4  (Total)
    Ku = eu_0*kinV*100**4*3600**4 # shear component now in cm4/hr4  (Total)
    Kwave = ewave_0*kinV*100**4*3600**4 # wave component now in cm4/hr4  (Total)
    Kall = e_0*kinV*100**4*3600**4       # turbulent kinetic energy now in cm4/hr4  (Total)

    #Schmidt number could be calculated as temperature dependent
    #Sc = 1568+(-86.04*Ts)+(2.142*Ts**2)+(-0.0216*Ts**3)
    k600org = nu*600**(-0.5)*(Kc+Ku)**(1/4)   # in cm/hr (Total)
    k600org = k600org*24/100 #now in units in m d-1

    k600 = nu*600**(-0.5)*Kall**(1/4)   # in cm/hr (Total)
    k600 = k600*24/100 #now in units in m d-1

    # ---Breaking Wave Component, Author: R I Woolway, 2014-11-13 ---
    # bubble mediated component - Woolf 1997
    kbi = W*2450
    beta_0 = 2.71*1e-2 # Ostwald gas solubility (Emerson and Hedges, 2008)
    Sc = 1568+(-86.04*Ts)+(2.142*Ts**2)+(-0.0216*Ts**3) # Schmidt number
    kbiii = (1+(1/(14*beta_0*Sc**(-0.5))**(1/1.2)))**1.2
    kb = kbi/((beta_0*kbiii))
    kb = kb*24/100 #units in m d-1
    #----------------------------------------------------------------

    k600b = k600+kb
    #allks = pd.DataFrame(data =(Ku,Kc,Kwave,kb,k600org,k600,k600b), columns= ["shear","convective","wave","bubble","k600org","k600",'k600b'])
    return k600b

def k_vachon(wnd, A0, params=np.array([2.51,1.48,0.39])):
    U10 = wnd  #This function uses just the wind speed it is supplied
    k600 = params[0] + params[1]*U10 + params[2]*U10*np.log10(A0/1000000) # units in cm h-1
    k600 = k600*24/100 #units in m d-1
    return(k600)
    

def k600_2_kGAS(k600,temperature,gas="O2"):

    n	=	0.5
    schmidt	=	getSchmidt(temperature,gas)
    Sc600	=	schmidt/600

    kGAS	=	k600*(Sc600**-n)
    return(kGAS)

