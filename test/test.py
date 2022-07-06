import pylake
import numpy as np
import seawater as sw

Temp = np.array([14.3,14,12.1,10,9.7,9.5,6,5])
depth = np.array([1,2,3,4,5,6,7,8])

thermoD, thermoInd = pylake.thermocline(Temp, depth)
epilimnion, hypolimnion = pylake.metalimnion(Temp, depth)
SthermoD, SthermoInd = pylake.seasonal_thermocline(Temp,depth)
hML = pylake.mixed_layer(Temp,depth, threshold=0.4)
n2 = pylake.buoyancy_freq(Temp, depth, g=9.81)
avg_ep_T = pylake.Average_layer_temp(Temp, depth=depth, depth_ref=epilimnion, layer='top')
avg_ep_rho = sw.dens0(s=0.2,t=avg_ep_T)
avg_hyp_T = pylake.Average_layer_temp(Temp, depth=depth, depth_ref=hypolimnion, layer='bot')
avg_hyp_rho = sw.dens0(s=0.2,t=avg_hyp_T)
delta_rho = avg_hyp_rho-avg_ep_rho

bthA = np.array([100,90,86,82,20,1])
bthD = np.array([0,2.3,2.5,4.2,5.8,7])
Lo = 50
ustar = 0.5
W = pylake.wedderburn(delta_rho, metaT=epilimnion, uSt=ustar, AvHyp_rho=avg_hyp_rho, Lo=Lo, g=9.81)
Schmidt_stability = pylake.schmidt_stability(Temp, depth=depth, bthA=bthA, bthD=bthD, sal = 0.2, g=9.81, dz=0.1)
heat_content = pylake.heat_content(Temp, bthA=bthA, bthD=bthD, depth=depth, s=0.2)
seiche_period_1 = pylake.seiche_period_1(depth=depth, Zt=thermoD, Lt=Lo, delta_rho=delta_rho, AvHyp_rho=avg_hyp_rho, g= 9.81)
Lake_number = pylake.Lake_number(bthA=bthA, bthD=bthD, ustar=ustar, St=Schmidt_stability, metaT=epilimnion, metaB=hypolimnion, averageHypoDense=avg_hyp_rho, g=9.81)
