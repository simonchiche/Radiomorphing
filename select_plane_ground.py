#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 19:08:09 2021

@author: chiche
"""


import numpy as np
import glob
import sys

# =============================================================================
#                 Fonctions to generate a parametrized Xmax
# =============================================================================

def Xmax_param(primary, energy, fluctuations = False):

    #input energy in EeV
    
    
    if(primary == 'Iron'):
        a =65.2
        c =270.6
        
        Xmax = a*np.log10(energy*1e6) + c
        
        if(fluctuations):
            a = 20.9
            b = 3.67
            c = 0.21
            
            sigma_xmax = a + b/energy**c
            Xmax = np.random.normal(Xmax, sigma_xmax)
        
        return Xmax
    
    elif(primary == 'Proton'):
        a = 57.4
        c = 421.9
        Xmax = a*np.log10(energy*1e6) + c
        
        if(fluctuations):
            a = 66.5
            b = 2.84
            c = 0.48
            
            sigma_xmax = a + b/energy**c
            Xmax = np.random.normal(Xmax, sigma_xmax)
        
        return Xmax
        
    else:
        print("missing primary")  
        

def _getAirDensity(_height, model):

    '''Returns the air density at a specific height, using either an 
    isothermal model or the Linsley atmoshperic model as in ZHAireS

    Parameters:
    ---------
        h: float
            height in meters

    Returns:
    -------
        rho: float
            air density in g/cm3
    '''

    if model == "isothermal":
            #Using isothermal Model
            rho_0 = 1.225*0.001    #kg/m^3
            M = 0.028966    #kg/mol
            g = 9.81        #m.s^-2
            T = 288.        #
            R = 8.32        #J/K/mol , J=kg m2/s2
            rho = rho_0*np.exp(-g*M*_height/(R*T))  # kg/m3

    elif model == "linsley":
        #Using Linsey's Model
        bl = np.array([1222., 1144., 1305.5948, 540.1778,1])*10  # g/cm2  ==> kg/cm3
        cl = np.array([9941.8638, 8781.5355, 6361.4304, 7721.7016, 1e7])  #m
        hl = np.array([4,10,40,100,113])*1e3  #m
        if (_height>=hl[-1]):  # no more air
            rho = 0
        else:
            hlinf = np.array([0, 4,10,40,100])*1e3  #m
            ind = np.logical_and([_height>=hlinf],[_height<hl])[0]
            rho = bl[ind]/cl[ind]*np.exp(-_height/cl[ind])
            rho = rho[0]*0.001
    else:
        print("#### Error in GetDensity: model can only be isothermal or linsley.")
        return 0

    return rho   

def _get_CRzenith(zen, GdAlt, injh):
    ''' Corrects the zenith angle for CR respecting Earth curvature, zenith seen by observer
        ---fix for CR (zenith computed @ shower core position
    
    Arguments:
    ----------
    zen: float
        GRAND zenith in deg
    injh: float
        injection height wrt to sealevel in m
    GdAlt: float
        ground altitude of array/observer in m (should be substituted)
    
    Returns:
    --------
    zen_inj: float
        GRAND zenith computed at shower core position in deg
        
    Note: To be included in other functions   
    '''

    #Note: To be included in other functions
   
    Re= 6370949 # m, Earth radius

    a = np.sqrt((Re + injh)**2. - (Re+GdAlt)**2 *np.sin(np.pi-np.deg2rad(zen))**2) - (Re+GdAlt)*np.cos(np.pi-np.deg2rad(zen))
    zen_inj = np.rad2deg(np.pi-np.arccos((a**2 +(Re+injh)**2 -Re**2)/(2*a*(Re+injh))))

    return zen_inj    

        
def _dist_decay_Xmax(zen, GdAlt, injh, primary, energy): 
    ''' Calculate the height of Xmax and the distance injection point to Xmax along the shower axis
    
    Arguments:
    ----------
    zen: float
        GRAND zenith in deg, for CR shower use _get_CRzenith()
    injh2: float
        injectionheight above sealevel in m
    Xmax_primary: float
        Xmax in g/cm2 
        
    Returns:
    --------
    h: float
        vertical Xmax_height in m
    ai: float
        Xmax_distance injection to Xmax along shower axis in m
    '''
    
    zen = _get_CRzenith(zen, GdAlt, injh)
    injh2 = injh
    Xmax_primary = Xmax_param(primary, energy) 
    
    zen2 = np.deg2rad(zen)
    
    hD=injh2
    step=10 #m
    if hD>10000:
        step=100 #m
    Xmax_primary= Xmax_primary#* 10. # g/cm2 to kg/m2: 1g/cm2 = 10kg/m2
    gamma=np.pi-zen2 # counterpart of where it goes to
    Re= 6370949 # m, Earth radius
    X=0.
    i=0.
    h=hD
    ai=0
    while X< Xmax_primary:
        i=i+1
        ai=i*step #100. #m
        hi= -Re+np.sqrt(Re**2. + ai**2. + hD**2. + 2.*Re*hD - 2*ai*np.cos(gamma) *(Re+hD))## cos(gamma)= + to - at 90dg
        deltah= abs(h-hi) #(h_i-1 - hi)= delta h
        h=hi # new height
        rho = _getAirDensity(hi, "linsley")
        X=X+ rho * step*100. #(deltah*100) *abs(1./np.cos(np.pi-zen2)) # Xmax in g/cm2, slanted = Xmax, vertical/ cos(theta); density in g/cm3, h: m->100cm, np.pi-zen2 since it is defined as where the showers comes from, abs(cosine) so correct for minus values
       
    return h, ai # Xmax_height in m, Xmax_distance in m    

def getGroundXmaxDistance(zenith, glevel, injh, primary, energy):
    
    # zenith in cosmic ray convention here
    
    GroundAltitude = glevel
    XmaxHeight, DistDecayXmax = _dist_decay_Xmax(zenith, glevel, injh, primary, energy)

    Rearth = 6370949 
    zenith = (180 -zenith)*np.pi/180
    dist = np.sqrt((Rearth+ XmaxHeight)**2 - ((Rearth + GroundAltitude)*np.sin(zenith))**2) \
    - (Rearth + GroundAltitude)*np.cos(zenith)
            
    return dist   


        
def showerdirection(azimuth, zenith):
    
    # CR convention
    zenith = zenith*np.pi/180.0
    azimuth = azimuth*np.pi/180.0
    
    uv = np.array([np.sin(zenith)*np.cos(azimuth), \
                   np.sin(zenith)*np.sin(azimuth), np.cos(zenith)])
    
    return uv

def getXmaxPosition(azimuth, zenith, glevel, injh, primary, energy):
    
    uv = showerdirection(azimuth, zenith)
        
    showerDistance = getGroundXmaxDistance(zenith, glevel, injh, primary, energy)
    
    XmaxPosition = -uv*showerDistance 
    XmaxPosition[2] = XmaxPosition[2] + glevel  
            
    return XmaxPosition
  

# =============================================================================
#                 Fonctions to get the select the ref shower   
# =============================================================================
    

def select_zenith(target_zenith):
    
    target_zenith = 180 - target_zenith # cosmic ray convention
    
    zenith_sim  = np.array([67.8, 74.8, 81.3, 83.9, 86.5])
    
    min_index = np.argmin(abs(zenith_sim - target_zenith))
    
    selected_zenith =  zenith_sim[min_index]
    
    return selected_zenith


def select_azimuth(target_azimuth):

    target_azimuth = abs(180  - target_azimuth)

    azimuth_sim  = np.array([0, 90, 180])

    min_index = np.argmin(abs(azimuth_sim - target_azimuth))

    selected_azimuth =  azimuth_sim[min_index]

    return selected_azimuth


def select_path(path, dplane, selected_azimuth):
    
    sim = glob.glob(path)

    n = len(sim)
    dsim = np.zeros(n)
    
    for i in range(n):
        
        az_sim = float(sim[i].split("_")[-3])
        
        if(int(az_sim) == int(selected_azimuth)): 
        
            dsim[i] =  float(sim[i].split("_")[-1][:-5])
            
    
    index_all = np.argsort(abs(dsim - dplane))
    
    min_index = index_all[0]
    #min_index = index_all[1]  si on veut tester le radiomorphing pour des plans perp
    # en prenant plan test et plan simulé différent
        
    return sim[min_index], dsim[min_index]
        
        
def select_plane_ground(primary, energy, zenith, azimuth, injection, altitude, fluctuations, simulation, cross_check):
    
    
    Xmax = getXmaxPosition(azimuth, zenith, altitude, injection, primary, energy)
 
    dplane = get_distplane(zenith, azimuth, cross_check[:,0], cross_check[:,1], cross_check[:,2], Xmax[0], Xmax[1], Xmax[2])
    #print(zenith, azimuth, cross_check[:,0], cross_check[:,1], cross_check[:,2], Xmax[0], Xmax[1], Xmax[2])
    dplane = np.mean(dplane)
    target_zenith = select_zenith(zenith)
    target_azimuth = select_azimuth(azimuth)
    
    #target_azimuth = 90
    
    path = "./Simulations/SelectedPlane/theta_%.1f/*.hdf5" \
                      %(target_zenith)

    
    selected_plane, dsim = select_path(path, dplane, target_azimuth)    
    
    return selected_plane, dplane

def get_distplane(zenith, azimuth, x, y, z, x_Xmax, y_Xmax, z_Xmax):
 
    #function that returns "w" at each antenna, i.e. the angle between the direction that goes from Xmax to the core and the direction that goes from Xmax to a given antenna
    
    pi = np.pi
    zenith = zenith*pi/180.0
    azimuth = azimuth*pi/180.0
    
    x_antenna = x - x_Xmax # distance along the x-axis between the antennas postions and Xmax
    y_antenna = y - y_Xmax
    z_antenna = z - z_Xmax
    
    uv = np.array([np.sin(zenith)*np.cos(azimuth), np.sin(zenith)*np.sin(azimuth) , np.cos(zenith)]) # direction of the shower
    u_antenna = np.array([x_antenna, y_antenna, z_antenna]) # direction of the unit vectors that goes from Xmax to the position of the antennas
    distplane = np.dot(np.transpose(u_antenna), uv)
    
    
    return distplane

def print_plane(RefShower, TargetShower, target_dplane):
    
    print("-----------------------")
    print("Target shower: Energy = %.2f, Zenith = %.2f,  Azimuth = %.2f, \
          Dxmax = %.2d" %(TargetShower.energy, 180 -TargetShower.zenith,\
          TargetShower.azimuth, target_dplane))
    print("")
    print("Ref shower: Energy = %.2f, Zenith = %.2f,  Azimuth = %.2f, \
          Dxmax = %.2d" %(RefShower.energy, 180 -RefShower.zenith,\
          RefShower.azimuth, RefShower.distplane))
    print("-----------------------")
    
