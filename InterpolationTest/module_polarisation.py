# -*- coding: utf-8 -*-
"""
Created on Fri May 22 00:13:36 2020

@author: Simon
"""
import numpy as np
from scipy.signal import hilbert
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.fftpack import fft

def correct_early_late(x,y,z, x_Xmax, y_Xmax, z_Xmax, Traces):
    
    # function that corrects the early-late effect knowing the position (x, y, z) of the traces and the position (x_Xmax, y_Xmax, z_Xmax) of Xmax
    
    n = len(x) # number of antennas
    l0 = np.sqrt(x_Xmax**2 + y_Xmax**2 + (z_Xmax - z[0])**2) # distance that goes from the core to Xmax : this définition is effective only for a core with coordinates : (0,0,z[0])
    l = np.zeros(n) # distance that goes from Xmax to each antenna
    for i in range(n):
        
        l[i] = np.sqrt((x_Xmax - x[i])**2  + (y_Xmax - y[i])**2 + (z[i] - z_Xmax)**2)
        
        # We correct the early-late effect for the three components of the lectric field
        Traces[:,i + n] = Traces[:,i + n]*(l[i]/l0) 
        Traces[:,i + n*2] = Traces[:,i + n*2]*(l[i]/l0)
        Traces[:,i + n*3] = Traces[:,i + n*3]*(l[i]/l0)
                        
    return (Traces, l, l0) 


def get_in_shower_plane(x,y,z, Traces, time_sample, elevation, inclination,azimuth):
    
    # function that returns the trcaes in the shower plane (v, vxb, vxvxb) from the traces in the geographic plane (x, y, z)
    
    z = z - z[0] # We move the core position in (0,0,0) before changing the reference frame
    n = len(x) # number of antennas
    
    # antennas positions in the  shower reference frame (v, vxB, vxvxB)
    v = np.zeros(n)   
    vxb = np.zeros(n)
    vxvxb = np.zeros(n)
    
    #Traces in the shower reference frame
    Traces_Ev = np.zeros([time_sample,n])
    Traces_Evxb = np.zeros([time_sample,n])
    Traces_Evxvxb = np.zeros([time_sample,n])
    Time = np.zeros([time_sample, n])
    
    # we reper the direction of the shower
    pi = np.pi
    elevation = elevation*pi/180.0 # elevation = zenith - 90°
    inclination = inclination*pi/180.0 # inclination of the magnetic field
    azimuth = azimuth*pi/180.0 # azimuth of the shower
    
    # unit vectors 
    uv = np.array([np.cos(elevation)*np.cos(azimuth), np.cos(elevation)*np.sin(azimuth) , -np.sin(elevation)]) # direction of the shower
    uB = np.array([np.cos(inclination), 0, -np.sin(inclination)]) # direction of the magnetic field
    
    
    uv_x_uB = np.cross(uv, uB) # unit vector along the vxb direction
    uv_x_uB /= np.linalg.norm(uv_x_uB) # normalisation
    
    uv_x_uvxB  = np.cross(uv, uv_x_uB) # unit vector along the vxvxb direction
    uv_x_uvxB /= np.linalg.norm(uv_x_uB) # normalisation
    
    P = np.transpose(np.array([uv, uv_x_uB, uv_x_uvxB])) # matrix to go from the shower reference frame to the geographic reference frame
    
    P_inv = np.linalg.inv(P) # matrix to go from the geographic reference frame to the shower reference frame
    
    # We calculate the positions in the shower plane
    Position_geo = np.array([x,y,z]) # position in the geographic reference frame
    Position_shower = np.dot(P_inv, Position_geo) # position in the shower reference frame
    
    # We deduce the different components
    v = Position_shower[0, :] 
    vxb = Position_shower[1, :]
    vxvxb =  Position_shower[2, :]
    
    # We calulate the traces in the shower plane
    Traces_geo = np.zeros([time_sample,3])
    Traces_shower_temp = np.zeros([3, time_sample])
    
    for i in range(n):
        
        Traces_geo = np.array([Traces[:,i + n], Traces[:, i + 2*n], Traces[:, i + 3*n]])
        
        Traces_shower_temp = np.dot(P_inv, Traces_geo)
        
        Traces_Ev[:,i] = np.transpose(Traces_shower_temp[0,:]) # Ev component of the traces
        Traces_Evxb[:,i] = np.transpose(Traces_shower_temp[1,:]) # Evxb component of the traces
        Traces_Evxvxb[:,i] = np.transpose(Traces_shower_temp[2,:]) # Evxvxb component of the traces
        
        Time[:,i] = Traces[:,i]
    
    # We deduce the traces in the shower plane
    
    Traces_showerplane_all = np.transpose(np.concatenate((np.transpose(Time), np.transpose(Traces_Ev), np.transpose(Traces_Evxb), np.transpose(Traces_Evxvxb))))


    return (v,vxb, vxvxb, Traces_Ev, Traces_Evxb, Traces_Evxvxb, Traces_showerplane_all)


def get_stokes_parameters_time(x,y, time_sample):    

    # funtion that returns the stokes parameter at a given antenna from the x and y compoent of the traces  
    
    #hilbert transform of the traces
    xh = hilbert(x) 
    yh = hilbert(y)
     
    # We deduce the values of the stokes parameters as a function time 
    
    S0_i = x**2 + xh.imag**2 + y**2 + yh.imag**2 # first stokes parameter at a given antenna as a function of time
    S1_i = x**2 + xh.imag**2 - y**2 - yh.imag**2 # second stokes parameter
    S2_i = x*y +xh.imag*yh.imag # third stokes parameter 
    S3_i = xh.imag*y -x*yh.imag # fourth stokes parameter
   
    return (S0_i, S1_i, S2_i, S3_i)



def get_stokes_parameters_time_all(Traces,time_sample,n):

    # function that returns the values of the Stokes parameters as a function of time for all the antennas of the array 
    
    I = np.zeros([time_sample, n])
    Q = np.zeros([time_sample, n])
    U = np.zeros([time_sample, n])
    V = np.zeros([time_sample, n])

    for i in range(n):
        (I[:,i], Q[:,i], U[:,i], V[:,i]) = get_stokes_parameters_time(Traces[:,i + 2*n], Traces[:, i + 3*n], time_sample)
        
    return(I,Q,U,V)
    
def get_time_window(Traces,time_sample,n, I):
    
    # function that uses the first stokes parmeter "I" to decuce a time window at each antenna
    
    # To calculate the time window we deduce the fwhm of the peak by considering the roots at half width of the peak 
    r1 = np.zeros(n) # first root of the peak at each antenna
    r2 = np.zeros(n) # second root of the peak at each antenna
    index_min = np.zeros(n) # index corresponding to the first
    index_max = np.zeros(n) # index corresponding to the second root

    for i in range(n):
        spline = UnivariateSpline(Traces[:,i], I[:,i]-np.max(I[:,i])/2, s=0) # interpolation of the signal at which we substracted the half maximum value of the peak so as the roots of the  interpolated signal return the abscisse corresponding to the fwhm
        
        if(np.shape(spline.roots())!=(2,)): # if the signal is not a clear peak the spline may return more than 2 roots, in this case we restrict to the smallest time window given by the roots
            r_all = np.sort(spline.roots()) # sorted roots
            imax = np.argmax(I[:,i]) # index at the maximum value of the I parameter
            rmax = Traces[imax,i] # time of occurance of the maximal value of the I parameter
            
            k = 0
            for j in range(len(r_all)):
                if((r_all[j]-rmax)<0): r1[i] = r_all[j] # closest root before the peak value
                if((r_all[j] - rmax >0) & (k==0)):
                    r2[i] = r_all[j] # closest root after the peak value
                    k = k +1
        else:
        
        # if there is only two roots we can directly deduce the fwhm
            r1[i] = np.min(spline.roots())
            r2[i] = np.max(spline.roots())
        
        step = Traces[0,0] - Traces[1,0] # temporal step of the time traces
        
        index_min[i] = abs(round((r1[i] - np.min(Traces[:,i]))/step)) # We deduce the index corresponding to the first root knowing its time of occurance and the time at which the time traces begin
        index_max[i] = abs(round((r2[i] - np.min(Traces[:,i]))/step))  # We deduce the index corresponding to the second root knowing its time of occurance and the time at which the time traces begin
        
        if((index_max[i]>time_sample)): # If the peak is not clearly defined a root may be outside of the time window. In this case we restrict to a time window of 5ns arround the peak value
            index_min[i] = abs(round((rmax - np.min(Traces[:,i]))/step)) - 5
            index_max[i] = abs(round((rmax - np.min(Traces[:,i]))/step)) + 5
        
        if((index_max[i]>time_sample)): # If the peak value occcurs at the end of the time window a root may be outside of the time window. In this case we select the last 5ns of the signal for the time window
            index_min[i] = time_sample -10
            index_max[i] = time_sample
        
    return(r1, r2, index_min, index_max)
    
def get_mean_stokes_parameters(x,y, time_sample, time_min, time_max): # x = Traces_EvxB, y = Traces_EvxvxB  
 
    # function that returns the averaged stokes parameters on a given time window at a given antenna from the traces
    xh = hilbert(x)
    yh = hilbert(y)

    window = time_max - time_min 
    
    I = (np.sum(x[time_min:time_max]**2) + np.sum(xh.imag[time_min:time_max]**2) + np.sum(y[time_min:time_max]**2) + np.sum(yh.imag[time_min:time_max]**2))/window

    Q = (np.sum(x[time_min:time_max]**2) + np.sum(xh.imag[time_min:time_max]**2) - np.sum(y[time_min:time_max]**2) - np.sum(yh.imag[time_min:time_max]**2))/window
        
    U = 2*(np.sum(x[time_min:time_max]*y[time_min:time_max]+ xh.imag[time_min:time_max]*yh.imag[time_min:time_max]))/window
            
    V = 2*(np.sum(xh.imag[time_min:time_max]*y[time_min:time_max] - x[time_min:time_max]*yh.imag[time_min:time_max]))/window
    
    S = (I, Q, U, V)        

    return S

def get_mean_stokes_parameters_sp(Traces, time_min, time_max, time_sample, n, trigger):

    # function that returns the averagred Stokes parameters on a definite time window for all the antennas of the array
    
    # integrated Stokes paramreters
    I_int = np.zeros(n) 
    Q_int = np.zeros(n)
    U_int = np.zeros(n)
    V_int = np.zeros(n)

    for i in range(n):

        (I_int[i], Q_int[i], U_int[i], V_int[i]) = get_mean_stokes_parameters(Traces[:, i + 2*n], Traces[:, i + 3*n], time_sample, int(time_min[i]), int(time_max[i]))
    
    Ip = np.sqrt(Q_int**2 + U_int**2 + V_int**2) # polarised intensity
    
    # derivation of the fraction of polarised signal
    polar_frac = np.zeros(n) 
    for i in range(n):
        if(np.sqrt(I_int[i])<trigger):  # We impose a trigger condition and set the results of antennas that don't trigger as "np.nan"
            I_int[i] = np.nan
            Q_int[i] = np.nan
            U_int[i] = np.nan
            V_int[i] = np.nan
            Ip[i] = np.nan
        if(I_int[i] != 0): polar_frac[i] = Ip[i]/I_int[i]
    
    polar_frac[polar_frac==0] = np.nan
            
    return (I_int, Q_int, U_int, V_int, Ip, polar_frac)

def get_polarisation_from_stokes(Ip, Q_int, U_int):

    # function that returns the polarisation in the shower plane from the integrated Stokes parameters
    
    phiP = 0.5*np.arctan2(U_int,Q_int) + np.pi # polarisation angle
    Etot_sp = np.sqrt(Ip) # total electric field in the shower plane
    Evxb = Etot_sp*np.cos(phiP) # vxb commponent of the electric field
    Evxvxb = Etot_sp*np.sin(phiP) # vxvxb component of the electric field
    Evxb[Evxb == 0] = np.nan
    Evxvxb[Evxvxb == 0] = np.nan
    r = np.sqrt(Evxb**2 + Evxvxb**2)  # norm of the polarisation vector in the shower plane
    
    return (Etot_sp, Evxb, Evxvxb, r, phiP)

def get_max_stokes_parameters_sp(I,Q,U,V,n, trigger):
    
    # function that returns the maximum of the Stokes parameters
    
    I_max = np.zeros(n)
    Q_max = np.zeros(n)
    U_max = np.zeros(n)
    V_max = np.zeros(n)

    for i in range(n):
        I_max[i] = np.max(I[:,i])
        arg = np.argmax(I[:,i])
        Q_max[i] = Q[arg,i]
        U_max[i] = U[arg,i]
        V_max[i] = V[arg,i]
    
    Ipmax = np.sqrt(Q_max**2 + U_max**2 + V_max**2)
   
    polar_frac_max = np.zeros(n)
    for i in range(n):
        if(np.sqrt(I_max[i])<trigger): 
            I_max[i] =0
            Ipmax[i] = 0 
        if(I_max[i] != 0): polar_frac_max[i] = Ipmax[i]/I_max[i]
    
    polar_frac_max[polar_frac_max==0] = np.nan
    
    return(I_max, Q_max, U_max, V_max, Ipmax, polar_frac_max)

def get_polarisation_from_absolute_traces(Traces, time_min, time_max, n, Etot, trigger):
    
    #function that derived the polarisation from the absolute value of the traces
    
    Evxb_abs = np.zeros(n)    
    Evxvxb_abs = np.zeros(n)

    for i in range(n):
    
        window = int(time_max[i] - time_min[i]) # time window over which the signal is averaged
    
        # we integrate the absolute value of the traces on the selected time window 
        for j in range(window):
        
            Evxb_abs[i] = Evxb_abs[i] + np.abs(Traces[j + int(time_min[i]), i + 2*n ])
            Evxvxb_abs[i] = Evxvxb_abs[i] + np.abs(Traces[j + int(time_min[i]), i + 3*n ])

        # we then correct the sign of the traces considering the sign of the peak value
        arg = np.argmax(np.abs(Traces[:, i+2*n]))
        Evxb_abs[i] = (Evxb_abs[i]/window)*np.sign(Traces[arg, i+2*n])    
        Evxvxb_abs[i] = (Evxvxb_abs[i]/window)*np.sign(Traces[arg, i+3*n])  
    
    
    
    rabs = np.sqrt(Evxb_abs**2 + Evxvxb_abs**2) # norm of the polarisation vector in the shower plane
    Etot_sp_abs = rabs
    
    # other sign correction in case the sign of the peak doesn't corresponds to the expected sign of the traces
    for i in range(n):
                    
        if(i%8<4):
            Evxvxb_abs[i] = -np.abs(Evxvxb_abs)[i]
        else:
            Evxvxb_abs[i] = np.abs(Evxvxb_abs)[i]
        
        Evxb_abs[i] = -np.abs(Evxb_abs[i])
        
        # trigger condition
        if(Etot[i]<trigger):
            Evxb_abs[i] = 0
            Evxvxb_abs[i] = 0
            Etot_sp_abs[i] = 0
    
    Evxb_abs[Evxb_abs == 0] = np.nan
    Evxvxb_abs[Evxvxb_abs == 0] = np.nan
    Etot_sp_abs[Etot_sp_abs == 0] = np.nan
    
    return (Etot_sp_abs, Evxb_abs, Evxvxb_abs, rabs)

def get_field_along_axis(vxb, vxvxb, Evxb, Evxvxb):

    #function that returns the components of the lectric field alond the vxb and the vxvxb axis

    position_along_vxb = np.zeros(40) # position of the antennas along the vxb axis
    position_along_vxvxb = np.zeros(40) # position of the antennas along the vxvxb axis
    Evxvxb_along_vxvxb = np.zeros(40) 
    Evxb_along_vxvxb = np.zeros(40)
    Evxvxb_along_vxb = np.zeros(40)
    Evxb_along_vxb = np.zeros(40)
    Etot_along_vxb = np.zeros(40)
    Etot_along_vxvxb = np.zeros(40)

    
    k = 0
    l = 0
       
    # We get the position of the antennas along th vxb baseline
    for i in range(len(vxb)):
        if((i%4 == 0) & (i<160)): # if the antenna is on the vxb baseline 
            position_along_vxb[k] = vxb[i]        
            k = k + 1
            
    # We get the position of the antennas along th vxvxb baseline
    for i in range(len(vxb)):
        if(((i-2)%4 == 0) & (i<160)): # if the antenna is on the vxvxb baseline  
            position_along_vxvxb[l] = vxvxb[i]        
            l = l + 1
    
    # we sort the postition in order to get a plot the LDF with lines
    
    position_along_vxb = np.sort(position_along_vxb)
    position_along_vxvxb = np.sort(position_along_vxvxb)
    
    # we deduce the corresponding values of the electric field along each axis
        
    for i in range(len(vxb)):
        for j in range(len(position_along_vxb)):
            if(position_along_vxb[j] == vxb[i]):
                Evxvxb_along_vxb[j] = Evxvxb[i]
                Evxb_along_vxb[j] = Evxb[i]
                Etot_along_vxb = np.sqrt(Evxb_along_vxb**2 + Evxvxb_along_vxb**2)  

               
    for i in range(len(vxb)):
        for j in range(len(position_along_vxvxb)):
            if(position_along_vxvxb[j] == vxvxb[i]):

                Evxvxb_along_vxvxb[j] = Evxvxb[i]
                Evxb_along_vxvxb[j] = Evxb[i]
                Etot_along_vxvxb = np.sqrt(Evxb_along_vxvxb**2 + Evxvxb_along_vxvxb**2)  
     
         
    return (position_along_vxb, position_along_vxvxb, Evxb_along_vxb, Evxvxb_along_vxb, Evxb_along_vxvxb, Evxvxb_along_vxvxb, Etot_along_vxb, Etot_along_vxvxb)


def get_alpha(elevation, inclination, azimuth):
    
    # function that returns the angle between the direction of the shower and the direction of the magnetic field 
    
    pi = np.pi
    elevation = elevation*pi/180.0
    inclination = inclination*pi/180.0
    azimuth = azimuth*pi/180.0
    
    # unit vectors    
    uv = np.array([np.cos(elevation)*np.cos(azimuth), np.cos(elevation)*np.sin(azimuth) , -np.sin(elevation)]) # direction of the shower
    uB = np.array([np.cos(inclination), 0, -np.sin(inclination)]) # direction of the magnetic field
    cos_alpha = np.dot(uv,uB)
    alpha = np.arccos(cos_alpha) # angle between the direction of the shower and the direction of the magnetic field
    
    return alpha

def get_reconstruction(elevation,inclination, azimuth, vxb, vxvxb, Evxb, Evxvxb):
    
    # function that returns the expected component of the charge excess and the geomganetic emission from the polarisation in the shower plane
    
    alpha = get_alpha(elevation, inclination, azimuth) # angle between the direction of the shower and the direction of the magnetic field
    uvxb = vxb/np.sqrt(vxb**2 + vxvxb**2) 
    theta = np.arccos(uvxb) # observer angle 
    
    Egeo = np.zeros(120) # amplitude of the geomagnetic emission for any antenna that is not on the horizontal baseline
    E_ce = np.zeros(120) # amplitude of the charge excess emission for any antenna that is not on the hoizontal baseline
    k = 0
    
    # we reconstruct the amplitude of both emissions
    for i in range(160):
        if((i%4!=0)):
            Egeo[k] = np.abs(Evxb[i]) - np.abs(Evxvxb[i])*(uvxb[i])/np.abs(np.sin(theta[i]))
            E_ce[k] = (1/np.abs(np.sin(theta[i])))*np.abs(Evxvxb[i])
            k = k +1       
 
    a = np.sin(alpha)*E_ce/np.abs(Egeo) # charge excess to geomagnetic ratio corrected from sin(alpha)
    
    return (E_ce, Egeo, a, theta, alpha)

def get_w(elevation, inclination, azimuth, x, y, z, x_Xmax, y_Xmax, z_Xmax):
 
    #function that returns "w" at each antenna, i.e. the angle between the direction that goes from Xmax to the core and the direction that goes from Xmax to a given antenna
    
    pi = np.pi
    elevation = elevation*pi/180.0
    inclination = inclination*pi/180.0
    azimuth = azimuth*pi/180.0
    
    x_antenna = x - x_Xmax # distance along the x-axis between the antennas postions and Xmax
    y_antenna = y - y_Xmax
    z_antenna = z - z_Xmax
    
    uv = np.array([np.cos(elevation)*np.cos(azimuth), np.cos(elevation)*np.sin(azimuth) , -np.sin(elevation)]) # direction of the shower
    u_antenna = np.array([x_antenna, y_antenna, z_antenna]) # direction of the unit vectors that goes from Xmax to the position of the antennas
    u_antenna /= np.linalg.norm(u_antenna, axis =0)
    w = np.arccos(np.dot(np.transpose(u_antenna), uv))
    w = w*180.0/pi # we calculte w in degrees
    
    return w

def ce2geo_ratio(elevation, inclination, azimuth, x, y,z, x_Xmax, y_Xmax, z_Xmax, vxb, vxvxb, Evxb, Evxvxb, zenith, energy):   
  
    # function that retruns the charge excess to geomagnetic ratio along each arm ff the array (except the horizontal one)
    
    # We first get the amplitude and the ratio  of both emissions
    (E_ce, Egeo, a, theta, alpha) = get_reconstruction(elevation,inclination, azimuth, vxb, vxvxb, Evxb, Evxvxb)
    
    # we get the angle between the direction that goes from Xmax to the core and the direction that goes from Xmax to a given antenna
    w = get_w(elevation, inclination, azimuth, x, y, z, x_Xmax, y_Xmax, z_Xmax)
    
    # The ratio is definite only for antennas that are not on the horizontal baseline, consequently we deduce the values of "w" for the corresponding antennas
    w_cut = np.zeros(120) 
    k = 0
    
    for i in range(len(w)):
        if((i%4!=0)&(i<160)):
            w_cut[k] = w[i]
            k = k+1
    
    w_upr = np.zeros(20) #w for upper right antennas
    a_upr = np.zeros(20) # a for upper right antennas
    a_upl = np.zeros(20) # a for upper left antennas
    a_vertical = np.zeros(40) # a for vertical antennas (up and down)
    w_vertical = np.zeros(40) # w for vertical antennas (up and down)
    a_vertical_up = np.zeros(20) #for vertical antennas up
    a_vertical_down = np.zeros(20) # for vertical antennas down
    w_vertical_up = np.zeros(20)
    w_vertical_down = np.zeros(20)
    w_upl = np.zeros(20)
    w_bl = np.zeros(20)
    w_br = np.zeros(20)
    a_br = np.zeros(20)
    a_bl = np.zeros(20)
    
    g = 0
    h = 0
    m = 0
    n = 0
    b = 0
    c = 0
    d = 0
    
    # we derive the values of a and w for antennas along each arm of the array
    for i in range(len(w_cut)):
        if((i%6==0)):
            w_upl[g] = w_cut[i]
            a_upl[g] = a[i]
            g = g +1
            
        if(((i-4)%6==0)):
            w_vertical_down[c] = w_cut[i]
            a_vertical_down[c] = a[i]
            c = c +1
            
        if(((i-1)%6==0)):
            w_vertical_up[d] = w_cut[i]
            a_vertical_up[d] = a[i]
            d = d +1
            
        if((i-1)%3==0):
            w_vertical[h] = w_cut[i]
            a_vertical[h] = a[i]
            h = h +1
        if(((i-2)%6==0)):
            w_upr[m] = w_cut[i]
            a_upr[m] = a[i]
            m = m +1            
        if(((i-3)%6==0)):
            w_br[n] = w_cut[i]
            a_br[n] = a[i]
            n = n +1    
        if(((i-5)%6==0)):
            w_bl[b] = w_cut[i]
            a_bl[b] = a[i]
            b = b +1
    
    # We calculate the mean value of the ratio over all the arms for each circle of antennas
    a_meanTim = np.zeros(20)
    for i in range(len(a_meanTim)):
        a_meanTim[i] = (a_upl[i] + a_upr[i] + a_vertical_up[i] + a_bl[i] + a_br[i] + a_vertical_up[i])/6.0

    return (theta, w, w_upl, w_upr, w_vertical, w_bl, w_br, a_upl, a_upr, a_vertical, a_bl, a_br, a_meanTim)
    
    
def sinusoidal_fit(a, elevation, inclination, azimuth, Evxb, Evxvxb, vxb, vxvxb, zenith, energy,x, y, z, x_Xmax, y_Xmax, z_Xmax, save, start):

    # function that retruns the charge excess to geomagnetic ratio from a sinusoidal fit of the polarisation angle
    
    pi = np.pi
    elevation = elevation*pi/180.0
    inclination = inclination*pi/180.0
    azimuth = azimuth*pi/180.0
  
    # unit vectors           
    uv = np.array([np.cos(elevation)*np.cos(azimuth), np.cos(elevation)*np.sin(azimuth) , -np.sin(elevation)]) # direction of the shower
    uB = np.array([np.cos(inclination), 0, -np.sin(inclination)]) # direction of the magnetic field
    cos_alpha = np.dot(uv,uB)
    alpha = np.arccos(cos_alpha) #angle between the direction of the magnetic field and the direction of the shower

    
    phiG = pi   # geomagnetic angle: angle between the direction of the geomagnetic emission and the vxb axis
    
    phiP = np.arctan(Evxvxb/Evxb) # polarisation angle: angle between the total polarisation and the vxb axis
    phiobs = np.arctan2(vxvxb,vxb) # observer angle: angle between the direction that goes from the core to a give antenna and the vxb axis
    phiC = np.arctan2(-vxvxb,-vxb) # charge excess angle: phiobs + pi
    
    
    a = a/(np.abs(np.sin(alpha))) # charge excess to geomagnetic ratio corrected from sin(alpha)
    
    # analytical expression of the polarisation angle for a given charge excess to geomagnetic ratio
    y_phiP_a = (np.sin(phiG)+ a*np.sin(phiC)) 
    x_phiP_a = (np.cos(phiG) + a*np.cos(phiC))
    phiP_a = np.arctan(y_phiP_a/x_phiP_a) # expected polarisation angle for a given value for a
    
    # for a given circle of antenna we now fit the best value of the charge excess to geomagnetic ratio
    Number_antenna = 8 # Each cicle of antennas contains 8 antennas
    
    phiobs_cut = np.zeros(Number_antenna) # observer angle for a given circle of antennas
    phiP_cut = np.zeros(Number_antenna) # measured polarisation angle for a given circle of antennas
    phiP_a_cut = np.zeros(Number_antenna) # expected polarisation angle for a given circle of antennas (i.e. a given value of "a")
    
    
    # we calculate the various angles for a given circle of antennas
    for i in range(Number_antenna):
        index = i + start # "start" reper the circle of antennas we consider
        phiobs_cut[i] = phiobs[index]  
        phiP_cut[i] = phiP[index]
        phiP_a_cut[i] = phiP_a[index]
    

    azimuth = azimuth*180/pi # we switch the azimuth to degrees
    
    if(save == 'True'): # instructions to get a plot of the fit
        plt.scatter(phiobs_cut*180/pi, phiP_cut*180/pi)
        plt.plot(phiobs_cut*180/pi, phiP_a_cut*180/pi, color = 'orange')
        plt.legend(['model: a = %.3f' %a,'data'])
        plt.xlabel('Observer angle (degrees)')
        plt.ylabel('Polarisation angle (degrees)')
        plt.title('Azimuth $= %.0f \degree$ Zenith $= %.2f \degree$, Energy = %.3f Eev' %(azimuth, zenith, energy))
        plt.ylim(-7,7)
        plt.savefig('./images/Reconstruction/GeoCE_ratio/CE_geo_ratio_fit_%d.png' %start)
        plt.show()
    
    # we then calculate the quadratic deviation between the model and the data
    chi2 = 0 
    
    for i in range(Number_antenna):
        chi2 = chi2 + ((phiP_cut[i]*180/pi - phiP_a_cut[i]*180/pi)**2)
    
    return chi2

def sinusoidal_fit_min(elevation, inclination, azimuth, EvcrossB, Evcross_vcrossB, vcrossB, vcross_vcrossB, zenith, energy, x, y, z, x_Xmax, y_Xmax, z_Xmax, start, save):

    # function that returns the value of the charge excess to geomagnetic ratio that minimises the chi2 (i.e the quadratic deviation between the model and the data) for a given circle of antennas
    
    step = 1000 # we test 1000 values of "a" between 0 and 0.4 
    a_var = np.linspace(0.,0.4,step)  
    chi2 = np.zeros(step) # quadratic deviation between the observed polarisation angle and the theoritical value expected from the tested value of "a" 
    
    for i in range(step):    
        chi2[i] = sinusoidal_fit(a_var[i],elevation, inclination, azimuth, EvcrossB, Evcross_vcrossB, vcrossB, vcross_vcrossB, zenith, energy,x, y, z, x_Xmax, y_Xmax, z_Xmax, '', start)

    imin = np.argmin(chi2) # index corresponding to the minimal value of the chi2
    amin = a_var[imin] # value of the ratio that minimizes the chi2  
    
    if(save == 'True'): # instructions to get a plot of the chi2 minnimalisation
        plt.xlabel('a')
        plt.ylabel('$\chi^{2}$')
        plt.title('Azimuth $= %.0f \degree$ Zenith $= %.2f \degree$, Energy = %.3f Eev' %(azimuth, zenith, energy))
        plt.plot(a_var, chi2)
        plt.legend(['a_min $= %.3f$  $\chi^{2}$_min = %.3f ' %(amin, np.min(chi2))])
        plt.savefig('./images/Reconstruction/GeoCE_ratio/chi2_ce_geo_ratio%d.png' %start)
        plt.show()

    if(save == 'True'): # instruction to get a plot of the resulting fit
        sinusoidal_fit(amin,elevation, inclination, azimuth, EvcrossB, Evcross_vcrossB, vcrossB, vcross_vcrossB, zenith, energy, x, y, z, x_Xmax, y_Xmax, z_Xmax,save, start)

    return amin

def sinusoidal_fit_all(elevation, inclination, azimuth, Evxb, Evxvxb, vxb, vxvxb, zenith, energy, x, y, z, x_Xmax, y_Xmax, z_Xmax, save):
    
    # function that returns the values of the charge excesss to geomagnetic ratio that fit the the best the data for all the circle of antennas
    
    n = len(Evxb) - 16 # number of antennas without taking the cross check into account
    w_circle = np.zeros(int(n/8.0)) # value of w for each circle of antennas 
    a = np.zeros(int(n/8.0)) # value of the ce/geo ratio for each circle of antennas
    
    w =  get_w(elevation, inclination, azimuth, x, y, z, x_Xmax, y_Xmax, z_Xmax) 
    
    k = 0 
    
    # we derive the charge excess to geomganetic ratio for each circle of antennas that trigger
    for i in range(n):
        if((i%8==0) & (np.isfinite(Evxb[i]))): # if we are not on the horizontal baseline and that we consider a circle of antennas that trigger 
            a[k] = sinusoidal_fit_min(elevation, inclination, azimuth, Evxb, Evxvxb, vxb, vxvxb, zenith, energy, x, y, z, x_Xmax, y_Xmax, z_Xmax, i, save)
            w_circle[k] = np.mean(w[i:(i + 8)]) # we average the values of w for the antennas located on each circle
            k = k +1
    
    # we set the "w" and the "a" as np.nan for antennas that don't trigger
    w_circle[w_circle == 0] = np.nan 
    a[a==0] = np.nan
    
    return (a,w_circle)


def Stokes_parameters_geo(Traces, n, time_sample):
    
    # function that returns the Stokes parameters I as a function of time for each antenna (shorter version of the function get_stokes_parameters_time_all(x,y, time_sample)) 
    
    I = np.zeros([time_sample, n])
    
    for i in range(n):
        
        x = Traces[:, i +n]
        y = Traces[:, i + 2*n]
        z = Traces[:, i +3*n]
        xh = hilbert(x)
        yh = hilbert(y)
        zh = hilbert(z)
                
        for j in range(time_sample):
            
            I[j,i] = (x[j]**2 + xh.imag[j]**2 + y[j]**2 + yh.imag[j]**2 + z[j]**2 + zh.imag[j]**2)
    
    return I

def get_mean_stokes_parameters_geo(Traces, n, time_sample, trigger):     
   
    # function that returns the mean stokes parameters and the 3 components of the polarisation for all the antennas (generalised version of get_mean_stokes_parameters_sp)
    
    I_antenna = Stokes_parameters_geo(Traces, n, time_sample)
    (r1_geo, r2_geo, time_min_geo, time_max_geo) = get_time_window(Traces,time_sample,n, I_antenna)

    Ex = np.zeros(n)
    Ey = np.zeros(n)
    Ez = np.zeros(n)
    I = np.zeros(n)
    Q = np.zeros(n)
    U = np.zeros(n)
    V = np.zeros(n)
    
    for i in range(n):
        x = Traces[:, i +n]
        y = Traces[:, i + 2*n]
        z = Traces[:, i +3*n]
        xh = hilbert(x)
        yh = hilbert(y)
        zh = hilbert(z)
        
        time_min = int(time_min_geo[i])
        time_max = int(time_max_geo[i])
        window = time_max - time_min 
    
        I[i] = (np.sum(x[time_min:time_max]**2) + np.sum(xh.imag[time_min:time_max]**2) + np.sum(y[time_min:time_max]**2) + np.sum(yh.imag[time_min:time_max]**2))/window

        Ex[i] = np.sqrt((np.sum(x[time_min:time_max]**2) + np.sum(xh.imag[time_min:time_max]**2))/window)
            
        Ey[i] = np.sqrt((np.sum(y[time_min:time_max]**2) + np.sum(yh.imag[time_min:time_max]**2))/window)
            
        Ez[i] = np.sqrt((np.sum(z[time_min:time_max]**2) + np.sum(zh.imag[time_min:time_max]**2))/window)
        
        Etot = np.sqrt(Ex**2 + Ey**2 + Ez**2)
               
        if(Etot[i]>trigger): # trigger condition            
            
            Q[i] = (np.sum(x[time_min:time_max]**2) + np.sum(xh.imag[time_min:time_max]**2) - np.sum(y[time_min:time_max]**2) - np.sum(yh.imag[time_min:time_max]**2))/window
        
            U[i] = 2*(np.sum(x[time_min:time_max]*y[time_min:time_max]+ xh.imag[time_min:time_max]*yh.imag[time_min:time_max]))/window
            
            V[i] = 2*(np.sum(xh.imag[time_min:time_max]*y[time_min:time_max] - x[time_min:time_max]*yh.imag[time_min:time_max]))/window
            
        else:
            I[i] = np.nan
            Q[i] = np.nan
            U[i] = np.nan
            V[i] = np.nan
            Ex[i] = 0
            Ey[i] = 0
            Ez[i] = 0
        
        # polarised intensity
        
        Ip_geo = np.sqrt(Q**2 + U**2 + V**2)
        
    return (Ex, Ey, Ez, Ip_geo, I, Q, U, V, I_antenna, r1_geo, r2_geo, time_min_geo, time_max_geo) 


def get_Eb(n, Ex_geo, Ey_geo, Ez_geo, inclination):
    
   
    # function that returns the field along the direction of the magnetic field B
    
    # In GRAND conventions, the the direction of the magnetic field is orthogonal to the y direcition which implies that the field along B has no y component  
    
    Eb_x = Ex_geo*np.cos(inclination*np.pi/180) # x-component of the field along B
    Eb_z = -Ez_geo*np.sin(inclination*np.pi/180) # z-compoent of the field along B
    

    Eb = np.abs(Eb_x + Eb_z) # total value of the electric field along B
    Etot_geo = np.sqrt(Ex_geo**2 + Ey_geo**2 + Ez_geo**2) # total value of the electric field mesured at eache antenna
    
    # values for antennas that don't trigger are set to np.nan
    
    Eb[Eb==0] = np.nan 
    Etot_geo[Etot_geo==0] = np.nan

    return (Eb_x, Eb_z, Eb, Etot_geo)

def get_core_reconstruction(n, n_tries, n_estimation, Eb, Etot, x, y, azimuth, zenith, energy, trigger):

    # function that returns the estimation of the core position using the charge excess to geomagnetic ratio
    
    # we test "n_tries" positions as possible positions for the core
    
    x_core_estimation = np.zeros(n_estimation) # x coordinate of the estimated core postion
    y_core_estimation = np.zeros(n_estimation) # y component of the estimated core position
    x_err_all = np.zeros([n_tries,n_estimation]) # x component of the tested core positions
    y_err_all = np.zeros([n_tries,n_estimation]) # y component of the tested core positions

    Eb_test = Eb[~np.isnan(Eb)] # we check if there are antennas that trigger

    if(np.shape(Eb_test)!= (0,)):
        for k in range(n_estimation):
            # we generate aleatory positions for the tested core positions
            x_err = np.random.normal(0, 100, size=n_tries) 
            y_err = np.random.normal(0, 100, size=n_tries)
            
            x_err_all[:,k] = x_err # array that contains all the x-component of the tested positions for each estimation of the core position
            y_err_all[:,k] = y_err # array that contains all the y-component of the tested positions for each estimation of the core position
            r = np.zeros(n) # distance of the tested core position to the antennas of the array 
            Bmin = np.zeros(len(x_err)) # value of the field along B averaged over the 20 closest antennas of the tested core
        
            for i in range(len(x_err)):
                x_core = x_err[i]
                y_core = y_err[i]
            
                # for each tested core position we calculate the distance of the tested core to the antennas
                for j in range(n):
                    r[j] = np.sqrt((x[j] - x_core)**2 + (y[j] -y_core)**2)
                    antenna_min = r.argsort() # we sort the distances
            
                print(len(Etot))
                Etot_all = Etot[antenna_min[:20]]
                Eb_all = Eb[antenna_min[:20]] # we extract the field along B for the 20 closest antennas of the tested core
                Etot_trigger = Etot_all[~np.isnan(Etot_all)]
                Eb_trigger = Eb_all[~np.isnan(Eb_all)] # we keep only the signal of antennas that trigger  
                
                Bmin[i] = np.mean(Eb_trigger/Etot_trigger)  # we average the resulting values
                
            
            index_min = np.argmin(Bmin) # we extract the index corresponding to the minimal value of the averaged field along B
            
            # we deduce the core position corresponding to this index
            
            x_core_estimation[k] = x_err[index_min] 
            y_core_estimation[k] = y_err[index_min]
    
    
    
        xmean = np.mean(x_core_estimation)
        ymean = np.mean(y_core_estimation)
        
        #xstd = np.std(x_core_estimation)
        #ystd = np.std(y_core_estimation)
        
        Bmin_all = np.zeros([n_tries, n_estimation])
        Bmin_all[:,0] = Bmin
        
        return (x_err_all, y_err_all, Bmin_all, x_core_estimation, y_core_estimation, xmean, ymean)
        
    else: print('no trigger')
    