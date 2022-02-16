# -*- coding: utf-8 -*-
"""
Created on Thu May 14 19:16:10 2020

@author: Simon
"""

import matplotlib.pyplot as plt
import numpy as np
from module_polarisation import get_Eb, get_mean_stokes_parameters_geo
from module_signal_process import add_noise_traces

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 14}

plt.rc('font', **font)

def plot_polarisation(vxb, vxvxb, Etot_sp, Evxb, Evxvxb, r, azimuth, zenith, energy, method):
    
    # function that plots the normalised polarisation in a given plane
    
    plt.scatter(vxb, vxvxb, c=Etot_sp, cmap = 'jet', s= 10)
    cbar = plt.colorbar()
    plt.xlabel('k x b [m]')
    plt.ylabel('k x k x b [m]')
    #plt.xlim(-200,200)
    #plt.ylim(-200,200)
    plt.title(r'$\phi$ $= %.2f \degree$, $\theta$ $= %.2f \degree$, E = %.3f Eev' %(azimuth, zenith, energy), fontsize = 12)
    cbar.set_label(r"$ E\ [\mu V/m]$")
    plt.quiver(vxb, vxvxb, Evxb/r, Evxvxb/r)
    plt.tight_layout()
    plt.savefig('./images/reconstruction/polarisation/polarisation_sp.png', dpi = 500)
    plt.show()
    
    
def plot_polarisation_vxvxb(vcrossB, vcross_vcrossB, Etot_sp, Evxvxb, azimuth, zenith, energy, method):
    
    # function that plots the vxvxb component of the polarisation
    
    plt.scatter(vcrossB, vcross_vcrossB, c=Etot_sp, cmap = 'jet', s= 10)
    cbar = plt.colorbar()
    plt.xlabel('k x b [m]')
    plt.ylabel('k x k x b [m]')
    plt.title(r'$\phi$ $= %.2f \degree$, $\theta$ $= %.2f \degree$, E = %.3f Eev' %(azimuth, zenith, energy), fontsize = 12)
    plt.tight_layout()
    cbar.set_label(r"$ E\ [\mu V/m]$")
    plt.xlim(-300,300)
    plt.ylim(-300,300)    
    plt.quiver(vcrossB, vcross_vcrossB, 0, Evxvxb/np.abs(Evxvxb))
    plt.savefig('./images/reconstruction/polarisation/polarisation_sp_vxvxb_%s.png' %method, dpi = 500)
    plt.show()
    
    
def plot_stokes_parameters(vcrossB, vcross_vcrossB, I, Q, U, V, azimuth, zenith, energy, method):
    
    # function that plots the averaged Stokes parameters in a 2D plane
    
    plt.scatter(vcrossB, vcross_vcrossB, c=I/1e6, cmap = 'jet', s= 10)
    cbar = plt.colorbar()
    plt.xlabel('k x b [m]')
    plt.ylabel('k x k x b [m]')
    plt.title(r'$\phi$ $= %.2f \degree$, $\theta$ $= %.2f \degree$, E = %.3f Eev' %(azimuth, zenith, energy), fontsize = 12)
    plt.tight_layout()
    cbar.set_label(r"$ E\ [V^{2}/m^{2}]$")
    plt.savefig('./images/stokes_parameters/I_%s_sp.png' %method, dpi = 500)
    plt.show()

    plt.scatter(vcrossB, vcross_vcrossB, c=Q/1e6, cmap = 'jet', s= 10)
    cbar = plt.colorbar()
    plt.xlabel('k x b [m]')
    plt.ylabel('k x k x b [m]')
    plt.title(r'$\phi$ $= %.2f \degree$, $\theta$ $= %.2f \degree$, E = %.3f Eev' %(azimuth, zenith, energy), fontsize = 12)
    plt.tight_layout()
    cbar.set_label(r"$ E\ [V^{2}/m^{2}]$")
    plt.savefig('./images/stokes_parameters/Q_%s_sp.png' %method, dpi = 500)
    plt.show()

    plt.scatter(vcrossB, vcross_vcrossB, c=U/1e6, cmap = 'jet', s= 10)
    cbar = plt.colorbar()
    plt.xlabel('k x b [m]')
    plt.ylabel('k x k x b [m]')
    plt.title(r'$\phi$ $= %.2f \degree$, $\theta$ $= %.2f \degree$, E = %.3f Eev' %(azimuth, zenith, energy), fontsize = 12)
    plt.tight_layout()
    cbar.set_label(r"$ E\ [V^{2}/m^{2}]$")
    plt.savefig('./images/stokes_parameters/U_%s_sp.png' %method, dpi = 500)
    plt.show()

    plt.scatter(vcrossB, vcross_vcrossB, c=V/1e6, cmap = 'jet', s= 10)
    cbar = plt.colorbar()
    plt.xlabel('k x b [m]')
    plt.ylabel('k x k x b [m]')
    plt.title(r'$\phi$ $= %.2f \degree$, $\theta$ $= %.2f \degree$, E = %.3f Eev' %(azimuth, zenith, energy), fontsize = 12)
    plt.tight_layout()
    cbar.set_label(r"$ E\ [V^{2}/m^{2}]$")
    plt.savefig('./images/stokes_parameters/V_%s_sp.png' %method, dpi = 500)
    plt.show()
    
    
def plot_total_field(vcrossB, vcross_vcrossB, Etot_sp, azimuth, zenith, energy, method):
    
    # function that plots the total electric field in a 2D plane
    
    plt.scatter(vcrossB, vcross_vcrossB, c=Etot_sp, cmap = 'jet', s= 10)
    cbar = plt.colorbar()
    plt.xlabel('k x b [m]')
    plt.ylabel('k x k x b [m]')
    plt.title(r'$\phi$ $= %.2f \degree$, $\theta$ $= %.2f \degree$, E = %.3f Eev' %(azimuth, zenith, energy), fontsize = 12)
    plt.tight_layout()
    cbar.set_label(r"$ E\ [\mu V/m]$")
    plt.savefig('./images/reconstruction/polarisation/total_field_sp_%s.png' %method, dpi = 500)
    plt.show()
    
def plot_field_along_axis(position_along_vxb, position_along_vxvxb, Etot_along_vxB, Etot_along_vxvxB, azimuth, zenith, energy, method):

    # function that plots the ldf along the main axis of a 2D plane
    
    plt.plot(position_along_vxb, Etot_along_vxB) # On trace le champ le long de la ligne d'antenne
    plt.xlabel('k x b [m]')
    plt.ylabel(r"$ LDF \: along \: kxB$  $[\mu V/m]$")
    plt.title(r'$\phi$ $= %.2f \degree$, $\theta$ $= %.2f \degree$, E = %.3f Eev' %(azimuth, zenith, energy), fontsize = 14)
    plt.tight_layout()
    plt.savefig('./images/ldf/ldf_vxB_%s.png' %method)
    plt.show()
    
    
    plt.plot(position_along_vxvxb, Etot_along_vxvxB) # On trace le champ le long de la ligne d'antenne
    plt.xlabel('k x k x b [m]')
    plt.ylabel(r"$ LDF \: along \: kxkxB$  $[\mu V/m]$")
    plt.title(r'$\phi$ $= %.2f \degree$, $\theta$ $= %.2f \degree$, E = %.3f Eev' %(azimuth, zenith, energy), fontsize = 14)
    plt.tight_layout()
    plt.savefig('./images/ldf/ldf_vxvxB_%s.png' %method)
    plt.show()
    

def plot_time_window(Traces, I, r1, r2, azimuth, zenith, energy, ant):
    
    # function that plots the time window on which the Stokes parameters are averaged
    
    plt.plot(Traces[:,ant], I[:,ant])
    plt.xlim(r1[ant]-5, r2[ant]+5)
    plt.axvspan((r1[ant]), (r2[ant]), facecolor='g', alpha=0.5)
    plt.ylabel('I ($V^{2}/m^{-2}$)')
    plt.xlabel('Time (ns)')
    plt.title(r'$\phi$ $= %.2f \degree$, $\theta$ $= %.2f \degree$, E = %.3f Eev' %(azimuth, zenith, energy), fontsize = 14)
    plt.tight_layout()
    plt.savefig('./images/stokes_parameters/time_window_antenna%d.png' %ant)
    plt.show()

def plot_polar_frac(polar_frac, n, azimuth, zenith, energy, method):
    
    # function that plots the fraction of polarised intensity at each antenna
    
    antenna_id = np.arange(0,n,1)
    plt.plot(antenna_id,polar_frac)
    plt.title(r'$\phi$ $= %.2f \degree$, $\theta$ $= %.2f \degree$, E = %.3f Eev' %(azimuth, zenith, energy), fontsize = 14)
    plt.xlabel('antenna id')
    plt.ylabel('fraction of polarised signal')
    plt.tight_layout()
    plt.savefig('./images/stokes_parameters/polarised_intensity/polar_frac_%s.png' %method)
    plt.show()
    
def plot_ce2geo_antennaid(a, azimuth, zenith, energy, method):
    
    # function that plots the charge excess to geomagnetic ratio at each antenna
    
    antenna_ID = np.arange(0,120,1)    
    plt.xlabel('antenna ID')
    plt.ylabel('ce to geo ratio')
    plt.title(r'$\phi$ $= %.2f \degree$, $\theta$ $= %.2f \degree$, E = %.3f Eev' %(azimuth, zenith, energy), fontsize = 14)
    plt.plot(antenna_ID, a)
    plt.tight_layout()
    plt.savefig('./images/reconstruction/ce_geo_ratio/geo_ce_ratio_antennaID_%s.png' %method)
    plt.show()  

def plot_ce2geo_wcut(a, w, azimuth, zenith, energy, method):
    
    # function that plots the charge excess to geomagnetic ratio as a function of w
    
    w_cut = np.zeros(120) #w without cross check and horizontal (theta = 0[pi]) antennas
    k = 0
    
    for i in range(len(w)):
        if((i%4!=0)&(i<160)):
            w_cut[k] = w[i]
            k = k+1
    
    #w_cut = w_cut[a<0.2]
    #a = a[a<0.2]
    plt.scatter(w_cut, a)
    
    plt.xlabel('w (degrees)')
    plt.ylabel('ce to geo ratio')
    plt.title(r'$\phi$ $= %.2f \degree$, $\theta$ $= %.2f \degree$, E = %.3f Eev' %(azimuth, zenith, energy), fontsize = 14)
    plt.ylim(0,0.2)
    plt.tight_layout()
    #plt.savefig('./images/reconstruction/ce_geo_ratio/geo_ce_ratio_w_scatter_%s.png' %method)
    #plt.show()
    
    #index = w_cut.argsort()
    #w_cut_sort = w_cut[index]
    #a_sort = a[index]
    
    #plt.plot(w_cut_sort, a_sort)
    #plt.ylabel('ce to geo ratio')
    #plt.xlabel('w (degrees)')
    #plt.title(r'$\phi$ $= %.2f \degree$, $\theta$ $= %.2f \degree$, E = %.3f Eev' %(azimuth, zenith, energy), fontsize = 14)
    #plt.tight_layout()
    #plt.savefig('./images/reconstruction/ce_geo_ratio/geo_ce_ratio_w_%s.png' %method)
    #plt.show()
    
def plot_ce2geo_wall(a_upl, a_vertical, a_upr, a_br, a_bl, w_upl, w_vertical, w_upr, w_br, w_bl, azimuth, zenith, energy, method):
    
    # function that plots the charge excess to geomagnetic ratio as a function of w for all the different arms in the array
    
    plt.scatter(w_upl, a_upl)
    plt.scatter(w_vertical, a_vertical)
    plt.scatter(w_br, a_br)
    plt.scatter(w_bl, a_bl)
    plt.scatter(w_upr, a_upr)
    plt.legend(['upper left', 'vertical', 'bottom right', 'bottom left', 'upper right'])
    plt.xlabel('w (degrees)')
    plt.ylabel('ce to geo ratio')
    #plt.ylim(-0.01,0.15)
    #plt.xlim(0,3)
    plt.title(r'$\phi$ $= %.2f \degree$, $\theta$ $= %.2f \degree$, E = %.3f Eev' %(azimuth, zenith, energy), fontsize = 14)
    plt.tight_layout()
    plt.savefig('./images/reconstruction/ce_geo_ratio/geo_ce_ratio_%s.png' %method)
    plt.show()
    
def plot_ce2geo_harm_tim(w_harm, a_harm, w_bl, w_br, w_upl, w_upr, w_vertical, a_bl, a_br, a_upl, a_upr, a_vertical, a_meanTim, azimuth, zenith, energy, method):

    # function that plots the charge excess to geomagnetic from Harm's and Tim's methods
    
    plt.scatter(w_harm[:20], a_harm[:20])
    #plt.ylim(0, 0.15)
    plt.legend(['harm'])
    plt.xlabel('w (degrees)')
    plt.ylabel('ce to geo ratio')
    plt.title(r'$\phi$ $= %.2f \degree$, $\theta$ $= %.2f \degree$, E = %.3f Eev' %(azimuth, zenith, energy), fontsize = 14)
    plt.tight_layout()
    plt.savefig('./images/reconstruction/ce_geo_ratio/geoce_ratio_harm%s.png' %method)
    plt.show()

    plt.scatter(w_harm[:20], a_harm[:20])
    plt.scatter(w_vertical, a_vertical)
    plt.scatter(w_upl, a_upl)
    plt.scatter(w_br, a_br)
    plt.scatter(w_bl, a_bl)
    plt.scatter(w_upr, a_upr)
    #plt.ylim(0, 0.15)
    plt.legend(['harm','vertical','upper left', 'bottom right', 'bottom left', 'upper right'])
    plt.xlabel('w (degrees)')
    plt.ylabel('ce to geo ratio')
    plt.title(r'$\phi$ $= %.2f \degree$, $\theta$ $= %.2f \degree$, E = %.3f Eev' %(azimuth, zenith, energy), fontsize = 14)
    plt.tight_layout()
    plt.savefig('./images/reconstruction/ce_geo_ratio/geoce_ratio_harm_tim_all%s.png' %method)
    plt.show()
    
    plt.scatter(w_harm[:20], a_harm[:20])
    plt.scatter(w_vertical, a_vertical)
    #plt.ylim(0, 0.15)
    plt.legend(['harm', 'vertical'])
    plt.xlabel('w (degrees)')
    plt.ylabel('ce to geo ratio')
    plt.title(r'$\phi$ $= %.2f \degree$, $\theta$ $= %.2f \degree$, E = %.3f Eev' %(azimuth, zenith, energy), fontsize = 14)
    plt.tight_layout()
    plt.savefig('./images/reconstruction/ce_geo_ratio/geoce_ratio_harm_tim_vertical%s.png' %method)
    plt.show()

    plt.scatter(w_harm[:20], a_harm[:20])
    plt.scatter(w_harm[:20], a_meanTim)
    #plt.ylim(0, 0.15)
    plt.legend(['harm', 'Tim'])#, 'vertical', 'bottom right', 'bottom left', 'upper right'])
    plt.xlabel('w (degrees)')
    plt.ylabel('ce to geo ratio')
    plt.title(r'$\phi$ $= %.2f \degree$, $\theta$ $= %.2f \degree$, E = %.3f Eev' %(azimuth, zenith, energy), fontsize = 14)
    plt.tight_layout()
    plt.savefig('./images/reconstruction/ce_geo_ratio/geoce_ratio_harm_tim_mean%s.png' %method)
    plt.show()
    
    
def plot_identification_geographic(Traces, n, time_sample, x, y, Eb, Etot_geo, I_geo, Ip_geo, azimuth, zenith, energy, inclination):
    
    # function that plots the fraction of field along B and the fraction of polarised intensity measured at each antenna
    
    plt.scatter(x,y, c=Eb/Etot_geo, cmap = 'jet')
    plt.xlabel('x[m]')
    plt.ylabel('y[m]')
    cbar = plt.colorbar()
    cbar.set_label("Eb/Etot")
    plt.title(r'$\phi$ $= %.2f \degree$, $\theta$ $= %.2f \degree$, E = %.3f Eev' %(azimuth, zenith, energy), fontsize = 14)
    plt.tight_layout()
    plt.savefig('./images/reconstruction/signal_identification/B_frac2D_E%2.f_z%2.f' %(energy, zenith))
    plt.show()


    noise = np.copy(Traces)
    noise[:,n:] = 0
    noise = add_noise_traces(noise, n, time_sample)
    (Ex_noise, Ey_noise, Ez_noise, Ip_noise, I_noise) = get_mean_stokes_parameters_geo(noise, n, time_sample, 0)[0:5]
    (Eb_x_noise, Eb_z_noise, Eb_noise, Etot_noise) = get_Eb(n, Ex_noise, Ey_noise, Ez_noise, inclination)

    antenna_id = np.arange(0,n,1)
    plt.scatter(antenna_id,Eb/Etot_geo, c='orange')
    plt.plot(antenna_id, Eb_noise/Etot_noise)
    #plt.xlim(0,75)
    plt.xlabel('antenna id', fontsize = 15)
    plt.ylabel('$E_{B}/E_{tot}$', fontsize = 15)
    plt.title(r'$\phi$ $= %.2f \degree$, $\theta$ $= %.2f \degree$, E = %.3f Eev' %(azimuth, zenith, energy), fontsize = 14)
    plt.legend(['with signal', 'without signal'], loc= 1, fontsize = 15)
    plt.tight_layout()
    plt.savefig('./images/reconstruction/signal_identification/B_frac_E%2.f_z%2.f' %(energy, zenith), dpi = 400)
    plt.show()

    plt.scatter(antenna_id,Ip_geo/I_geo, c='orange')
    plt.plot(antenna_id, Ip_noise/I_noise)
    plt.xlabel('antenna id', fontsize = 15)
    plt.ylabel('$q = I_{p}/I$', fontsize = 15)
    plt.legend(['with signal', 'without signal'], fontsize = 15)
    #plt.xlim(0,75)
    plt.tight_layout()
    plt.title(r'$\phi$ $= %.2f \degree$, $\theta$ $= %.2f \degree$, E = %.3f Eev' %(azimuth, zenith, energy), fontsize = 14)
    plt.savefig('./images/reconstruction/signal_identification/polar_frac_E%2.f_z%2.f' %(energy, zenith), dpi = 400)
    plt.show()
    

def plot_core_reconstruction(x_err_all, y_err_all, Bmin, x_core_estimation, y_core_estimation, xmean, ymean, azimuth, zenith, energy, trigger):
    
    # function that plots the reconstruction of the core possition
    
    zenith = 180 - zenith
    plt.scatter(x_err_all, y_err_all, c= Bmin, cmap ='jet')
    cbar = plt.colorbar()
    cbar.set_label(r"$Eb/Etot$")
    plt.scatter(x_core_estimation, y_core_estimation, color = 'orange', marker = '.')
    plt.scatter(0,0, color = 'red')
    plt.xlabel('x [m]', fontsize = 15)
    plt.ylabel('y [m]', fontsize = 15)
    legend =plt.legend(['core estimation $x = %2.f$  $y = %.2f$ ' %(xmean, ymean)], fontsize =11)
    for item in legend.legendHandles:
        item.set_visible(False)
    plt.title(r'$\phi$ $= %.2f \degree$, $\theta$ $= %.2f \degree$, E = %.3f Eev' %(azimuth, zenith, energy), fontsize = 14)
    plt.tight_layout()
    plt.savefig('./images/reconstruction/core_reconstruction/core_geo_20antennas_E%2.f_z%2.f_trigger%d.png' %(energy, zenith, trigger), dpi = 400,bbox_inches = "tight")
    plt.show()
    
def histogram(data, _xlabel, _ylabel, _title, nsim, length, _bins, _color, _edgecolor, save):
    plt.xlabel(_xlabel)
    plt.ylabel(_ylabel)
    plt.title(_title %nsim, fontsize =14)
    plt.hist(data, range = (0, length), bins = _bins, color = _color, edgecolor = _edgecolor)
    plt.tight_layout()
    plt.savefig(save)
    plt.show()
    
def histogram2D(w_cut, data, _xlabel, _ylabel, cbar_label, _title, xlow, ylow, xhigh, yhigh, nsim, _bin1, _bin2, save):

    n_ant = len(w_cut)
    
    plt.hist2d(w_cut, data, bins=(_bin1, _bin2), cmap=plt.cm.jet, range=[[xlow,xhigh ], [ylow, yhigh]])
    cbar = plt.colorbar()
    cbar.set_label(cbar_label)
    plt.title(_title %(nsim, n_ant) , fontsize = 14)
    plt.xlabel(_xlabel)
    plt.ylabel(_ylabel)
    plt.tight_layout()
    plt.savefig(save)
    plt.show()