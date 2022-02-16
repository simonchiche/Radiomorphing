#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 16:39:16 2021

@author: chiche
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


Ldfvxb, Ldfvxvxb, Itot, krho, kE, ref_theta, target_theta, dplane, azimuth = \
np.loadtxt("scaling_check_all.txt", unpack = True)


# =============================================================================
#                           vs theta
# =============================================================================

Etarget=  3.98*kE

plt.scatter(180 -target_theta[Etarget>0], Itot[Etarget>0])
plt.xlabel("Target zenith [Deg.]")
plt.ylabel("residuals total integral")
plt.legend(["E = 3.98 EeV, $\Phi = 90\degree$"])
plt.tight_layout()
plt.savefig("residuals_total_Integral_vs_target_zenith_all_noEcut.pdf")
plt.show()

plt.scatter(180 -target_theta[Etarget>0.1], Itot[Etarget>0.1])
plt.xlabel("Target zenith [Deg.]")
plt.ylabel("residuals total integral")
plt.legend(["E = 3.98 EeV, $\Phi = 90\degree$"])
plt.tight_layout()
plt.savefig("residuals_total_Integral_vs_target_zenith_all_Ecut_0.1.pdf")
plt.show()

plt.scatter(180 -target_theta[Etarget>0.2], Itot[Etarget>0.2])
plt.xlabel("Target zenith [Deg.]")
plt.ylabel("residuals total integral")
plt.legend(["E = 3.98 EeV, $\Phi = 90\degree$"])
plt.tight_layout()
plt.savefig("residuals_total_Integral_vs_target_zenith_all_Ecut_0.2.pdf")
plt.show()

plt.scatter(180 -target_theta[Etarget>0.5], Itot[Etarget>0.5])
plt.xlabel("Target zenith [Deg.]")
plt.ylabel("residuals total integral")
plt.legend(["E = 3.98 EeV, $\Phi = 90\degree$"])
plt.tight_layout()
plt.savefig("residuals_total_Integral_vs_target_zenith_all_Ecut_0.5.pdf")
plt.show()

target_theta_cut = 180 -target_theta[Etarget>0.2]

target_theta_cut_sort = np.sort(180 -target_theta[Etarget>0.2])
Itot_cut = Itot[Etarget>0.2]

theta_bins = [target_theta_cut_sort[0]]

for i in range(len(target_theta_cut)):
    
    if(target_theta_cut[i]!= theta_bins[-1]):
        theta_bins.append(target_theta_cut_sort[i])

Imean = []   
Istd = []
for i in range(len(theta_bins)):
    
    Imean.append(np.mean(Itot_cut[target_theta_cut == theta_bins[i]]))
    Istd.append(np.std(Itot_cut[target_theta_cut == theta_bins[i]]))

plt.scatter(180 -target_theta[Etarget>0.2], Itot[Etarget>0.2])

plt.errorbar(theta_bins, Imean, yerr = Istd, fmt = 'o')
plt.xlabel("Target zenith [Deg.]")
plt.ylabel("residuals total integral")
plt.legend(["E = 3.98 EeV, $\Phi = 90\degree$"])
plt.tight_layout()
plt.savefig("residuals_total_Integral_vs_target_zenith_all_bins_Ecut_0.2.pdf")
plt.show()


target_theta = target_theta[Etarget>0.2]
Ldfvxb = Ldfvxb[Etarget>0.2]
Ldfvxvxb = Ldfvxvxb[Etarget>0.2]
krho = krho[Etarget>0.2]
dplane = dplane[Etarget>0.2]
kE = kE[Etarget>0.2]



plt.scatter(180 -target_theta, Ldfvxb)
plt.xlabel("Target zenith [Deg.]")
plt.ylabel("residuals total integral")
plt.legend(["E = 3.98 EeV, $\Phi = 90\degree$"])
plt.tight_layout()
plt.savefig("residuals_ldfvxb_vs_target_zenith_all.pdf")
plt.show()

plt.scatter(180 -target_theta, Ldfvxvxb)
plt.xlabel("Target zenith [Deg.]")
plt.ylabel("residuals total integral")
plt.legend(["E = 3.98 EeV, $\Phi = 90\degree$"])
plt.tight_layout()
plt.savefig("residuals_ldfvxvxb_vs_target_zenith_all.pdf")
plt.show()


# =============================================================================
#                            vs dplane 
# =============================================================================

plt.scatter(dplane/1e3, Itot[Etarget>0.2])
plt.xlabel("Dist plane xmax [km]")
plt.ylabel("residuals total integral")
plt.legend(["E = 3.98 EeV, $\Phi = 90\degree$"])
plt.tight_layout()
plt.savefig("residuals_total_Integral_vs_dplane_all.pdf")
plt.show()

plt.scatter(dplane/1e3, Ldfvxb)
plt.xlabel("Dist plane xmax [km]")
plt.ylabel("residuals total integral")
plt.legend(["E = 3.98 EeV, $\Phi = 90\degree$"])
plt.tight_layout()
plt.savefig("residuals_ldfvxb_vs_dplane_all.pdf")
plt.show()

plt.scatter(dplane/1e3, Ldfvxvxb)
plt.xlabel("Dist plane xmax [km]")
plt.ylabel("residuals total integral")
plt.legend(["E = 3.98 EeV, $\Phi = 90\degree$"])
plt.tight_layout()
plt.savefig("residuals_ldfvxvxb_vs_dplane_all.pdf")
plt.show()

# =============================================================================
#                               vs krho
# =============================================================================

plt.scatter(abs(1-krho), Itot[Etarget>0.2])
plt.xlabel("|1- $k_{\\rho}^{geo}$|")
plt.ylabel("residuals total integral")
plt.legend(["E = 3.98 EeV, $\Phi = 90\degree$"])
plt.tight_layout()
plt.savefig("residuals_total_Integral_vs_krho_geo_all.pdf")
plt.show()

plt.scatter(abs(1-krho), Ldfvxb)
plt.xlabel("|1- $k_{\\rho}^{geo}$|")
plt.ylabel("residuals total integral")
plt.legend(["E = 3.98 EeV, $\Phi = 90\degree$"])
plt.tight_layout()
plt.savefig("residuals_ldfvxb_vs_krho_geo_all.pdf")
plt.show()

plt.scatter(abs(1-krho), Ldfvxvxb)
plt.xlabel("|1- $k_{\\rho}^{geo}$|")
plt.ylabel("residuals total integral")
plt.legend(["E = 3.98 EeV, $\Phi = 90\degree$"])
plt.tight_layout()
plt.savefig("residuals_ldfvxvxb_vs_krho_geo_all.pdf")
plt.show()


# =============================================================================
#                                   vs kE
# =============================================================================



plt.scatter(abs(1-kE), Itot[Etarget>0.2])
plt.xlabel("|1- $k_{\\rho}^{geo}$|")
plt.ylabel("residuals total integral")
plt.legend(["E = 3.98 EeV, $\Phi = 90\degree$"])
plt.tight_layout()
plt.savefig("residuals_total_Integral_vs_kE_all.pdf")
plt.show()

plt.scatter(abs(1-kE), Ldfvxb)
plt.xlabel("|1- $k_{\\rho}^{geo}$|")
plt.ylabel("residuals total integral")
plt.legend(["E = 3.98 EeV, $\Phi = 90\degree$"])
plt.tight_layout()
plt.savefig("residuals_ldfvxb_vs_kE_all.pdf")
plt.show()

plt.scatter(abs(1-kE), Ldfvxvxb)
plt.xlabel("|1- $k_{\\rho}^{geo}$|")
plt.ylabel("residuals total integral")
plt.legend(["E = 3.98 EeV, $\Phi = 90\degree$"])
plt.tight_layout()
plt.savefig("residuals_ldfvxvxb_vs_kE_all.pdf")
plt.show()

