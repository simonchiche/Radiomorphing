#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 16:39:16 2021

@author: chiche
"""


import numpy as np
import matplotlib.pyplot as plt



Ldfvxb, Ldfvxvxb, Itot, krho, ref_theta, target_theta, dplane = \
np.loadtxt("zenith_scaling_check_full.txt", unpack = True)


# =============================================================================
#                           vs theta
# =============================================================================
plt.scatter(180 -target_theta, Itot)
plt.xlabel("Target zenith [Deg.]")
plt.ylabel("residuals total integral")
plt.legend(["E = 3.98 EeV, $\Phi = 90\degree$"])
plt.tight_layout()
#plt.savefig("residuals_total_Integral_vs_target_zenith.pdf")
plt.show()

plt.scatter(180 -target_theta, Ldfvxb)
plt.xlabel("Target zenith [Deg.]")
plt.ylabel("residuals total integral")
plt.legend(["E = 3.98 EeV, $\Phi = 90\degree$"])
plt.tight_layout()
#plt.savefig("residuals_ldfvxb_vs_target_zenith.pdf")
plt.show()

plt.scatter(180 -target_theta, Ldfvxvxb)
plt.xlabel("Target zenith [Deg.]")
plt.ylabel("residuals total integral")
plt.legend(["E = 3.98 EeV, $\Phi = 90\degree$"])
plt.tight_layout()
#plt.savefig("residuals_ldfvxvxb_vs_dplane.pdf")
plt.show()


# =============================================================================
#                            vs dplane 
# =============================================================================

plt.scatter(dplane/1e3, Itot)
plt.xlabel("Dist plane xmax [km]")
plt.ylabel("residuals total integral")
plt.legend(["E = 3.98 EeV, $\Phi = 90\degree$"])
plt.tight_layout()
#plt.savefig("residuals_total_Integral_vs_dplane.pdf")
plt.show()

plt.scatter(dplane/1e3, Ldfvxb)
plt.xlabel("Dist plane xmax [km]")
plt.ylabel("residuals total integral")
plt.legend(["E = 3.98 EeV, $\Phi = 90\degree$"])
plt.tight_layout()
#plt.savefig("residuals_ldfvxb_vs_dplane.pdf")
plt.show()

plt.scatter(dplane/1e3, Ldfvxvxb)
plt.xlabel("Dist plane xmax [km]")
plt.ylabel("residuals total integral")
plt.legend(["E = 3.98 EeV, $\Phi = 90\degree$"])
plt.tight_layout()
#plt.savefig("residuals_ldfvxvxb_vs_dplane.pdf")
plt.show()

# =============================================================================
#                               vs krho
# =============================================================================

plt.scatter(abs(1-krho), Itot)
plt.xlabel("|1- $k_{\\rho}^{geo}$|")
plt.ylabel("residuals total integral")
plt.legend(["E = 3.98 EeV, $\Phi = 90\degree$"])
plt.tight_layout()
#plt.savefig("residuals_total_Integral_vs_krho_geo.pdf")
plt.show()

plt.scatter(abs(1-krho), Ldfvxb)
plt.xlabel("|1- $k_{\\rho}^{geo}$|")
plt.ylabel("residuals total integral")
plt.legend(["E = 3.98 EeV, $\Phi = 90\degree$"])
plt.tight_layout()
#plt.savefig("residuals_ldfvxb_vs_krho_geo.pdf")
plt.show()

plt.scatter(abs(1-krho), Ldfvxvxb)
plt.xlabel("|1- $k_{\\rho}^{geo}$|")
plt.ylabel("residuals total integral")
plt.legend(["E = 3.98 EeV, $\Phi = 90\degree$"])
plt.tight_layout()
#plt.savefig("residuals_ldfvxvxb_vs_krho_geo.pdf")
plt.show()


plt.scatter(dplane[dplane==38815], Ldfvxb[dplane==38815])
