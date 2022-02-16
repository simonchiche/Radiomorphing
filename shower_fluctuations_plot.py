#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 02:46:12 2021

@author: chiche
"""


import numpy as np
import matplotlib.pyplot as plt


w1, ldf1 = np.loadtxt("LDF_data1.txt", unpack = True)
w2, ldf2 = np.loadtxt("LDF_data2.txt", unpack = True)
w3, ldf3 = np.loadtxt("LDF_data3.txt", unpack = True)


plt.scatter(w1, ldf1)
plt.scatter(w2, ldf2, marker = 'v')
plt.scatter(w3, ldf3, marker = '+')
plt.xlabel("$\omega$ [Deg.]")
plt.ylabel("Lateral distribution function (LDF)", fontsize = 13)
plt.legend(["Radio Morphing 1", "Radio Morphing 2", "Radio Morphing 3"], fontsize = 11)
plt.tight_layout()
#plt.savefig("LDF_shower_fluctuations.pdf")
plt.show()

