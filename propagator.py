# -*- coding: utf-8 -*-
"""
Created on Mon Dec 05 14:49:51 2016

@author: ariasoca
"""

import numpy as np
import time
import random
import matplotlib.pyplot as plt
start_time = time.time()

## Parameters ##
# General parameter #
k1 = 50.0 #simulation box
deltat = 10**-3 #timestep
l = 512 #mesh points
dx = k1/l #mesh-grid in x-axis
w = 0.5/dx**2 #hopping parameter

# Gaussian wavefunction parameter #
x0 = 12.5
s0 = 3.0

# Potential barrier parameter #
e0 = 5.0
bh = 5.0
bw = 1.0
eh = 50.0

T = 2000 # Simulation time 

## Arrays ##
psi_c = np.zeros(l,dtype=np.complex_) #array to save initial wavefunction
pot = np.zeros(l,dtype=np.complex_) #array to save potential
pwrk = np.zeros(l,dtype=np.complex_) #array to save potential propagator
wrk = np.zeros(l,dtype=np.complex_) #array to save propagated wavefunvtion

pos = []

## Create Initial Wavefunction ##
def entry():
    for i in range (l):
        x = dx*(i+1)-x0
        gauss = np.exp(-0.25*x*x/(s0*s0))
        a = gauss*np.cos(np.sqrt(2.0*e0)*x)
        b = gauss*np.sin(np.sqrt(2.0*e0)*x)
        psi_c[i] = complex(a,b)
        pos.append(dx*(i+1))
## Create Normalization Function ##
def normalize(vr):
    nor = 0.0
    for i in range (l):
        nor += vr.real[i]**2+vr.imag[i]**2
    nor *= dx
    norm = 1.0/np.sqrt(nor)
    for i in range (l):
        vr[i] = vr[i]*norm
## Create potential barrier ##
def P():
    for i in range (l):
        x = dx*(i+1)
        if i==0 or i==l:
            pot[i] = eh
        elif x>(0.5*(k1-bw)) and x<(0.5*(k1+bw)):
            pot[i] = bh
        else:
            pot[i] = 0.0

## Potential propagator ##
def pprop(vc,dt):
    for i in range(l):
        pwrk[i] = np.exp(-1.0j*dt*pot[i])
        wr = vc[i]*pwrk[i]
        vc[i] = wr

## Propagation through real space ##
def H1(vc,dt,s):
    c1 = np.cos(dt*(-w))
    s1 = -1.0J*np.sin(dt*(-w))
    if l%2 == 0:
        ran = l//2
    elif l%2 != 0:
        ran = l-((l//2)+s)
    for i in range (ran):
        i0 = (i*2)+s
        i1 = i0+1
        if i1 == l:
            i1 = 0
        a = c1*vc[i0]+s1*vc[i1]
        b = s1*vc[i0]+c1*vc[i1]
        vc[i0] = a
        vc[i1] = b

## On-site potetntial ##
def H2(vc,dt):
    for i in range(l):
        ea = np.exp(-1.0j*dt*2.0*w)
        wr = vc[i]*ea
        vc[i] = wr

## Time Evolution ##
def evolution():
    t0 = 0.0
    wrk = np.copy(psi_c)
    for i in range (T*2):
        if i == 0:
            dt = 0.0
        else:
            dt = deltat
        # Propagation in x-axis #
        pprop(wrk,dt)
        H1(wrk,dt,0)
        H1(wrk,dt,1)
        H2(wrk,dt)
        normalize(wrk)
        t0 = t0+deltat
    plt.plot(pos,wrk.real,'--') #plot of wavefunction after propagation
    plt.plot(pos,psi_c.real) #plot of initial wavefunction
    plt.show()
    
## Main Initialization ##
def main_random():
    entry()
    P()
    normalize(psi_c)
    evolution()
main_random()
print("--- %s seconds ---" % (time.time() - start_time))
