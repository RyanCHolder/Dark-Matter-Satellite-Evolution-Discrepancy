# Here we try to obtain 
import numpy as np
import matplotlib.pyplot as plt
import pdb

hubble = 0.67
rho_crit = 277.5*hubble**2
G = 4.3e-6

M200 = 4e9
sigma = 2.5
c200 = 10**(0.905+sigma*0.11)/(M200*hubble/1.0e12)**0.101
#c200 = 29.5
r200 = (3*M200/(4*np.pi*200*rho_crit))**(1./3)
print('r200:' , r200)
r_s = r200/c200
#c200 = 11.80
rho_s = 200.0/3 * rho_crit * c200**3/(np.log(1+c200)-c200/(1+c200))
#r_s = 20.1
#rho_s = 6.86e6
print('rhos:', rho_s)
print('rmax:', 2.16*r_s)
print('Vmax:', 1.64*r_s*np.sqrt(G*rho_s))
#V_max = 3.38e-3*r_s*np.sqrt(rho_s)
r_cutoff = 50.0 # some cutoff radius
r_decay = 0.3*r_cutoff # free parameter
q = r_cutoff/r_s

#------ Values for NFW -------#
alpha = 1.
beta = 3.
gamma = 1.
#-----------------------------#
delta = r_cutoff/r_decay - (gamma+ beta *q**alpha)/(1+q**alpha)
def IM_integrand(x):
	p1 = 2-gamma
	p2 = (beta-gamma)/alpha
	return x**p1 / (1+x**alpha)**p2

def IMcutoff_integrand(x):
	p2 = (beta-gamma)/alpha
	pre = r_s**3 * q**gamma * (1+q**alpha)**p2
	pre = 1/pre
	return pre * (x/(1-x)+r_cutoff)**2 *(1/(1-x)**2 ) * (1+x/(r_cutoff*(1-x)))**delta * np.exp(-x/(r_decay*(1-x)) ) 
# IMPORANT: calculate cutoff integral from 0 to 0.999

N = 1000
a = 0 # lower band for both integrals are zero
b1 = q # upper bound for first integral
b2 = 0.9999 # upper bound for second integral
h1 = (b1-a)/N
h2 = (b2-a)/N

I_M = 0.5*IM_integrand(b1)
for i in range(1,N):
	I_M += IM_integrand(i*h1)
I_M *=h1

I_Mcutoff = 0.5*IMcutoff_integrand(b2)
for i in range(1,N):
	I_Mcutoff += IMcutoff_integrand(i*h2)
I_Mcutoff *=h2

M_tot = 4*np.pi*r_s**3*rho_s*(I_M+I_Mcutoff)
print('M_200=', M200)
print('M_tot=',  M_tot)
print('rho_s = ', rho_s)
print('r_s = ', r_s)
print('r_cutoff = ', r_cutoff)
#print r200
print('c200 = ', c200)
##### Stellar Mass
def M_star(M200, z):
  zfactor = z/(1.+z)
  M1 = 11.590 + 1.195*zfactor
  M1 = 10**M1
  N = 0.0351 - 0.0247*zfactor
  beta = 1.376 - 0.826*zfactor
  gamma = 0.608 + 0.329*zfactor
  m = 2*N*M200/( (M1/M200)**beta + (M200/M1)**gamma )
  return m
#print "Stellar Mass:", M_star(M200, 0)
#pdb.set_trace()
