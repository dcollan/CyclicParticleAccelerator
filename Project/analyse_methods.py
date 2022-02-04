import log
import numpy as np
import scipy.constants as const
import matplotlib.pyplot as plt
import matplotlib
import copy
from EMField import EMField
from ProtonBunch import ProtonBunch

"""
This file will load the data from the file generated from the RecordCyclotron.py file and plot the
fractional kinetic energy and momentum of a proton bunch.

If you do not have the file, the RecordCyclotron.py file will be ran and the data file will be generated
for you. The revolution axis on the plots marks theoretical revolutions, not revolutions measured by the 
simulation, see analyse_period.py which shows that meausred and theoretical revolutions are very similar.
"""
try:
    simulation_data= np.load('methods_data.npz',allow_pickle=True)
except FileNotFoundError:
    try:
        simulation_data = np.load('cyclotron_data.npz',allow_pickle=True)
    except FileNotFoundError:
        import RecordCyclotron
        simulation_data = np.load('cyclotron_data.npz',allow_pickle=True)

def timeTOrev(t):
    T = 2*const.pi*const.m_p / (const.e*0.07)
    return t*10**(-6)/T

def revTOtime(r):
    T = 2*const.pi*const.m_p / (const.e*0.07)
    return r*T*10**(6)

timeSeries = simulation_data['time']
time = [t*10**(6) for t in timeSeries]
eulerData = simulation_data['euler']
cromerData = simulation_data['cromer']
verletData = simulation_data['verlet']
RK4Data = simulation_data['rk']

fractional_kinetic_euler = [i.KineticEnergy()/eulerData[0].KineticEnergy() for i in eulerData]
fractional_kinetic_cromer = [i.KineticEnergy()/cromerData[0].KineticEnergy() for i in cromerData]
fractional_kinetic_verlet = [i.KineticEnergy()/verletData[0].KineticEnergy() for i in verletData]
fractional_kinetic_rk4 = [i.KineticEnergy()/RK4Data[0].KineticEnergy() for i in RK4Data]

fractional_momentum_euler = [np.linalg.norm(i.momentum())/np.linalg.norm(eulerData[0].momentum()) for i in eulerData]
fractional_momentum_cromer = [np.linalg.norm(i.momentum())/np.linalg.norm(cromerData[0].momentum()) for i in cromerData]
fractional_momentum_verlet = [np.linalg.norm(i.momentum())/np.linalg.norm(verletData[0].momentum()) for i in verletData]
fractional_momentum_rk4 = [np.linalg.norm(i.momentum())/np.linalg.norm(RK4Data[0].momentum()) for i in RK4Data]

euler_cromer_kinetic = []
euler_cromer_momentum = []
for i,j in zip(eulerData,cromerData):
    euler_cromer_kinetic.append(j.KineticEnergy()/i.KineticEnergy())
    euler_cromer_momentum.append(np.linalg.norm(j.momentum())/np.linalg.norm(i.momentum()))

fig, ax = plt.subplots()
ax.plot(time, fractional_kinetic_euler, label='Euler')
ax.plot(time, fractional_kinetic_cromer, label='Euler-Cromer')
secax = ax.secondary_xaxis('top', functions=(timeTOrev,revTOtime))
secax.set_xlabel('Revolutions')
ax.set_xlabel(r'time [$\mu$s]')
ax.set_ylabel('Fractional Kinetic Energy')
ax.legend()
ax.tick_params(which='both',direction='in',right=True,top=False)
secax.tick_params(direction='in')
ax.ticklabel_format(useOffset=False)

fig, ax = plt.subplots()
ax.plot(time, fractional_momentum_euler, label='Euler')
ax.plot(time, fractional_momentum_cromer, label='Euler-Cromer')
secax = ax.secondary_xaxis('top', functions=(timeTOrev,revTOtime))
secax.set_xlabel('Revolutions')
ax.set_xlabel(r'time [$\mu$s]')
ax.set_ylabel(r'Fractional $\parallel\vec{p}\parallel$')
ax.legend()
ax.tick_params(which='both',direction='in',right=True,top=False)
secax.tick_params(direction='in')
ax.ticklabel_format(useOffset=False)

fig, ax = plt.subplots()
ax.plot(time, fractional_kinetic_verlet, label='Velocity Verlet')
ax.plot(time, fractional_kinetic_rk4, label='4th Order RK')
secax = ax.secondary_xaxis('top', functions=(timeTOrev,revTOtime))
secax.set_xlabel('Revolutions')
ax.set_xlabel(r'time [$\mu$s]')
ax.set_ylabel('Fractional Kinetic Energy')
ax.legend()
ax.tick_params(which='both',direction='in',right=True,top=False)
secax.tick_params(direction='in')
ax.ticklabel_format(useOffset=False)

fig, ax = plt.subplots()
ax.plot(time, fractional_momentum_verlet, label='Velocity Verlet')
ax.plot(time, fractional_momentum_rk4, label='4th Order RK')
secax = ax.secondary_xaxis('top', functions=(timeTOrev,revTOtime))
secax.set_xlabel('Revolutions')
ax.set_xlabel(r'time [$\mu$s]')
ax.set_ylabel(r'Fractional $\parallel\vec{p}\parallel$')
ax.legend()
ax.tick_params(which='both',direction='in',right=True,top=False)
secax.tick_params(direction='in')
ax.ticklabel_format(useOffset=False)

fig, ax = plt.subplots()
ax.plot(time, euler_cromer_kinetic)
secax = ax.secondary_xaxis('top', functions=(timeTOrev,revTOtime))
secax.set_xlabel('Revolutions')
ax.set_xlabel(r'time [$\mu$s]')
ax.set_ylabel(r'$E_{k,Euler} \ / \ E_{k,Euler-Cromer}$')
ax.tick_params(which='both',direction='in',right=True,top=False)
secax.tick_params(direction='in')
ax.ticklabel_format(useOffset=False)

fig, ax = plt.subplots()
ax.plot(time, euler_cromer_momentum)
secax = ax.secondary_xaxis('top', functions=(timeTOrev,revTOtime))
secax.set_xlabel('Revolutions')
ax.set_xlabel(r'time [$\mu$s]')
ax.set_ylabel(r'$\parallel\vec{p}_{Euler}\parallel \ / \ \parallel\vec{p}_{Euler-Cromer}\parallel$')
ax.tick_params(which='both',direction='in',right=True,top=False)
secax.tick_params(direction='in')
ax.ticklabel_format(useOffset=False)

plt.show()