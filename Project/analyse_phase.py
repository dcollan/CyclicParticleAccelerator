import log
import numpy as np
import scipy.constants as const
import matplotlib.pyplot as plt
import matplotlib
import copy
from EMField import EMField
from ProtonBunch import ProtonBunch

"""
This file will analyse several different electric field phase shifts in a cyclotron. It will then plot the 
spread of the y-position of a proton bunch as well as the bunch's spread of kinetic energy against time.

The 'phase_data.npz' contains the position and kinetic energy spreads for a proton bunch containing
100 protons over approximately 10 revolutions. If you do not have the file in the same location as
this file, the simulation will run and generate it for you. I would recommend you reduce the size of the
bunch and the simulation's duration otherwise the run time will be quite long.
"""

phase_keys = [
    r'$0$', r'$\frac{1}{8} \pi$', r'$\frac{1}{4} \pi$', r'$\frac{3}{8} \pi$',
    r'$\frac{1}{2} \pi$', r'$\frac{5}{8} \pi$', r'$\frac{3}{4} \pi$', r'$\frac{7}{8} \pi$',
    r'$\pi$', r'$\frac{9}{8} \pi$', r'$\frac{5}{4} \pi$', r'$\frac{11}{8} \pi$',
    r'$\frac{3}{2} \pi$', r'$\frac{13}{8} \pi$', r'$\frac{7}{4} \pi$', r'$\frac{15}{8} \pi$'
]
phase_values = [i*np.pi/8 for i in range(0,16)]
phase_dict = {phase_keys[i]:phase_values[i] for i in range(len(phase_keys)) if np.cos(phase_values[i])>=-0.00001}
phases = list(phase_dict.values())

def generate_file(phases):
    field = EMField([50000,0,0], [0,0,0.07]) 
    inital_bunch = ProtonBunch(10**(6),100,10**(-12))
    field.setFrequency(inital_bunch)

    deltaT, duration = 10**(-9), 10**(-6)*10
    timeSeries = []
    positionSpread = []
    energySpread = []
    for _ in range(len(phases)):
        positionSpread.append([])
        energySpread.append([])

    log.logger.info('starting simulation')

    for (angle,loop_1,loop_2) in zip(phases,positionSpread,energySpread):
        time = 0
        field.phase = angle # apply the new phase shift to the electric field
        protons = copy.deepcopy(inital_bunch) # reset the proton bunch
        log.logger.info('phase currently being investigated: %s radians' % angle)
        while time <= duration:
            time += deltaT
            if angle == phases[0]:
                timeSeries.append(time)
            field.getAcceleration(protons.bunch, time, deltaT)
            protons.update(deltaT,field,time,2)
            temp_spread = copy.deepcopy(protons.positionSpread())
            temp_energy = copy.deepcopy(protons.energySpread())
            loop_1.append(temp_spread)
            loop_2.append(temp_energy)
    log.logger.info('simulation finished')

    log.logger.info('writing to file')
    np.savez('phase_data', time=timeSeries, positions=positionSpread, energies=energySpread)
    log.logger.info('file written')
    return np.load('phase_data.npz',allow_pickle=True)

def timeTOrev(t):
    T = 2*const.pi*const.m_p / (const.e*0.07)
    return t*10**(-6)/T

def revTOtime(r):
    T = 2*const.pi*const.m_p / (const.e*0.07)
    return r*T*10**(6)

try:
    simulation_data = np.load('phase_data.npz',allow_pickle=True)
except FileNotFoundError:
    simulation_data = generate_file(phases)

times = simulation_data['time']
positions = simulation_data['positions']
energies = simulation_data['energies']
timeSeries = [i*10**(6) for i in times]

fig, ax = plt.subplots()
for (i,phase) in zip(positions,phase_dict.keys()):
    y_spread = [j[1] for j in i]
    ax.plot(timeSeries,y_spread,label='Phase: '+ phase)
secax = ax.secondary_xaxis('top', functions=(timeTOrev,revTOtime))
secax.set_xlabel('Revolutions')
ax.set_xlabel(r'time [$\mu$s]')
ax.set_ylabel(r'$\sigma_y$ [m]')
ax.legend(loc='upper left')
secax.tick_params(direction='in')
ax.tick_params(which='both',direction='in',right=True,top=False)

fig, ax = plt.subplots()
for (i,phase) in zip(energies,phase_dict.keys()):
    ax.plot(timeSeries,[j*10**(-6) for j in i],label='Phase: '+ phase)
secax = ax.secondary_xaxis('top', functions=(timeTOrev,revTOtime))
secax.set_xlabel('Revolutions')
ax.set_xlabel(r'time [$\mu$s]')
ax.set_ylabel(r'Kinetic Energy Spread ($1\sigma$) [MeV]')
ax.legend(loc='upper left')
secax.tick_params(direction='in')
ax.tick_params(which='both',direction='in',right=True,top=False)

plt.figure('Phases')
theta = np.linspace(0,2*np.pi,10000)
cosine = [np.cos(x) for x in theta]
phase_points = [np.cos(x) for x in phases]
plt.plot(theta, cosine, label=r'$cos(\theta)$', color='red',zorder=1)
plt.scatter(phases, phase_points, label='Electric field phases',color='black',zorder=10)
for x,y,name in zip(phases,phase_points,phase_dict.keys()):
    if phases.index(x) < 5:
        plt.text(x+0.15,y+0.1,name)
    else:
        plt.text(x-0.25,y+0.1,name)
plt.ylim(-1.25,1.25)
plt.xlabel(r'$\theta$ (radians)')
plt.legend(loc='lower left')
plt.tick_params(which='both',direction='in',right=True,top=True)

plt.show()