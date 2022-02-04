import log
import numpy as np
import scipy.constants as const
import matplotlib.pyplot as plt
import matplotlib
import copy
from EMField import EMField
from ProtonBunch import ProtonBunch

"""
This file will analyse the how time step effects the accuracy of the simulation. It will record a cyclotron
simulation using velocity Verlet method using three different time steps. It will then plot the fractional 
kinetic energy and momentum against time for each time step.

If you do not have the data file: timestep_data.npz which contains the kinetic energy and momentum data then
the file will be generated for you. I would recommend you reduce the size of the bunch and the simulation's 
duration otherwise the run time will be quite long.
"""

timesteps = [10**(-8), 4.5*10**(-9), 10**(-9)]
step_strings = ['1e-8 s', '5e-9 s', '1e-9 s']
step_dict = {step_strings[i]:timesteps[i] for i in range(len(timesteps))}
field = EMField([0,0,0], [0,0,0.07], [0,0])
protons = ProtonBunch(10**(6),1)
field.setFrequency(protons)

def generate_file(protons,field):
    inital_bunch = copy.deepcopy(protons)

    duration = 10**(-6)*100
    timeSeries = []
    kineticEnergies = []
    momenta = []
    timeSeries = []
    for _ in range(len(timesteps)):
        kineticEnergies.append([])
        momenta.append([])
        timeSeries.append([])

    log.logger.info('starting simulation')

    for (step,loop_1,loop_2,loop_3) in zip(timesteps,timeSeries,kineticEnergies,momenta):
        time = 0
        deltaT = step
        protons = copy.deepcopy(inital_bunch)
        log.logger.info('time step currently being investigated: %s s' % step)
        while time <= duration:
            time += deltaT
            field.getAcceleration(protons.bunch, time, deltaT)
            protons.update(deltaT,field,time,3)
            temp_energy = copy.deepcopy(protons.KineticEnergy())
            temp_momentum = copy.deepcopy(protons.momentum())
            loop_1.append(time)
            loop_2.append(temp_energy)
            loop_3.append(np.linalg.norm(temp_momentum))
    log.logger.info('simulation finished')

    log.logger.info('writing to file')
    np.savez('timestep_data', time=timeSeries, energies=kineticEnergies, momenta=momenta)
    log.logger.info('file written')
    return np.load('timestep_data.npz',allow_pickle=True)

def timeTOrev(t):
    T = 2*const.pi*const.m_p / (const.e*field.magneticMag())
    return t*10**(-6)/T

def revTOtime(r):
    T = 2*const.pi*const.m_p / (const.e*field.magneticMag())
    return r*T*10**(6)

try:
    simulation_data = np.load('timestep_data.npz',allow_pickle=True)
except FileNotFoundError:
    simulation_data = generate_file(protons,field)

time = simulation_data['time']
energies = simulation_data['energies']
momenta = simulation_data['momenta']
timeSeries = []
for series in time:
    timeSeries.append([])
    for i in series:
        timeSeries[len(timeSeries)-1].append(i*10**(6))

fig, ax = plt.subplots()
for i,time,step in zip(momenta,timeSeries,step_dict.keys()):
    fractional_momenta = [j/i[0] for j in i]
    ax.plot(time, fractional_momenta,label=step)
ax.set_xlabel(r'Time [$\mu$s]')
ax.set_ylabel(r'Fractional $\|\vec{p}\|$')
secax = ax.secondary_xaxis('top', functions=(timeTOrev,revTOtime))
secax.set_xlabel('Revolutions')
ax.legend(loc='lower left')
ax.ticklabel_format(useOffset=False)
ax.tick_params(which='both',direction='in',right=True,top=False)
secax.tick_params(direction='in')

fig, ax = plt.subplots()
for i,time,step in zip(energies,timeSeries,step_dict.keys()):
    fractional_energies = [j/i[0] for j in i]
    ax.plot(time, fractional_energies,label=step)
ax.set_xlabel(r'Time [$\mu$s]')
ax.set_ylabel(r'Fractional $E_k$')
secax = ax.secondary_xaxis('top', functions=(timeTOrev,revTOtime))
secax.set_xlabel('Revolutions')
ax.legend(loc='lower left')
ax.ticklabel_format(useOffset=False)
ax.tick_params(which='both',direction='in',right=True,top=False)
secax.tick_params(direction='in')

plt.show()