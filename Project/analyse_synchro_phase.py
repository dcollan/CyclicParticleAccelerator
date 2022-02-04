import log
import numpy as np
import scipy.constants as const
import matplotlib.pyplot as plt
import matplotlib
import copy
from EMField import EMField
from ProtonBunch import ProtonBunch

"""
This file will analyse several different electric field phase shifts in a synchrocyclotron. It will then 
plot the spread of the y-position of a proton bunch as well as the bunch's spread of kinetic energy against 
time. It will then fit a linear curve to the energy spreads so the variation of the spreads against time 
can be more easily visualised.

The 'phase_synchro_data.npz' contains the position and kinetic energy spreads for a proton bunch containing
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
    field = EMField([500*10**3,0,0], [0,0,2.82], [-0.029,0.029]) 
    protons = ProtonBunch(500*10**(6),100)
    inital_bunch = copy.deepcopy(protons)

    deltaT, duration = 10**(-11), 2.78*10**(-8)*10
    timeSeries = []
    positionSpread = []
    energySpread = []
    for _ in range(len(phases)):
        positionSpread.append([])
        energySpread.append([])
        timeSeries.append([])

    log.logger.info('starting simulation')

    for (angle,loop_1,loop_2,loop_3) in zip(phases,positionSpread,energySpread,timeSeries):
        time = 0
        field.phase = angle # apply the new phase shift to the electric field
        protons = copy.deepcopy(inital_bunch) # reset the proton bunch
        log.logger.info('phase currently being investigated: %s radians' % angle)
        while time <= duration:
            dt = protons.adaptiveStep(deltaT,field)
            time += dt
            field.setFrequency(protons)
            field.getAcceleration(protons.bunch, time, dt)
            protons.update(dt,field,time)
            temp_spread = copy.deepcopy(protons.positionSpread())
            temp_energy = copy.deepcopy(protons.energySpread())
            loop_1.append(temp_spread)
            loop_2.append(temp_energy)
            loop_3.append(time)
    log.logger.info('simulation finished')

    log.logger.info('writing to file')
    np.savez('phase_synchro_data', time=timeSeries, positions=positionSpread, energies=energySpread)
    log.logger.info('file written')
    return np.load('phase_synchro_data.npz',allow_pickle=True)

try:
    simulation_data = np.load('phase_synchro_data.npz',allow_pickle=True)
except FileNotFoundError:
    simulation_data = generate_file(phases)

times = simulation_data['time']
positions = simulation_data['positions']
energies = simulation_data['energies']
timeSeries = list(map(lambda x: [i*10**(6) for i in x],times))

plt.figure('Synchro Phase Position Spreads')
for (i,phase,time) in zip(positions,phase_dict.keys(),timeSeries):
    y_spread = [j[1] for j in i]
    plt.plot(time,y_spread,label='Phase: '+ phase)
plt.xlabel(r'time [$\mu$s]')
plt.ylabel(r'$\sigma_y$ [m]')
plt.legend(loc='upper left')
plt.tick_params(which='both',direction='in',right=True,top=True)

plt.figure('Synchro Phase Energy Spreads')
for (i,phase,time) in zip(energies,phase_dict.keys(),timeSeries):
    plt.plot(time,[j*10**(-6) for j in i],label='Phase: '+ phase)
plt.xlabel(r'time [$\mu$s]')
plt.ylabel(r'Kinetic Energy Spread ($1\sigma$) [MeV]')
plt.legend(loc='upper left')
plt.tick_params(which='both',direction='in',right=True,top=True)

plt.figure('linear synchro phase energy')
for (i,phase,time) in zip(energies,phase_dict.keys(),timeSeries):
    adjusted_energies = [j*10**(-6) for j in i]
    linear_fit = np.polynomial.polynomial.Polynomial.fit(time,adjusted_energies,1)
    line = np.poly1d(linear_fit.coef)
    a, b = line[0], line[1]
    spread = [a*t+b for t in time]
    plt.plot(time,spread,label='Phase: '+phase)
plt.xlabel(r'time [$\mu$s]')
plt.ylabel(r'Kinetic Energy Spread ($1\sigma$) [MeV]')
plt.legend(loc='upper left')
plt.tick_params(which='both',direction='in',right=True,top=True)

plt.show()