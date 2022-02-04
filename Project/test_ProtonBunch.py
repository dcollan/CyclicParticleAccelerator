import pytest
import scipy.constants as const
import numpy as np
import math
from unittest.mock import patch
from ProtonBunch import ProtonBunch
from EMField import EMField
from ChargedParticle import ChargedParticle

proton_1 = ChargedParticle('proton-1', const.m_p, const.e, [50,-20,10], [1000,-2000,3000])
proton_2 = ChargedParticle('proton-2', const.m_p, const.e, [-75,-30,80], [4000,5000,-6000])
bunch_of_protons = ProtonBunch(3,2) # initialise a ProtonBunch
bunch_of_protons.bunch = [proton_1,proton_2] # overwrite the bunch attribute to replace pseudo-random ChargedParticle objects

def test_averagePosition():
    """
    This will test that the averagePosition method correctly returns a 3D numpy array, containing
    the mean position in x, y and z for a Bunch object.
    """
    calculated_position = np.array([-12.5,-25,45],dtype=float)
    assert np.array_equal(calculated_position, bunch_of_protons.averagePosition())

def test_averageVelocity():
    """
    This will test that the averageVelocity method correctly returns a 3D numpy array, containing
    the mean velocity in x, y and z for a Bunch object.
    """
    calculated_velocity = np.array([2500,1500,-1500],dtype=float)
    assert np.array_equal(calculated_velocity, bunch_of_protons.averageVelocity())

def test_momentum():
    """
    This will test that the momentum function in the Bunch ABC is correctly returning the average
    kinetic energy for a particle in the bunch. If method is called and True is passed into it, then
    then this will also test that the method is correctly returning the total momentum of the bunch.
    """
    calculated_average_momentum = const.m_p * np.array([2500,1500,-1500],dtype=float)
    calculated_total_momentum = const.m_p * np.array([5000,3000,-3000],dtype=float)

    assert np.allclose(calculated_average_momentum,bunch_of_protons.momentum(),rtol=10**(-9))
    assert np.allclose(calculated_total_momentum,bunch_of_protons.momentum(True),rtol=10**(-9))

def test_KineticEnergy():
    """
    This will test that the KineticEnergy function in the Bunch ABC is correctly returning the 
    average kinetic energy for a particle in the bunch. If method is called and True is passed into 
    it, then then this will also test that the method is correctly returning the total kinetic 
    energy of the bunch.
    """
    calculated_average_kinetic = 0.237502831 # units: eV
    calculated_total_kinetic = 0.475005663 # units: eV

    assert calculated_average_kinetic == pytest.approx(bunch_of_protons.KineticEnergy())
    assert calculated_total_kinetic == pytest.approx(bunch_of_protons.KineticEnergy(True))

@patch.object(ProtonBunch, 'distributeEnergies')
def test_assignVelocities(mock_energies):
    """
    This will test that the assignVelocities method correctly calculates a linear speed given a
    kinetic energy. It also tests that the function returns a list of velocity vectors where the
    calculated speeds are the x-components in these velocity vectors.
    """
    mock_energies.return_value=np.array([10**8,0.75]) # return these values instead of values sampled from normal distribution
    velocity_1, velocity_2 = [128369776.9,0,0], [11986.4508,0,0]
    calculated_velocities = [velocity_1, velocity_2]
    assigned_velocities = bunch_of_protons.assignVelocities()
    assert np.allclose(calculated_velocities, assigned_velocities, rtol=10**(-4))

@patch.object(ProtonBunch, 'assignVelocities')
def test_gamma(mock_velocities):
    """
    This will test that the Lorentz factor of a particle bunch is correctly calculated using the average
    velocity of a particle in the bunch.
    """
    mock_velocities.return_value = [np.array([0.6*const.c,0,0]), np.array([0.7*const.c,0,0]), np.array([0.8*const.c,0,0])]
    protons = ProtonBunch(1)
    expected_gamma = 1.400280084
    assert protons.gamma() == pytest.approx(expected_gamma)

def test_positionSpread():
    """
    This will test that the positionSpread method (numpy standard deviation) is correctly returning the standard
    deviation of all the particles' positions in a bunch.
    """
    calculated_spread = np.array([62.5,5,35],dtype=float)
    assert np.array_equal(calculated_spread, bunch_of_protons.positionSpread())

def test_energySpread():
    """
    This will test that the energySpread (numpy standard deviation) is correctly returning the standard
    deviation of all the particles' kinetic energies in a bunch
    """
    calculated_spread = 0.164390625025
    assert calculated_spread == pytest.approx(bunch_of_protons.energySpread(),rel=1e-3)

@patch.object(ProtonBunch, 'assignVelocities')
@patch.object(ProtonBunch, 'assignPositions')
def test_createBunch(mock_positions, mock_velocities):
    """
    Tests that a Bunch object is successfully created. The patched methods above mock out
    the normally sampled positions and velocites so that a created bunch can be compared the
    expected one.
    """
    mock_positions.return_value = [[1,2,3], [4,5,6]]
    mock_velocities.return_value = [[10,20,30], [40,50,60]]
    particle_1 = ChargedParticle('proton-1', const.m_p, const.e, [1,2,3], [10,20,30])
    particle_2 = ChargedParticle('proton-2', const.m_p, const.e, [4,5,6], [40,50,60])
    expected_bunch = [particle_1, particle_2]
    actual_bunch = bunch_of_protons.createBunch()
    assert np.array_equal(expected_bunch, actual_bunch)

@pytest.mark.parametrize('test_input,expected',[(np.array([0,0,0],dtype=float),1),
                        (np.array([10,10,10],dtype=float),100),
                        (np.array([1.05,0,0],dtype=float),1),
                        (np.array([-1.05,0,0],dtype=float),1)])
def test_adaptiveStep(test_input,expected):
    """
    This will test that the step size in any integrator correctly reduces when any of the particles
    in the bunch are either, in the electric field, or within ±10% of the electric field's
    boundaries.
    """
    deltaT = 100
    protons = ProtonBunch(100,1)
    protons.bunch[0].position = test_input
    field = EMField(ElectricFieldWidth=[-1,1])
    assert protons.adaptiveStep(deltaT,field) == expected