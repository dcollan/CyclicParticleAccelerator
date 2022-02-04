import pytest
from ChargedParticle import ChargedParticle
from Particle import Particle
from EMField import EMField
import scipy.constants as const
import numpy as np

def test_exceedingC():
    """
    This test will check that if the user tries to create a particle whose speed exceeds the speed
    of light in a vacuum, a ValueError is raised. It will then check that if a particle's speed is
    ever equal to or greater than the speed of the light in vacuum, the selected integrator will 
    raise a ValueError.
    """
    with pytest.raises(ValueError):
        Particle('proton', const.m_p, [0,0,0], [300000000,0,0])
    proton = ChargedParticle()
    field = EMField()
    proton.velocity = np.array([3*10**(8),0,0],dtype=float)
    with pytest.raises(ValueError):
        proton.euler(0.1)
    with pytest.raises(ValueError):
        proton.eulerCromer(0.1)
    with pytest.raises(ValueError):
        proton.velocityVerlet(0.1,field,0)
    with pytest.raises(ValueError):
        proton.RungeKutta4(0.1,field,0)

def test_gamma():
    """
    This test checks that the gamma function correctly calculates the Lorenzt factor. Using the 
    approximation function accounts for the floating point error. It will also test that if a
    singularity occurs (particle's speed equal to the speed of light) or if a math domain error occurs
    (particle's velocity is greater than the speed of light), the correct error is raised.
    """
    calculated_value = 1.342384701 # calculated by hand using c = 299792458 [ms^(-1)]
    proton = Particle('proton', const.m_p, [0,0,0], [200000000,0,0])
    assert calculated_value == pytest.approx(proton.gamma())
    proton.velocity = np.array([1.1*const.c,0,0])
    with pytest.raises(ValueError):
        proton.gamma()
    proton.velocity = np.array([const.c,0,0])
    with pytest.raises(ZeroDivisionError):
        proton.gamma()

@pytest.mark.parametrize('test_input,expected',[(Particle('proton', const.m_p, [0,0,0], [3000,0,0]),7.526911023*10**(-21)),
                        (Particle('proton', const.m_p, [0,0,0], [200000000,0,0]),5.146992568*10**(-11))])
def test_KineticEnergy(test_input,expected):
    """
    This test checks that the Kinetic Energy method correctly returns the kinetic energy 
    of a particle in both the relativistic and non-relativistic case.
    """    
    assert expected == pytest.approx(test_input.KineticEnergy(),rel=10**(-4))

@pytest.mark.parametrize('test_input,expected',[(Particle('proton', const.m_p, [0,0,0], [3000,0,0]),const.m_p*np.array([3000,0,0],dtype=float)),
                        (Particle('proton', const.m_p, [0,0,0], [200000000,0,0]),const.m_p*1.342384701*np.array([200000000,0,0],dtype=float))])
def test_momentum(test_input,expected):
    """
    This test checks that the momentum method correctly returns the momentum 
    of a particle in both the relativistic and non-relativistic case.
    """    
    assert np.allclose(test_input.momentum(),expected,rtol=10**(-10))

def test_Euler():
    """
    This test checks that the Euler method is correctly updating a particle's position and
    velocity.
    """
    proton = Particle('proton', const.m_p, [0,0,0], [-100,200,300], [10,20,-30])
    deltaT = 0.5
    calculated_position = np.array([-50,100,150], dtype=float)
    calculated_velocity = np.array([-95,210,285], dtype=float)
    proton.euler(deltaT)
    assert np.array_equal(calculated_position,proton.position)
    assert np.array_equal(calculated_velocity,proton.velocity)

def test_EulerCromer():
    """
    This test checks that the Euler Cromer method is correctly updating a particle's position and
    velocity.
    """
    proton = Particle('proton', const.m_p, [0,0,0], [-100,200,300], [10,20,-30])
    deltaT = 0.5
    calculated_velocity = np.array([-95,210,285], dtype=float)
    calculated_position = np.array([-47.5,105,142.5], dtype=float)
    proton.eulerCromer(deltaT)
    assert np.array_equal(calculated_velocity,proton.velocity)
    assert np.array_equal(calculated_position,proton.position)

def test_velocityVerlet():
    """
    This tests that the velocity Verlet update method is correctly updating a particle's
    position and velocity.
    """
    proton = ChargedParticle(Position=[0,0,0],Velocity=[3000,0,0],Acceleration=[0,-9000,0],Charge=2,Mass=1)
    field = EMField(MagneticField=[0,0,1.5],ElectricFieldWidth=[0,0])
    proton.velocityVerlet(0.1,field,0)
    expected_position = [300., -45., 0.]
    expected_velocity = [2865., -879.75, 0.]
    assert np.array_equal(expected_position,proton.position)
    assert np.allclose(expected_velocity,proton.velocity,atol=0.01)

def test_RungeKutta4():
    """
    This tests that the fourth order Runge-Kutta update method is correctly updating a particle's
    position and velocity.
    """
    proton = ChargedParticle(Position=[0,0,0],Velocity=[3000,0,0],Acceleration=[0,-9000,0],Charge=2,Mass=1)
    field = EMField(MagneticField=[0,0,1.5],ElectricFieldWidth=[0,0])
    proton.RungeKutta4(0.1,field,0)
    expected_position = [295.5, -44.6625, 0.]
    expected_velocity = [2866.0125, -886.5, 0.]
    assert np.allclose(expected_position,proton.position,atol=0.0001)
    assert np.allclose(expected_velocity,proton.velocity,atol=0.01)