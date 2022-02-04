import pytest
from ChargedParticle import ChargedParticle
from EMField import EMField
from ProtonBunch import ProtonBunch
import scipy.constants as const
import numpy as np

@pytest.mark.parametrize('test_input,expected',
                        [(ChargedParticle(Charge=const.e,Mass=const.m_p,Velocity=[0.01*const.c,0,0]),11694.31882),
                        (ChargedParticle(Charge=const.e,Mass=const.m_p,Velocity=[0.5*const.c,0,0]), 10122.64424)])
def test_frequency(test_input,expected):
    """
    This function tests that the (sychro)cyclotron frequency is correctly being calculated for both non-relativistic
    and relativistic particle bunches.
    """
    field = EMField([-0.3,0.1,0.2], [6*10**(-5),-7*10**(-5),8*10**(-5)])
    protons = ProtonBunch(100,1)
    protons.bunch = [test_input]
    field.setFrequency(protons)
    assert expected == pytest.approx(field.frequency,rel=0.001)


@pytest.mark.parametrize('test_input,expected',
                        [((ChargedParticle('proton-1', const.m_p, const.e, [0,0,0], [2000,0,0]),EMField([0.1,0,0])), [0.1,0,0]),
                        ((ChargedParticle('proton-1', const.m_p, const.e, [10,0,0], [2000,0,0]),EMField([0.1,0,0])), [0,0,0]),
                        ((ChargedParticle('proton-1', const.m_p, const.e, [10,0,0], [2000,-4000,6000]),EMField([0,0,0], [6*10**(-5),-7*10**(-5),8*10**(-5)])), [0.1,0.2,0.1]),
                        ((ChargedParticle('proton-1', const.m_p, const.e, [0,0,0], [2000,-4000,6000]),EMField([0,0,0], [6*10**(-5),-7*10**(-5),8*10**(-5)])), [0,0,0])])
def test_getAcceleration(test_input,expected):
    # BUG if you run the tests, they pass as expected but if you debug the tests you get an 
    # unbound local error in the EMField getAcceleration method but the tests still pass?
    """
    This function will test the getAcceleration method in the EMField class, it will check that in
    the acceleration is correctly calculated. It will also check that the electric field is being 
    correctly constrained. The four test cases are:
        1) An electric field exists within the specified coordinates
        2) The electric field is zero outside of these coordinates.
        3) The magnetic fields exists within the dees.
        4) The magentic field is zero outside the dees.
    """
    proton, field = test_input
    field.getAcceleration([proton],0,0) # get Lorentz force, set time to 0 so the electric field is equal to its magnitude 
    calculated_lorentz = np.array(expected,dtype=float)
    calculated_lorentz *= const.physical_constants['proton charge to mass quotient'][0] # convert to acceleration
    assert np.allclose(calculated_lorentz, proton.acceleration, rtol=10**(-9))