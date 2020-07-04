from atm76 import ATM76
from atm76.tables import (ALTITUDE, PRESSURE, TEMPERATURE, DENSITY, DELTA, THETA, SIGMA,
                          SPEED_OF_SOUND, VISCOSITY, KINEMATIC_VISCOSITY)
import numpy as np
from scipy.optimize import check_grad, approx_fprime


def rsquare(Y_pred, Y_true):
    """
    Compute R-square for a single response.

    NOTE: If you have more than one response, then you'll either have to modify this method to handle many responses at
          once or wrap a for loop around it (i.e. treat one response at a time).

    Arguments:
    Y_pred -- predictions,  numpy array of shape (K, m) where n_y = no. of outputs, m = no. of examples
    Y_true -- true values,  numpy array of shape (K, m) where n_y = no. of outputs, m = no. of examples

    Return:
    R2 -- the R-square value,  numpy array of shape (K, 1)
    """
    epsilon = 1e-8  # small number to avoid division by zero
    Y_bar = np.mean(Y_true)
    SSE = np.sum(np.square(Y_pred - Y_true), axis=1)
    SSTO = np.sum(np.square(Y_true - Y_bar) + epsilon, axis=1)
    R2 = 1 - SSE / SSTO
    return R2


def test_singleton():

    atmos_1 = ATM76()
    atmos_2 = ATM76()

    assert id(atmos_1) == id(atmos_2)


def test_values():
    atm = ATM76()
    alt = np.array(ALTITUDE).reshape((1, -1))
    R2_k_mu = rsquare(atm.kinematic_viscosity(alt).reshape((1, -1)), np.array(KINEMATIC_VISCOSITY).reshape((1, -1)))
    R2_mu = rsquare(atm.viscosity(alt).reshape((1, -1)), np.array(VISCOSITY).reshape((1, -1)) * 10**(-6))
    R2_P = rsquare(atm.pressure(alt).reshape((1, -1)), np.array(PRESSURE).reshape((1, -1)))
    R2_T = rsquare(atm.temperature(alt).reshape((1, -1)), np.array(TEMPERATURE).reshape((1, -1)))
    R2_rho = rsquare(atm.density(alt).reshape((1, -1)), np.array(DENSITY).reshape((1, -1)))
    R2_sos = rsquare(atm.speed_of_sound(alt).reshape((1, -1)), np.array(SPEED_OF_SOUND).reshape((1, -1)))

    assert 0.95 < R2_k_mu
    assert 0.95 < R2_mu
    assert 0.95 < R2_P
    assert 0.95 < R2_T
    assert 0.95 < R2_rho
    assert 0.95 < R2_sos


def test_partials():
    atm = ATM76()

    error = check_grad(func=lambda x: atm.temperature(x),
                       grad=lambda x: atm.grad_temperature(x),
                       x0=np.zeros(1,))

    assert np.allclose(error, 0., atol=1e-5)

    error = check_grad(func=lambda x: atm.density(x),
                       grad=lambda x: atm.grad_density(x),
                       x0=np.zeros(1,))

    assert np.allclose(error, 0., atol=1e-5)

    error = check_grad(func=lambda x: atm.speed_of_sound(x),
                       grad=lambda x: atm.grad_speed_of_sound(x),
                       x0=np.zeros(1,))

    assert np.allclose(error, 0., atol=1e-5)

    error = check_grad(func=lambda x: atm.kinematic_viscosity(x),
                       grad=lambda x: atm.grad_kinematic_viscosity(x),
                       x0=np.zeros(1,))

    assert np.allclose(error, 0., atol=1e-5)

    error = check_grad(func=lambda x: atm.viscosity(x),
                       grad=lambda x: atm.grad_viscosity(x),
                       x0=np.zeros(1,))

    assert np.allclose(error, 0., atol=1e-5)

    error = check_grad(func=lambda x: atm.pressure(x),
                       grad=lambda x: atm.grad_pressure(x),
                       x0=np.zeros(1,))

    assert np.allclose(error, 0., atol=0.01, rtol=1e-6)


if __name__ == "__main__":
    test_singleton()
    test_values()
    test_partials()
