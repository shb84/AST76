# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
#  Copyright (c) 2019, Georgia Tech Applied Research Corporation.
#
#  Distributed under the terms as defined in the file LICENSE.txt, distributed
#  with this software.
# -----------------------------------------------------------------------------

"""
This module provides standard atmosphere model.  It is based on the data
contained in tables 1 and 2 from `Properties Of The U.S. Standard Atmosphere
1976 <http://www.pdas.com/atmos.html>`_ The data was used to train partial-
Enhanced Neural Networks (GENN), such that all predictions and partial are
pure analytical functions (as opposed to linear table interpolation).
"""
from genn import GENN
from atm76.surrogates import (load_surrogate, GENN_PRESSURE, GENN_TEMPERATURE, GENN_DENSITY,
                              GENN_SPEED_OF_SOUND, GENN_VISCOSITY, GENN_KINEMATIC_VISCOSITY)
import numpy as np


# Import trained surrogate models
genn_pressure = load_surrogate(GENN_PRESSURE)
genn_temperature = load_surrogate(GENN_TEMPERATURE)
genn_density = load_surrogate(GENN_DENSITY)
genn_viscosity = load_surrogate(GENN_VISCOSITY)
genn_k_viscosity = load_surrogate(GENN_KINEMATIC_VISCOSITY)
genn_speed_of_sound = load_surrogate(GENN_SPEED_OF_SOUND)


# Helper function to avoid repeating code
def _property(model: GENN, altitude: np.ndarray, partial: bool = False) -> np.ndarray:
    """
    Compute atmospheric property given altitude in km

    Parameters
    ----------
        model : numpy.ndarray
            GENN model associated with atmospheric property

        altitude : numpy.ndarray
            Altitude, h (units: km, bounds: 0 <= h <= 86 km)

        partial : bool
            False = return viscosity, mu (units: mg / (m * s))
            True  = return partial, d(mu)/dh (units: (mg/(m*s))/km)
    """
    try:
        x = altitude.reshape((1, -1))
    except AttributeError:
        altitude = np.array(altitude)
        x = altitude.reshape((1, -1))
    finally:
        TypeError(f'type(altitude) = {type(altitude)}. Expected numpy.ndarray')
    if np.any(x < -2) or np.any(x > 86):
        raise ValueError(f'Altitude(s) out of bound. Allowed values: 0 <= alt <= 86 km')
    if partial:
        y = model.gradient(x).reshape((1, -1))
    else:
        y = model.predict(x)
    return y.reshape(altitude.shape)


def pressure(altitude: np.ndarray, partial: bool = False) -> np.ndarray:
    """
    Compute static pressure in Pa given altitude in km

    Parameters
    ----------
        altitude : numpy.ndarray
            Altitude, h (units: km, bounds: 0 <= h <= 86 km)

        partial : bool
            False = return static pressure, P (units: Pa)
            True  = return partial, dP/dh (units: Pa/km)
    """
    return _property(genn_pressure, altitude, partial)


def temperature(altitude: np.ndarray, partial: bool = False) -> np.ndarray:
    """
    Compute static temperature in K given altitude in km

    Parameters
    ----------
        altitude : numpy.ndarray
            Altitude, h (units: km, bounds: 0 <= h <= 86 km)

        partial : bool
            False = return static temperature, T (units: K)
            True  = return partial, dT/dh (units: K/km)
    """
    return _property(genn_temperature, altitude, partial)


def density(altitude: np.ndarray, partial: bool = False) -> np.ndarray:
    """
    Compute density in kg/m^3 given altitude in km

    Parameters
    ----------
        altitude : numpy.ndarray
            Altitude, h (units: km, bounds: 0 <= h <= 86 km)

        partial : bool
            False = return density, rho (units: kg/m^3)
            True  = return partial, d(rho)/dh (units: (kg/m^3)/km)
    """
    return _property(genn_density, altitude, partial)


def viscosity(altitude: np.ndarray, partial: bool = False) -> np.ndarray:
    """
    Compute viscosity in mg / (m * s) given altitude in km

    Parameters
    ----------
        altitude : numpy.ndarray
            Altitude, h (units: km, bounds: 0 <= h <= 86 km)

        partial : bool
            False = return viscosity, mu (units: mg / (m * s))
            True  = return partial, d(mu)/dh (units: (mg/(m*s))/km)
    """
    return _property(genn_viscosity, altitude, partial) * 10**(-6)


def k_viscosity(altitude: np.ndarray, partial: bool = False) -> np.ndarray:
    """
    Compute kinetic viscosity in m^2/s given altitude in km

    Parameters
    ----------
        altitude : numpy.ndarray
            Altitude, h (units: km, bounds: 0 <= h <= 86 km)

        partial : bool
            False = return kin. viscosity, k-mu (units: m**2/s)
            True  = return partial, d(k-mu)/dh (units: (m**2/s)/km)
    """
    return _property(genn_k_viscosity, altitude, partial)


def speed_of_sound(altitude: np.ndarray, partial: bool = False) -> np.ndarray:
    """
    Compute speed of sound in m/s given altitude in km

    Parameters
    ----------
        altitude : numpy.ndarray
            Altitude, h (units: km, bounds: 0 <= h <= 86 km)

        partial : bool
            False = return kin. viscosity, k-mu (units: m**2/s)
            True  = return partial, d(k-mu)/dh (units: (m**2/s)/km)
    """
    return _property(genn_speed_of_sound, altitude, partial)


def total_temperature(altitude: np.ndarray,
                      mach_number: np.ndarray,
                      gamma: float = 1.4, partial: bool = False) -> np.ndarray:
    """
    Compute total temperature in K given Mach and altitude in km

    Parameters
    ----------
        altitude : numpy.ndarray
            Altitude, h (units: km, bounds: 0 <= h <= 86 km)

        partial : bool
            False = return total temperature, Tt (units: K)
            True  = return partial, [d(Tt)/dh (units: K/km),
                                      d(Tt)/dM (units: K)  ],
    """
    try:
        h = altitude.reshape((1, -1))
    except AttributeError:
        altitude = np.array(altitude)
        h = altitude.reshape((1, -1))
    finally:
        TypeError(f'type(altitude) = {type(altitude)}. Expected numpy.ndarray')

    try:
        M = mach_number.reshape((1, -1))
    except AttributeError:
        mach_number = np.array(mach_number)
        M = mach_number.reshape((1, -1))
    finally:
        TypeError(f'type(altitude) = {type(altitude)}. Expected numpy.ndarray')

    if M.shape[1] > 1 and h.shape[1] > 1:
        assert M.shape == h.shape

    if np.any(h < -2) or np.any(h > 86):
        raise ValueError(f'Altitude(s) out of bound. Allowed values: 0 <= alt <= 86 km')

    if partial:
        T = genn_temperature.predict(h)
        dT_dh = genn_temperature.gradient(h).reshape((1, -1))
        dTt_dh = dT_dh * (1. + 0.5 * (gamma - 1.) * M ** 2)
        dTt_dM = T * (gamma - 1.) * M
        y = np.array([dTt_dh[0],
                      dTt_dM[0]])
    else:
        T = genn_temperature.predict(h)
        y = T * (1. + 0.5 * (gamma - 1.) * M ** 2)
        y = y.reshape(altitude.shape)
    return y


def total_pressure(altitude: np.ndarray,
                   mach_number: np.ndarray,
                   gamma: float = 1.4, partial: bool = False) -> np.ndarray:
    """
    Compute total pressure in K given Mach and altitude in km

    Parameters
    ----------
        altitude : numpy.ndarray
            Altitude, h (units: km, bounds: 0 <= h <= 86 km)

        partial : bool
            False = return total pressure, Tt (units: K)
            True  = return partial, [d(Pt)/dh (units: K/km),
                                      d(Pt)/dM (units: K)  ],
    """
    try:
        h = altitude.reshape((1, -1))
    except AttributeError:
        altitude = np.array(altitude)
        h = altitude.reshape((1, -1))
    finally:
        TypeError(f'type(altitude) = {type(altitude)}. Expected numpy.ndarray')

    try:
        M = mach_number.reshape((1, -1))
    except AttributeError:
        mach_number = np.array(mach_number)
        M = mach_number.reshape((1, -1))
    finally:
        TypeError(f'type(altitude) = {type(altitude)}. Expected numpy.ndarray')

    if M.shape[1] > 1 and h.shape[1] > 1:
        assert M.shape == h.shape

    if np.any(h < -2) or np.any(h > 86):
        raise ValueError(f'Altitude(s) out of bound. Allowed values: 0 <= alt <= 86 km')

    if partial:
        P = genn_pressure.predict(h)
        dP_dh = genn_pressure.gradient(h).reshape((1, -1))

        dPt_dh = dP_dh * (1. + 0.5 * (gamma - 1.) * M ** 2) ** (gamma / (gamma - 1.))

        z = 1. + 0.5 * (gamma - 1.) * M ** 2
        dz_dM = (gamma - 1.) * M
        dPt_dM = dz_dM * (gamma / (gamma - 1.)) * P * z ** (gamma / (gamma - 1.) - 1.)

        y = np.array([dPt_dh[0],
                      dPt_dM[0]])

    else:

        P = genn_pressure.predict(h)
        y = P * (1. + 0.5 * (gamma - 1.) * M ** 2) ** (gamma / (gamma - 1.))
        y = y.reshape(altitude.shape)

    return y


class ATM76:
    """ Singleton class to compute atmospheric properties """

    _instance = None

    # Singleton
    def __new__(cls):
        if not cls._instance:
            cls._instance = object.__new__(cls)
        return cls._instance

    def pressure(self, altitude: np.ndarray) -> np.ndarray:
        """ Compute static pressure in Pa given altitude in km """
        return pressure(altitude)

    def temperature(self, altitude: np.ndarray) -> np.ndarray:
        """ Compute static temperature in K given altitude in km """
        return temperature(altitude)

    def speed_of_sound(self, altitude: np.ndarray) -> np.ndarray:
        """ Compute speed of sound in m/s given altitude in km """
        return speed_of_sound(altitude)

    def density(self, altitude: np.ndarray) -> np.ndarray:
        """ Compute density in kg/m^3 given altitude in km """
        return density(altitude)

    def viscosity(self, altitude: np.ndarray) -> np.ndarray:
        """ Compute viscosity in mg / (m * s) given altitude in km """
        return viscosity(altitude)

    def kinematic_viscosity(self, altitude: np.ndarray) -> np.ndarray:
        """ Compute viscosity in mg / (m * s) given altitude in km """
        return k_viscosity(altitude)

    def total_pressure(self, altitude: np.ndarray, mach_number: np.ndarray, gamma: float = 1.4):
        """ Compute total pressure P_t given altitude in km """
        return total_pressure(altitude, mach_number, gamma)

    def total_temperature(self, altitude: np.ndarray, mach_number: np.ndarray, gamma: float = 1.4):
        """ Compute total temperature T_t given altitude in km """
        return total_temperature(altitude, mach_number, gamma)

    def grad_pressure(self, altitude: np.ndarray) -> np.ndarray:
        """ Compute partial static pressure in Pa w.r.t. altitude in km """
        return pressure(altitude, partial=True)

    def grad_temperature(self, altitude: np.ndarray) -> np.ndarray:
        """ Compute partial of static temperature in K w.r.t. altitude in km """
        return temperature(altitude, partial=True)

    def grad_speed_of_sound(self, altitude: np.ndarray) -> np.ndarray:
        """ Compute partial of speed of sound in m/s w.r.t. altitude in km """
        return speed_of_sound(altitude, partial=True)

    def grad_density(self, altitude: np.ndarray) -> np.ndarray:
        """ Compute partial of density in kg/m^3 w.r.t. altitude in km """
        return density(altitude, partial=True)

    def grad_viscosity(self, altitude: np.ndarray) -> np.ndarray:
        """ Compute partial of viscosity in kg/m^3 w.r.t. altitude in km """
        return viscosity(altitude, partial=True)

    def grad_kinematic_viscosity(self, altitude: np.ndarray) -> np.ndarray:
        """ Compute partial of viscosity in kg/m^3 w.r.t. altitude in km """
        return k_viscosity(altitude, partial=True)

    def grad_total_pressure(self, altitude: np.ndarray, mach_number: np.ndarray, gamma: float = 1.4):
        """ Compute partial of total pressure P_t given altitude in km """
        return total_pressure(altitude, mach_number, gamma, partial=True)

    def grad_total_temperature(self, altitude: np.ndarray, mach_number: np.ndarray, gamma: float = 1.4):
        """ Compute partial of total temperature P_t given altitude in km """
        return total_temperature(altitude, mach_number, gamma, partial=True)


if __name__ == "__main__":
    atm = ATM76()
    print(atm.grad_total_temperature(altitude=[10, 9], mach_number=0.5))
