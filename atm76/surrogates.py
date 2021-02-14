"""
This module provides standard atmosphere model.  It is based on the data
contained in tables 1 and 2 from `Properties Of The U.S. Standard Atmosphere
1976 <http://www.pdas.com/atmos.html>`_ The data was used to train Gradient-
Enhanced Neural Networks (GENN), such that all predictions and gradient are
pure analytical functions (as opposed to linear table interpolation).
"""
from atm76.tables import (ALTITUDE, PRESSURE, TEMPERATURE, DENSITY,
                          SPEED_OF_SOUND, VISCOSITY, KINEMATIC_VISCOSITY)

from importlib.util import find_spec

if find_spec("matplotlib"):
    import matplotlib.pyplot as plt
    MATPLOTLIB_INSTALLED = True
else:
    MATPLOTLIB_INSTALLED = False

import pickle
import numpy as np
import os

from genn import GENN

# Path to surrogate models
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

GENN_TEMPERATURE = os.path.join(DATA_DIR, 'genn_temperature.pkl')
GENN_SPEED_OF_SOUND = os.path.join(DATA_DIR, 'genn_speed_of_sound.pkl')
GENN_PRESSURE = os.path.join(DATA_DIR, 'genn_pressure.pkl')
GENN_DENSITY = os.path.join(DATA_DIR, 'genn_density.pkl')
GENN_VISCOSITY = os.path.join(DATA_DIR, 'genn_viscosity.pkl')
GENN_KINEMATIC_VISCOSITY = os.path.join(DATA_DIR, 'genn_kinematic_viscosity.pkl')


def fit_model(x_train, y_train, x_label, y_label,
              show_fit=False, save_plot=False, verbose=False):
    model = GENN(hidden_layer_sizes=[12] * 3,
                 activation="tanh",
                 alpha=0,
                 gamma=0,
                 learning_rate_init=0.01,
                 num_epochs=1,
                 max_iter=1000, verbose=verbose)
    model.fit(x_train.reshape((-1, 1)),
              y_train.reshape((-1, 1)), is_normalize=True)
    if show_fit and MATPLOTLIB_INSTALLED:
        y_pred = model.predict(x_train)
        plt.figure()
        plt.scatter(x_train, y_train)
        plt.plot(x_train, y_pred, 'r-')
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend(['fit', 'data'])
        if save_plot:
            path = os.path.join(DATA_DIR, 'plot_temperature.png')
            plt.savefig(path, dpi=300)
        plt.show()
    return model


if __name__ == "__main__":

    show_fit = True

    # train data
    temperature = fit_model(x_train=np.array(ALTITUDE),
                            y_train=np.array(TEMPERATURE),
                            show_fit=show_fit, save_plot=True,
                            x_label='alt (km)', y_label='T (K)')
    pressure = fit_model(x_train=np.array(ALTITUDE),
                         y_train=np.array(PRESSURE),
                         show_fit=show_fit, save_plot=True,
                         x_label='alt (km)', y_label='P (Pa)')
    speed_of_sound = fit_model(x_train=np.array(ALTITUDE),
                         y_train=np.array(SPEED_OF_SOUND),
                         show_fit=show_fit, save_plot=True,
                         x_label='alt (km)', y_label='sos (m/s)')
    density = fit_model(x_train=np.array(ALTITUDE),
                        y_train=np.array(DENSITY),
                        show_fit=show_fit, save_plot=True,
                        x_label='alt (km)', y_label='rho (kg/m**3)')
    viscosity = fit_model(x_train=np.array(ALTITUDE),
                          y_train=np.array(VISCOSITY),
                          show_fit=show_fit, save_plot=False,
                          x_label='alt (km)',
                          y_label='visc (N * s / m ** 2) * 10 ** (-6)')
    kinematic_viscosity = fit_model(
        x_train=np.array(ALTITUDE),
        y_train=np.array(KINEMATIC_VISCOSITY),
        show_fit=show_fit, save_plot=True,
        x_label='alt (km)', y_label='k. visc (m ** 2 / s)')

    # Save to pickle
    for model, filename in {temperature: GENN_TEMPERATURE,
                            pressure: GENN_PRESSURE,
                            speed_of_sound: GENN_SPEED_OF_SOUND,
                            density: GENN_DENSITY,
                            viscosity: GENN_VISCOSITY,
                            kinematic_viscosity: GENN_KINEMATIC_VISCOSITY
                            }.items():
        pickle.dump(model, open(filename, 'wb'))
