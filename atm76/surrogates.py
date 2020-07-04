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

import numpy as np
import pickle
import os

from genn.model import GENN

# Path to surrogate models
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

GENN_TEMPERATURE = os.path.join(DATA_DIR, 'genn_temperature.pkl')
GENN_SPEED_OF_SOUND = os.path.join(DATA_DIR, 'genn_speed_of_sound.pkl')
GENN_PRESSURE = os.path.join(DATA_DIR, 'genn_pressure.pkl')
GENN_DENSITY = os.path.join(DATA_DIR, 'genn_density.pkl')
GENN_VISCOSITY = os.path.join(DATA_DIR, 'genn_viscosity.pkl')
GENN_KINEMATIC_VISCOSITY = os.path.join(DATA_DIR, 'genn_kinematic_viscosity.pkl')


def load_surrogate(trained_parameters_pkl):
    if os.path.exists(trained_parameters_pkl):
        content = open(trained_parameters_pkl, 'rb')
        trained_parameters, scale_factors = pickle.load(content)
        content.close()
        genn = GENN.initialize()
        genn.load_parameters(trained_parameters, scale_factors)
        return genn
    return None


def save_parameters(genn_model, pkl_file):
    output = open(pkl_file, 'wb')
    parameters = genn_model.parameters
    scale_factors = genn_model.scale_factors
    pickle.dump((parameters, scale_factors), output)
    output.close()


def _load_model(trained_parameters_pkl):
    if os.path.exists(trained_parameters_pkl):
        content = open(trained_parameters_pkl, 'rb')
        trained_parameters, scale_factors = pickle.load(content)
        content.close()
        genn = GENN.initialize()
        genn.load_parameters(trained_parameters, scale_factors)
        return genn
    return None


def fit_temperature(show_fit=False, save_plot=False, verbose=False):
    X_train = np.array(ALTITUDE).reshape((1, -1))
    y_train = np.array(TEMPERATURE).reshape((1, -1))

    model = GENN.initialize(n_x=1, n_y=1, deep=3, wide=12)

    model.train(X=X_train,
                Y=y_train,
                alpha=0.01,
                lambd=0.00,
                gamma=0.00,
                beta1=0.90,
                beta2=0.99,
                mini_batch_size=128,
                num_iterations=10,
                num_epochs=200,
                silent=not verbose)

    if show_fit and MATPLOTLIB_INSTALLED:
        import matplotlib.pyplot as plt
        y_pred = model.predict(X_train)
        plt.scatter(X_train.T, y_train.T)
        plt.plot(X_train.T, y_pred.T, c='red')
        plt.xlabel('alt (km)')
        plt.ylabel('temp (K)')
        plt.legend(['data', 'fit'])
        if save_plot:
            path = os.path.join(DATA_DIR, 'plot_temperature.png')
            plt.savefig(path, dpi=300)
        plt.show()

    # model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.01)
    # model.fit(X_train, y_train)
    #
    # if show_fit:
    #     import matplotlib.pyplot as plt
    #     y_pred = model.predict(X_train)
    #     plt.scatter(X_train, y_train)
    #     plt.plot(X_train, y_pred, c='red')
    #     plt.xlabel('alt (km)')
    #     plt.ylabel('temp (K)')
    #     plt.legend(['data', 'fit'])
    #     plt.show()

    return model


def fit_pressure(show_fit=False, save_plot=False, verbose=False):
    X_train = np.array(ALTITUDE).reshape((1, -1))
    y_train = np.array(PRESSURE).reshape((1, -1))

    model = GENN.initialize(n_x=1, n_y=1, deep=2, wide=12)

    model.train(X=X_train,
                Y=y_train,
                alpha=0.01,
                lambd=0.00,
                beta1=0.90,
                beta2=0.99,
                mini_batch_size=128,
                num_iterations=10,
                num_epochs=200,
                silent=not verbose)

    if show_fit and MATPLOTLIB_INSTALLED:
        y_pred = model.predict(X_train)
        plt.scatter(X_train.T, y_train.T)
        plt.plot(X_train.T, y_pred.T, c='red')
        plt.xlabel('alt (km)')
        plt.ylabel('press (Pa)')
        plt.legend(['data', 'fit'])
        if save_plot:
            path = os.path.join(DATA_DIR, 'plot_pressure.png')
            plt.savefig(path, dpi=300)
        plt.show()

    return model


def fit_viscosity(show_fit=False, save_plot=False, verbose=False):
    X_train = np.array(ALTITUDE).reshape((1, -1))
    y_train = np.array(VISCOSITY).reshape((1, -1))

    # model = GENN.initialize(n_x=1, n_y=1, deep=3, wide=12)
    model = GENN.initialize(n_x=1, n_y=1, deep=3, wide=25)

    model.train(X=X_train,
                Y=y_train,
                alpha=0.001,
                lambd=0.00,
                gamma=0.00,
                beta1=0.90,
                beta2=0.99,
                mini_batch_size=128,
                num_iterations=25,
                num_epochs=300,
                silent=not verbose)

    if show_fit and MATPLOTLIB_INSTALLED:
        y_pred = model.predict(X_train)
        plt.scatter(X_train.T, y_train.T)
        plt.plot(X_train.T, y_pred.T, c='red')
        plt.xlabel('alt (km)')
        plt.ylabel('visc (N * s / m ** 2) * 10 ** (-6)')
        plt.legend(['data', 'fit'])
        if save_plot:
            path = os.path.join(DATA_DIR, 'plot_viscosity.png')
            plt.savefig(path, dpi=300)
        plt.show()

    return model


def fit_k_viscosity(show_fit=False, save_plot=False, verbose=False):
    X_train = np.array(ALTITUDE).reshape((1, -1))
    y_train = np.array(KINEMATIC_VISCOSITY).reshape((1, -1))

    model = GENN.initialize(n_x=1, n_y=1, deep=2, wide=12)

    model.train(X=X_train,
                Y=y_train,
                alpha=0.01,
                lambd=0.00,
                gamma=0.00,
                beta1=0.90,
                beta2=0.99,
                mini_batch_size=128,
                num_iterations=10,
                num_epochs=200,
                silent=not verbose)

    if show_fit and MATPLOTLIB_INSTALLED:
        y_pred = model.predict(X_train)
        plt.scatter(X_train.T, y_train.T)
        plt.plot(X_train.T, y_pred.T, c='red')
        plt.xlabel('alt (km)')
        plt.ylabel('k. visc (m ** 2 / s)')
        plt.legend(['data', 'fit'])
        if save_plot:
            path = os.path.join(DATA_DIR, 'plot_k_viscosity.png')
            plt.savefig(path, dpi=300)
        plt.show()

    return model


def fit_speed_of_sound(show_fit=False, save_plot=300, verbose=False):
    X_train = np.array(ALTITUDE).reshape((1, -1))
    y_train = np.array(SPEED_OF_SOUND).reshape((1, -1))

    model = GENN.initialize(n_x=1, n_y=1, deep=3, wide=12)

    model.train(X=X_train,
                Y=y_train,
                alpha=0.01,
                lambd=0.00,
                gamma=0.00,
                beta1=0.90,
                beta2=0.99,
                mini_batch_size=128,
                num_iterations=10,
                num_epochs=200,
                silent=not verbose)

    if show_fit and MATPLOTLIB_INSTALLED:
        y_pred = model.predict(X_train)
        plt.scatter(X_train.T, y_train.T)
        plt.plot(X_train.T, y_pred.T, c='red')
        plt.xlabel('alt (km)')
        plt.ylabel('speed of sound (m/s)')
        plt.legend(['data', 'fit'])
        if save_plot:
            path = os.path.join(DATA_DIR, 'plot_speed_of_sound.png')
            plt.savefig(path, dpi=300)
        plt.show()

    return model


def fit_density(show_fit=False, save_plot=False, verbose=False):
    X_train = np.array(ALTITUDE).reshape((1, -1))
    y_train = np.array(DENSITY).reshape((1, -1))

    model = GENN.initialize(n_x=1, n_y=1, deep=2, wide=12)

    model.train(X=X_train,
                Y=y_train,
                alpha=0.01,
                lambd=0.00,
                gamma=0.00,
                beta1=0.90,
                beta2=0.99,
                mini_batch_size=128,
                num_iterations=10,
                num_epochs=200,
                silent=not verbose)

    if show_fit:
        y_pred = model.predict(X_train)
        plt.scatter(X_train.T, y_train.T)
        plt.plot(X_train.T, y_pred.T, c='red')
        plt.xlabel('alt (km)')
        plt.ylabel('dens (kg / m ** 3)')
        plt.legend(['data', 'fit'])
        if save_plot:
            path = os.path.join(DATA_DIR, 'plot_density.png')
            plt.savefig(path, dpi=300)
        plt.show()

    return model


if __name__ == "__main__":

    # train data
    temperature = fit_temperature(show_fit=True, save_plot=True)
    speed_of_sound = fit_speed_of_sound(show_fit=True, save_plot=True)
    pressure = fit_pressure(show_fit=True, save_plot=True)
    density = fit_density(show_fit=True, save_plot=True)
    viscosity = fit_viscosity(show_fit=True, save_plot=True)
    kinematic_viscosity = fit_k_viscosity(show_fit=True, save_plot=True)

    # save trained parameters
    save_parameters(temperature, GENN_TEMPERATURE)
    save_parameters(speed_of_sound, GENN_SPEED_OF_SOUND)
    save_parameters(pressure, GENN_PRESSURE)
    save_parameters(density, GENN_DENSITY)
    save_parameters(viscosity, GENN_VISCOSITY)
    save_parameters(kinematic_viscosity, GENN_KINEMATIC_VISCOSITY)
