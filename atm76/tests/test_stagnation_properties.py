import numpy as np
from scipy.optimize import check_grad, approx_fprime
from atm76.atmosphere import ATM76


def test_dTt_dh():
    atm = ATM76()

    alt = np.array([10.])
    M_0 = np.array([0.5])

    def f(h: np.ndarray):
        return atm.total_temperature(altitude=h, mach_number=M_0)

    def df_dh(h: np.ndarray):
        return atm.grad_total_temperature(altitude=h, mach_number=M_0)[0]

    fprime = df_dh(alt)
    fprime_approx = approx_fprime(alt, f, epsilon=1e-5)
    error = check_grad(f, df_dh, alt)

    # print(f'partial of T_t wrt alt: '
    #       f'fprime = {fprime.ravel()[0]:.7f}, '
    #       f'fprime_approx = {fprime_approx.ravel()[0]:.7f}, '
    #       f'error = {error:.9f}')

    assert np.allclose(fprime.ravel()[0], fprime_approx.ravel()[0], rtol=1e-3)
    assert np.sign(fprime_approx) == np.sign(fprime)


def test_dTt_dM():
    atm = ATM76()

    alt = np.array([10.])
    M_0 = np.array([0.5])

    def f(M: np.ndarray):
        return atm.total_temperature(altitude=alt, mach_number=M)

    def df_dM(M: np.ndarray):
        return atm.grad_total_temperature(altitude=alt, mach_number=M)[1]

    fprime = df_dM(M_0)
    fprime_approx = approx_fprime(M_0, f, epsilon=1e-5)
    error = check_grad(f, df_dM, M_0)

    # print(f'partial of T_t wrt M: '
    #       f'fprime = {fprime.ravel()[0]:.7f}, '
    #       f'fprime_approx = {fprime_approx.ravel()[0]:.7f}, '
    #       f'error = {error:.9f}')

    assert np.allclose(fprime.ravel()[0], fprime_approx.ravel()[0], rtol=1e-3)
    assert np.sign(fprime_approx) == np.sign(fprime)


def test_dPt_dh():
    atm = ATM76()

    alt = np.array([10.])
    M_0 = np.array([0.5])

    def f(h: np.ndarray):
        return atm.total_pressure(altitude=h, mach_number=M_0)

    def df_dh(h: np.ndarray):
        return atm.grad_total_pressure(altitude=h, mach_number=M_0)[0]

    fprime = df_dh(alt)
    fprime_approx = approx_fprime(alt, f, epsilon=1e-5)
    error = check_grad(f, df_dh, alt)

    # print(f'partial of P_t wrt alt: '
    #       f'fprime = {fprime.ravel()[0]:.7f}, '
    #       f'fprime_approx = {fprime_approx.ravel()[0]:.7f}, '
    #       f'error = {error:.9f}')

    assert np.allclose(fprime.ravel()[0], fprime_approx.ravel()[0], rtol=1e-3)
    assert np.sign(fprime_approx) == np.sign(fprime)


def test_dPt_dM():
    atm = ATM76()

    alt = np.array([10.])
    M_0 = np.array([0.5])

    def f(M: np.ndarray):
        return atm.total_pressure(altitude=alt, mach_number=M)

    def df_dM(M: np.ndarray):
        return atm.grad_total_pressure(altitude=alt, mach_number=M)[1]

    fprime = df_dM(M_0)
    fprime_approx = approx_fprime(M_0, f, epsilon=1e-5)
    error = check_grad(f, df_dM, M_0)

    # print(f'partial of P_t wrt M: '
    #       f'fprime = {fprime.ravel()[0]:.7f}, '
    #       f'fprime_approx = {fprime_approx.ravel()[0]:.7f}, '
    #       f'error = {error:.9f}')

    assert np.allclose(fprime.ravel()[0], fprime_approx.ravel()[0], rtol=1e-3)
    assert np.sign(fprime_approx) == np.sign(fprime)


if __name__ == '__main__':
    test_dTt_dh()
    test_dTt_dM()
    test_dPt_dh()
    test_dPt_dM()
