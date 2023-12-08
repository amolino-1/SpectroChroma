import numpy as np


# CIE 1931 color matching functions
def color_match_function_X(wavelength):
    """
    Calculate the CIE 1931 color matching function X value for
    a given wavelength.

    Parameters:
    x (numpy.ndarray): Array of wavelengths (in nanometers).

    Returns:
    numpy.ndarray: Array of CIE 1931 color matching function X values
    corresponding to the input wavelengths.
    """
    t1 = (wavelength - 442.0) * np.where(wavelength < 442.0, 0.0624, 0.0374)
    t2 = (wavelength - 599.8) * np.where(wavelength < 599.8, 0.0264, 0.0323)
    t3 = (wavelength - 501.1) * np.where(wavelength < 501.1, 0.0490, 0.0382)
    return (
        0.362 * np.exp(-0.5 * t1 * t1)
        + 1.056 * np.exp(-0.5 * t2 * t2)
        - 0.065 * np.exp(-0.5 * t3 * t3)
    )


def color_match_function_Y(wavelength):
    t1 = (wavelength - 568.8) * np.where(wavelength < 568.8, 0.0213, 0.0247)
    t2 = (wavelength - 530.9) * np.where(wavelength < 530.9, 0.0613, 0.0322)
    return 0.821 * np.exp(-0.5 * t1 * t1) + 0.286 * np.exp(-0.5 * t2 * t2)


def color_match_function_Z(wavelength):
    t1 = (wavelength - 437.0) * np.where(wavelength < 437.0, 0.0845, 0.0278)
    t2 = (wavelength - 459.0) * np.where(wavelength < 459.0, 0.0385, 0.0725)
    return 1.217 * np.exp(-0.5 * t1 * t1) + 0.681 * np.exp(-0.5 * t2 * t2)
