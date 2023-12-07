import numpy as np
import pandas as pd

# Constants
eV_cm = 8065.540106923572


# Function to read spectral data
def read_spectral_data(file_path):
    data = pd.read_csv(file_path)
    return data["Wavelength"].values, data["Intensity"].values


def normalize_yaxis(intensities):
    max_intensity = np.max(intensities)
    return intensities / max_intensity if max_intensity != 0 else intensities


# CIE 1931 color matching functions (placeholders)
def CMFX(x):
    t1 = (x - 442.0) * np.where(x < 442.0, 0.0624, 0.0374)
    t2 = (x - 599.8) * np.where(x < 599.8, 0.0264, 0.0323)
    t3 = (x - 501.1) * np.where(x < 501.1, 0.0490, 0.0382)
    return (
        0.362 * np.exp(-0.5 * t1 * t1)
        + 1.056 * np.exp(-0.5 * t2 * t2)
        - 0.065 * np.exp(-0.5 * t3 * t3)
    )


def CMFY(x):
    t1 = (x - 568.8) * np.where(x < 568.8, 0.0213, 0.0247)
    t2 = (x - 530.9) * np.where(x < 530.9, 0.0613, 0.0322)
    return 0.821 * np.exp(-0.5 * t1 * t1) + 0.286 * np.exp(-0.5 * t2 * t2)


def CMFZ(x):
    t1 = (x - 437.0) * np.where(x < 437.0, 0.0845, 0.0278)
    t2 = (x - 459.0) * np.where(x < 459.0, 0.0385, 0.0725)
    return 1.217 * np.exp(-0.5 * t1 * t1) + 0.681 * np.exp(-0.5 * t2 * t2)


# Function to calculate XYZ values using NumPy
def calculate_XYZ(wavelengths, intensities):
    delta_lambda = np.abs(np.diff(wavelengths))
    X = np.sum(
        (
            CMFX(wavelengths[:-1]) * intensities[:-1]
            + CMFX(wavelengths[1:]) * intensities[1:]
        )
        * delta_lambda
        / 2
    )
    Y = np.sum(
        (
            CMFY(wavelengths[:-1]) * intensities[:-1]
            + CMFY(wavelengths[1:]) * intensities[1:]
        )
        * delta_lambda
        / 2
    )
    Z = np.sum(
        (
            CMFZ(wavelengths[:-1]) * intensities[:-1]
            + CMFZ(wavelengths[1:]) * intensities[1:]
        )
        * delta_lambda
        / 2
    )
    return X, Y, Z


# Main function
def main():
    file_path = "test_data.csv"
    wavelengths, intensities = read_spectral_data(file_path)
    # Normalize the intensity values
    normalized_intensities = normalize_yaxis(intensities)
    # Calculate the XYZ values
    X, Y, Z = calculate_XYZ(wavelengths, normalized_intensities)
    # Output the results
    print(f"XYZ Color Space\n\tX = {X:.4f}\n\tY = {Y:.4f}\n\tZ = {Z:.4f}")


if __name__ == "__main__":
    main()
