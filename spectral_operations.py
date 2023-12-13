import numpy as np
import pandas as pd

# Constants
eV_cm = 8065.540106923572


# Function to read spectral data
def read_spectral_data(file_path):
    # Determine the file extension
    if file_path.endswith(".csv"):
        delimiter = ","
        # Reading the csv file with a comma delimiter
        data = pd.read_csv(file_path, delimiter=delimiter)
        wavelength = data["Wavelength"].values
        intensity = data["Intensity"].values
    elif file_path.endswith(".spectrum"):
        delimiter = "\t"
        # Reading the specfile with a tab delimiter, skipping the first row
        # these steps are necessary because ORCA .sepectrum file has wierd formating
        data = pd.read_csv(
            file_path, delimiter=delimiter, usecols=[0, 2], header=None, skiprows=1
        )
        wavelength = data.iloc[:, 0].values  # First column for Wavelength
        intensity = data.iloc[:, 1].values  # Third column (index 2) for Intensity

    else:
        raise ValueError("Unsupported file format")

    return wavelength, intensity


def normalize_yaxis(intensities):
    max_intensity = np.max(intensities)
    return intensities / max_intensity if max_intensity != 0 else intensities
