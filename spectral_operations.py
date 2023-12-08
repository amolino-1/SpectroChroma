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
