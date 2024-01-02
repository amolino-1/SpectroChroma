import argparse
import os

import numpy as np
from color_matching import (
    color_match_function_X,
    color_match_function_Y,
    color_match_function_Z,
)
from spectral_operations import normalize_yaxis, read_spectral_data

# Unicode characters
DEGREE_SYMBOL = "\u00B0"


def loadSpec():
    # Initialize parser
    parser = argparse.ArgumentParser()

    # Adding optional argument
    parser.add_argument(
        "-S",  # Argument flag (short)
        "--spectrumFile",  # Argument flag (long)
        type=str,  # Accept both files and paths
        required=True,  # Required argument
        help="Input spectrum file or path to spectrum file",  # Help description
        metavar="\b",  # Remove space before help
    )

    # Read arguments from command line and return them
    return parser.parse_args()


# Call the function and use the returned values
args = loadSpec()
spectrumFile = args.spectrumFile


# Function to calculate XYZ values using NumPy
def calculate_XYZ(wavelengths, intensities):
    """
    Calculate the CIE XYZ color space values from given spectral data.

    Parameters:
    wavelengths (numpy.ndarray): Array of wavelengths (in nanometers).
    intensities (numpy.ndarray): Array of corresponding intensity values.

    Returns:
    tuple: Returns the calculated X, Y, and Z values.
    """
    delta_wavelength = np.abs(
        np.diff(wavelengths)
    )  # Difference in successive wavelengths

    X_values = (
        (
            color_match_function_X(wavelengths[:-1]) * intensities[:-1]
            + color_match_function_X(wavelengths[1:]) * intensities[1:]
        )
        * delta_wavelength
        / 2
    )
    X = np.sum(X_values)

    Y_values = (
        (
            color_match_function_Y(wavelengths[:-1]) * intensities[:-1]
            + color_match_function_Y(wavelengths[1:]) * intensities[1:]
        )
        * delta_wavelength
        / 2
    )
    Y = np.sum(Y_values)

    Z_values = (
        (
            color_match_function_Z(wavelengths[:-1]) * intensities[:-1]
            + color_match_function_Z(wavelengths[1:]) * intensities[1:]
        )
        * delta_wavelength
        / 2
    )
    Z = np.sum(Z_values)

    return X, Y, Z


def XYZ_to_Yxy(XYZ_X, XYZ_Y, XYZ_Z):
    """
    Convert XYZ color space values to Yxy color space.

    Parameters:
    XYZ_X, XYZ_Y, XYZ_Z (float): The X, Y, and Z values in the XYZ color space.

    Returns:
    tuple: Returns the Y, x, and y values in the Yxy color space.
    """
    total = XYZ_X + XYZ_Y + XYZ_Z
    if total == 0:
        return XYZ_Y, 0, 0  # Avoid division by zero

    yxy_x = XYZ_X / total
    yxy_y = XYZ_Y / total
    return XYZ_Y, yxy_x, yxy_y


# sRGB to XYZ conversion
# Linear Transformation
def XYZ_to_linear_RGB(xyz_x, xyz_y, xyz_z):
    """
    Convert XYZ color space values to linear RGB values.

    This function applies a linear transformation using
    the sRGB color space matrix.

    Parameters:
    xyz_x, xyz_y, xyz_z (float): The X, Y, and Z values in the XYZ color space.

    Returns:
    numpy.ndarray: Array of linear RGB values.
    """
    # sRGB transformation matrix for XYZ to linear RGB conversion
    srgb_transformation_matrix = np.array(
        [
            [3.2404542, -1.5371385, -0.4985314],  # Red coefficient vector
            [-0.9692660, 1.8760108, 0.0415560],  # Green coefficient vector
            [0.0556434, -0.2040259, 1.0572252],  # Blue coefficient vector
        ]
    )

    linear_rgb = np.dot(srgb_transformation_matrix, np.array([xyz_x, xyz_y, xyz_z]))
    return linear_rgb


# Gamma Correction
def gamma_correct_sRGB(channel):
    if channel <= 0.0031308:
        return 12.92 * channel
    else:
        return 1.055 * (channel ** (1 / 2.4)) - 0.055


# Scaling
def scale_to_255(RGB):
    return np.clip(RGB * 255, 0, 255)


def XYZ_to_sRGB(xyz_x, xyz_y, xyz_z):
    """
    Convert XYZ color space values to sRGB color space.

    This function first scales the XYZ values, converts them
    to linear RGB, applies gamma correction, and then scales
    the RGB values to the range [0, 255].

    Parameters:
    xyz_x, xyz_y, xyz_z (float): The X, Y, and Z values in the XYZ color space.

    Returns:
    numpy.ndarray: Array of sRGB values scaled to the range [0, 255].
    """
    # Scale XYZ values by 100 as per the standard conversion process
    scaled_xyz_x = xyz_x / 100
    scaled_xyz_y = xyz_y / 100
    scaled_xyz_z = xyz_z / 100

    # Convert scaled XYZ to linear RGB
    linear_rgb = XYZ_to_linear_RGB(scaled_xyz_x, scaled_xyz_y, scaled_xyz_z)

    # Apply gamma correction to the linear RGB values
    gamma_corrected_rgb = np.array(
        [gamma_correct_sRGB(channel) for channel in linear_rgb]
    )

    # Scale the gamma-corrected RGB values to the standard sRGB range of [0, 255]
    standard_srgb = scale_to_255(gamma_corrected_rgb)

    return standard_srgb


def sRGB_to_HSL(srgb_r, srgb_g, srgb_b):
    """
    Convert sRGB color space values to HSL (Hue, Saturation, Lightness) color space.

    Parameters:
    srgb_r, srgb_g, srgb_b (float): Red, Green, and Blue values in the sRGB color space,
                                    scaled in the range [0, 255].

    Returns:
    tuple: Returns the Hue (H) in the range [0, 1], Saturation (S), and Lightness (L) values,
           and also the Hue in degrees (H_deg).
    """
    # Normalize the sRGB values to the range [0, 1]
    normalized_r = srgb_r / 255.0
    normalized_g = srgb_g / 255.0
    normalized_b = srgb_b / 255.0

    # Find the maximum and minimum of the RGB values
    c_max = max(normalized_r, normalized_g, normalized_b)
    c_min = min(normalized_r, normalized_g, normalized_b)
    delta = c_max - c_min

    # Calculate Lightness
    lightness = (c_max + c_min) / 2

    # Calculate Saturation
    saturation = 0 if delta == 0 else delta / (1 - abs(2 * lightness - 1))

    # Calculate Hue
    if delta == 0:
        hue = 0
    elif c_max == normalized_r:
        hue = (((normalized_g - normalized_b) / delta) % 6) / 6
    elif c_max == normalized_g:
        hue = (((normalized_b - normalized_r) / delta) + 2) / 6
    else:  # c_max == normalized_b
        hue = (((normalized_r - normalized_g) / delta) + 4) / 6

    # Hue in degrees
    hue_deg = hue * 360

    return hue, saturation, lightness, hue_deg


def sRGB_to_CMYK(srgb_r, srgb_g, srgb_b):
    """
    Convert sRGB color space values to CMYK color space.

    Parameters:
    srgb_r, srgb_g, srgb_b (float): Red, Green, and Blue values in the sRGB color space,
                                    scaled in the range [0, 255].

    Returns:
    tuple: Returns the Cyan (C), Magenta (M), Yellow (Y), and Key (black) (K) values.
    """
    # Normalize the sRGB values to the range [0, 1]
    normalized_r = srgb_r / 255.0
    normalized_g = srgb_g / 255.0
    normalized_b = srgb_b / 255.0

    # Convert RGB to CMY
    c = 1 - normalized_r
    m = 1 - normalized_g
    y = 1 - normalized_b

    # Convert CMY to CMYK
    k = min(c, m, y)

    if k == 1:
        c = m = y = 0  # Pure black
    else:
        c = (c - k) / (1 - k)
        m = (m - k) / (1 - k)
        y = (y - k) / (1 - k)

    return c, m, y, k


# RGB to Hex conversion
def rgb_to_hex(R, G, B):
    return "#{:02x}{:02x}{:02x}".format(R, G, B)


# Main function
def main():
    # Determine the directory of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the path to 'header.txt' relative to the script's directory
    header_path = os.path.join(script_dir, "header.txt")

    # Read and print the program header
    with open(header_path, "r") as file:
        header = file.read()
    print(header)

    # Calculate the XYZ values
    wavelengths, intensities = read_spectral_data(spectrumFile)
    # Normalize the intensity values
    normalized_intensities = normalize_yaxis(intensities)
    # Calculate the XYZ values
    X, Y, Z = calculate_XYZ(wavelengths, normalized_intensities)
    Y_yxy, x, y = XYZ_to_Yxy(X, Y, Z)
    sRGB = XYZ_to_sRGB(X, Y, Z)
    H, S, L, H_deg = sRGB_to_HSL(sRGB[0], sRGB[1], sRGB[2])
    C_cmyk, M_cmyk, Y_cmyk, K_cmyk = sRGB_to_CMYK(sRGB[0], sRGB[1], sRGB[2])
    hex_color = rgb_to_hex(int(sRGB[0]), int(sRGB[1]), int(sRGB[2]))
    # Output the results
    print(
        f"XYZ Color Space D65/2{DEGREE_SYMBOL}\n\tX = {X:.4f}\n\tY = {Y:.4f}\n\tZ = {Z:.4f}\n"
    )
    print(f"Yxy Color Space\n\tY = {Y_yxy:.4f}\n\tx = {x:.4f}\n\ty = {y:.4f}\n")
    print(
        f"sRGB Color Space [0-255] D65/2{DEGREE_SYMBOL}\n\tR = {sRGB[0]:.4f}\n\tG = {sRGB[1]:.4f}\n\tB = {sRGB[2]:.4f}\n"
    )
    print(
        f"HSL Color Space [0-1]\n\tH = {H:.4f} ({H_deg:.2f}°)\n\tS = {S:.4f}\n\tL = {L:.4f}\n"
    )
    print(
        f"CMYK Color Space [0-1]\n\tC = {C_cmyk:.4f}\n\tM = {M_cmyk:.4f}\n\tY = {Y_cmyk:.4f}\n\tK = {K_cmyk:.4f}\n"
    )
    print(f"CSS/HTML Hex Color Code: {hex_color}\n")


if __name__ == "__main__":
    main()
