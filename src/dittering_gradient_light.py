# coding:utf-8

from PIL import Image, ImageDraw
import numpy as np
import math


def generate_toon_gradient(width, height, light_strength, radius, center_x, center_y, num_colors) -> Image:
    """
    Generate a radial light gradient image with a toon effect (stepped gradient).

    Args:
        width (int): The width of the image.
        height (int): The height of the image.
        light_strength (float): The maximum brightness at the center (0 to 255).
        radius (float): The radius of the gradient effect.
        center_x (int): X-coordinate of the gradient center.
        center_y (int): Y-coordinate of the gradient center.
        num_colors (int): Number of distinct color levels in the gradient.

    Returns:
        Image: The generated toon gradient image.
    """
    # Create an empty image with black background
    image = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(image)

    # Calculate the brightness step for each color level
    color_step = light_strength / (num_colors - 1)  # -1 to ensure inclusive range

    # Calculate brightness based on the distance from the center
    for y in range(height):
        for x in range(width):
            # Distance from the center
            distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

            # Calculate base brightness based on distance
            if distance <= radius:
                brightness = light_strength * (1 - (distance / radius) ** 2)

                # Apply toon effect by snapping brightness to the nearest color level
                toon_brightness = int(math.ceil(brightness / color_step) * color_step)
                draw.point((x, y), fill=toon_brightness)

    return image


def generate_toon_oval_gradient(width, height, light_strength, radius_x, radius_y, center_x, center_y, num_colors) -> Image:
    """
    Generate an oval-shaped radial light gradient with a toon effect using NumPy.

    Args:
        width (int): The width of the image.
        height (int): The height of the image.
        light_strength (float): The maximum brightness at the center (0 to 255).
        radius_x (float): The horizontal radius of the gradient effect.
        radius_y (float): The vertical radius of the gradient effect.
        center_x (int): X-coordinate of the gradient center.
        center_y (int): Y-coordinate of the gradient center.
        num_colors (int): Number of distinct color levels in the gradient.

    Returns:
        np.ndarray: The generated oval toon gradient image as a 2D array.
    """
    # Initialize an empty 2D array for the gradient
    gradient = np.zeros((height, width), dtype=np.uint8)

    # Calculate the brightness step for each color level
    color_step = light_strength / (num_colors - 1)  # -1 to ensure inclusive range

    # Create a meshgrid of x and y coordinates
    y, x = np.ogrid[:height, :width]

    # Compute the normalized elliptical distance from the center
    norm_distance = np.sqrt(((x - center_x) / radius_x) ** 2 + ((y - center_y) / radius_y) ** 2) ** 2

    # Calculate brightness for each pixel, limited by the normalized distance <= 1 (within ellipse)
    brightness = np.clip(light_strength * (1 - norm_distance), 0, light_strength)

    # Apply toon effect by snapping brightness to nearest color level
    toon_brightness = (np.ceil(brightness / color_step) * color_step).astype(np.uint8)

    # Set the gradient array to the toon brightness values
    gradient = toon_brightness.astype(np.uint8)

    return gradient


def sample_convolution(array, kernel_size=3) -> np.ndarray:
    """
    Perform a 2D convolution on a 2D array with a given kernel.

    Args:
        array (np.ndarray): 2D input array.
        kernel_size (int): -

    Returns:
        np.ndarray: Convolved 2D array.
    """
    # Get dimensions of the input array and kernel
    array_height, array_width = array.shape
    kernel_height, kernel_width = kernel_size, kernel_size

    # Calculate the padding needed for 'same' convolution
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2

    # Pad the input array to handle borders
    padded_array = np.pad(array, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')

    # Prepare output array
    output = np.zeros_like(array, dtype=float)

    # Perform convolution
    for i in range(array_height):
        for j in range(array_width):
            # Extract the current region of interest from the padded array
            region = padded_array[i:i + kernel_height, j:j + kernel_width]

            # Element-wise multiplication and summation to calculate convolution value
            original_value = array[i, j]
            sample_value = np.random.choice(region.flatten(), size=1, replace=False)
            if sample_value > original_value:
                if i % 2 == 0 and j % 2 == 0 or i % 2 == 1 and j % 2 == 1:
                    output[i, j] = sample_value
                else:
                    output[i, j] = original_value
            else:
                if i % 2 == 1 and j % 2 == 0 or i % 2 == 0 and j % 2 == 1:
                    output[i, j] = sample_value
                else:
                    output[i, j] = original_value
    return output


def run():
    gra = generate_toon_oval_gradient(192, 128, 224, 64, 52, 64, 72, 5)
    cnv = sample_convolution(gra, 8)
    cnv = sample_convolution(cnv, 6)
    # img.show()
    # img = Image.open('c:/Users/Administrator/Downloads/swat2-basic.png')
    img = Image.fromarray(cnv)
    w, h = img.size
    res = img.resize((w * 4, h * 4), resample=Image.Resampling.NEAREST)
    # res.save('../data/upscale-res.png')
    res.show()


if __name__ == '__main__':
    run()
