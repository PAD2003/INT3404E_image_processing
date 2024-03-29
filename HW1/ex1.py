import cv2
import matplotlib.pyplot as plt
import numpy as np

def load_image(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def flip_image(image):
    flipped_image = cv2.flip(image, 1)  # Flip horizontally
    return flipped_image

def rotate_image(image, angle = 45):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, matrix, (w, h))
    return rotated_image

def grayscale_image(image):
    """
    Convert an image to grayscale. Convert the original image to a grayscale image. In a grayscale image, the pixel value of the
    3 channels will be the same for a particular X, Y coordinate. The equation for the pixel value
    [1] is given by:
        p = 0.299R + 0.587G + 0.114B
    Where the R, G, B are the values for each of the corresponding channels. We will do this by
    creating an array called img_gray with the same shape as img
    """
    
    # Extract R, G, B channels
    R = image[:, :, 0]
    G = image[:, :, 1]
    B = image[:, :, 2]

    # Calculate the grayscale value using the provided equation
    img_gray = 0.299 * R + 0.587 * G + 0.114 * B

    # Convert the dtype to uint8
    img_gray = img_gray.astype(np.uint8)
    return img_gray


def show_diference(image, func):
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    
    plt.subplot(1, 2, 2)
    image = func(image)
    if len(image.shape) == 2:
        plt.imshow(image, cmap='gray')
    else:
        plt.imshow(image)
    
    plt.show()

if __name__ == "__main__":
    image = load_image("uet.png")
    
    # show_diference(image, flip_image)
    # show_diference(image, rotate_image)
    show_diference(image, grayscale_image)