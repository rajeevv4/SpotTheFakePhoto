import cv2
import numpy as np

def load_image(path: str) -> np.ndarray:
    """Loads an image from the specified path using OpenCV.

    Args:
        path: Path to the image file.

    Returns:
        np.ndarray: Loaded image in BGR format.

    Raises:
        FileNotFoundError: If the image cannot be read or doesn't exist.
    """
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Could not read image at path: {path}")
    return image

def get_resized_gray(image: np.ndarray, size: int = 512) -> np.ndarray:
    """Converts the image to grayscale and resizes it to a square of the given size.

    Args:
        image: Source BGR image.
        size: Target width and height.

    Returns:
        np.ndarray: Grayscale resized image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)

def get_center_crop(image: np.ndarray, size: int = 256) -> np.ndarray:
    """Extracts a square crop from the center of the image at its original resolution
    and converts it to grayscale. This preserves original high-frequency details.

    Args:
        image: Source BGR image.
        size: Target width and height of the crop.

    Returns:
        np.ndarray: Grayscale cropped image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # If the image is smaller than the requested crop size, return the resized version
    if h < size or w < size:
        return cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
        
    start_y = (h - size) // 2
    start_x = (w - size) // 2
    return gray[start_y:start_y + size, start_x:start_x + size]

def get_hsv_value_channel(image: np.ndarray, size: int = 512) -> np.ndarray:
    """Converts the image to HSV and extracts the Value (brightness) channel,
    resized to a square of the given size.

    Args:
        image: Source BGR image.
        size: Target width and height.

    Returns:
        np.ndarray: Resized Value channel.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]
    return cv2.resize(v_channel, (size, size), interpolation=cv2.INTER_AREA)
