import cv2
import numpy as np

# Created once at import time — reused for every image
_HOG = cv2.HOGDescriptor(
    _winSize=(64, 128), _blockSize=(16, 16), _blockStride=(8, 8),
    _cellSize=(8, 8), _nbins=9
)


def extract_hog_features(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    resized_image = cv2.resize(gray_image, (64, 128))
    return _HOG.compute(resized_image).flatten()


def extract_color_histogram(image, bins=(8, 8, 8)):
    hist = cv2.calcHist([image], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
    return cv2.normalize(hist, hist).flatten()


def extract_spatial_histogram(image, size=(32, 32)):
    downscaled_image = cv2.resize(image, size).flatten()
    return downscaled_image


def extract_feature_dict(image, use_hog=True, use_color_hist=True, use_spatial=True):
    """Extract selected per-image descriptors as a dict, keyed by block name.

    Downstream code can pick any subset by key. Keys mirror the flags:
    'hog', 'color_hist', 'spatial'. Returned arrays are float32.
    """
    if not isinstance(image, np.ndarray):
        image = np.array(image, dtype=np.uint8)
    image = np.ascontiguousarray(image[:, :, :3], dtype=np.uint8)

    out = {}
    if use_hog:
        out['hog']        = extract_hog_features(image).astype(np.float32)
    if use_color_hist:
        out['color_hist'] = extract_color_histogram(image).astype(np.float32)
    if use_spatial:
        out['spatial']    = extract_spatial_histogram(image).astype(np.float32)
    return out


def feature_block_sizes():
    """Return the dimensionality of each descriptor block with the defaults
    used above. Useful for splitting a legacy `multimodal_features` vector
    (stored as HOG||color_hist||spatial) back into its blocks."""
    # HOG on a 64x128 grey image with blockSize 16, blockStride 8, cellSize
    # 8, nbins 9 → (64-16)/8+1 = 7 blocks per row, (128-16)/8+1 = 15 per
    # col, (16/8)^2 = 4 cells/block, 9 bins/cell → 7*15*4*9 = 3780.
    return {
        'hog':        3780,
        'color_hist': 8 * 8 * 8,
        'spatial':    32 * 32 * 3,
    }


def extract_combined_features(image,
                               landmark_mask=None,
                               robot_theta=None,
                               use_hog=True,
                               use_color_hist=True,
                               use_spatial=True):
    """
    Extract a concatenated multimodal feature vector from a single POV image.

    Parameters
    ----------
    image         : np.ndarray or list — RGB image from the Webots camera
    landmark_mask : list[int] or None  — binary mask of visible landmarks
    robot_theta   : float or None      — robot heading in radians
    use_hog       : bool — include HOG features          (default True)
    use_color_hist: bool — include colour histogram       (default True)
    use_spatial   : bool — include downsampled spatial map (default True)

    Returns
    -------
    np.ndarray — flat concatenated feature vector containing only the
                 requested descriptors, in the order: HOG · colour hist · spatial.
    """
    if not isinstance(image, np.ndarray):
        image = np.array(image, dtype=np.uint8)
    image = np.ascontiguousarray(image[:, :, :3], dtype=np.uint8)

    parts = []
    if use_hog:
        parts.append(extract_hog_features(image))
    if use_color_hist:
        parts.append(extract_color_histogram(image))
    if use_spatial:
        parts.append(extract_spatial_histogram(image))

    if landmark_mask is not None and robot_theta is not None:
        parts.append(np.asarray(landmark_mask, dtype=np.float32))
        parts.append(np.array([robot_theta], dtype=np.float32))

    if not parts:
        raise ValueError("At least one feature type must be enabled.")

    return np.concatenate(parts)
