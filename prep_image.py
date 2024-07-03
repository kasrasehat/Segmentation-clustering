import cv2
import numpy as np
import pywt
from skimage.feature import local_binary_pattern, hog


def extract_features(image_rgb, kernel_size=5, lbp_radius=1, lbp_points=8, hog_orientations=8,
                     hog_pixels_per_cell=(16, 16), hog_cells_per_block=(1, 1), gabor_frequencies=[0.1, 0.2, 0.3],
                     gabor_orientations=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]):

    # Convert the image to HSV color space
    image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    image_with_hsv = np.concatenate((image_rgb, image_hsv), axis=2)

    # Convert the image to LAB color space
    image_lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    image_with_lab = np.concatenate((image_with_hsv, image_lab), axis=2)

    # Compute the mean color for each channel in the neighborhood
    mean_color_r = cv2.blur(image_rgb[:, :, 0], (kernel_size, kernel_size))
    mean_color_g = cv2.blur(image_rgb[:, :, 1], (kernel_size, kernel_size))
    mean_color_b = cv2.blur(image_rgb[:, :, 2], (kernel_size, kernel_size))
    mean_color_image = np.stack((mean_color_r, mean_color_g, mean_color_b), axis=2) / 255.0
    image_with_mean_color = np.concatenate((image_with_lab, mean_color_image), axis=2)

    # Compute the standard deviation for each channel in the neighborhood
    def compute_local_std(image_channel, kernel_size):
        mean = cv2.blur(image_channel, (kernel_size, kernel_size))
        mean_sq = cv2.blur(image_channel ** 2, (kernel_size, kernel_size))
        variance = mean_sq - mean ** 2
        return np.sqrt(variance)

    std_color_r = compute_local_std(image_rgb[:, :, 0], kernel_size)
    std_color_g = compute_local_std(image_rgb[:, :, 1], kernel_size)
    std_color_b = compute_local_std(image_rgb[:, :, 2], kernel_size)
    std_color_image = np.stack((std_color_r, std_color_g, std_color_b), axis=2)
    image_with_std_color = np.concatenate((image_with_mean_color, std_color_image), axis=2)

    # Compute the LBP for the grayscale image
    image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    lbp_image = local_binary_pattern(image_gray, lbp_points, lbp_radius, method='uniform')
    image_with_lbp = np.concatenate((image_with_std_color, lbp_image[:, :, np.newaxis]), axis=2)

    # Apply Gabor filters to the grayscale image
    def apply_gabor_filters(image, frequencies, orientations):
        gabor_features = []
        for frequency in frequencies:
            for theta in orientations:
                kernel = cv2.getGaborKernel((5, 5), 4.0, theta, frequency, 0.5, 0, ktype=cv2.CV_32F)
                filtered_image = cv2.filter2D(image, cv2.CV_8UC3, kernel)
                gabor_features.append(filtered_image)
        return np.stack(gabor_features, axis=2)

    gabor_features = apply_gabor_filters(image_gray, gabor_frequencies, gabor_orientations)
    image_with_gabor = np.concatenate((image_with_lbp, gabor_features), axis=2)

    # Compute the Sobel edge magnitude
    sobel_x = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=5)
    sobel_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    sobel_magnitude = cv2.normalize(sobel_magnitude, None, 0, 255, cv2.NORM_MINMAX)
    sobel_magnitude = sobel_magnitude[:, :, np.newaxis]
    image_with_sobel = np.concatenate((image_with_gabor, sobel_magnitude), axis=2)

    # Compute the HOG features
    hog_features, hog_image = hog(image_gray, orientations=hog_orientations, pixels_per_cell=hog_pixels_per_cell,
                                  cells_per_block=hog_cells_per_block, visualize=True, feature_vector=False)
    hog_features_reshaped = np.zeros_like(image_gray, dtype=float)
    cell_size = hog_pixels_per_cell
    for i in range(hog_features.shape[0]):
        for j in range(hog_features.shape[1]):
            hog_features_reshaped[i * cell_size[0]:(i + 1) * cell_size[0], j * cell_size[1]:(j + 1) * cell_size[1]] = \
            hog_features[i, j].mean()
    hog_features_reshaped = hog_features_reshaped[:, :, np.newaxis]
    image_with_hog = np.concatenate((image_with_sobel, hog_features_reshaped), axis=2)

    # Generate the coordinate features
    height, width = image_gray.shape
    y_coords, x_coords = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    y_coords_normalized = y_coords / height
    coord_features = y_coords_normalized[:, :, np.newaxis]
    image_with_coords = np.concatenate((image_with_hog, coord_features), axis=2)

    # Compute the mean intensity of the surrounding pixels
    mean_intensity = cv2.blur(image_gray, (kernel_size, kernel_size))
    mean_intensity_normalized = mean_intensity / 255.0
    mean_intensity_normalized = mean_intensity_normalized[:, :, np.newaxis]
    image_with_mean_intensity = np.concatenate((image_with_coords, mean_intensity_normalized), axis=2)

    # Compute the standard deviation of the surrounding pixels
    std_intensity = compute_local_std(image_gray, kernel_size)
    min_val = np.min(std_intensity)
    max_val = np.max(std_intensity)
    std_intensity_normalized = (std_intensity - min_val) / (max_val - min_val)
    std_intensity_normalized = std_intensity_normalized[:, :, np.newaxis]
    image_with_std_intensity = np.concatenate((image_with_mean_intensity, std_intensity_normalized), axis=2)

    # Compute the standard deviation for each channel in the neighborhood
    std_color_r = compute_local_std(image_rgb[:, :, 0], kernel_size)
    std_color_g = compute_local_std(image_rgb[:, :, 1], kernel_size)
    std_color_b = compute_local_std(image_rgb[:, :, 2], kernel_size)
    std_color_image = np.stack((std_color_r, std_color_g, std_color_b), axis=2)
    std_color_image_normalized = np.zeros_like(std_color_image, dtype=float)
    for i in range(3):
        min_val = np.min(std_color_image[:, :, i])
        max_val = np.max(std_color_image[:, :, i])
        std_color_image_normalized[:, :, i] = (std_color_image[:, :, i] - min_val) / (max_val - min_val)
    image_with_std_color_all = np.concatenate((image_with_std_intensity, std_color_image_normalized), axis=2)

    # Compute DWT for each channel in the RGB image
    def compute_dwt(image_channel):
        coeffs2 = pywt.dwt2(image_channel, 'haar')
        LL, (LH, HL, HH) = coeffs2
        return LL, LH, HL, HH

    LL_r, LH_r, HL_r, HH_r = compute_dwt(image_rgb[:, :, 0])
    LL_g, LH_g, HL_g, HH_g = compute_dwt(image_rgb[:, :, 1])
    LL_b, LH_b, HL_b, HH_b = compute_dwt(image_rgb[:, :, 2])

    LL_r_resized = cv2.resize(LL_r, (image_rgb.shape[1], image_rgb.shape[0]))
    LH_r_resized = cv2.resize(LH_r, (image_rgb.shape[1], image_rgb.shape[0]))
    HL_r_resized = cv2.resize(HL_r, (image_rgb.shape[1], image_rgb.shape[0]))
    HH_r_resized = cv2.resize(HH_r, (image_rgb.shape[1], image_rgb.shape[0]))

    LL_g_resized = cv2.resize(LL_g, (image_rgb.shape[1], image_rgb.shape[0]))
    LH_g_resized = cv2.resize(LH_g, (image_rgb.shape[1], image_rgb.shape[0]))
    HL_g_resized = cv2.resize(HL_g, (image_rgb.shape[1], image_rgb.shape[0]))
    HH_g_resized = cv2.resize(HH_g, (image_rgb.shape[1], image_rgb.shape[0]))

    LL_b_resized = cv2.resize(LL_b, (image_rgb.shape[1], image_rgb.shape[0]))
    LH_b_resized = cv2.resize(LH_b, (image_rgb.shape[1], image_rgb.shape[0]))
    HL_b_resized = cv2.resize(HL_b, (image_rgb.shape[1], image_rgb.shape[0]))
    HH_b_resized = cv2.resize(HH_b, (image_rgb.shape[1], image_rgb.shape[0]))

    LL = np.stack((LL_r_resized, LL_g_resized, LL_b_resized), axis=2)
    LH = np.stack((LH_r_resized, LH_g_resized, LH_b_resized), axis=2)
    HL = np.stack((HL_r_resized, HL_g_resized, HL_b_resized), axis=2)
    HH = np.stack((HH_r_resized, HH_g_resized, HH_b_resized), axis=2)

    LL_normalized = cv2.normalize(LL, None, 0, 1, cv2.NORM_MINMAX)
    LH_normalized = cv2.normalize(LH, None, 0, 1, cv2.NORM_MINMAX)
    HL_normalized = cv2.normalize(HL, None, 0, 1, cv2.NORM_MINMAX)
    HH_normalized = cv2.normalize(HH, None, 0, 1, cv2.NORM_MINMAX)

    image_with_dwt = np.concatenate(
        (image_with_std_color_all, LL_normalized, LH_normalized, HL_normalized, HH_normalized), axis=2)

    # Compute the Canny edge detection
    edges = cv2.Canny(image_gray, threshold1=120, threshold2=160)

    # Append the edges channel to the original image
    edges = edges[:, :, np.newaxis]  # Add a new axis to match the image dimensions
    image_with_edges = np.concatenate((image_with_dwt, edges), axis=2)

    # Normalize each channel separately
    for i in range(image_with_edges.shape[2]):
        channel = image_with_edges[:, :, i]
        min_val = np.min(channel)
        max_val = np.max(channel)
        image_with_edges[:, :, i] = (channel - min_val) / (max_val - min_val)

    return image_with_edges


