from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np
from prep_image import extract_features
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import cv2
import matplotlib.pyplot as plt
import os


def cluster_sky_non_sky(features, image, n_clusters=3, sample_points=None, sample_labels=None):
    # Flatten the feature tensor
    height, width, num_features = features.shape
    features_flattened = features.reshape(-1, num_features)

    # If sample points and labels are provided, use them to initialize GMM
    if sample_points is not None and sample_labels is not None:
        # Convert sample points to indices
        sample_indices = np.array([point[0] * width + point[1] for point in sample_points])
        # Initialize cluster centers with the provided samples
        init_means = np.array([features_flattened[sample_indices[sample_labels == label]].mean(axis=0)
                               for label in np.unique(sample_labels)])
        gmm = GaussianMixture(n_components=n_clusters, means_init=init_means)
    else:
        gmm = GaussianMixture(n_components=n_clusters)

    # Fit GMM to the flattened features
    gmm.fit(features_flattened)

    # Get the cluster labels for each pixel
    labels = gmm.predict(features_flattened)

    # Reshape the labels back to the image shape
    labels_image = labels.reshape(height, width)

    return labels_image


class SegmentationProcessor_sky:
    def __init__(self, processor_name="nvidia/segformer-b5-finetuned-ade-640-640"):
        self.image_processor = AutoImageProcessor.from_pretrained(processor_name)
        self.model = SegformerForSemanticSegmentation.from_pretrained(processor_name)

    def get_bounding_box_v2(self, mask):
        """
        Get the bounding box of the object in a mask array, returned as a tuple of a list.

        Parameters:
        mask (np.ndarray): A 2D NumPy array representing the mask,
                           where 'True' values indicate the object.

        Returns:
        tuple of list: Coordinates of the bounding box as a list in the format [min_x, min_y, max_x, max_y].
        """
        # Find the indices of the True values
        true_indices = np.argwhere(mask)

        # Determine min and max for x (columns) and y (rows)
        min_y, min_x = true_indices.min(axis=0)
        max_y, max_x = true_indices.max(axis=0)

        # The bounding box coordinates are combined into a single list
        bounding_box = [min_x, min_y, max_x, max_y]

        # Return the list as a tuple
        return bounding_box

    def process_logits(self, model, logits, original_image_size):
        # Squeeze the logits to remove the batch dimension
        squeezed_logits = logits.squeeze(0)  # shape (num_labels, height, width)

        # Resize the squeezed tensor to match the original image size
        num_labels, height, width = squeezed_logits.shape
        resized_logits = F.interpolate(squeezed_logits.unsqueeze(0), size=(original_image_size[1], original_image_size[0]), mode='bilinear',
                                       align_corners=False)
        resized_logits = resized_logits.squeeze(0)  # shape (num_labels, original_height, original_width)

        # Apply argmax on the first axis to get the predicted labels
        predicted_labels = torch.argmax(resized_logits, dim=0)  # shape (original_height, original_width)
        masks = {}
        for label in np.unique(predicted_labels):
            masks[model.config.id2label[label]] = (predicted_labels == label).numpy().astype(
                np.uint8) * 255  # Convert to numpy array and binary mask
            # if label == self.model.config.label2id['sky']:
            #     masks[model.config.id2label[label]] = (predicted_labels == label).numpy().astype(np.uint8) * 255  # Convert to numpy array and binary mask

        return masks, predicted_labels

    def ensure_pil_image(self, image):
        if isinstance(image, np.ndarray):
            return Image.fromarray(image)
        elif isinstance(image, torch.Tensor):
            image = image.cpu().numpy()  # Move to CPU if it's on GPU
            return Image.fromarray(image)
        elif not isinstance(image, Image.Image):
            raise TypeError("Unsupported image type")
        return image

    def process_image(self, image):
        # Ensure the image is a PIL image
        image = self.ensure_pil_image(image)

        # Prepare the inputs
        inputs = self.image_processor(images=image, return_tensors="pt")

        # Get the logits from both models
        outputs_sky = self.model(**inputs)
        logits_sky = outputs_sky.logits  # shape (batch_size, num_labels, height/4, width/4)

        # Process logits to get masks
        masks, labels_mask = self.process_logits(self.model, logits_sky, image.size)

        # Combine masks
        desired_label = 'sky'
        mask_sky = masks[desired_label]
        log_mask = mask_sky > 0
        bounding_box = self.get_bounding_box_v2(log_mask)

        return log_mask, masks, bounding_box, labels_mask, self.model


def get_random_unmasked_points(mask, num_points=1000):
    """
    Randomly selects a specified number of points from an image that are masked (True values in the mask).

    Parameters:
    mask (np.array): A 2D boolean array where True values indicate the region of interest.
    num_points (int): The number of points to select. Default is 1000.

    Returns:
    np.array: An array of coordinates (x, y) of the selected points.
    """
    # Get coordinates of True values in the mask
    true_coords = np.argwhere(mask==True)

    # If there are fewer True values than num_points, return all available points
    if len(true_coords) < num_points:
        raise ValueError("The mask contains fewer True values than the requested number of points.")

    # Randomly select num_points indices from the True coordinates
    selected_indices = np.random.choice(len(true_coords), num_points, replace=False)

    # Get the selected points
    selected_points = true_coords[selected_indices]
    # selected_points = selected_points[:, [1, 0]]  # Swap x and y coordinates

    return selected_points


def concatenate_points_and_labels(*arrays_with_labels):
    """
    Concatenate multiple arrays of points with corresponding labels.

    Parameters:
    *arrays_with_labels (tuple): Tuples where each tuple contains an array of points and the corresponding label.

    Returns:
    combined_points (list): List of combined points as tuples.
    combined_labels (np.ndarray): Array of combined labels for the points.
    """
    combined_points = []
    combined_labels = []

    for points, label in arrays_with_labels:
        # Convert points to a list of tuples
        points_list = [tuple(row) for row in points]
        combined_points.extend(points_list)

        # Create labels for these points
        labels = np.full(len(points), label, dtype=int)
        combined_labels.extend(labels)

    combined_labels = np.array(combined_labels)

    return combined_points, combined_labels


def merge_masks(masks, exclude_keys=['sky', 'tree']):
    """
    Merge all masks in the dictionary except those specified in exclude_keys.

    Parameters:
    masks (dict): Dictionary of masks with keys as labels.
    exclude_keys (list): List of keys to exclude from merging.

    Returns:
    integrated_mask (np.ndarray): The merged mask.
    """
    # Initialize the integrated mask with the same shape as the first mask in the dictionary
    mask_shape = next(iter(masks.values())).shape
    integrated_mask = np.zeros(mask_shape, dtype=np.uint8)

    # Iterate through the dictionary and merge masks except the excluded ones
    for key, mask in masks.items():
        if key not in exclude_keys:
            integrated_mask = (integrated_mask + mask).astype(np.uint8)

    return ~integrated_mask


def non_max_suppression(img, D):
    M, N = img.shape
    Z = np.zeros((M, N), dtype=np.int32)
    angle = D * 180. / np.pi
    angle[angle < 0] += 180

    for i in range(1, M - 1):
        for j in range(1, N - 1):
            try:
                q = 255
                r = 255

                # angle 0
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = img[i, j + 1]
                    r = img[i, j - 1]
                # angle 45
                elif (22.5 <= angle[i, j] < 67.5):
                    q = img[i + 1, j - 1]
                    r = img[i - 1, j + 1]
                # angle 90
                elif (67.5 <= angle[i, j] < 112.5):
                    q = img[i + 1, j]
                    r = img[i - 1, j]
                # angle 135
                elif (112.5 <= angle[i, j] < 157.5):
                    q = img[i - 1, j - 1]
                    r = img[i + 1, j + 1]

                if (img[i, j] >= q) and (img[i, j] >= r):
                    Z[i, j] = img[i, j]
                else:
                    Z[i, j] = 0

            except IndexError as e:
                pass

    return Z


def sobel_filters(img):
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

    img = img.astype(float)
    Ix = cv2.filter2D(img, -1, Kx)
    Iy = cv2.filter2D(img, -1, Ky)

    G = np.hypot(Ix, Iy)
    G = G / G.max() * 255
    theta = np.arctan2(Iy, Ix)

    return (G, theta)


# Save the clustered image with a new name
def save_image_with_suffix(image_path, image):
    base, ext = os.path.splitext(image_path)
    new_image_path = f"{base}_after_gaussian-{ext}"
    cv2.imwrite(new_image_path, image)
    print(f"Image saved as: {new_image_path}")


# Define the paths to your models
processor_name = "nvidia/segformer-b5-finetuned-ade-640-640"
# Instantiate the SegmentationProcessor
seg_processor = SegmentationProcessor_sky()

# Process an image and get the masks
for i in range(1, 20):

    try:

        image_path = f"/home/kasra/PycharmProjects/DataFiltering/d2d_post_processor/assets/{i}.jpg"
        # Load the image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        feature_image = extract_features(image_rgb)
        image = Image.open(image_path)
        mask, masks, box, labels_mask, model = seg_processor.process_image(image)
        mask_tree = masks["tree"] > 0
        non_sky_tree_mask = ~np.logical_or(mask_tree, mask)
        sky_pixels = get_random_unmasked_points(mask=mask, num_points=10000)
        tree_pixels = get_random_unmasked_points(mask=mask_tree, num_points=10000)
        non_sky_pixels = get_random_unmasked_points(mask=non_sky_tree_mask, num_points=10000)

        init_points, init_labels = concatenate_points_and_labels(
                                            (non_sky_pixels, 0),
                                                           (sky_pixels, 1),
                                                           (tree_pixels, 2)
                                                            )# labels_mask[pixels[1,:][1], pixels[1,:][2]
        # Cluster the pixels
        labels_image = cluster_sky_non_sky(feature_image, image_rgb, sample_points=init_points,
                                                   sample_labels=init_labels)

        labels_image = labels_image == 1
        integrated_mask = merge_masks(masks, exclude_keys=['sky', 'tree'])
        # final_mask = labels_image[integrated_mask > 0]
        final_mask = np.zeros_like(labels_image, dtype=np.uint8)
        final_mask[integrated_mask > 0] = labels_image[integrated_mask > 0]
        final_mask = 255.0 * final_mask
        # grad, angle1 = sobel_filters(final_mask)
        # final_mask = non_max_suppression(final_mask, angle1)
        save_image_with_suffix(image_path, final_mask.astype(np.uint8))

    except:
        print("An exception occurred")

# Visualize the clustered image
# plt.figure(figsize=(20, 20))
# plt.imshow(final_mask, cmap='gray')
# plt.title('Sky and Non-Sky Clusters')
# plt.show()

