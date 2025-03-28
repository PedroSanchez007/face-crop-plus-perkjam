import cv2
import numpy as np
import torch

# Import your detector class; adjust the import based on your repo structure.
from face_crop_plus.models import RetinaFace

def extend_bbox(bbox, image_shape, expansion_ratio):
    """
    Extends a bounding box outward by a given percentage,
    stopping at the image boundaries.

    Args:
        bbox (tuple or list): Original bounding box (x1, y1, x2, y2).
        image_shape (tuple): Image shape as (height, width, ...).
        expansion_ratio (float): Fraction to extend each side (e.g., 0.2 for 20% expansion).

    Returns:
        tuple: Extended bounding box (new_x1, new_y1, new_x2, new_y2), clipped to image boundaries.
    """
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1

    # Calculate expansion amounts.
    delta_x = expansion_ratio * width
    delta_y = expansion_ratio * height

    # Compute new coordinates, ensuring they stay within image bounds.
    new_x1 = int(max(0, x1 - delta_x))
    new_y1 = int(max(0, y1 - delta_y))
    image_height, image_width = image_shape[:2]
    new_x2 = int(min(image_width, x2 + delta_x))
    new_y2 = int(min(image_height, y2 + delta_y))

    return new_x1, new_y1, new_x2, new_y2

def show_extended_bbox_from_detector(image, expansion_ratio=0.2):
    """
    Uses the detector to compute the face bounding box, extends it,
    and draws the extended box on the original image.

    Args:
        image (np.ndarray): Original image in BGR format.
        expansion_ratio (float): Fraction to extend the box (e.g., 0.2 for 20% expansion).

    Returns:
        np.ndarray: A copy of the original image with the extended bounding box drawn.
    """
    # Initialize the detector (using strategy "best" as per your config).
    device = torch.device("cpu")
    detector = RetinaFace(strategy="best", vis=0.6)
    detector.load(device)

    # Convert image from BGR to RGB since the detector expects RGB.
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Create a torch tensor of shape (1, 3, H, W) and float type.
    image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).unsqueeze(0).float()

    # Call the detector's predict function.
    # Expected to return landmarks, indices, and bounding boxes.
    landmarks, indices, bboxes = detector.predict(image_tensor)

    # If no faces were detected, return the original image.
    if bboxes.size == 0 or len(bboxes) == 0:
        print("No faces detected.")
        return image.copy()

    # Since strategy is "best", assume one face is returned.
    bbox = bboxes[0]  # [x1, y1, x2, y2]

    # Compute the extended bounding box.
    extended_bbox = extend_bbox(bbox, image.shape, expansion_ratio)

    # Draw the extended bounding box on a copy of the original image.
    image_with_box = image.copy()
    cv2.rectangle(
        image_with_box,
        (int(extended_bbox[0]), int(extended_bbox[1])),
        (int(extended_bbox[2]), int(extended_bbox[3])),
        (0, 0, 255),  # Red color for the extended box.
        2
    )

    return image_with_box

if __name__ == "__main__":
    # Replace this path with your test image path.
    image_path = "C:/source/repos/face-crop-plus-perkjam/demo/input_images/000004.jpg"
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not load the test image.")

    # Get the image with the extended detector's bounding box drawn on it.
    result_image = show_extended_bbox_from_detector(image, expansion_ratio=0.2)

    # Display the result in a single window.
    cv2.imshow("Extended Detector Bounding Box", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
