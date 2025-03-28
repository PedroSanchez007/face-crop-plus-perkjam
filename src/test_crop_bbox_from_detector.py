import cv2
import numpy as np
import torch

# Import your detector class and helper functions.
from face_crop_plus.models import RetinaFace
from face_crop_plus.utils import extend_bbox, scale_bbox

def show_extended_bbox_from_detector(image, expansion_ratio=0.2):
    """
    Uses the detector to compute the face bounding box, scales it back to the original
    image dimensions, extends it, prints debug information, and draws the extended box on the image.

    Args:
        image (np.ndarray): Original image in BGR format.
        expansion_ratio (float): Fraction to extend the box (e.g., 0.2 for 20% expansion).

    Returns:
        np.ndarray: A copy of the original image with the extended bounding box drawn.
    """
    # Define the resized shape used by the detector.
    resized_shape = (1024, 1024)

    # Initialize the detector (using strategy "best" as per your config).
    device = torch.device("cpu")
    detector = RetinaFace(strategy="best", vis=0.6)
    detector.load(device)

    # Convert image from BGR to RGB since the detector expects RGB.
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Create a torch tensor of shape (1, 3, H, W) from the resized image.
    # Here, we assume the detector resizes the image internally to 1024x1024.
    # In many pipelines, the image is resized to a common size for batching.
    image_tensor = torch.from_numpy(cv2.resize(image_rgb, resized_shape)).permute(2, 0, 1).unsqueeze(0).float()

    # Call the detector's predict function.
    landmarks, indices, bboxes = detector.predict(image_tensor)

    # If no faces were detected, return the original image.
    if bboxes.size == 0 or len(bboxes) == 0:
        print("No faces detected.")
        return image.copy()

    # For strategy "best", assume one face is returned.
    bbox_resized = bboxes[0]  # Bounding box in the resized coordinate system

    # Print the original image size (in case it differs from the resized shape).
    print("Original image size (height, width):", image.shape[:2])
    print("Detector bounding box (resized coordinates):", bbox_resized)

    # Scale the bounding box from resized (1024x1024) back to the original dimensions.
    bbox_original = scale_bbox(bbox_resized, image.shape, resized_shape)
    print("Scaled bounding box (original coordinates):", bbox_original)

    # Extend the scaled bounding box.
    extended_bbox = extend_bbox(bbox_original, image.shape, expansion_ratio)
    print("Extended bounding box coordinates:", extended_bbox)

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
    # Replace with your test image path.
    image_path = "C:/source/repos/face-crop-plus-perkjam/demo/input_images/000001.jpg"
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not load the test image.")

    result_image = show_extended_bbox_from_detector(image, expansion_ratio=0.2)
    cv2.imshow("Extended Detector Bounding Box", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
