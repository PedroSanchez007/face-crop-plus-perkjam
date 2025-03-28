import cv2
import numpy as np
import torch

# Import your detector and helper functions.
from face_crop_plus.models import RetinaFace
from face_crop_plus.utils import extend_bbox, scale_bbox

def show_extended_bbox_from_detector(image, expansion_top, expansion_bottom, expansion_left, expansion_right):
    """
    Uses the detector to compute the face bounding box, scales it from the resized (padded) coordinate
    system back to the original image dimensions, extends it, prints debug info, and draws it on the image.

    Args:
        image (np.ndarray): Original image in BGR format.
        expansion_ratio (float): Fraction to extend the box (e.g., 0.2 for 20% expansion).

    Returns:
        np.ndarray: A copy of the original image with the extended bounding box drawn.
    """
    # Define the resized shape used by the detector.
    resized_shape = (1024, 1024)
    # For this test, assume no padding was applied:
    padding = (0, 0, 0, 0)

    # Initialize the detector.
    device = torch.device("cpu")
    detector = RetinaFace(strategy="best", vis=0.6)
    detector.load(device)

    # Convert image from BGR to RGB.
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Resize the image to 1024x1024 (without padding) for this test.
    resized_image_rgb = cv2.resize(image_rgb, resized_shape)
    image_tensor = torch.from_numpy(resized_image_rgb).permute(2, 0, 1).unsqueeze(0).float()

    # Call the detector's predict function.
    landmarks, indices, bboxes = detector.predict(image_tensor)
    if bboxes.size == 0 or len(bboxes) == 0:
        print("No faces detected.")
        return image.copy()

    # Assume one face is returned.
    bbox_resized = bboxes[0]
    print("Original image size (height, width):", image.shape[:2])
    print("Detector bounding box (resized coordinates):", tuple(float(x) for x in bbox_resized))

    # Scale the bounding box to original coordinates.
    bbox_original = scale_bbox(bbox_resized, image.shape, resized_shape, padding)
    print("Scaled bounding box (original coordinates):", tuple(float(x) for x in bbox_original))

    # Extend the scaled bounding box.
    extended_bbox = extend_bbox(bbox_original, image.shape, expansion_top, expansion_bottom, expansion_left, expansion_right)
    print("Extended bounding box coordinates:", tuple(float(x) for x in extended_bbox))

    # Draw the extended bounding box on a copy of the original image.
    image_with_box = image.copy()
    cv2.rectangle(
        image_with_box,
        (int(extended_bbox[0]), int(extended_bbox[1])),
        (int(extended_bbox[2]), int(extended_bbox[3])),
        (0, 0, 255), 2
    )
    return image_with_box

if __name__ == "__main__":
    image_path = "C:/source/repos/face-crop-plus-perkjam/demo/input_images/000001.jpg"
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not load the test image.")

    result_image = show_extended_bbox_from_detector(image, expansion_top=0.5, expansion_bottom=0.2, expansion_left=0.3, expansion_right=0.3)
    cv2.imshow("Extended Detector Bounding Box", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
