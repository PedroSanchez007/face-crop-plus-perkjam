import cv2
import numpy as np
import torch

# Import your detector class; adjust the import based on your repo structure.
from face_crop_plus.models import RetinaFace


def crop_bbox_from_detector(image):
    """
    Calls the detector to compute the face boundary box, then draws the box
    on the image and returns both the image with the drawn boundary and the cropped face region.

    Args:
        image (np.ndarray): Original image in BGR format.

    Returns:
        tuple: (image_with_box, cropped_face) where:
            - image_with_box is a copy of the original image with the detected face box drawn.
            - cropped_face is the face region cropped using the detector's bounding box.
              If no face is detected, cropped_face is None.
    """
    # Initialize the detector (using strategy "best" as per your config).
    device = torch.device("cpu")
    detector = RetinaFace(strategy="best", vis=0.6)
    # Load the model weights (this call may vary in your implementation)
    detector.load(device)

    # Convert image from BGR to RGB since the detector expects RGB input.
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Convert the image to a torch tensor of shape (1, 3, H, W) and float type.
    image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).unsqueeze(0).float()

    # Call the detector's predict function.
    # It should return landmarks, indices, and bounding boxes.
    landmarks, indices, bboxes = detector.predict(image_tensor)

    # If no faces were detected, return the original image.
    if bboxes.size == 0 or len(bboxes) == 0:
        print("No faces detected.")
        return image.copy(), None

    # Since strategy is "best", assume one face is returned.
    bbox = bboxes[0]  # [x1, y1, x2, y2]

    # Draw the bounding box on a copy of the original image.
    image_with_box = image.copy()
    cv2.rectangle(
        image_with_box,
        (int(bbox[0]), int(bbox[1])),
        (int(bbox[2]), int(bbox[3])),
        (0, 255, 0),
        2
    )

    # Crop the face region from the image.
    cropped_face = image[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

    return image_with_box, cropped_face


if __name__ == "__main__":
    # Replace this path with your test image path.
    image_path = "C:/source/repos/face-crop-plus-perkjam/demo/input_images/000004.jpg"
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not load the test image.")

    # Get image with the detector's boundary box and the cropped face region.
    image_with_box, cropped_face = crop_bbox_from_detector(image)

    # Display the image with bounding box.
    cv2.imshow("Detector Bounding Box", image_with_box)

    # Display the cropped face if available.
    if cropped_face is not None:
        cv2.imshow("Cropped Face", cropped_face)
    else:
        print("No cropped face to display.")

    cv2.waitKey(0)
    cv2.destroyAllWindows()
