def extend_bbox(bbox, image_shape, expansion_ratio):
    """
    Extends a bounding box outward by a given percentage,
    stopping at the image boundaries.

    Args:
        bbox (tuple or list): The original bounding box (x1, y1, x2, y2).
        image_shape (tuple): The shape of the image as (height, width, ...).
        expansion_ratio (float): Fractional amount to extend each side (e.g., 0.2 for 20%).

    Returns:
        tuple: Extended bounding box (new_x1, new_y1, new_x2, new_y2),
               clipped to the image boundaries.
    """
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1

    # Calculate expansion amounts.
    delta_x = expansion_ratio * width
    delta_y = expansion_ratio * height

    # Compute new coordinates.
    new_x1 = int(max(0, x1 - delta_x))
    new_y1 = int(max(0, y1 - delta_y))

    # Image boundaries.
    image_height, image_width = image_shape[:2]
    new_x2 = int(min(image_width, x2 + delta_x))
    new_y2 = int(min(image_height, y2 + delta_y))

    return new_x1, new_y1, new_x2, new_y2


# For testing purposes, you can run this block independently.
if __name__ == "__main__":
    # Dummy bounding box and image shape.
    original_bbox = (100, 150, 300, 350)
    dummy_image_shape = (512, 512, 3)  # height, width, channels
    expansion_ratio = 0.2  # 20% expansion

    extended_bbox = extend_bbox(original_bbox, dummy_image_shape, expansion_ratio)
    print("Original bbox:", original_bbox)
    print("Extended bbox:", extended_bbox)
