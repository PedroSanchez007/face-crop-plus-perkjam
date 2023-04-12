import cv2
import json
import torch
import numpy as np


STANDARD_LANDMARKS_5 = np.float32([
    [0.31556875000000000, 0.4615741071428571],
    [0.68262291666666670, 0.4615741071428571],
    [0.50026249999999990, 0.6405053571428571],
    [0.34947187500000004, 0.8246919642857142],
    [0.65343645833333330, 0.8246919642857142],
])


def parse_landmarks_file(
    file_path: str,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    if file_path.endswith(".json"):
        with open(file_path, 'r') as f:
            # Read and parse
            data = json.load(f)
            filenames = np.array(data.keys())
            landmarks = np.array(data.values())
    else:
        if file_path.endswith(".csv"):
            # Set default params for csv files
            kwargs.setdefault("delimiter", ',')
            kwargs.setdefault("skip_header", 1)

        # Use the first column for filenames, the rest for landmarks
        filenames = np.genfromtxt(file_path, usecols=0, **kwargs)
        landmarks = np.genfromtxt(file_path, **kwargs)[:, 1:]
    
    return filenames, landmarks.reshape(len(landmarks), -1, 2)

def get_landmark_indices_5(num_landmarks: int) -> dict[str, int | slice]:
    match num_landmarks:
        case 5:
            indices = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
        case 12:
            indices = [(10, 11), (11, 12), (2, 3), (3, 4), (4, 5)]
        case 17:
            indices = [(2, 5), (7, 10), (10, 11), (13, 14), (16, 17)]
        case 21:
            indices = [(6, 9), (9, 12), (14, 15), (17, 18), (19, 20)]
        case 29:
            indices = [(4, 9), (13, 18), (19, 20), (22, 23), (27, 28)]
        case 49: # same as 51
            indices = [(19, 25), (25, 31), (13, 14), (31, 32), (37, 38)]
        case 68:
            indices = [(36, 42), (42, 48), (30, 31), (48, 49), (54, 55)]
        case 98:
            indices = [(60, 68), (68, 76), (54, 55), (76, 77), (82, 83)]
        case 106:
            indices = [(66, 75), (75, 84), (54, 55), (85, 86), (91, 92)]
        case _:
            raise ValueError(f"Invalid number of landmarks: {num_landmarks}")

    return [slice(*x) for x in indices]


def create_batch_from_img_path_list(
    path_list: list[str],
    padding_mode: str = "constant",
    size: int | tuple[int, int] = 512,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Creates image batch from a list of image paths

    For every image path in the given list, the image is read, resized
    to not exceed either of the dimensions specified in `size` while
    keeping the same aspect ratio and the shorter dimension is padded to
    fully match the specified size. All the images are stacked and
    returned as a batch. Variables required to transform the images back
    to the original ones (padding and scale) are also returned as a
    batch.

    Example:
        If some loaded image dimension is (1280×720) and the desired
        output `size` is specified as `(512, 256)`, then the image is
        first be resized to (455, 256) and then the width is padded from
        both sides. The final image size is (512, 256).

    Args:
        path_list: The list of paths to images.
        padding_mode: The type of padding to apply to pad the shorter
            dimension. For the available options, see <https://docs.opencv.org/3.4/d2/de8/group__core__array.html#ga209f2f4869e304c82d07739337eae7c5>.
            It can be all lowercase. Defaults to "constant".
        size: The width and the height each image should be resized +
            padded to. I.e., the spacial dimensions of the batch. If
            a single number is specified then it is the same for width
            height. Defaults to 512.

    Returns:
        tuple: A tuple of stacked torch tensors representing 3 batches - 
            a resized + padded images, unscale factor, applied padding.
    """
    # Init lists, resize dims, border type
    images, unscales, paddings = [], [], []
    size = (size, size) if isinstance(size, int) else size
    border_type = getattr(cv2, f"BORDER_{padding_mode.upper()}")

    for i, path in enumerate(path_list):
        # Read the image from the given path, convert to RGB form
        image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    
        # Get width, height, padding & check interpolation
        (h, w), m = image.shape[:2], max(*image.shape[:2])
        interpolation = cv2.INTER_AREA if m > max(size) else cv2.INTER_CUBIC

        if (ratio_w := size[0] / w) < (ratio_h := size[1] / h):
            # Based on width 
            unscale = ratio_w
            (ww,hh) = size[0], int(h * ratio_w)
            padding = [(size[1] - hh) // 2, (size[1] - hh + 1) // 2, 0, 0]
        else:
            # Based on height
            unscale = ratio_h
            (ww,hh) = int(w * ratio_h), size[1]
            padding = [0, 0, (size[0] - ww) // 2, (size[0] - ww + 1) // 2]
    
        # Pad the lower dimension with specific border type, then resize
        image = cv2.resize(image, (ww, hh), interpolation=interpolation)
        image = cv2.copyMakeBorder(image, *padding, borderType=border_type)

        # img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # cv2.imwrite(f"ddemo2/processed-{i}.jpg", img)

        # Add images, unscale, padding to lists
        images.append(torch.from_numpy(image))
        unscales.append(torch.tensor(unscale))
        paddings.append(torch.tensor(padding))

    return torch.stack(images), torch.stack(unscales), torch.stack(paddings)