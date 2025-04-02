import os
import cv2
import tqdm
import torch
import numpy as np

from functools import partial
from collections import defaultdict
from multiprocessing.pool import ThreadPool

from sympy.sets.fancysets import normalize_theta_set

from .models import BiSeNet
from .models import RRDBNet
from .models import RetinaFace

from .utils import (
    STANDARD_LANDMARKS_5,
    parse_landmarks_file,
    get_ldm_slices,
    as_numpy,
    as_tensor,
    read_images,
    as_batch,
    extend_bbox,
    scale_bbox
)


class Cropper():
    """Face cropper class with bonus features.

    This class is capable of automatically aligning and center-cropping 
    faces, enhancing image quality and grouping the extracted faces 
    according to specified face attributes, as well as generating masks 
    for those attributes.
    
    Capabilities
    ------------

    This class has the following 3 main features:

        1. **Face cropping** - automatic face alignment and cropping 
           based on landmarks. The landmarks can either be predicted via 
           face detection model (see :class:`.RetinaFace`) or they 
           can be provided as txt, csv, json etc. file. It is possible 
           to control face factor in the extracted images and strategy 
           of extraction (e.g., largest face, all faces per image).
        2. **Face enhancement** - automatic quality enhancement of 
           images where the relative face area is small. For instance, 
           there may be images with many faces, but the quality of those 
           faces, if zoomed in, is low. Quality enhancement feature 
           allows to remove the blurriness. It can also enhance the 
           quality of every image, if desired (see 
           :class:`.RRDBNet`).
        3. **Face parsing** - automatic face attribute parsing and 
           grouping to sub-directories according selected attributes. 
           Attributes can indicate to group faces that contain specific 
           properties, e.g., "earrings and necklace", "glasses". They 
           can also indicate what properties the faces should not 
           include to form a group, e.g., "no accessories" group would 
           indicate to include faces without hats, glasses, earrings, 
           necklace etc. It is also possible to generate masks for 
           selected face attributes, e.g., "glasses", 
           "eyes and eyebrows". For more intuition on how grouping 
           works, see :class:`.BiSeNet` and 
           :meth:`save_groups`.

    The class is designed to perform all or some combination of the 
    functions in one go, however, each feature is independent of one 
    another and can work one by one. For example, it is possible to 
    first extract all the faces in some output directory, then apply 
    quality enhancement for every face to produce better quality faces 
    in another output directory and then apply face parsing to group 
    faces into different sub-folders according to some common attributes 
    in a final output directory.

    It is possible to configure the number of processing units and the 
    batch size for significant speedups., if the hardware allows.

    Examples
    --------

    Command line example
        >>> python face_crop_plus -i path/to/images -o path/to/out/dir

    Auto face cropping (with face factor) and quality enhancement:
        >>> cropper = Cropper(face_factor=0.7, enh_threshold=0.01)
        >>> cropper.process_dir(input_dir="path/to/images")

    Very fast cropping with already known landmarks (no enhancement):
        >>> cropper = Cropper(landmarks="path/to/landmarks.txt", 
                              num_processes=24,
                              enh_threshold=None)
        >>> cropper.process_dir(input_dir="path/to/images")

    Face cropping to attribute groups to custom output dir:
        >>> attr_groups = {"glasses": [6], "no_glasses_hats": [-6, -18]}
        >>> cropper = Cropper(attr_groups=attr_groups)
        >>> inp, out = "path/to/images", "path/to/parent/out/dir"
        >>> cropper.process_dir(input_dir=inp, output_dir=out)

    Face cropping and grouping by face attributes (+ generating masks):
        >>> groups = {"glasses": [6], "eyes_and_eyebrows": [2, 3, 4, 5]}
        >>> cropper = Cropper(output_format="png", mask_groups=groups)
        >>> cropper.process_dir("path/to/images")

    For grouping by face attributes, see documented face attribute 
    indices in :class:`.BiSeNet`.

    Class Attributes
    ----------------

    For how to initialize the class and to understand its functionality 
    better, please refer to class attributes initialized via 
    :meth:`__init__`. Here, further class attributes are 
    described automatically initialized via  :meth:`_init_models` and 
    :meth:`_init_landmarks_target`.

    Attributes:
        det_model (RetinaFace): Face detection model 
            (:class:`torch.nn.Module`) that is capable of detecting 
            faces and predicting landmarks used for face alignment. See 
            :class:`.RetinaFace`.
        enh_model (RRDBNet): Image quality enhancement model 
            (torch.nn.Module) that is capable of enhancing the quality 
            of images with faces. It can automatically detect which 
            faces to enhance based on average face area in the image, 
            compared to the whole image area. See :class:`.RRDBNet`.
        par_model (BiSeNet): Face parsing model (torch.nn.Module) that 
            is capable of classifying pixels according to specific face 
            attributes, e.g., "left_eye", "earring". It is able to group 
            faces to different groups and generate attribute masks. See 
            :class:`.BiSeNet`.
        landmarks_target (numpy.ndarray): Standard normalized landmarks 
            of shape  (``self.num_std_landmarks``, 2). These are scaled 
            by ``self.face_factor`` and used as ideal landmark 
            coordinates for the extracted faces. In other words, they 
            are reference landmarks used to estimate the transformation 
            of an image based on some actual set of face landmarks for 
            that image.
    """

    def __init__(
            self,
            output_size: int | tuple[int, int] | list[int] = 256,
            output_format: str | None = None,
            resize_size: int | tuple[int, int] | list[int] = 1024,
            face_factor: float = 0.65,
            strategy: str = "largest",
            padding: str = "constant",
            allow_skew: bool = False,
            landmarks: str | tuple[np.ndarray, np.ndarray] | None = None,
            attr_groups: dict[str, list[int]] | None = None,
            mask_groups: dict[str, list[int]] | None = None,
            det_threshold: float | None = 0.6,
            enh_threshold: float | None = None,
            batch_size: int = 8,
            num_processes: int = 1,
            device: str | torch.device = "cpu",
            crop_mode: str = "aligned",
            expansion_top: float = 0.5,
            expansion_bottom: float = 0.2,
            expansion_left: float = 0.3,
            expansion_right: float = 0.3,
            **kwargs):
        """Initializes the cropper.

        Initializes class attributes. 

        Args:
            output_size: The output size (width, height) of cropped 
                image faces. If provided as a single number, the same 
                value is used for both width and height. Defaults to
                256.
            output_format: The output format of the saved face images. 
                For available options, see 
                `OpenCV imread <https://docs.opencv.org/4.x/d4/da8/group__imgcodecs.html#ga288b8b3da0892bd651fce07b3bbd3a56>`_. 
                If not specified, then the same image extension will not 
                be changed, i.e., face images will be of the same format 
                as the images from which they are extracted. Defaults to 
                None.
            resize_size: The interim size (width, height) each image 
                should be resized to before processing images. This is 
                used to resize images to a common size to allow to make 
                a batch. It should ideally be the mean width and height 
                of all the images to be processed (but can simply be a
                square). Images will be resized to to the specified size 
                while maintaining the aspect ratio (one of the 
                dimensions will always match either the specified width 
                or height). The shorter dimension would afterwards be 
                padded - for more information on how it works, see 
                :func:`.utils.create_batch_from_files`. Defaults to 
                1024.
            face_factor: The fraction of the face area relative to the 
                output image. Defaults to 0.65.
            strategy: The strategy to use to extract faces from each 
                image. The available options are:

                    * "all" - all faces will be extracted form each 
                      image.
                    * "best" - one face with the largest confidence 
                      score will be extracted from each image.
                    * "largest" - one face with the largest face area 
                      will be extracted from each image.

                For more info, see :meth:`.RetinaFace.__init__`. 
                Defaults to "largest".
            padding: The padding type (border mode) to apply when 
                cropping out faces. If faces are near edge, some part of 
                the resulting center-cropped face image may be blank, in 
                which case it can be padded with specific values. For 
                available options, see 
                `OpenCV BorderTypes <https://docs.opencv.org/3.4/d2/de8/group__core__array.html#ga209f2f4869e304c82d07739337eae7c5>`_. 
                If specified as "constant", the value of 0 will be used. 
                Defaults to "reflect".
            allow_skew: Whether to allow skewing when aligning the face 
                according to its landmarks. If True, then facial points 
                will be matched very closely to the ideal standard 
                landmark points (which is a set of reference points 
                created internally when preforming the transformation). 
                If all faces face forward, i.e., in portrait-like 
                manner, then this could be set to True which results in 
                minimal perspective changes. However, most of the time 
                this should be set to False to preserve the face 
                perspective. For more details, see 
                :meth:`.crop_align`. Defaults to False.
            landmarks: If landmarks are already known, they should be 
                specified via this variable. If specified, landmark 
                estimation will not be performed. There are 2 ways to 
                specify landmarks:

                    1. As a path to landmarks file, in which case str 
                       should be provided. The specified file should 
                       contain file (image) names and corresponding 
                       landmark coordinates. Duplicate file names are 
                       allowed (in case multiple faces are present in 
                       the same image). For instance, it could be 
                       .txt file where each row contains space-separated 
                       values: the first value is the file name and the 
                       other 136 values represent landmark coordinates 
                       in x1, y1, x2, y2, ... format. For more details 
                       about the possible file formats and how they are 
                       parsed, see 
                       :func:`~.utils.parse_landmarks_file`.
                    2. As a tuple of 2 numpy arrays. The first one is of 
                       shape (``num_faces``, ``num_landm``, 2) of type 
                       :attr:`numpy.float32` and represents the 
                       landmarks of every face that is going to be 
                       extracted from images. The second is a numpy 
                       array of shape (``num_faces``,) of type 
                       :class:`numpy.str_` where each value specifies a 
                       file name to which a corresponding set of 
                       landmarks belongs.

                If not specified, 5 landmark coordinates will be 
                estimated for each face automatically. Defaults to None.
            attr_groups: Attribute groups dictionary that specifies how
                to group the output face images according to some common
                attributes. The keys are names describing some common
                attribute, e.g., "glasses", "no_accessories" and the 
                values specify which attribute indices belong (or don't 
                belong, if negative) to that group, e.g., [6], 
                [-6, -9, -15]. For more information, see 
                :class:`.BiSeNet` and :meth:`save_groups`. 
                If not provided, output images will not be grouped by 
                attributes and no attribute sub-folders will be created
                in the desired output directory. Defaults to None.
            mask_groups: Mask groups dictionary that specifies how to 
                group the output face images according to some face
                attributes that make up a segmentation mask. The keys 
                are mask type names, e.g., "eyes", and the values 
                specify which attribute indices should be considered for 
                that mask, e.g., [4, 5]. For every group, not only face 
                images will be saved in a corresponding sub-directory, 
                but also black and white face attribute masks (white 
                pixels indicating the presence of a mask attribute). For 
                more details, see For more info, see 
                :class:`.BiSeNet` and :py:meth:`save_groups`.
                If not provided, no grouping is applied. Defaults to 
                None.
            det_threshold: The visual threshold, i.e., minimum 
                confidence score, for a detected face to be considered 
                an actual face. See :meth:`.RetinaFace.__init__` for 
                more details. If None, no face detection will be 
                performed. Defaults to 0.6.
            enh_threshold: Quality enhancement threshold that tells when 
                the image quality should be enhanced (it is an expensive 
                operation). It is the minimum average face factor, i.e., 
                face area relative to the image, below which the whole 
                image is enhanced. It is advised to set this to a low 
                number, like 0.001 - very high fractions might 
                unnecessarily cause the image quality to be improved.
                Defaults to None.
            batch_size: The batch size. It is the maximum number of 
                images that can be processed by every processor at a 
                single time-step. Large values may result in memory 
                errors, especially, when GPU acceleration is used. 
                Increase this if less models (i.e., landmark detection, 
                quality enhancement, face parsing models) are used and 
                decrease otherwise. Defaults to 8.
            num_processes: The number of processes to launch to perform 
                image processing. Each process works in parallel on 
                multiple threads, significantly increasing the 
                performance speed. Increase if less prediction models 
                are used and increase otherwise. Defaults to 1.
            device: The device on which to perform the predictions, 
                i.e., landmark detection, quality enhancement and face 
                parsing. If landmarks are provided, no enhancement and 
                no parsing is desired, then this has no effect. Defaults
                to "cpu".
        """
        # Init specified attributes
        self.output_size = output_size
        self.output_format = output_format
        self.resize_size = resize_size
        self.face_factor = face_factor
        self.strategy = strategy
        self.padding = padding
        self.allow_skew = allow_skew
        self.landmarks = landmarks
        self.attr_groups = attr_groups
        self.mask_groups = mask_groups
        self.det_threshold = det_threshold
        self.enh_threshold = enh_threshold
        self.batch_size = batch_size
        self.num_processes = num_processes
        self.device = device
        self.crop_mode = crop_mode
        self.expansion_top = expansion_top
        self.expansion_bottom = expansion_bottom
        self.expansion_left = expansion_left
        self.expansion_right = expansion_right

        # The only option for STD
        self.num_std_landmarks = 5

        # Modify attributes to have proper type
        if isinstance(self.output_size, int):
            self.output_size = (self.output_size, self.output_size)

        if len(self.output_size) == 1:
            self.output_size = (self.output_size[0], self.output_size[0])

        if isinstance(self.resize_size, int):
            self.resize_size = (self.resize_size, self.resize_size)

        if len(self.resize_size) == 1:
            self.resize_size = (self.resize_size[0], self.resize_size[0])

        if isinstance(self.device, str):
            self.device = torch.device(device)

        if isinstance(self.landmarks, str):
            self.landmarks = parse_landmarks_file(self.landmarks)

        # Further attributes
        self._init_models()
        self._init_landmarks_target()

    def _init_models(self):
        """Initializes detection, enhancement and parsing models.

        The method initializes 3 models:
            1. If ``self.det_threshold`` is provided and no landmarks 
               are known in advance, the detection model is initialized 
               to estimate 5-point landmark coordinates. For more info, 
               see :class:`.RetinaFace`.
            2. If ``self.enh_threshold`` is provided, the quality 
               enhancement model is initialized. For more info, see
               :class:`.RRDBNet`.
            3. If ``self.attr_groups`` or ``self.mask_groups`` is 
               provided, then face parsing model is initialized. For 
               more info, see :class:`.BiSeNet`.

        Note:
            This is a useful initializer function if multiprocessing is 
            used, in which case copies of all the models can be created 
            on separate cores.
        """
        # Init models as None
        self.det_model = None
        self.enh_model = None
        self.par_model = None

        if torch.cuda.is_available() and self.device.index is not None:
            # Helps to prevent CUDA memory errors
            torch.cuda.set_device(self.device.index)
            torch.cuda.empty_cache()

        if self.det_threshold is not None and self.landmarks is None:
            # If detection threshold is set, we will predict landmarks
            self.det_model = RetinaFace(self.strategy, self.det_threshold)
            self.det_model.load(device=self.device)

        if self.enh_threshold is not None:
            # If enhancement threshold is set, we might enhance quality
            self.enh_model = RRDBNet(self.enh_threshold)
            self.enh_model.load(device=self.device)

        if self.attr_groups is not None or self.mask_groups is not None:
            # If grouping by attributes or masks is set, use parse model
            args = (self.attr_groups, self.mask_groups, self.batch_size)
            self.par_model = BiSeNet(*args)
            self.par_model.load(device=self.device)

    def _init_landmarks_target(self):
        """Initializes target landmarks set.

        This method initializes a set of standard landmarks. Standard, 
        or target, landmarks refer to an average set of landmarks with 
        ideal normalized coordinates for each facial point. The source 
        facial points will be rotated, scaled and translated to match 
        the standard landmarks as close as possible.

        Both source (computed separately for each image) and target 
        landmarks must semantically match, e.g., the left eye coordinate 
        in target landmarks also corresponds to  the left eye coordinate 
        in source landmarks.

        There should be a standard landmarks set defined for a desired 
        number of landmarks. Each coordinate in that set is normalized, 
        i.e., x and y values are between 0 and 1. These values are then 
        scaled based on face factor and resized to match the desired 
        output size as defined by ``self.output_size``.

        Note:
            Currently, only 5 standard landmarks are supported.

        Raises:
            ValueError: If the number of standard landmarks is not 
                supported. The number of standard landmarks is 
                ``self.num_std_landmarks``.
        """
        match self.num_std_landmarks:
            case 5:
                # If the number of std landmarks is 5
                std_landmarks = STANDARD_LANDMARKS_5.copy()
            case _:
                # Otherwise the number of STD landmarks is not supported
                raise ValueError(f"Unsupported number of standard landmarks "
                                 f"for estimating alignment transform matrix: "
                                 f"{self.num_std_landmarks}.")

        # Apply appropriate scaling based on face factor and out size
        std_landmarks[:, 0] *= self.output_size[0] * self.face_factor
        std_landmarks[:, 1] *= self.output_size[1] * self.face_factor

        # Add an offset to standard landmarks to center the cropped face
        std_landmarks[:, 0] += (1 - self.face_factor) * self.output_size[0] / 2
        std_landmarks[:, 1] += (1 - self.face_factor) * self.output_size[1] / 2

        # Pass STD landmarks as target landms
        self.landmarks_target = std_landmarks

    def align_face(self, image: np.ndarray, landmarks_source: np.ndarray) -> np.ndarray:
        """
        Applies an affine transformation to the unpadded image based on the detected landmarks,
        producing a rotated (aligned) image that contains the entire rotated content.

        Args:
            image (np.ndarray): The unpadded image (for example, the original image resized
                                while preserving aspect ratio, without extra padding).
            landmarks_source (np.ndarray): Detected landmarks for the face in the image (shape: (num_landmarks, 2)).

        Returns:
            np.ndarray: The rotated (aligned) image. Its dimensions are determined based on the rotated corners.
        """
        # Choose transformation function.
        transform_function = cv2.estimateAffinePartial2D if not self.allow_skew else cv2.estimateAffine2D

        # Compute the affine transformation matrix mapping source landmarks to target landmarks.
        result = transform_function(landmarks_source, self.landmarks_target, ransacReprojThreshold=np.inf)
        M = result[0] if result is not None else None
        if M is None:
            print("Warning: Could not compute transformation matrix. Returning original image.")
            return image

        # Compute the new canvas size and translation offset based on the rotated image corners.
        (out_w, out_h), (min_x, min_y) = Cropper.compute_rotated_size(image, M)
        # Adjust the transformation matrix so that the rotated image fits entirely.
        M_adjusted = M.copy()
        M_adjusted[0, 2] -= min_x
        M_adjusted[1, 2] -= min_y

        # Apply the affine transformation.
        aligned = cv2.warpAffine(image, M_adjusted, (out_w, out_h),
                                 borderMode=getattr(cv2, f"BORDER_{self.padding.upper()}"))
        return aligned

    def crop_aligned_face(self, aligned_image: np.ndarray, bbox: tuple[float, float, float, float]) -> np.ndarray:
        """
        Crops the aligned image using the provided bounding box.

        Args:
            aligned_image (np.ndarray): The rotated (aligned) image.
            bbox (tuple[float, float, float, float]): Bounding box (x1, y1, x2, y2) to crop from the aligned image.

        Returns:
            np.ndarray: The cropped region of the aligned image.
        """
        x1, y1, x2, y2 = bbox
        return aligned_image[int(round(y1)):int(round(y2)), int(round(x1)):int(round(x2))]

    def save_group(
            self,
            faces: np.ndarray,
            file_names: list[str],
            output_dir: str,
    ):
        """Saves a group of images to output directory.

        Takes in a batch of faces or masks as well as corresponding file 
        names from where the faces were extracted and saves the 
        faces/masks to a specified output directory with the same names 
        as those image files (appends counter suffixes if multiple faces 
        come from the same file). If the batch of face images/masks is 
        empty, then the output directory is not created either.

        Args:
            faces: Face images (cropped and aligned) represented as a
                numpy array of shape (N, H, W, 3) with values of type
                :attr:`numpy.uint8` ranging from 0 to 255. It may also 
                be face mask of shape (N, H, W) with values of 255 where 
                some face attribute is present and 0 elsewhere.
            file_names: The list of filenames of length N. Each face 
                comes from a specific file whose name is also used to 
                save the extracted face. If ``self.strategy`` allows 
                multiple faces to be extracted from the same file, such 
                as "all", counters at the end of filenames are added.
            output_dir: The output directory to save ``faces``.
        """
        if len(faces) == 0:
            # Just return
            return

        # Create output directory, name counts
        os.makedirs(output_dir, exist_ok=True)
        file_name_counts = defaultdict(lambda: -1)

        for face, file_name in zip(faces, file_names):
            # Split each filename to base name, ext
            name, ext = os.path.splitext(file_name)

            if self.output_format is not None:
                # If specific img format given
                ext = '.' + self.output_format

            if self.strategy == "all":
                # Attach numbering to filenames
                file_name_counts[file_name] += 1
                name += f"_{file_name_counts[file_name]}"

            if face.ndim == 3:
                # If it's a colored img (not a mask), to BGR
                face = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)

            # Make image path based on file format and save
            file_path = os.path.join(output_dir, name + ext)
            cv2.imwrite(file_path, face)

    def save_groups(
            self,
            faces: np.ndarray,
            file_names: np.ndarray,
            output_dir: str,
            attr_groups: dict[str, list[int]] | None,
            mask_groups: dict[str, tuple[list[int], np.ndarray]] | None,
    ):
        """Saves images (and masks) group-wise.

        This method takes a batch of face images of equal dimensions, a 
        batch of file names identifying which image each face comes 
        from, and, optionally, attribute and/or mask groups telling how 
        to split the face images (and masks) across different folders.
        This method then loops through all the groups and saves images 
        accordingly.

        Example 1:
            If neither ``attr_groups`` nor ``mask_groups`` are provided, 
            the face images will be saved according to this structure::

                ├── output_dir
                |    ├── face_image_0.jpg
                |    ├── face_image_1.png
                |    ...

        Example 2:
            If only ``attr_groups`` is provided (keys are names 
            describing common attributes across faces in that group and 
            they are also sub-directories of ``output_dir``), the 
            structure is as follows::

                ├── output_dir
                |    ├── attribute_group_1
                |    |    ├── face_image_0.jpg
                |    |    ├── face_image_1.png
                |    |    ...
                |    ├── attribute_group_2
                |    ...

        Example 3:
            If only ``mask_groups`` is provided (keys are names 
            describing the mask type and they are also sub-directories 
            of ``output_dir``), the structure is as follows::

                ├── output_dir
                |    ├── group_1
                |    |    ├── face_image_0.jpg
                |    |    ├── face_image_1.png
                |    |    ...
                |    ├── group_1_mask
                |    |    ├── face_image_0.jpg
                |    |    ├── face_image_1.png
                |    |    ...
                |    ├── group_2
                |    |    ...
                |    ├── group_2_mask
                |    |    ...
                |    ...

        Example 4:
            If both ``attr_groups`` and ``mask_groups`` are provided, 
            then all images and masks will first be grouped by 
            attributes and then by mask groups. The structure is then as 
            follows::

                ├── output_dir
                |    ├── attribute_group_1
                |    |    ├── group_1_mask
                |    |    |    ├── face_image_0.jpg
                |    |    |    ├── face_image_1.png
                |    |    |    ...
                |    |    ├── group_1_mask
                |    |    |    ├── face_image_0.jpg
                |    |    |    ├── face_image_1.png
                |    |    |    ...
                |    |    ├── group_2
                |    |    |    ...
                |    |    ├── group_2_mask
                |    |    |    ...
                |    |    ...
                |    |
                |    ├── attribute_group_2
                |    |    ...
                |    ...

        Args:
            faces: Face images (cropped and aligned) represented as a
                numpy array of shape (N, H, W, 3) with values of type
                :attr:`numpy.uint8` ranging from 0 to 255.
            file_names: File names of images from which the faces were 
                extracted. This value is a numpy array of shape (N,) 
                with values of type :class:`numpy.str_`. Each nth 
                face in ``faces`` maps to exactly one file nth name in
                this array, thus there may be duplicate file names
                (because different faces may come from the same file).
            output_dir: The output directory where the faces or folders 
                of faces will be saved to.
            attr_groups: Face groups by attributes. Each key represents 
                the group name (describes common attributes across
                faces) and each value is a list of indices identifying 
                faces (from `faces`) that should go to that group.
            mask_groups: Face groups by extracted masks. Each key
                represents group name (describes the mask type) and each 
                value is a tuple where the first element is a list of 
                indices identifying faces (from ``faces``) that should 
                go to that group and the second element is a batch of 
                masks corresponding to indexed faces represented as 
                numpy arrays of shape (N, H, W) with values of type 
                :attr:`numpy.uint8` and being either 0 (negative) or 255 
                (positive).
        """
        if attr_groups is None:
            # No-name group of idx mapping to all faces
            attr_groups = {'': list(range(len(faces)))}

        if mask_groups is None:
            # No-name group mapping to all faces, with no masks
            mask_groups = {'': (list(range(len(faces))), None)}

        for attr_name, attr_indices in attr_groups.items():
            for mask_name, (mask_indices, masks) in mask_groups.items():
                # Make mask group values that fall under attribute group
                group_idx = list(set(attr_indices) & set(mask_indices))
                group_dir = os.path.join(output_dir, attr_name, mask_name)

                # Retrieve group values & save
                face_group = [faces[idx] for idx in group_idx]
                file_name_group = file_names[group_idx]
                self.save_group(face_group, file_name_group, group_dir)

                if masks is not None:
                    # Save to masks dir
                    group_dir += "_mask"
                    masks = masks[[mask_indices.index(i) for i in group_idx]]
                    self.save_group(masks, file_name_group, group_dir)

    def crop_bbox_extended(self, images, indices, bboxes, paddings):
        """
        Crops faces from images using extended bounding boxes.
        The detector's bounding box (in the resized+padded coordinate system) is first scaled back to the original image
        and then extended asymmetrically using the four expansion parameters.

        Args:
            images (list[np.ndarray]): List of original images.
            indices (list[int]): List mapping each detection to its corresponding image index.
            bboxes (np.ndarray): Array of shape (N, 4) with bounding boxes from the detector (in resized coordinates).
            paddings (np.ndarray): Array of shape (N, 4) with padding applied as [top, bottom, left, right].

        Returns:
            list[np.ndarray]: List of cropped face images.
        """
        # The resized shape is taken from self.resize_size. Note: we treat it as (height, width)
        resized_shape = (self.resize_size[1], self.resize_size[0])  # e.g., (1024, 1024)
        cropped_faces = []
        for i, img_idx in enumerate(indices):
            print(f"Image {img_idx} original size:", images[img_idx].shape[:2])
            bbox_resized = bboxes[i]
            pad_vals = paddings[img_idx]  # [top, bottom, left, right]
            # Scale the detector's bbox from the resized (padded) coordinate system back to original image.
            bbox_original = scale_bbox(bbox_resized, images[img_idx].shape, resized_shape, pad_vals)
            print(f"Scaled bounding box for detection {i}:", tuple(float(x) for x in bbox_original))
            # Extend the bounding box asymmetrically using the custom extension function.
            ext_bbox = extend_bbox(
                bbox_original,
                images[img_idx].shape,
                self.expansion_top,
                self.expansion_bottom,
                self.expansion_left,
                self.expansion_right)
            print(f"Extended bounding box for detection {i}:", tuple(float(x) for x in ext_bbox))
            cropped_face = images[img_idx][int(ext_bbox[1]):int(ext_bbox[3]), int(ext_bbox[0]):int(ext_bbox[2])]
            cropped_faces.append(cropped_face)
        return cropped_faces

    def process_batch_without_rotation(self, file_names: list[str], input_dir: str, output_dir: str):
        """
        Processes a batch of images.
        """
        images, file_names = read_images(file_names, input_dir)
        images_batch, _, paddings = as_batch(images, self.resize_size)
        images_batch = as_tensor(images_batch, self.device)

        # Get detector predictions from padded images.
        landmarks, indices, bboxes = self.det_model.predict(images_batch)
        if landmarks is not None and len(landmarks) == 0:
            return

        cropped_faces = self.crop_bbox_extended(as_numpy(images), indices, bboxes, paddings)
        self.save_group(cropped_faces, file_names[indices], output_dir)

    def process_batch_with_rotation(self, file_names: list[str], input_dir: str, output_dir: str):
        """
        Processes a batch of images and outputs the final cropped faces by:
          1. Reading the original images.
          2. Generating padded images using as_batch.
          3. Running the detector on the padded images to obtain landmarks.
          4. Converting the detected landmark coordinates from the padded image system to the original image system.
          5. Aligning (rotating) the original image using align_face with the adjusted landmarks.
          6. Generating padded images using as_batch on the rotated images.
          7. Running the detector on the new images to obtain landmarks.
          8. Extend the original images using crop_bbox_extended.
          9. Saving the final cropped face images.
        """
        # Step 1: Read original images.
        original_images, file_names = read_images(file_names, input_dir)

        # Step 2: Generate padded images (e.g., 1024x1024) using as_batch.
        padded_original_images, original_unscales, original_paddings = as_batch(original_images, self.resize_size)
        padded_original_images = as_numpy(padded_original_images)

        # Step 3: Run detector on padded images.
        images_tensor = as_tensor(padded_original_images, self.device)
        padded_original_landmarks, padded_original_indices, _ = self.det_model.predict(images_tensor)
        if padded_original_landmarks is None or len(padded_original_landmarks) == 0:
            print("No faces detected.")
            return

        # Step 4: For each detection, adjust landmarks from padded to original coordinates.
        padded_shape = (self.resize_size[1], self.resize_size[0])
        adjusted_landmarks = []
        for j, img_idx in enumerate(padded_original_indices):
            # Get the padding values for this image.
            if original_paddings is not None:
                pad_vals = original_paddings[img_idx]  # (top, bottom, left, right)
            else:
                pad_vals = (0, 0, 0, 0)
            orig_shape = original_images[img_idx].shape[:2]  # (height, width)
            adjusted = Cropper.adjust_landmarks_to_original(padded_original_landmarks[j], orig_shape, padded_shape, pad_vals)
            adjusted_landmarks.append(adjusted)
        adjusted_landmarks = np.array(adjusted_landmarks)

        # Step 5: For each detection, use the original image and the adjusted landmarks to align (rotate) the image.
        aligned_images = []
        for k, img_idx in enumerate(padded_original_indices):
            # Use the original image here.
            aligned_image = self.align_face(original_images[img_idx], adjusted_landmarks[k])
            cropped_aligned_image = Cropper.crop_empty_borders(aligned_image, threshold=10)
            aligned_images.append(cropped_aligned_image)

        # Step 6: Scale and pad the images to 1024x1024 for detection
        padded_rotated_images, padded_rotated_unscales, padded_rotated_paddings = as_batch(aligned_images, self.resize_size)
        padded_rotated_images = as_tensor(padded_rotated_images, self.device)

        # Step 7: Get detector predictions from padded images.
        landmarks, indices, bboxes = self.det_model.predict(padded_rotated_images)
        if landmarks is not None and len(landmarks) == 0:
            return

        # Step 8: Get an extended boundary box crop of the image. The extended boundary box is the box around the face
        # extended out to contain more of the image around the face and head.
        cropped_faces = self.crop_bbox_extended(as_numpy(padded_rotated_images), indices, bboxes, padded_rotated_paddings)

        # Step 9: Save the final cropped images.
        self.save_group(cropped_faces, file_names[indices], output_dir)

    def process_dir(self, input_dir: str, output_dir: str | None = None, desc: str | None = "Processing"):
        """
        Processes images in the specified input directory.

        Splits the file names into batches and processes each batch on multiple cores.
        The processing function is chosen internally based on self.crop_mode:
           - If crop_mode is "aligned", then process_batch_aligned_output is used.
           - Otherwise (e.g., "bbox"), process_batch is used.

        If output_dir is not specified, it will be set to:
             input_dir + "_" + self.crop_mode + "_faces"
        so that the output folder reflects the crop_mode.
        """
        if output_dir is None:
            output_dir = input_dir + "_" + self.crop_mode + "_faces"

        files = os.listdir(input_dir)
        bs = self.batch_size
        file_batches = [files[i:i + bs] for i in range(0, len(files), bs)]
        if len(file_batches) == 0:
            return

        from multiprocessing.pool import ThreadPool
        import tqdm
        from functools import partial

        # Choose the processing function based on crop_mode.
        if self.crop_mode == "aligned":
            print("Using aligned (rotated) processing approach.")
            process_fn = self.process_batch_with_rotation
        else:
            print("Using bounding box (non-rotated) processing approach.")
            process_fn = self.process_batch_without_rotation

        # Process batches in parallel.
        worker = partial(process_fn, input_dir=input_dir, output_dir=output_dir)
        with ThreadPool(self.num_processes, self._init_models) as pool:
            imap = pool.imap_unordered(worker, file_batches)
            if desc is not None:
                imap = tqdm.tqdm(imap, total=len(file_batches), desc=desc)
            list(imap)

    def compute_landmarks_on_aligned_image(self, aligned_images: np.ndarray) -> tuple[np.ndarray, list[int], np.ndarray]:
        """
        Detects facial landmarks on a batch of aligned images.

        Args:
            aligned_images (np.ndarray): An array of aligned images of shape (N, H, W, 3) in RGB format.

        Returns:
            tuple: A tuple (landmarks, indices, bboxes) where:
                - landmarks is a NumPy array of shape (num_faces, 5, 2) containing the detected landmarks.
                - indices is a list mapping each detection to its corresponding image index.
                - bboxes is a NumPy array of shape (num_faces, 4) containing the bounding boxes.
        """
        # Convert aligned images from (N, H, W, 3) to (N, 3, H, W) as required by the model.
        image_tensor = torch.from_numpy(aligned_images).permute(0, 3, 1, 2).float().to(self.device)
        # Run the detector on the aligned images.
        landmarks, indices, bboxes = self.det_model.predict(image_tensor)
        return landmarks, indices, bboxes

    @staticmethod
    def adjust_landmarks_to_original(landmarks: np.ndarray, original_shape: tuple[int, int],
                                     padded_shape: tuple[int, int], padding: tuple[int, int, int, int]) -> np.ndarray:
        """
        Converts landmark coordinates from the padded image coordinate system back to the original image coordinate system.

        Args:
            landmarks (np.ndarray): Array of shape (num_landmarks, 2) with landmark coordinates detected on the padded image.
            original_shape (tuple[int, int]): The original image shape as (height, width).
            padded_shape (tuple[int, int]): The shape of the padded image as (height, width) (e.g. 1024x1024).
            padding (tuple[int, int, int, int]): The padding applied to the original image to create the padded image, as (top, bottom, left, right).

        Returns:
            np.ndarray: The adjusted landmark coordinates in the original image coordinate system.
        """
        orig_h, orig_w = original_shape
        padded_h, padded_w = padded_shape
        pad_top, pad_bottom, pad_left, pad_right = padding

        # The effective region in the padded image is the padded image minus the padding.
        effective_w = padded_w - (pad_left + pad_right)
        effective_h = padded_h - (pad_top + pad_bottom)

        # Scale factors from effective region to original image.
        scale_x = orig_w / effective_w
        scale_y = orig_h / effective_h

        # Adjust the landmarks: first subtract the left/top padding, then scale.
        adjusted = (landmarks - np.array([pad_left, pad_top])) * np.array([scale_x, scale_y])
        return adjusted

    @staticmethod
    def crop_empty_borders(image: np.ndarray, threshold: int = 10) -> np.ndarray:
        """
        Crops away border rows and columns that are 'empty' (i.e. all pixel values below a threshold).

        Args:
            image (np.ndarray): Input image with shape (H, W, C).
            threshold (int): Pixel intensity threshold (0-255) below which a pixel is considered empty.

        Returns:
            np.ndarray: The cropped image with empty borders removed.
        """
        # Convert image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Create a binary mask where non-empty pixels are 1.
        mask = (gray > threshold).astype(np.uint8)

        # Find rows and columns that contain at least one non-empty pixel.
        rows = np.where(mask.sum(axis=1) > 0)[0]
        cols = np.where(mask.sum(axis=0) > 0)[0]

        if rows.size == 0 or cols.size == 0:
            # If all pixels are empty, return the original image.
            return image

        top, bottom = rows[0], rows[-1]
        left, right = cols[0], cols[-1]

        return image[top:bottom + 1, left:right + 1]

    @staticmethod
    def compute_bbox_from_landmarks(landmarks: np.ndarray) -> tuple[float, float, float, float]:
        """
        Computes a tight axis-aligned bounding box around the given landmarks.

        Args:
            landmarks (np.ndarray): Array of shape (num_landmarks, 2) with landmark coordinates.

        Returns:
            tuple: (x1, y1, x2, y2) representing the bounding box.
        """
        x1 = float(np.min(landmarks[:, 0]))
        y1 = float(np.min(landmarks[:, 1]))
        x2 = float(np.max(landmarks[:, 0]))
        y2 = float(np.max(landmarks[:, 1]))
        return (x1, y1, x2, y2)

    @staticmethod
    def compute_rotated_size(image: np.ndarray, M: np.ndarray) -> tuple[tuple[int, int], tuple[float, float]]:
        """
        Computes the size of the output canvas needed to contain the entire rotated image,
        and the minimum (x, y) coordinates of the rotated corners.

        Args:
            image (np.ndarray): The input image.
            M (np.ndarray): The 2x3 affine transformation matrix.

        Returns:
            output_size (tuple[int, int]): (width, height) of the rotated image canvas.
            offset (tuple[float, float]): The minimum (x, y) coordinates of the rotated corners.
        """
        h, w = image.shape[:2]
        # Define the image corners in homogeneous coordinates.
        corners = np.array([
            [0, 0, 1],
            [w, 0, 1],
            [w, h, 1],
            [0, h, 1]
        ], dtype=np.float32)
        # Transform the corners.
        rotated_corners = (M @ corners.T).T
        min_x = np.min(rotated_corners[:, 0])
        max_x = np.max(rotated_corners[:, 0])
        min_y = np.min(rotated_corners[:, 1])
        max_y = np.max(rotated_corners[:, 1])
        out_w = int(np.ceil(max_x - min_x))
        out_h = int(np.ceil(max_y - min_y))
        return (out_w, out_h), (min_x, min_y)
