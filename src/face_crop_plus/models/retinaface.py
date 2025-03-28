import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.models._utils as _utils
from ._layers import LoadMixin, PriorBox, SSH, FPN, Head


class RetinaFace(nn.Module, LoadMixin):
    """RetinaFace face detector and 5-point landmark predictor.

    This class is capable of predicting 5-point landmarks from a batch 
    of images and filter them based on strategy, e.g., "all landmarks in 
    the image", "a single set of landmarks per image of the largest
    face". For more information, see the main method of this class
    :meth:`predict`. For main attributes, see :meth:`__init__`.

    This class also inherits ``load`` method from ``LoadMixin`` class. 
    The method takes a device on which to load the model and loads the 
    model with a default state dictionary loaded from 
    ``WEIGHTS_FILENAME`` file. It sets this model to eval mode and 
    disables gradients.

    For more information on how RetinaFace model works, see this repo:
    `PyTorch Retina Face <https://github.com/biubug6/Pytorch_Retinaface>`_. 
    Most of the code was taken from that repository.

    Note:
        Whenever an input shape is mentioned, N corresponds to batch 
        size, C corresponds to the number of channels, H - to input
        height, and W - to input width. ``out_dim`` corresponds to the
        total guesses (the number of priors) the model made about each
        sample. Within those guesses, there typically exists at least 1 
        face but can be more. By default, it should be 43,008.
    
    Be default, this class initializes the following attributes which 
    can be changed after initialization of the class (but, typically, 
    should not be changed):

    Attributes:
        nms_threshold (float): The threshold, based on which 
            multiple bounding box or landmark predictions for the same 
            face are merged into one. Defaults to 0.4.
        variance (list[int]): The variance of the bounding boxes 
            used to undo the encoding of coordinates of raw  bounding 
            box and landmark predictions.
    """
    #: WEIGHTS_FILENAME (str): The constant specifying the name of 
    #: ``.pth`` file from which the weights for this model should be 
    #: loaded. Defaults to "retinaface_detector.pth".
    WEIGHTS_FILENAME = "retinaface_detector.pth"

    def __init__(self, strategy: str = "all", vis: float = 0.6):
        """Initializes RetinaFace model.            
        
        This method initializes ResNet-50 backbone and further 
        layers required for face detection and bbox/landm predictions.

        Args:
            strategy: The strategy used to retrieve the landmarks when
                :meth:`predict` is called. The available options are:

                    * "all" - landmarks for all faces per single image
                      (single batch entry) will be considered.
                    * "best" - landmarks for a single face with the
                      highest confidence score per image will be 
                      considered.
                    * "largest" - landmarks for a single largest face
                      per image will be considered.

                The most efficient option is 'best' and the least
                efficient is "largest". Defaults to "all".
            vis: The visual threshold, i.e., minimum confidence score,
                for a face to be considered an actual face. Lower
                scores will allow the detection of more faces per image
                but can result in non-actual faces, e.g., random
                surfaces somewhat representing faces. Higher scores will 
                prevent detecting faulty faces but may result in only a
                few faces detected, whereas there can be more, e.g., 
                higher will prevent the detection of blurry faces. 
                Defaults to 0.6.
        """
        super().__init__()

        # Initialize attributes
        self.strategy = strategy
        self.vis_threshold = vis
        self.nms_threshold = 0.4
        self.variance = [0.1, 0.2]

        # Set up backbone and config
        backbone = models.resnet50()
        in_channels, out_channels = 256, 256
        in_channels_list = [in_channels * x for x in [2, 4, 8]]
        return_layers = {'layer2': 1, 'layer3': 2, 'layer4': 3}

        # Construct the backbone by retrieving intermediate layers
        self.body = _utils.IntermediateLayerGetter(backbone, return_layers)

        # Construct sub-layers to extract features for heads
        self.fpn = FPN(in_channels_list, out_channels)
        self.ssh1 = SSH(out_channels, out_channels)
        self.ssh2 = SSH(out_channels, out_channels)
        self.ssh3 = SSH(out_channels, out_channels)

        # Construct 3 heads - score, bboxes & landms
        self.ClassHead = Head.make(2, out_channels)
        self.BboxHead = Head.make(4, out_channels)
        self.LandmarkHead = Head.make(10, out_channels)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Performs forward pass.

        Takes an input batch and performs inference based on the modules 
        it has. Returns an unfiltered tuple of scores, bounding boxes 
        and landmarks for all the possible detected faces. The 
        predictions are encoded to comfortably compute the loss during 
        training and thus should be decoded to coordinates.

        Args:
            x: The input tensor of shape (N, 3, H, W).

        Returns:
            A tuple of torch tensors where the first element is
            confidence scores for each prediction of shape
            (N, out_dim, 2) with values between 0 and 1 representing
            probabilities, the second element is bounding boxes of shape 
            (N, out_dim, 4) with unbounded values and the last element 
            is landmarks of shape (N, ``out_dim``, 10) with unbounded
            values.
        """
        # Extract FPN + SSH features
        fpn = self.fpn(self.body(x))
        fts = [self.ssh1(fpn[0]), self.ssh2(fpn[1]), self.ssh3(fpn[2])]

        # Create head list and use each to process feature list
        hs = [self.ClassHead, self.BboxHead, self.LandmarkHead]
        pred = [torch.cat([h[i](f) for i, f in enumerate(fts)], 1) for h in hs]
        
        return F.softmax(pred[0], dim=-1), pred[1], pred[2]
    
    def decode_bboxes(
        self,
        loc: torch.Tensor,
        priors: torch.Tensor,
    ) -> torch.Tensor:
        """Decodes bounding boxes from predictions.

        Takes the predicted bounding boxes (locations) and undoes the 
        encoding for offset regression used at training time.

        Args:
            loc: Bounding box (location) predictions for loc layers of
                shape (N, out_dim, 4). 
            priors: Prior boxes in center-offset form of shape
                (out_dim, 4).

        Returns:
            A tensor of shape (N, out_dim, 4) representing decoded
            bounding box predictions where the last dim can be
            interpreted as x1, y1, x2, y2 coordinates - the start and
            the end corners defining the face box.
        """
        # Concatenate priors
        boxes = torch.cat((
            priors[:, :2] + loc[..., :2] * self.variance[0] * priors[:, 2:],
            priors[:, 2:] * torch.exp(loc[..., 2:] * self.variance[1])
        ), 2)
        
        # Adjust values for proper xy coords
        boxes[..., :2] -= boxes[..., 2:] / 2
        boxes[..., 2:] += boxes[..., :2]

        return boxes

    def decode_landms(
        self,
        pre: torch.Tensor,
        priors: torch.Tensor,
    ) -> torch.Tensor:
        """Decodes landmarks from predictions.

        Takes the predicted landmarks (pre) and undoes the encoding for
        offset regression used at training time.

        Args:
            pre: Landmark predictions for loc layers of shape
                (N, out_dim, 10).
            priors: Prior boxes in center-offset form of shape
                (out_dim, 4).

        Returns:
            A tensor of shape (N, out_dim, 10) representing decoded
            landmark predictions where the last dim can be
            interpreted as x1, y1, ..., x10, y10 coordinates - one for 
            each of the 5 landmarks.
        """
        # Concatenate priors
        var = self.variance
        landms = torch.cat((
            priors[..., :2] + pre[..., :2] * var[0] * priors[..., 2:],
            priors[..., :2] + pre[..., 2:4] * var[0] * priors[..., 2:],
            priors[..., :2] + pre[..., 4:6] * var[0] * priors[..., 2:],
            priors[..., :2] + pre[..., 6:8] * var[0] * priors[..., 2:],
            priors[..., :2] + pre[..., 8:10] * var[0] * priors[..., 2:],
        ), dim=2)

        return landms

    def filter_preds(
            self,
            scores: torch.Tensor,
            bboxes: torch.Tensor,
            landms: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        """
        Filters predictions for each sample and returns:
          - filtered_landms: Landmarks for each kept detection.
          - filtered_bboxes: Bounding boxes for each kept detection.
          - sample_indices: A list mapping each detection to its original image index.
        """
        cumsum = 0
        people_indices = []
        sample_indices = []
        # Create a mask of predictions above the visual threshold.
        masks = scores > self.vis_threshold

        # Apply the mask to scores, bboxes, and landmarks.
        scores = scores[masks]
        bboxes = bboxes[masks]
        landms = landms[masks]
        # Compute the area of each bounding box.
        areas = (bboxes[:, 2] - bboxes[:, 0] + 1) * (bboxes[:, 3] - bboxes[:, 1] + 1)

        # Process detections image by image.
        for i, num_valid in enumerate(masks.sum(dim=1)):
            start = cumsum
            end = cumsum + num_valid
            keep = []
            bbox = bboxes[start:end]
            area = areas[start:end]
            scores_sorted = scores[start:end].argsort(descending=True)

            while scores_sorted.numel() > 0:
                j = scores_sorted[0]
                keep.append(j)
                if scores_sorted.numel() == 1:
                    break
                # Compute the intersection over union (IoU) between the chosen bbox and the rest.
                xy1 = torch.maximum(bbox[j, :2], bbox[scores_sorted[1:], :2])
                xy2 = torch.minimum(bbox[j, 2:], bbox[scores_sorted[1:], 2:])
                w = torch.clamp(xy2[:, 0] - xy1[:, 0] + 1, min=0)
                h = torch.clamp(xy2[:, 1] - xy1[:, 1] + 1, min=0)
                inter = w * h
                ovr = inter / (area[j] + area[scores_sorted[1:]] - inter)
                # Keep detections with overlap less than the NMS threshold.
                inds = torch.where(ovr <= self.nms_threshold)[0]
                scores_sorted = scores_sorted[inds + 1]

            people_indices.extend([cumsum + k for k in keep])
            sample_indices.extend([i] * len(keep))
            cumsum += num_valid

        filtered_bboxes = bboxes[people_indices, :]
        filtered_landms = landms[people_indices, :]

        return filtered_landms, filtered_bboxes, sample_indices

    def take_by_strategy(
            self,
            landms: torch.Tensor,
            bboxes: torch.Tensor,
            idx: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        """
        Filters landmarks and bounding boxes per image according to the strategy.

        Args:
            landms: Tensor of shape (N, num_landm*2) with detected landmarks.
            bboxes: Tensor of shape (N, 4) with bounding box coordinates.
            idx: List of length N mapping each detection to its source image index.

        Returns:
            A tuple of (selected_landms, selected_bboxes, selected_indices) where:
              - selected_landms is a tensor of shape (M, num_landm*2),
              - selected_bboxes is a tensor of shape (M, 4),
              - selected_indices is a list of length M mapping each selected detection to its image.
        """
        if len(idx) == 0:
            return (torch.tensor([], device=landms.device),
                    torch.tensor([], device=bboxes.device),
                    [])

        selected_landms = []
        selected_bboxes = []
        selected_indices = []
        cache = {"idx": [], "bboxes": [], "landms": []}

        for i in range(len(idx)):
            # Accumulate detections belonging to the same image
            cache["idx"].append(idx[i])
            cache["bboxes"].append(bboxes[i])
            cache["landms"].append(landms[i])

            # If next detection is from the same image, keep accumulating
            if i != len(idx) - 1 and cache["idx"][-1] == idx[i + 1]:
                continue

            # Process accumulated detections for this image according to the strategy
            if self.strategy == "all":
                selected_landms.extend(cache["landms"])
                selected_bboxes.extend(cache["bboxes"])
                selected_indices.extend(cache["idx"])
            elif self.strategy == "best":
                # "Best": choose the first (already sorted) detection for the image.
                selected_landms.append(cache["landms"][0])
                selected_bboxes.append(cache["bboxes"][0])
                selected_indices.append(cache["idx"][0])
            elif self.strategy == "largest":
                # "Largest": choose the detection with the largest bounding box area.
                bbs_stack = torch.stack(cache["bboxes"])
                areas = (bbs_stack[:, 2] - bbs_stack[:, 0] + 1) * (bbs_stack[:, 3] - bbs_stack[:, 1] + 1)
                best_idx = torch.argmax(areas).item()
                selected_landms.append(cache["landms"][best_idx])
                selected_bboxes.append(cache["bboxes"][best_idx])
                selected_indices.append(cache["idx"][0])
            else:
                raise ValueError(f"Unsupported strategy: {self.strategy}")

            # Reset cache for the next image group.
            cache = {"idx": [], "bboxes": [], "landms": []}

        selected_landms = torch.stack(selected_landms)
        selected_bboxes = torch.stack(selected_bboxes)
        return selected_landms, selected_bboxes, selected_indices

    @torch.no_grad()
    def predict(self, images: torch.Tensor) -> tuple[np.ndarray, list[int], np.ndarray]:
        """
        Predicts face landmarks and bounding boxes from a batch of images.

        Returns:
            - landmarks: A NumPy array of shape (num_faces, 5, 2)
            - indices: A list mapping each face detection to its source image index
            - bboxes: A NumPy array of shape (num_faces, 4) with bounding box coordinates
        """
        # Preprocess: convert images from RGB to BGR order and subtract mean offset.
        x, offset = images[:, [2, 1, 0]], torch.tensor([104, 117, 123], device=images.device)
        x = x - offset.view(3, 1, 1)
        scores, bboxes, landms = self(x)

        # Create priors and compute scale factors.
        priors = PriorBox((x.size(2), x.size(3))).forward().to(x.device)
        scale_b = torch.tensor([x.size(3), x.size(2)] * 2, device=x.device)
        scale_l = torch.tensor([x.size(3), x.size(2)] * 5, device=x.device)

        # Decode raw predictions.
        scores = scores[..., 1]
        bboxes = self.decode_bboxes(bboxes, priors) * scale_b
        landms = self.decode_landms(landms, priors) * scale_l

        # Filter predictions.
        filtered_landms, filtered_bboxes, sample_indices = self.filter_preds(scores, bboxes, landms)

        # Apply strategy to select the best detection per image.
        selected_landms, selected_bboxes, selected_indices = self.take_by_strategy(
            filtered_landms, filtered_bboxes, sample_indices
        )

        # Reshape landmarks to (num_faces, 5, 2) and convert tensors to NumPy arrays.
        selected_landms = selected_landms.view(-1, 5, 2).cpu().numpy()
        selected_bboxes = selected_bboxes.cpu().numpy()

        return selected_landms, selected_indices, selected_bboxes


