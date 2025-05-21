import os
import pathlib
from typing import Optional, Tuple

import numpy as np
import torch

import yolo.models.experimental
import yolo.utils.datasets
import yolo.utils.general

if "YOLO_PATH" not in os.environ:
    raise Exception("Environment variable 'YOLO_PATH' is not set")


def model_file() -> str:
    return (pathlib.Path(os.environ["YOLO_PATH"]) / "models" / "yolov7x.pt").resolve()


def reid_file() -> str:
    return (pathlib.Path(os.environ["YOLO_PATH"]) / "reids" / "ReID.pb").resolve()


class YOLOPersonDetector:
    """
    YOLOv7 detector wrapper, exposing 2 methods: one method loads the model from
    a checkpoint file, and one that accepts an image to return a tensor of detections.
    """

    def __init__(
        self, conf_th: float = 0.25, iou_th: float = 0.45, img_size: int = 640
    ) -> None:
        self.conf_th = conf_th
        self.iou_th = iou_th
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.half = self.device.type != "cpu"

        self.model = yolo.models.experimental.attempt_load(
            model_file(), map_location=self.device
        )
        self.stride = int(self.model.stride.max())
        self.img_size = yolo.utils.general.check_img_size(img_size, self.stride)

        if self.half:
            self.model.half()

    @torch.no_grad()
    def detect(self, img: np.ndarray) -> Optional[np.ndarray]:
        """
        Takes an input RGB image, preprocesses it and gets all person detections in the image,
        applies non-maximum suppression on the detected bounding boxes, and returns refined
        detections as a NumPy array
        """
        # prepare the input image
        img, orig = self._preprocess(img)
        img = img.unsqueeze(0)

        # inference and non-maximum suppression
        pred = self.model(img, augment=False)[0]
        dets = yolo.utils.general.non_max_suppression(
            pred, self.conf_th, self.iou_th, classes=[0]
        )[0]
        dets[:, :4] = yolo.utils.general.scale_coords(
            img.shape[2:], dets[:, :4], orig.shape
        ).round()

        return dets.detach().cpu().numpy()

    def _preprocess(self, img: np.ndarray) -> Tuple[torch.Tensor, np.ndarray]:
        """
        - Modifies the input's resolution to a standard size while maintaining its aspect ratio
        - Changes the order of values from (H,W,C) to (C,H,W)
        - Transforms the color range from (0,255) to (0,1)

        Returns
        -------
        torch.Tensor
        """
        new = yolo.utils.datasets.letterbox(
            img, self.img_size, stride=self.stride, scaleup=False
        )[0]
        new = np.ascontiguousarray(new.transpose(2, 0, 1))
        new = torch.from_numpy(new).to(self.device)
        new = new.half() if self.half else new.float()

        return new / 255, img
