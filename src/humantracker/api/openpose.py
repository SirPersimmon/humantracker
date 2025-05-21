import os
import pathlib
import sys
from typing import List, Optional

import numpy as np

import deepsort.my_kalman_filter

if "OPENPOSE_PATH" not in os.environ:
    raise Exception("Environment variable 'OPENPOSE_PATH' is not set")

op_path = str((pathlib.Path(os.environ["OPENPOSE_PATH"]) / "build/python").resolve())
if op_path not in sys.path:
    sys.path.append(op_path)
del op_path

from openpose import pyopenpose as op


NMS_MAX_OVERLAP = 1.0


def models_dir() -> str:
    return (pathlib.Path(os.environ["OPENPOSE_PATH"]) / "models").resolve()


def reid_file() -> str:
    return (
        pathlib.Path(os.environ["OPENPOSE_PATH"]) / "reids" / "mars-small128.pb"
    ).resolve()


def kalman_filter() -> deepsort.my_kalman_filter.MyKalmanFilter:
    return deepsort.my_kalman_filter.MyKalmanFilter(
        transition_covariance=np.eye(8, 8) * 10,
        observation_covariance=np.eye(4, 4) * 500,
    )


class OpenposePersonDetector:
    """
    Openpose detector wrapper, exposing just one method which receives an image
    to return a tensor of detections.
    """

    def __init__(self) -> None:
        params = {"model_folder": models_dir(), "net_resolution": "-1x320"}
        self.openpose = op.WrapperPython()
        self.openpose.configure(params)
        self.openpose.start()

    def detect(self, img: np.ndarray) -> Optional[np.ndarray]:
        datum = op.Datum()
        datum.cvInputData = img
        self.openpose.emplaceAndPop(op.VectorDatum([datum]))

        keypoints, img = np.array(datum.poseKeypoints), datum.cvOutputData

        # Doesn't use keypoint confidence.
        try:
            poses = keypoints[:, :, :2]
        except:
            return None

        # Get (nonempty) containing box for each seen body.
        bboxes = np.array(
            [
                [x1, y1, x2, y2]
                for [x1, y1, x2, y2] in self._poses2boxes(poses)
                if x2 != x1 and y2 != y1
            ]
        )
        scores = np.ones(len(bboxes))

        # Run non-maxima suppression.
        indices = self._non_max_suppression(bboxes, NMS_MAX_OVERLAP, scores)

        return np.concatenate((bboxes[indices], np.ones((len(indices), 1))), axis=1)

    def _poses2boxes(self, poses: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        poses: ndarray
            Array of human 2D poses [People * BodyPart].

        Returns
        -------
        boxes: ndarray
            Array of containing boxes [People * [x1,y1,x2,y2]].
        """
        boxes = []
        for person in poses:
            seen_bodyparts = person[np.where((person[:, 0] != 0) | (person[:, 1] != 0))]
            mean = np.mean(seen_bodyparts, axis=0)
            deviation = np.std(seen_bodyparts, axis=0)
            box = [
                int(mean[0] - deviation[0]),
                int(mean[1] - deviation[1]),
                int(mean[0] + deviation[0]),
                int(mean[1] + deviation[1]),
            ]
            boxes.append(box)

        return np.array(boxes)

    def _non_max_suppression(
        self,
        boxes: np.ndarray,
        max_bbox_overlap: float,
        scores: Optional[np.ndarray] = None,
    ) -> List[int]:
        """Suppress overlapping detections.
        Original code from http://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python
        has been adapted to include confidence score.

        Parameters
        ----------
        boxes : ndarray
            Array of ROIs (x, y, width, height).
        max_bbox_overlap : float
            ROIs that overlap more than this values are suppressed.
        scores : Optional[array_like]
            Detector confidence score.

        Returns
        -------
        List[int]
            Returns indices of detections that have survived non-maxima suppression.

        Examples
        --------
            >>> boxes = [d.roi for d in detections]
            >>> scores = [d.confidence for d in detections]
            >>> indices = non_max_suppression(boxes, max_bbox_overlap, scores)
            >>> detections = [detections[i] for i in indices]
        """
        if len(boxes) == 0:
            return []

        boxes = boxes.astype(float)
        pick = []

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        if scores is not None:
            idxs = np.argsort(scores)
        else:
            idxs = np.argsort(y2)

        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            overlap = (w * h) / area[idxs[:last]]

            idxs = np.delete(
                idxs, np.concatenate(([last], np.where(overlap > max_bbox_overlap)[0]))
            )

        return pick
