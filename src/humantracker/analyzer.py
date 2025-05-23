from enum import Enum
from typing import List, Optional

import cv2
import numpy as np

import api.deepsort
import api.yolo
import api.openpose


MAX_COSINE_DISTANCE = 1
MAX_AGE = 100
NN_BUDGET = None


class DetectorType(Enum):
    YOLO = 1
    OPENPOSE = 2


class Analyzer:
    def __init__(
        self,
        detector_type: DetectorType,
        input_video_path: str = "",
        output_video_path: str = "",
        fourcc: str = "mp4v",
        area: Optional[List[int]] = None,
    ) -> None:
        match detector_type:
            case DetectorType.YOLO:
                self.detector = api.yolo.YOLOPersonDetector()
                self.tracker = api.deepsort.DeepSORTTracker(
                    api.yolo.reid_file(), MAX_COSINE_DISTANCE, MAX_AGE, NN_BUDGET
                )
            case DetectorType.OPENPOSE:
                self.detector = api.openpose.OpenposePersonDetector()
                self.tracker = api.deepsort.DeepSORTTracker(
                    api.openpose.reid_file(),
                    MAX_COSINE_DISTANCE,
                    MAX_AGE,
                    NN_BUDGET,
                    api.openpose.kalman_filter(),
                )
            case _:
                raise ValueError("Unknown detector type")

        self.capture = cv2.VideoCapture(input_video_path if input_video_path else 0)
        if self.capture.isOpened():
            self.frame_size = (
                int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            )
            self.frame_rate = int(self.capture.get(cv2.CAP_PROP_FPS))
        else:
            self.frame_size = (0, 0)
            self.frame_rate = 0

        if area:
            if area[1] > self.frame_size[0] or area[3] > self.frame_size[1]:
                raise ValueError("Area extends beyond the frame border")

        self.output_video_path = output_video_path
        self.fourcc = fourcc
        self.area = area

    def run(self) -> None:
        fourcc = cv2.VideoWriter_fourcc(*self.fourcc)
        output_video = cv2.VideoWriter(
            self.output_video_path,
            fourcc=fourcc,
            fps=self.frame_rate,
            frameSize=self.frame_size,
        )

        frame_number = 0
        while True:
            result, frame = self.capture.read()

            if not result:
                break

            frame_number += 1

            detections = self.detector.detect(frame, self.area)

            try:
                bboxes, scores, _ = np.hsplit(detections, [4, 5])
                bboxes[:, 2:] = bboxes[:, 2:] - bboxes[:, :2]
                n_objects = detections.shape[0]
            except ValueError:
                bboxes = np.empty(0)
                scores = np.empty(0)
                n_objects = 0

            self.tracker.track(frame, bboxes, scores.flatten(), frame_number, self.area)

            output_video.write(frame)

        output_video.release()

    def get_stats(self) -> dict[str, int]:
        return self.tracker.get_stats()
