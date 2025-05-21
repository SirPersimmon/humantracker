"""
This package contains (slightly adapted) modules from the official DeepSORT repository:
    [ https://github.com/nwojke/deep_sort ]
FROM "deep_sort/tools"
    - generate_detections.py
FROM "deep_sort/deep_sort"
    - detection.py
    - iou_matching.py
    - kalman_filter.py
    - linear_assignment.py
    - nn_matching.py
    - track.py
    - tracker.py
This package also contains module "my_kalman_filter.py" which implements a custom Kalman filter.
"""

import os

# suppress tensorflow logging to console
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
