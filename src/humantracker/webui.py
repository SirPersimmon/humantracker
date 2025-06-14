import pprint
import tempfile
import time
import traceback
from typing import Tuple

# Workaround for error 'Examining the path of torch.classes raised' in containers.
import torch

torch.classes.__path__ = []

import streamlit as st  # noqa: E402

import analyzer  # noqa: E402


MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB


def cast_detector_value(detector):
    result = None
    match detector:
        case "YOLO":
            result = analyzer.DetectorType.YOLO
        case "OpenPose":
            result = analyzer.DetectorType.OPENPOSE
    return result


def cast_area_value(area):
    result = None
    if area:
        try:
            result = [int(x) for x in area.split(",") if int(x) >= 0]
            if len(result) != 4 or result[0] > result[2] or result[1] > result[3]:
                raise ValueError
        except ValueError:
            raise ValueError("Incorrect area settings")
    return result


def process_video(video_bytes: bytes, detector: str, area: str) -> Tuple[bytes, dict]:
    with (
        tempfile.NamedTemporaryFile(suffix=".mp4") as input_file,
        tempfile.NamedTemporaryFile(suffix=".webm") as output_file,
    ):
        input_file.write(video_bytes)

        task = analyzer.Analyzer(
            cast_detector_value(detector),
            input_file.name,
            output_file.name,
            fourcc="VP80",
            area=cast_area_value(area),
        )

        task.run()

        video = output_file.read()
        stats = task.get_stats()

    return video, stats


def analyze_video(
    upload: st.runtime.uploaded_file_manager.UploadedFile, detector: str, area: str
) -> None:
    try:
        start_time = time.time()
        progress_bar = st.sidebar.progress(0)
        status_text = st.sidebar.empty()

        status_text.text("Loading video...")
        progress_bar.progress(10)

        video_bytes = upload.getvalue()

        status_text.text("Processing video...")
        progress_bar.progress(30)

        video, stats = process_video(video_bytes, detector, area)
        if video is None or stats is None:
            return

        progress_bar.progress(80)
        status_text.text("Displaying results...")

        col1.write("Processed Video :camera:")
        col1.video(video)

        col2.write("Tracker Data :wrench:")
        col2.code(
            pprint.pformat(stats),
            language="json",
        )

        progress_bar.progress(100)
        processing_time = time.time() - start_time
        status_text.text(f"Completed in {processing_time:.2f} seconds")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.sidebar.error("Failed to process video")
        # Log the full error for debugging.
        print(f"Error in analyze_video: {traceback.format_exc()}")


st.set_page_config(layout="wide", page_title="Human Tracker")

st.write("## Human Tracker")
st.write(
    """Try uploading a video to track human in it.
    This code is open source and available
    [here](https://github.com/SirPersimmon/humantracker) on GitHub.
    Special thanks to the
    [YOLOv7-DeepSORT-Human-Tracking](https://github.com/dasupradyumna/YOLOv7-DeepSORT-Human-Tracking)
    and [liveposetracker](https://github.com/ortegatron/liveposetracker).
    """
)

col1, col2 = st.columns(2)

st.sidebar.write("## Upload :gear:")

my_upload = st.sidebar.file_uploader("Upload a video", type=["mp4"])

# Information about limitations.
with st.sidebar.expander("ℹ️ Video Guidelines"):
    st.write(
        """
    - Maximum file size: 100MB
    - Supported formats: MP4
    - Processing time depends on video size
    """
    )

with st.sidebar:
    my_detector = st.radio("Select detector", ["YOLO", "OpenPose"])
    my_area = st.text_input(
        "Limit area to", help="left,top,right,bottom", placeholder="All frame"
    )

# Process the video.
if my_upload is not None:
    if my_upload.size > MAX_FILE_SIZE:
        st.error(
            f"The uploaded file is too large."
            f"Please upload a video smaller than {MAX_FILE_SIZE/1024/1024:.1f}MB."
        )
    else:
        analyze_video(upload=my_upload, detector=my_detector, area=my_area)
else:
    st.info("Please upload a video to get started!")
