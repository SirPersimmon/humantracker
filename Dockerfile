FROM gcr.io/kaggle-gpu-images/python AS builder
WORKDIR /app
RUN --mount=target=/data /data/build-openpose.sh cpu

FROM gcr.io/kaggle-gpu-images/python

WORKDIR /app

ENV OPENPOSE_PATH="/app/openpose"
ENV YOLO_PATH="/app/yolo"

COPY --from=builder /app/openpose/build ./openpose/build
COPY --from=builder /app/openpose/models ./openpose/models

COPY mars-small128.pb ./openpose/reids/
COPY yolov7x.pt ./yolo/models/
COPY ReID.pb ./yolo/reids/

RUN apt update && apt install libgoogle-glog0v5 && \
    uv pip install --system pykalman streamlit && \
    git clone https://github.com/SirPersimmon/humantracker.git /tmp/humantracker && \
    mv /tmp/humantracker/src/humantracker . && \
    rm -rf /tmp/humantracker

EXPOSE 8501

ENTRYPOINT ["streamlit", "run"]

CMD ["/app/humantracker/webui.py"]
