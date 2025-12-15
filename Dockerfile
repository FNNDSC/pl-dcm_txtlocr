# Python version can be changed, e.g.
FROM python:3.10

LABEL org.opencontainers.image.authors="FNNDSC <dev@babyMRI.org>" \
      org.opencontainers.image.title="My ChRIS Plugin to locate text in DICOMs" \
      org.opencontainers.image.description="A ChRIS plugin to locate text in DICOM images and optionally detect PHI"

ARG SRCDIR=/usr/local/src/pl-dcm_txtlocr
WORKDIR ${SRCDIR}

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

COPY requirements.txt .
RUN --mount=type=cache,sharing=private,target=/root/.cache/pip pip install -r requirements.txt

# EasyOCR model cache location
ENV EASY_OCR_MODEL_DIR=/opt/easyocr

# Create model directory with open permissions
RUN mkdir -p /opt/easyocr && chmod -R a+rX /opt/easyocr

# Copy your preload script into the container
COPY preload_model.py .

# Preload easyocr models
RUN python preload_model.py


COPY . .
ARG extras_require=none
RUN pip install ".[${extras_require}]" \
    && cd / && rm -rf ${SRCDIR}
WORKDIR /

CMD ["dcm_txtlocr"]
