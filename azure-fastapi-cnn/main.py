"""
FastAPI backend for CNN image classification.

- Loads a Keras model saved in .h5 format
- Exposes a POST /predict endpoint that accepts an image upload
- Designed to run on Azure Functions (see function_app.py) but also runs
  locally with:  uvicorn main:app --reload
"""

import io
import logging
import os
import threading
from pathlib import Path
from typing import Dict

# Reduce TensorFlow's noisy startup logs (must be set BEFORE importing tensorflow)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
logger = logging.getLogger("cnn-api")

BASE_DIR = Path(__file__).resolve().parent

# Path to the model file. Can be overridden with an app setting called MODEL_PATH.
#MODEL_PATH = os.getenv("MODEL_PATH", str(BASE_DIR /'model' / "cnn_model.h5"))
MODEL_PATH = "C:\\CODING\\Deep Learning\\02_Projects\\02_Skin_Disease_Predictor\\cnn_model.h5"

# TODO: replace with YOUR class names, in the SAME order used during training
# (e.g. the order of train_generator.class_indices).
CLASS_NAMES = ["class_0", "class_1", "class_2"]

# Reject uploads larger than this (in megabytes)
MAX_UPLOAD_MB = 10

# Accepted image MIME types
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}

# Fallback input size, only used if the model has a dynamic (None) input size
DEFAULT_IMG_SIZE = 224

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="CNN Image Classification API",
    description="Upload an image and get the predicted class from a CNN model.",
    version="1.0.0",
)

# Allow browser-based frontends to call this API.
# In production, replace "*" with your frontend's domain(s).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Response schema
# ---------------------------------------------------------------------------
class PredictionResponse(BaseModel):
    filename: str
    predicted_class: str
    predicted_index: int
    confidence: float                 # probability of the predicted class (0-1)
    probabilities: Dict[str, float]   # probability for every class


# ---------------------------------------------------------------------------
# Model loading (lazy + cached)
# ---------------------------------------------------------------------------
# The model is loaded on the first request and kept in memory, so later
# requests on the same instance are fast. Only a cold start pays the load cost.
_model = None
_model_lock = threading.Lock()  # prevents two requests loading the model at once


def get_model():
    """Return the cached Keras model, loading it from disk on first use."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:  # double-check after acquiring the lock
                if not os.path.exists(MODEL_PATH):
                    raise RuntimeError(f"Model file not found at: {MODEL_PATH}")
                logger.info("Loading model from %s ...", MODEL_PATH)
                # compile=False: we only need inference, so custom losses/optimizers
                # used during training don't need to be available here.
                _model = tf.keras.models.load_model(MODEL_PATH, compile=False)
                logger.info("Model loaded. Input shape: %s", _model.input_shape)
    return _model


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------
def preprocess_image(image_bytes: bytes, model) -> np.ndarray:
    """
    Convert raw image bytes into a model-ready batch of shape (1, H, W, C).

    IMPORTANT: this must match the preprocessing used during training.
    By default we resize to the model's input size and scale pixels to [0, 1].
    If you trained with something else (e.g. MobileNet's preprocess_input,
    mean/std normalization), change it here.
    """
    # model.input_shape looks like (None, height, width, channels)
    _, height, width, channels = model.input_shape
    height = height or DEFAULT_IMG_SIZE
    width = width or DEFAULT_IMG_SIZE

    # Decode the image; also converts RGBA / palette / grayscale as needed
    mode = "L" if channels == 1 else "RGB"
    image = Image.open(io.BytesIO(image_bytes)).convert(mode)

    # PIL's resize takes (width, height)
    image = image.resize((width, height))

    # Convert to float32 array and normalize to [0, 1]
    array = np.asarray(image, dtype="float32") / 255.0

    # Grayscale images need an explicit channel axis: (H, W) -> (H, W, 1)
    if channels == 1:
        array = np.expand_dims(array, axis=-1)

    # Add the batch dimension: (H, W, C) -> (1, H, W, C)
    return np.expand_dims(array, axis=0)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    """Simple health check (does not load the model)."""
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(file: UploadFile = File(...)):
    """Classify an uploaded image and return the predicted class."""

    # 1) Validate the upload type
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{file.content_type}'. "
                   f"Allowed: {sorted(ALLOWED_CONTENT_TYPES)}",
        )

    # 2) Read the file and validate its size
    contents = file.file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(contents) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413, detail=f"File too large (max {MAX_UPLOAD_MB} MB)."
        )

    # 3) Load the model (cached after the first call)
    try:
        model = get_model()
    except Exception as exc:
        logger.exception("Could not load model")
        raise HTTPException(status_code=500, detail=f"Model loading failed: {exc}")

    # 4) Preprocess the image
    try:
        batch = preprocess_image(contents, model)
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="Could not read the image file.")

    # 5) Run inference
    try:
        raw = model.predict(batch, verbose=0)[0]
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")

    # Binary models with a single sigmoid output -> build [P(class_0), P(class_1)]
    if raw.shape[0] == 1:
        p = float(raw[0])
        probs = np.array([1.0 - p, p])
    else:
        probs = raw  # softmax output: one probability per class

    # 6) Map class indexes to names (fall back to generic names if the list
    #    in CLASS_NAMES doesn't match the model's number of outputs)
    if len(CLASS_NAMES) == len(probs):
        names = CLASS_NAMES
    else:
        logger.warning(
            "CLASS_NAMES has %d entries but model outputs %d classes",
            len(CLASS_NAMES), len(probs),
        )
        names = [f"class_{i}" for i in range(len(probs))]

    best = int(np.argmax(probs))

    return PredictionResponse(
        filename=file.filename or "unknown",
        predicted_class=names[best],
        predicted_index=best,
        confidence=round(float(probs[best]), 4),
        probabilities={n: round(float(p), 4) for n, p in zip(names, probs)},
    )
