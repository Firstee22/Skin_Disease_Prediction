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
from typing import Dict, List

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
#MODEL_PATH = os.getenv("MODEL_PATH", str(BASE_DIR / "model" / "cnn_model.h5"))
MODEL_PATH = "C:\\CODING\\Deep Learning\\02_Projects\\02_Skin_Disease_Predictor\\cnn_model.h5"

# Disease names, in the SAME order as the model's output labels (index 0 -> 9).
CLASS_NAMES = [
    "Atopic Dermatitis",                                        # Label 0
    "Basal Cell Carcinoma",                                     # Label 1
    "Benign Keratosis-like Lesions",                            # Label 2
    "Eczema",                                                   # Label 3
    "Melanocytic Nevi",                                         # Label 4
    "Melanoma",                                                 # Label 5
    "Psoriasis pictures Lichen Planus and related diseases",    # Label 6
    "Seborrheic Keratoses and other Benign Tumors",             # Label 7
    "Tinea Ringworm Candidiasis and other Fungal Infections",   # Label 8
    "Warts Molluscum and other Viral Infections",               # Label 9
]

# Reject uploads larger than this (in megabytes)
MAX_UPLOAD_MB = 10

# Maximum number of images allowed in one /predict-batch request
MAX_BATCH_FILES = 10

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
    disease: str                      # predicted disease name
    predicted_index: int              # label number (0-9)
    confidence: float                 # probability of the predicted disease (0-1)
    probabilities: Dict[str, float]   # probability for every disease


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
# Response schemas for the batch endpoint
# ---------------------------------------------------------------------------
class BatchError(BaseModel):
    filename: str
    error: str


class BatchPredictionResponse(BaseModel):
    total: int                          # number of files received
    results: List[PredictionResponse]   # one entry per successfully processed image
    errors: List[BatchError]            # files that were rejected, with the reason


# ---------------------------------------------------------------------------
# Helper functions shared by both endpoints
# ---------------------------------------------------------------------------
def read_and_validate_upload(file: UploadFile) -> bytes:
    """Check the file type and size, and return its raw bytes."""
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{file.content_type}'. "
                   f"Allowed: {sorted(ALLOWED_CONTENT_TYPES)}",
        )

    contents = file.file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(contents) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413, detail=f"File too large (max {MAX_UPLOAD_MB} MB)."
        )
    return contents


def load_model_or_500():
    """Return the cached model, or raise a 500 error if it can't be loaded."""
    try:
        return get_model()
    except Exception as exc:
        logger.exception("Could not load model")
        raise HTTPException(status_code=500, detail=f"Model loading failed: {exc}")


def build_prediction(filename: str, raw: np.ndarray) -> PredictionResponse:
    """Turn the model's raw output for ONE image into a PredictionResponse."""

    # Binary models with a single sigmoid output -> build [P(class_0), P(class_1)]
    if raw.shape[0] == 1:
        p = float(raw[0])
        probs = np.array([1.0 - p, p])
    else:
        probs = raw  # softmax output: one probability per class

    # Map class indexes to names (fall back to generic names if the list
    # in CLASS_NAMES doesn't match the model's number of outputs)
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
        filename=filename,
        disease=names[best],
        predicted_index=best,
        confidence=round(float(probs[best]), 4),
        probabilities={n: round(float(p), 4) for n, p in zip(names, probs)},
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    """Simple health check (does not load the model)."""
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(file: UploadFile = File(...)):
    """Classify ONE uploaded image (form field name: 'file')."""

    # 1) Validate the upload (type, empty, size)
    contents = read_and_validate_upload(file)

    # 2) Load the model (cached after the first call)
    model = load_model_or_500()

    # 3) Preprocess the image
    try:
        batch = preprocess_image(contents, model)
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="Could not read the image file.")

    # 4) Run inference
    try:
        raw = model.predict(batch, verbose=0)[0]
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")

    return build_prediction(file.filename or "unknown", raw)


@app.post("/predict-batch", response_model=BatchPredictionResponse)
def predict_batch(files: List[UploadFile] = File(...)):
    """
    Classify MULTIPLE uploaded images in one request
    (repeat the form field name 'files' once per image).

    A bad file does not fail the whole request: it is reported in "errors"
    while the valid images are still classified.
    """
    if len(files) > MAX_BATCH_FILES:
        raise HTTPException(
            status_code=400,
            detail=f"Too many files ({len(files)}). Maximum is {MAX_BATCH_FILES}.",
        )

    model = load_model_or_500()

    valid_names: List[str] = []       # filenames of images that passed validation
    valid_arrays: List[np.ndarray] = []  # their preprocessed arrays, shape (1, H, W, C)
    errors: List[BatchError] = []

    # 1) Validate + preprocess every image individually
    for file in files:
        name = file.filename or "unknown"
        try:
            contents = read_and_validate_upload(file)
            valid_arrays.append(preprocess_image(contents, model))
            valid_names.append(name)
        except HTTPException as exc:
            errors.append(BatchError(filename=name, error=str(exc.detail)))
        except (UnidentifiedImageError, OSError):
            errors.append(BatchError(filename=name, error="Could not read the image file."))

    # 2) Run ONE prediction call for all valid images (faster than one-by-one)
    results: List[PredictionResponse] = []
    if valid_arrays:
        batch = np.concatenate(valid_arrays, axis=0)  # shape (N, H, W, C)
        try:
            raw_batch = model.predict(batch, verbose=0)
        except Exception as exc:
            logger.exception("Batch prediction failed")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")

        results = [build_prediction(n, r) for n, r in zip(valid_names, raw_batch)]

    return BatchPredictionResponse(total=len(files), results=results, errors=errors)