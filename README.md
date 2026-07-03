# YOLO Detector API

This project is built using FastAPI and YOLO model for image based tasks.

It can perform:

- Object Detection
- Image Segmentation
- Image Classification

The API takes an image as input, processes it using YOLO model, and returns prediction results with output image.

---

## Files Used

- yolo_fastapi.py
- yolo_detector_model.py
- processing.py
- config.py
- logger.py
- validation.py

---

## Libraries Used

- Python
- FastAPI
- Uvicorn
- NumPy
- OpenCV
- YOLO

---

## How It Works

1. User uploads an image
2. Image is preprocessed
3. YOLO model runs prediction
4. Prediction results are processed
5. Output image and prediction results are returned in base64 format


---

## Running Project

Install required packages:

pip install -r requirements.txt

Run server:

py yolo_fastapi.py

or

python yolo_fastapi.py

---

## Open in Browser

After server starts, open:

http://127.0.0.1:8000/

This shows basic API information.

Swagger documentation:

http://127.0.0.1:8000/docs

Use this to test all API endpoints.

---

## Endpoints

### Home

/

Shows API information.

### Health Check

/health

Returns server status.

### Object Detection

/api/v1/detect

Upload image and detect objects.

### Segmentation

/api/v1/segment

Upload image and perform segmentation.

### Classification

/api/v1/classify

Upload image and classify image.

---

## Output

API returns:

- Prediction results
- Confidence scores
- Output image in base64 format

---

## Notes

Model is cached after first load so repeated predictions are faster.
