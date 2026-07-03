import numpy as np
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from yolo_detector_model import Yolo_Detection_Model
from processing import process_uploaded_file, mask_serialization, convert_base64_from_image
import uvicorn

fastapi_app = FastAPI(title = "Yolo Detector", version= "1.0.0")
fastapi_app.add_middleware(CORSMiddleware, allow_origins = ["*"], allow_methods = ["*"], allow_headers = ["*"])

model_cache = {}

def retrieve_model(model_selection, model_size):
    cache_key = (model_selection, model_size)

    if cache_key not in model_cache:
        model_cache[cache_key] = Yolo_Detection_Model(task = model_selection, size = model_size)

    return model_cache[cache_key]

@fastapi_app.get("/", response_class=HTMLResponse)
def endpoint():
    return """ <h1>YOLO DETECTOR 26</h1>
    
    <ul>
    <li>Object Detection</li>
    <li>Image Segmentation</li>
    <li>Image Classification</li>
    </ul>
    
    <p><a href = "/docs"> Documnetation </a> | <a href = "/health"> System Status </a></p>
    <p>Endpoints: POST /api/v1/detect, /api/v1/segment, /api/v1/classify</p>"""
    

@fastapi_app.post("/api/v1/detect")
async def object_detection(
    file: UploadFile = File(...),
    model_size: str = Form("nano")
):
    image = await process_uploaded_file(file)

    detector = retrieve_model("detect", model_size)
    detected, render_image = detector.detection_model(image)
    return{
        "model" : "detect",
        "success" : True,
        "model_size" : model_size,
        "result" : detected,
        "Image" : {
            "format" : "jpeg",
            "height" : image.shape[0],
            "width" : image.shape[1],
            "base64" : convert_base64_from_image(render_image, "jpeg")
        }

    }

@fastapi_app.post("/api/v1/segment")

async def object_segmentation(
    file: UploadFile = File(...),
    model_size: str = Form("nano")):
    image = await process_uploaded_file(file)

    segment = retrieve_model("segment", model_size)
    segmented, render_image = segment.segmentation_model(image)

    segmented = mask_serialization(segmented)

    return{
        "model" : "segment",
        "success" : True,
        "model_size" : model_size,
        "result" : segmented,
        "Image" : {
            "format" : "jpeg",
            "height" : image.shape[0],
            "width" : image.shape[1],
            "base64" : convert_base64_from_image(render_image, "jpeg")
        }
    }


@fastapi_app.post("/api/v1/classify")
async def object_classification(
    file: UploadFile = File(...),
    model_size: str = Form("nano")
):
    image = await process_uploaded_file(file)

    classify_detect = retrieve_model("classify", model_size)

    classification, render_image = classify_detect.classification_model(image)

    return{
        "model" : "classify",
        "success" : True,
        "model_size" : model_size,
        "result" : classification,
        "Image" : {
            "format" : "jpeg",
            "height" : image.shape[0],
            "width" : image.shape[1],
            "base64" : convert_base64_from_image(render_image, "jpeg")
        }
    }

@fastapi_app.get("/health")
def system_status():
    return {"status" : "healthy", "version" : "1.0.0"}


if __name__ == "__main__":
    uvicorn.run("yolo_fastapi:fastapi_app", host="0.0.0.0", port=8000, reload= False)



    