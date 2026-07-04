import numpy as np
import cv2
import base64
from io import BytesIO
from PIL import Image
from fastapi import UploadFile, HTTPException

async def process_uploaded_file(file: UploadFile):
    try:
        file = await file.read()

        image = Image.open(BytesIO(file))

        if image.mode != "RGB":
            image = image.convert("RGB")

        numpy_image = np.array(image, dtype = np.uint8)

        bgr_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)

        return bgr_image
    
    except Exception as e:
        raise HTTPException(status_code = 400,
                            detail = f"Prepocessing Failed. Encountered error: {e}")
    

def mask_serialization(segmentation):
    serializated = []

    for seg in segmentation:
        seg_copy = seg.copy()

        if "masks" in seg_copy and seg_copy["masks"] is not None:
            masks = seg_copy["masks"]

            if isinstance(masks, np.ndarray):
                seg_copy["masks"] = masks.tolist()

        serializated.append(seg_copy)

    return serializated


def convert_base64_from_image(image: np.ndarray, format = "jpeg"):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    pil_image = Image.fromarray(rgb_image)

    memory_buffered = BytesIO()

    if format.lower() == "jpeg":
        save_format = "JPEG"
    else:
        save_format = "PNG"

    pil_image.save(memory_buffered, format = save_format, quality = 95)

    encoded_image_str = base64.b64encode(memory_buffered.getvalue()).decode()
    
    image_format = f"image/{format}"
    return f"data:{image_format};base64,{encoded_image_str}"

