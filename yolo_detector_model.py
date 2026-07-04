import torch
import requests
import numpy as np
from pathlib import Path
from tqdm import tqdm
import config
import logger
from ultralytics import YOLO

log = logger.get_logger(__name__)

class Yolo_Detection_Model:

    def __init__(self, task = "classify", size = "small", device = None):

        if task not in config.TASK_TYPES:
            raise ValueError("Selected task does not exist")
        
        if size not in config.MODEL_SIZES:
            raise ValueError("Model size is invalid")
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if device not in ["cuda", "cpu"]:
            raise ValueError("The device must be cuda or cpu")
        
        self.task = task
        self.size = size
        self.device = device
        self.model = None
        self.classes = None

        path_dir = Path(__file__).parent.absolute()
        self.model_path = path_dir/ "models"
        self.model_path.mkdir(parents= True, exist_ok= True)

        self.model_path_dir = self.model_path/ f"yolo26_{self.size_suffix()}{self.model_suffix()}.pt"

        self._load_model()

    def size_suffix(self):
        model_size_suffix = {
            "nano": "n",
            "small": "s",
            "medium": "m",
            "large": "l",
            "xlarge": "x"
        }
        return model_size_suffix[self.size]
    
    def model_suffix(self):
        model_suffixes = {
            "detect" : "",
            "segment" : "-seg",
            "classify" : "-cls"
        }
        return model_suffixes[self.task]
        
    def fetch_url(self):
        return config.MODEL_URLS[self.task][self.size]
    
    def _download_model(self):
        url = self.fetch_url()

        if self.model_path_dir.exists():
            log.info("Model already exists")
            return

        try:
            response = requests.get(url, stream= True, timeout=60)
            response.raise_for_status()

            content_length = int(response.headers.get('content-length', 0))

            with (
                open (self.model_path_dir, "wb") as f,
                tqdm(
                    total = content_length,
                    unit = "iB",
                    unit_scale = True,
                    desc = (f"Downloading model : yolo26_{self.size_suffix()}{self.model_suffix()}.pt")
                ) as tqdm_bar
            ):
                for chunks in response.iter_content(chunk_size= 1024):
                    if chunks:
                        written = f.write(chunks)
                        tqdm_bar.update(written)
            
        except Exception as e:
            
            if self.model_path_dir.exists():
                self.model_path_dir.unlink()

            log.error(f"Downloading failed for model : yolo26_{self.size_suffix()}{self.model_suffix()}.pt")
            raise RuntimeError(f"Downloading yolo26_{self.size_suffix()}{self.model_suffix()}.pt failed: {e}")

    def _load_model(self):
        try:
            self._download_model()

            try:
                self.model = YOLO(str(self.model_path_dir))

                self.classes = self.model.names if hasattr(self.model, "names") else None

                if self.classes is None:
                    if self.task == "classify":
                        self.classes = [f"class{i}" for i in range(1000)]

                    else:
                        self.classes = [
                            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", 
                            "stop sign","parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
                            "cow", "elephant", "bear", "zebra", "giraffe"
                        ]

            except Exception as e:
                if "cuda out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    self.device = "cpu"
                    self.model = YOLO(str(self.model_path_dir))
                    self.classes = self.model.names

                else:
                    raise

        
        except Exception as e:
            raise RuntimeError(f"Failed to load the model: {e}")


    def _predict(self, image):
        if self.model is None:
            raise RuntimeError("Model not loaded")

        predicted = self.model.predict(
            image,
            device=self.device,
            verbose=False
        )
        return predicted[0] if predicted else None


    def detection_model(self, image):

        result = self._predict(image)

        detected = []

        if result is None:
            return [], None
        
        render_image = result.plot()
        
        if result.boxes is not None:

            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                label = self.classes[class_id] if self.classes else str(class_id)

                detected.append({
                    "box" : [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
                    "confidence": confidence,
                    "label": label
                    })
        return detected, render_image
    
    def segmentation_model(self, image):
        
        segmentation_result = self._predict(image)

        if segmentation_result is None:
            return [], None
        
        render_image = segmentation_result.plot()

        mask = segmentation_result.masks if hasattr(segmentation_result, "masks") else None

        if mask is None:
            return [], render_image
        
        segmented = []
        
        if segmentation_result.boxes is not None:
            for i, box in enumerate(segmentation_result.boxes):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                label = self.classes[class_id] if self.classes else str(class_id)

                masks = None
                if mask is not None and hasattr(mask, "xy"):
                    masks = mask.xy[i]
                

                segmented.append({
                    "box" : [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
                    "confidence": confidence,
                    "label" : label,
                    "masks" : masks
                    })
                
        return segmented, render_image               
    
    def classification_model(self, image, topk = 5):

        classification_result = self._predict(image)

        classification = []
        if classification_result is None:
            return classification, None

        probability = classification_result.probs if hasattr(classification_result, "probs") else None

        if probability is None:
            return [], None

        render_image = classification_result.plot()
        
        top5_index = probability.top5[:topk]
        top5_confidence = probability.top5conf[:topk]

        for indx, conf in zip(top5_index, top5_confidence):
            indexes = indx
            confidence = conf
            labels = self.classes[indexes] if self.classes else str(indexes)

            classification.append({
                "index" : indexes,
                "confidence": float(confidence),
                "labels" : labels 
                })
                
        return classification, render_image
    

