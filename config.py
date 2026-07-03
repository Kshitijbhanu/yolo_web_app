from pathlib import Path

Base_DIR = Path(__file__).parent.absolute()

LOG_CONFIG = {
    "LOG_DIR" : Base_DIR/"logs",
    "MAX_BYTES" : 10*1024*1024,
    "BACKUP_COUNT" : 5,
    "FILE_LEVEL": "DEBUG",
    "CONSOLE_LEVEL" : "INFO",
    "FORMAT" : "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
}

MODEL_CONFIG = {
    "MODELS_DIR" : Base_DIR/ "models",
    "PREDICTS_DIR" : Base_DIR/ "pred_images"
}

TASK_TYPES = ["detect", "segment", "classify"]

MODEL_SIZES = ["nano", "small", "medium", "large", "xlarge"]

MODEL_URLS = {
    "detect": {
        "nano": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt",
        "small": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s.pt",
        "medium": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt",
        "large": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26l.pt",
        "xlarge": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26x.pt",
    },
    "segment": {
        "nano": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-seg.pt",
        "small": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s-seg.pt",
        "medium": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m-seg.pt",
        "large": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26l-seg.pt",
        "xlarge": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26x-seg.pt",
    },
    "classify": {
        "nano": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-cls.pt",
        "small": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s-cls.pt",
        "medium": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m-cls.pt",
        "large": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26l-cls.pt",
        "xlarge": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26x-cls.pt",
    },
}

DEFAULT_PARAMS = {
    "detect" : {"confidence" : 0.25, "iou" : 0.45},
    "segment" : {"confidence" : 0.25, "iou" : 0.45},
    "classify" : {"topk" : 5}
}

for directory in [
    LOG_CONFIG["LOG_DIR"],
    MODEL_CONFIG["MODELS_DIR"],
    MODEL_CONFIG["PREDICTS_DIR"]
]:
    directory.mkdir(parents= True, exist_ok= True)