import numpy as np
from fastapi import HTTPException

def validate_image(format_type):
    valid_format = ["image/jpeg", "image/webp", "image/png"]
    return format_type in valid_format

def validate_model_size(model_size, valid_size):
    if model_size not in valid_size:
        raise HTTPException(status_code = 400,
                            detail = f"Invalid model size {model_size}")
    
def classification_topk_values(topk, min_value = 1, max_value = 5):
    if not(min_value <= topk <= max_value):
        raise HTTPException(status_code = 400,
                            detail = f"Prepocessing Failed. Must should be between {min_value} - {max_value}")