import streamlit as st
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
import logger
import config
from yolo_detector_model import Yolo_Detection_Model

log = logger.get_logger(__name__)

st.sidebar.title("Selection")

model_task = st.sidebar.selectbox(
    "Select Task",
    config.TASK_TYPES,
    index=2,
    help = "Select models according to objective"
)

model_size = st.sidebar.selectbox(
    "Model Size",
    config.MODEL_SIZES,
    index=1,
    help = "Select model size accordingly: nano :Fastest ,xlarge:Very slow but accurate "
)

if model_task in ["detect", "segment"]:
    confidence = st.sidebar.slider(
        "Confidence",
        min_value=0.01,
        value=config.DEFAULT_PARAMS[model_task]["confidence"],
        max_value=1.0,
        step=0.01,
        help="Select confidence value for the model"
    )

else:
    topk = st.sidebar.slider(
        "Predictions",
        min_value=1,
        value=config.DEFAULT_PARAMS[model_task]["topk"],
        max_value=5,
        step=1,
        help="Select the number of predictions"
    )

@st.cache_resource
def load_model(selected_task, selected_size):
    try:
        return Yolo_Detection_Model(
            task=selected_task,
            size=selected_size
        )

    except Exception as e:
        st.error(f"Failed to load the model: {e}")
        raise

if "last_task" not in st.session_state:
    st.session_state.last_task = model_task

elif st.session_state.last_task != model_task:
    load_model.clear()
    st.session_state.last_task = model_task

yolo_model = load_model(model_task, model_size)

if yolo_model is None:
    st.error(f"Could not load the model")
    st.stop()

st.title("YOLO26")

if model_task == "detect":
    st.info("Object Detection: Identifies objects and marks their positions in the image.")

elif model_task =="segment":
    st.info("Segmentation: Separates detected objects using precise object boundaries.")

else:
    st.info("Classification: Analyzes the image and predicts its most probable class.")

file = st.file_uploader("Upload image", type = ["png", "jpg", "jpeg"])

if file is not None:
    try:
        image = Image.open(file)

        if image.mode != "RGB":
            image = image.convert("RGB")

        numpy_image = np.array(image, dtype=np.uint8)
        numpy_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)

        if model_task == "detect":
            detected, predicted_image = yolo_model.detection_model(numpy_image)

            display_image = cv2.cvtColor(predicted_image, cv2.COLOR_BGR2RGB)
            st.image(display_image, caption= "Detected Image", use_container_width= True)

            if detected:
                st.write("Detections:")
                for i, detec in enumerate(detected, 1):
                    st.write(f"{i}. {detec['label']} = Confidence: {detec['confidence']}")

            else:
                st.warning("Could not find image for Detection")

        elif model_task == "segment":
            detected_segmented, predicted_image = yolo_model.segmentation_model(numpy_image)

            display_image = cv2.cvtColor(predicted_image, cv2.COLOR_BGR2RGB)
            st.image(display_image, caption= "Detected Image", use_container_width= True)

            if detected_segmented:
                st.write("Segmentation:")
                for i, seg in enumerate(detected_segmented, 1):
                    st.write(f"{i}. {seg['label']} = Confidence: {seg['confidence']}")

            else:
                st.warning("Could not find image for Segmentation")

        else:
            classified, predicted_image = yolo_model.classification_model(numpy_image)

            display_image = cv2.cvtColor(predicted_image, cv2.COLOR_BGR2RGB)
            st.image(display_image, caption="Detected Image", use_container_width=True)

            if classified:
                st.write("Classification:")
                for i, clf in enumerate(classified, 1):
                    st.write(f"{i}. {clf['labels']} = Confidence: {clf['confidence']}")

            else:
                st.warning("Could not find image for Classification")

        path_dir = Path(__file__).parent.absolute()
        path_dir = path_dir/"predictions"
        path_dir.mkdir(parents= True, exist_ok= True)
        save_path = path_dir/f"{file.name}_{model_size}_{model_task}.jpeg"
        cv2.imwrite(str(save_path), predicted_image)

        with open(save_path, "rb") as f:
            st.download_button(label= "Download result", data= f, file_name= save_path.name, mime= "image/jpeg")

    except Exception as e:
        st.error(f"Unable to process image : {e}")
        raise


