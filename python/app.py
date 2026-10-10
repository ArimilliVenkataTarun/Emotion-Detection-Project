import cv2
import numpy as np
import streamlit as st

from emotion_detector import EmotionDetector, annotate

st.set_page_config(page_title="Facial Emotion Detection", layout="centered")


@st.cache_resource(show_spinner="Loading emotion model...")
def load_detector() -> EmotionDetector:
    return EmotionDetector()


def decode_image(uploaded_file) -> np.ndarray | None:
    data = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


st.title("Facial Emotion Detection")
st.caption("Detects faces with OpenCV and classifies each face's emotion with a deep learning model running on OpenCV DNN.")

source = st.radio("Image source", ["Upload an image", "Use webcam"], horizontal=True)

if source == "Upload an image":
    uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "bmp", "webp"])
else:
    uploaded = st.camera_input("Take a photo")

if uploaded is not None:
    image = decode_image(uploaded)
    if image is None:
        st.error("Could not read that image. Please try a different file.")
        st.stop()

    try:
        detector = load_detector()
    except Exception as exc:
        st.error(f"Failed to load the emotion model: {exc}")
        st.stop()

    results = detector.predict(image)
    st.image(cv2.cvtColor(annotate(image, results), cv2.COLOR_BGR2RGB), use_container_width=True)

    if not results:
        st.warning("No face detected. Try a well-lit, front-facing photo.")
    else:
        st.success(f"Detected {len(results)} face{'s' if len(results) > 1 else ''}.")
        for index, result in enumerate(results, start=1):
            st.subheader(f"Face {index}: {result.emotion.capitalize()} ({result.confidence * 100:.1f}%)")
            ranked = sorted(result.scores.items(), key=lambda item: item[1], reverse=True)
            for label, score in ranked:
                st.progress(score, text=f"{label.capitalize()} - {score * 100:.1f}%")
