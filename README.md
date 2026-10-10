**Facial Emotion Detection using OpenCV**

This project detects human emotions from facial expressions using Computer Vision. OpenCV's Haar cascade locates faces in an image, and each face is classified by the pretrained FER+ deep learning model (ONNX) running on OpenCV's DNN module.

**Detected emotions:** neutral, happiness, surprise, sadness, anger, disgust, fear, contempt

**Technologies Used:**

* Python
* OpenCV (Haar cascade face detection + DNN module)
* FER+ emotion recognition model (ONNX)
* NumPy
* Streamlit

**Key Features:**

* Emotion detection from uploaded images or webcam snapshots (Streamlit app)
* Real-time emotion detection from a live webcam feed
* Multiple faces per image, with confidence scores for every emotion
* Model is downloaded automatically on first run (`models/emotion-ferplus-8.onnx`)

**Setup:**

```bash
pip install -r requirements.txt
python emotion_detector.py   # optional: pre-download the model
```

**Run the web app:**

```bash
streamlit run app.py
```

**Run real-time webcam detection** (press `q` to quit):

```bash
python webcam.py
```

**Objective:** To explore how computer vision and deep learning can recognize emotions from facial expressions to support emotion-aware applications.
