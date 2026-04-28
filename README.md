# 🧠 AI Vision Navigation System

## 📌 Project Description
This project is a Deep Learning-based real-time vision assistant that uses a custom trained YOLOv8 model to detect objects, estimate distance, and provide navigation guidance with voice output.

---

## 🚀 Features
- Real-time object detection using YOLOv8 (custom trained model)
- Distance estimation using bounding box geometry
- Spatial understanding (Left / Center / Right)
- Intelligent navigation guidance system
- Voice feedback using gTTS
- Streamlit web-based UI dashboard

---

## 🧠 Technologies Used
- Python
- YOLOv8 (Ultralytics)
- Streamlit
- OpenCV
- NumPy
- gTTS

---

## 📂 Project Structure
app.py → main application  
best.pt → trained model  
requirements.txt → dependencies  

---

## ▶️ How to Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py

🌐 Deployment

This project is deployed using Streamlit Cloud for public access.
https://ai-blind-navigation-system-8palztmei76tgjtubdhesa.streamlit.app/

📊 Model Info
Model: YOLOv8 custom trained
Classes: person, chair, car, etc.
Dataset: Custom labeled dataset
