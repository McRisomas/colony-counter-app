# Automatic Colony Counter Web Application

**Colony Counter Web App:** [Try the Colony Counter Web App Using Streamlit](https://colony-counter-app-eett3oohw3k3f23ajqkw8m.streamlit.app/~/+/#automatic-bacterial-colony-counter)

An interactive computer vision web application that automatically detect and count bacterial or fungi colony in an petri dishes image by using YOLOv11 model

🚀 **Live Demo:** [Web App Live Demo](assets/demo.gif)

## 📌 Problem Statement & Solution

Manual colony counting in microbiology laboratories is time-consuming, repetitive, and vulnerable to human fatigue, especially when handling dozens of agar plates daily. 

This application automates the counting process:
- **Reduces analysis time** from minutes to seconds per plate.
- **Minimizes subjective human error** across different laboratory analysts.
- **Supports batch processing** for processing multi-sample workflows seamlessly.

---

## ✨ Key Features

- **🎯 High-Precision Detection:** Powered by a fine-tuned YOLOv11 model trained to identify dense, overlapping, or faint agar colonies.
- **📁 Batch Image Upload:** Analyze single or multiple petri dish photos in one run.
- **⚙️ Dynamic Sensitivity Slider:** Adjustable confidence threshold allowing users to tune detection sensitivity based on lighting and agar contrast.
- **📊 Collapsible Results (UI/UX):** Uses interactive expanders to display individual CFU counts and annotated detection images without cluttering the screen.
- **⚡ Memory-Safe Architecture:** Sequential image stream handling to prevent Out-Of-Memory (OOM) crashes on cloud servers.

---

## 🛠️ Tech Stack

- **Computer Vision Model:** [Ultralytics YOLOv11](https://github.com/ultralytics/ultralytics)
- **Web Framework:** [Streamlit](https://streamlit.io/)
- **Image Processing:** OpenCV, PIL, NumPy
- **Language:** Python 3.10+

---

## 💻 Local Setup & Installation

If you want to run this application locally on your machine:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/colony-counter-app.git](https://github.com/YOUR_USERNAME/colony-counter-app.git)
   cd colony-counter-app
