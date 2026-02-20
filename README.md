# 🌱 Smart Agriculture AI System

An AI-powered Smart Agriculture Web Application that performs soil analysis, crop recommendation, micronutrient evaluation, IoT simulation, and chatbot assistance using Machine Learning and Flask.

---

## 🚀 Features

- 🌾 Crop Recommendation using Machine Learning (Random Forest, Decision Tree, Gradient Boosting)
- 📊 Soil Health Analysis with Micronutrient Evaluation
- 📈 Real-time IoT Simulation (Temperature, Humidity, Soil Moisture, NPK)
- 🤖 AI Chatbot Integration (GPT4All)
- 🔊 Text-to-Speech (gTTS)
- ☁ Firebase Realtime Database Integration
- 📉 Model Evaluation (Accuracy, RMSE, Classification Report)
- 📊 Confusion Matrix & Data Visualization

---

## 🛠 Tech Stack

### Backend
- Flask
- Python
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn

### AI / ML
- Random Forest Classifier
- Decision Tree Regressor
- Gradient Boosting
- GPT4All (Local LLM)

### Database
- Firebase Realtime Database

### Other Tools
- gTTS (Text-to-Speech)
- REST API Architecture

---

## 📂 Project Structure
│
├── app.py
├── core_app.py
├── tts_api.py
├── Crop_recommendation1.csv
├── requirements.txt
├── templates/
├── static/


---

## ⚙ Installation

### 1️⃣ Clone Repository
go through my repo

### 2️⃣ Create Virtual Environment
python -m venv venv
venv\Scripts\activate # Windows


### 3️⃣ Install Dependencies


pip install -r requirements.txt


---

## 🔐 Firebase Setup

1. Create Firebase project  
2. Generate Service Account Key  
3. Place JSON file locally (DO NOT push to GitHub)  
4. Update credential path inside `core_app.py`  

---

## ▶ Run Application


python app.py


Server runs at:


http://localhost:5000


---

## 📊 Machine Learning Workflow

1. Upload Dataset (CSV)  
2. Process Dataset  
3. Train Model  
4. Evaluate Model  
5. Perform Crop Prediction  
6. Analyze Soil Health  
7. Generate Fertilizer Recommendations  

---

## 🤖 Chatbot Feature

Uses GPT4All local LLM model:
- Loads model  
- Creates chat session  
- Generates response  
- Supports text-to-speech  

---

## 📈 Model Evaluation

- Accuracy Score  
- RMSE  
- Classification Report  
- Confusion Matrix  

---

## 🌍 Future Improvements

- Deploy to Cloud (Render / AWS)  
- Add Authentication System  
- Improve IoT Real Sensor Integration  
- Add Dashboard Analytics  
- Optimize Model Performance  

---

## 👨‍💻 Author

Sai Sathwik  
AI & Full Stack Developer  

---

## 📜 License

This project is for academic and research purposes.
