# 🩺 Medical AI Diagnosis System

> **An advanced multimodal AI system** that predicts diseases using **symptoms (NLP)**, **heart health parameters (ML)**, and **chest X-ray images (CNN)** — all integrated into a unified prediction web app built with **Flask**.

---

## 🚀 Overview
This project combines **Natural Language Processing**, **Machine Learning**, and **Computer Vision** models into one unified AI pipeline.  
Users can input their **symptoms**, **heart metrics**, and **X-ray scans**, and the system predicts the **most probable diseases** using an ensemble model.

It demonstrates real-world **multimodal AI integration** — combining **text**, **structured data**, and **image analysis** within a single intelligent system.

---

## 🧠 Tech Stack

| Category | Tools / Frameworks |
|-----------|--------------------|
| **Frontend** | HTML, CSS, JavaScript (Flask templates) |
| **Backend** | Flask (Python) |
| **Machine Learning** | Scikit-learn, Pandas, NumPy |
| **Deep Learning** | PyTorch |
| **NLP** | TF-IDF Vectorizer, Logistic Regression |
| **Computer Vision** | Custom CNN for X-ray classification |
| **Data** | UCI Heart Dataset, Symptom2Disease dataset, Chest X-ray dataset |

---

## ⚙️ Features

✅ Predicts diseases from **text-based symptoms**  
✅ Performs **heart disease risk assessment** using medical parameters  
✅ Analyzes **chest X-rays** to detect pneumonia or related issues  
✅ Combines all three predictions into a **single ensemble output**  
✅ Clean, responsive **Flask web UI**  
✅ Modular, extendable, and fully reproducible  


## 🧩 How It Works

1. **Symptom Model (NLP)**  
   → Uses TF-IDF + Logistic Regression to classify diseases from textual symptoms.  
2. **Heart Model (ML)**  
   → Logistic Regression predicts heart disease risk from structured health data.  
3. **X-ray Model (CNN)**  
   → PyTorch CNN identifies pneumonia vs. normal X-ray.  
4. **Unified Ensemble**  
   → All predictions are weighted and fused for a final multimodal diagnosis.

---

## 💻 How to Run Locally

```bash
# 1️⃣ Clone this repository
git clone https://github.com/swastik-2004/medical-diagnosis-system.git
cd medical-diagnosis-system

# 2️⃣ Create virtual environment (optional)
python -m venv venv
venv\Scripts\activate   # On Windows
# or
source venv/bin/activate  # On Mac/Linux

# 3️⃣ Install dependencies
pip install -r requirements.txt

# 4️⃣ Run Flask app
python app.py
Now open your browser and visit:
http://localhost:5000/
🖥️ Web App Preview
<img width="1920" height="1080" alt="Screenshot (499)" src="https://github.com/user-attachments/assets/3e84f49f-1a8a-4691-802d-8890738be0ac" />

<img width="1920" height="1080" alt="Screenshot (500)" src="https://github.com/user-attachments/assets/942d7f4c-9177-476d-8fb8-7452a5a5aa60" />



📊 Model Performance
Model	Accuracy	Dataset
Symptom2Disease	95%	Symptom2Disease.csv
Heart Model	87%	UCI Heart Dataset
X-ray CNN	92%	Chest X-ray Dataset
Unified Ensemble	94%	Combined evaluation

📦 Future Improvements
 Cloud deployment (AWS / Render / Hugging Face Spaces)

 Add Docker containerization

 Include Kubernetes orchestration for scalability

 Expand datasets for real-world variety

 Add LLM-based medical Q&A assistant

👨‍💻 Author
Swastik Dasgupta
3rd Year, MSRIT — Artificial Intelligence & Machine Learning

🔗 GitHub | 💼 LinkedIn (add your link)

🧾 License
This project is released under the MIT License — free for personal and academic use.

⭐ Final Notes
This project demonstrates strong skills in:

Machine Learning & Deep Learning model development

Flask-based web application design

Multimodal AI integration (NLP + Vision + Tabular)


