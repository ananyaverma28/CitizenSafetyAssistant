# Citizen Safety Assistant
### Road Accident Severity Prediction System

---

## Overview
Citizen Safety Assistant is a full-stack machine learning application that predicts road accident severity using historical accident data and weather-related features. The system supports data-driven safety analysis through a web-based interface.

---

## Problem Statement
Road accident severity is influenced by factors such as weather conditions, visibility, time, and location. Traditional analysis methods fail to capture these complex relationships effectively. This project models accident severity using supervised machine learning and exposes predictions through an API-driven application.

---

## Key Features
- Accident severity prediction using trained ML models  
- Backend REST API for inference  
- Interactive frontend interface  
- Model comparison and evaluation  
- Reproducible preprocessing pipeline  

---

## Technical Approach

### Data Processing
- Data cleaning and missing value handling  
- Feature selection and transformation  
- Derived features such as accident duration  
- Train-test split with consistent preprocessing  

### Models Implemented
- Logistic Regression  
- Decision Tree  
- Random Forest  
- **XGBoost (final selected model)**  

### Evaluation Metrics
- Accuracy  
- Precision  
- Recall  
- F1-Score  

XGBoost achieved the best overall performance and was selected for deployment.

---

## Tech Stack

### Backend
- Python  
- Flask  
- scikit-learn  
- XGBoost  

### Frontend
- React.js  
- Vite  
- HTML, CSS, JavaScript  

### Tools
- Git & GitHub  
- Jupyter Notebook  

---

## Project Structure
CitizenSafetyAssistant/
├── backend/
│ ├── app.py
│ ├── requirements.txt
│ ├── model/
│ └── utils/
├── frontend/
│ ├── index.html
│ ├── package.json
│ ├── vite.config.js
│ ├── public/
│ └── src/
├── notebooks/
│ └── Accident_Hotspots.html
├── .gitignore
└── README.md


---

## How to Run

### Backend
```bash
cd backend
pip install -r requirements.txt
python app.py
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

---

## Dataset
The project uses publicly available U.S. road accident data covering multiple years.  
The dataset includes temporal, environmental, and weather-related attributes such as visibility, temperature, and accident duration.

Due to size constraints, raw dataset files are not included in this repository.  
All data cleaning, feature engineering, and preprocessing steps are implemented programmatically.

---

## Results
Multiple supervised learning models were evaluated for accident severity prediction.

The XGBoost classifier achieved the best overall performance with approximately 90% accuracy and balanced precision-recall scores.  
Weather conditions and visibility-related features were found to have the strongest influence on accident severity.

The final model generalized well on unseen test data.

---

## Limitations
- The model relies on historical data and may not capture sudden or rare events
- Traffic density and driver behavior data were not available
- Predictions are limited to the quality of input weather data

---

## Future Enhancements
- Integration of real-time traffic and weather streams
- Deployment on cloud infrastructure
- Improved geospatial visualization of accident hotspots
- Periodic model retraining with new data

---

## Author
**Ananya Verma** 