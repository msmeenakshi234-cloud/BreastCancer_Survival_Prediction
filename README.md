# 🎗️ Breast Cancer 10-Year Survival Prediction

Predicting whether breast cancer patients will survive 10 years using Machine Learning.

---

## 📌 What Does This Project Do?

This project helps doctors predict if a breast cancer patient will survive for 10 years or more. It uses patient data like age, tumor size, and treatment history to make predictions.

---

## 📊 Dataset

- **Name**: METABRIC Breast Cancer Dataset
- **Size**: 2509 patients
- **Features**: 34 ( Age, Tumor Size, ER/PR/HER2 Status, Treatment info, etc.)

---

## 🛠️ Tools Used

- **Python** - Programming language
- **Jupyter Notebook** - For analysis
- **Scikit-learn** - Machine Learning library
- **Streamlit** - Web app
- **Pandas, Matplotlib, Seaborn** - Data analysis and visualization

---

## 📁 Project Files

- `breast_cancer_analysis.ipynb` - Main analysis code
- `app.py` - Web application
- `breast_cancer_clinical_model.pkl` - Trained model
- `README.md` - This file

---

## 🚀 How to Run

### 1. Install Required Libraries
```bash
pip install pandas numpy scikit-learn streamlit matplotlib seaborn lifelines
```

### 2. Run the Notebook
```bash
jupyter notebook breast_cancer_analysis.ipynb
```

### 3. Run the Web App
```bash
streamlit run app.py
```

---

## 📈 What was done

1. **Loaded the data** - Breast cancer patient records
2. **Cleaned the data** - Fixed missing values
3. **Analyzed patterns** - Found which factors affect survival
4. **Created survival curves** - Kaplan-Meier analysis
5. **Trained 8 ML models** - Tested different algorithms
6. **Found the best model** - Random Forest-Config 3 with ROC AUC : 0.8245
7. **Built a web app** - Easy-to-use prediction tool

---

## 🏆 Results

- **Best Model**:  Random Forest-Config 3
- **Recall**: 0.85%
- **ROC-AUC Score**: 0.8245
- **Can correctly identify 85% of high-risk patients**

---

## 💻 Web App Demo

The app lets you:
1. Enter patient information (age, tumor size, etc.)
2. Click "Predict"
3. See if patient is high-risk or low-risk
4. Demo video uploaded in the repository
---

## 📚 What I Learned

- How to clean and prepare medical data
- Survival analysis (Kaplan-Meier curves)
- Training and comparing ML models
- Building web applications with Streamlit
- Using Python for healthcare predictions

---

## 🔮 Future Improvements

- Add more patient data
- Try more advanced models (XGBoost, Neural Networks)
- Make the app mobile-friendly
- Test with real hospital data

---

## ⭐ Like this project?

Give it a star on GitHub!

---

*Note: This is a learning project. Not for actual medical use.*
