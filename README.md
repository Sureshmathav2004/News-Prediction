# News Authenticity Predictor 📰

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange)
![pandas](https://img.shields.io/badge/pandas-1.3%2B-lightgrey)
![NLP](https://img.shields.io/badge/NLP-TF--IDF-brightgreen)

A machine learning application that classifies news articles as genuine or fake using multiple classification algorithms.

## Features ✨

- Three different classification models:
  - Logistic Regression
  - Decision Tree
  - Random Forest
- Text preprocessing pipeline:
  - HTML tag removal
  - URL removal
  - Special character cleaning
  - Digit removal
  - Case normalization
- TF-IDF vectorization for text feature extraction
- Model performance evaluation with classification reports
- Interactive prediction interface

## Dataset 📊

The model is trained on two datasets:
- `True.csv`: Genuine news articles (labeled 'T')
- `Fake.csv`: Fake news articles (labeled 'F')

Combined dataset contains approximately 44,000 news articles.

## Performance Metrics 📈

| Model                | Accuracy | Precision | Recall | F1-Score |
|----------------------|----------|-----------|--------|----------|
| Logistic Regression  | 98.7%    | 0.99      | 0.99   | 0.99     |
| Decision Tree        | 99.1%    | 0.99      | 0.99   | 0.99     |
| Random Forest        | 98.9%    | 0.99      | 0.99   | 0.99     |

*(Example metrics - replace with your actual performance)*

## Installation ⚙️

1. Clone the repository:
   ```bash
   git clone [https://github.com/yourusername/news-prediction.git](https://github.com/Sureshmathav2004/News-Prediction.git)
   cd news-prediction
