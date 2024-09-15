# Fake News Detection Project

This project focuses on detecting fake news using machine learning techniques. Initially, a Support Vector Machine (SVM) classifier was employed for classification, and now the project is exploring the use of transformer-based models, such as BERT.

## Table of Contents
1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Setup](#setup)
4. [Training the Model](#Step-2-Train-the-Model)
5. [Running the App](#running-app)
6. [Future Work](#future-work)
7. [Acknowledgments](#acknowledgments)

## Overview
This repository contains the code and resources needed to train a model that classifies news articles as either "Fake" or "Real." The project uses:
- SVM (Support Vector Machine) for classification
- TF-IDF for text feature extraction
- A preprocessed dataset of news articles

In future versions, we aim to explore transformer models like **BERT** for enhanced accuracy.

## Project Structure
The repository consists of the following files:
### 1. `fakenews_model.py`
- Loads the **cleaned_news_dataset.csv** file.
- Preprocesses the text using **TfidfVectorizer**.
- Trains an SVM model using Scikit-learn’s `SVC` classifier.
- Evaluates the model using metrics like accuracy score and confusion matrix.
- Saves the trained model and vectorizer using **pickle** for future use in the Streamlit app.

### 2. `app.py`
- A **Streamlit** app that lets users classify news articles as fake or real.
- It loads the pre-trained SVM model and vectorizer, processes user input, and displays the classification result.

### 3. `cleaner.py`
- Contains functions for cleaning the dataset.
- Removes unwanted characters, stop words, and applies text normalization techniques.

### 4. `cleaned_news_dataset.csv`
- The dataset used for model training.
- Includes the following columns:
  - **title**: News article title.
  - **text**: Full content of the news article.
  - **subject**: Category of the news article (e.g., politics, sports).
  - **date**: Date the article was published.
  - **label**: Whether the article is **fake** or **real**.
  - **cleaned_text**: Preprocessed text.

## Setup

### Step 1: Install Dependencies 
``` pip install -r requirements.txt ```

### Step 2: Train the Model 
```python fakenews_model.py ```
This script will preprocess the data, train the SVM model, and save it as svm_model.pkl.

## Running App
After everything is trained, you could use the Streamlit app in order to actually use the model. 

```streamlit run app.py```

- Enter a news article in the input box on the app.
- Click "Predict" to classify the article as either Fake or Real.

## Future Work
- Integrating transformer models like BERT for better accuracy.
- Implement more advanced text preprocessing steps.
- Exploring multi-language support and more nuanced classifications.


## Acknowledgments
- Scikit-learn for providing a robust machine learning framework.
- Hugging Face for providing pre-trained transformer models.
- Streamlit for making it easy to create interactive web applications.
