from scipy.sparse import csr_matrix
import streamlit as st
import pickle

# Step 1: Load the pre-trained model and vectorizer
model = pickle.load(open('svm_model.pkl', 'rb'))  # Load the trained SVM model
vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))  # Load the fitted TF-IDF vectorizer

# Streamlit app title
st.title("Fake News Detection App")

# Step 2: Create a text input area for user input
input_text = st.text_area("Enter a news article to classify it as Fake or Real:")

# Step 3: When the user clicks the "Predict" button
if st.button("Predict"):
    if input_text:  # Check if the user has entered text
        # Step 4: Preprocess and vectorize the input text using the loaded vectorizer
        input_vectorized = vectorizer.transform([input_text])  # Vectorize the input text
        input_dense = input_vectorized.toarray()  # Convert sparse matrix to dense format

        # Step 5: Make a prediction using the loaded model
        prediction = model.predict(input_dense)


        # Step 6: Display the prediction
        if prediction[0] == 0:  # 0 corresponds to 'fake'
            st.write("Prediction: Fake News")
        else:  # 1 corresponds to 'real'
            st.write("Prediction: Real News")
    else:
        st.write("Please enter some text for classification.")
