import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

#Initialize the lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

#Load the datasets
#Insert your file paths here
fake_news_path = r'D:\Fake News Project\Fake.csv'
true_news_path = r'D:\Fake News Project\True.csv'

#Load the CSV files
fake_news_df = pd.read_csv(fake_news_path)
true_news_df = pd.read_csv(true_news_path)

#Add a new column 'label' to both datasets to indicate real or fake
fake_news_df['label'] = 'fake'
true_news_df['label'] = 'real'

#Combine both datasets into one
combined_df = pd.concat([fake_news_df, true_news_df], ignore_index=True)

# Function to clean the text
def clean_text(text):

    text = text.lower()
    #Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', '', text)
    #Remove numbers
    text = re.sub(r'\d+', '', text)
    words = text.split() #Tokenize
    
    #Remove stopwords and lemmatize each word
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    
    #Join the cleaned words back into a single string
    cleaned_text = ' '.join(words)
    
    return cleaned_text

#Apply the cleaning function to the 'text' column
combined_df['cleaned_text'] = combined_df['text'].apply(clean_text)

#Save the cleaned dataset to a CSV file
output_path = r'D:\Fake News Project\cleaned_news_dataset.csv'
combined_df.to_csv(output_path, index=False)

#Display a preview of the cleaned dataset
print(combined_df.head())
