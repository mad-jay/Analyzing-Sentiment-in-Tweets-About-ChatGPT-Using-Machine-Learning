from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd
import re
from sklearn.preprocessing import LabelEncoder
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from keras.preprocessing.text import Tokenizer
import tensorflow as tf

# Load your data
data = pd.read_csv("file.csv")
data = data.drop([data.columns[0]], axis=1)

# Encode your target labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(data['labels'])
original_classes = label_encoder.classes_

# Load your pre-trained model
model1 = tf.keras.models.load_model('cnnModel.h5')

# Define your preprocessing functions
lem = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def is_alpha(word):
    for part in word.split('-'):
        if not part.isalpha():
            return False
    return True

def clean_dataset(text):
    text = re.sub(r'http\S+', '', text) # removing links
    text = re.sub(r'\\n', ' ', text) # removing \\n
    text = re.sub(r"\s*#\S+", "", text) # removing hash tags
    text = re.sub(r"\s*@\S+", "", text) # removing @
    text = text.lower()
    words = [word for word in word_tokenize(text) if is_alpha(word)]
    words = [lem.lemmatize(word) for word in words]
    words = [w for w in words if not w in stop_words]
    text = " ".join(words)
    return text.strip()

# Apply preprocessing to your data
data['cleaned_tweets'] = data['tweets'].apply(clean_dataset)

# Convert text data to numerical sequences
num_words = 7000
tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(data['cleaned_tweets'])
sequences = tokenizer.texts_to_sequences(data['cleaned_tweets'])
maxlen = 140
pad_seqs = pad_sequences(sequences, maxlen=maxlen)

def perform_sentiment_analysis(tweets , model):
    # Clean the input tweets
    cleaned_data = [clean_dataset(tweet) for tweet in tweets]
    # Convert cleaned tweets to numerical sequences
    sequences = tokenizer.texts_to_sequences(cleaned_data)
    pad_seqs = pad_sequences(sequences, maxlen=maxlen)
    # Predict sentiment labels
    predicted_labels = model.predict(pad_seqs)
    # Decode predicted labels
    original_labels = label_encoder.inverse_transform(np.argmax(predicted_labels, axis=1))
    # Create a DataFrame with predicted labels
    data = pd.DataFrame({ 'Tweets': tweets, 'labels': original_labels })
    # Print predicted labels and data
    #print("Predicted Labels:")
    #print(predicted_labels)
    #print("\nData:")
    #print(data)
    return original_labels

def sentimenthehe(sent):
    if 'bad' in sent:
        return 'Negative'
    elif 'good' in sent:
        return 'Positive'
    elif 'neutral' in sent:
        return 'Neutral'
    else:
        return 'unknown'



# Test the sentiment analysis function
#perform_sentiment_analysis(["The worst result, I did not expect that unwanted results. it is a useless tool",
#    "my name is ahmed i want to became a data scientist"] , model1)

