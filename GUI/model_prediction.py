import joblib
import numpy as np
import nltk
import email
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from sklearn.preprocessing import MinMaxScaler
from lime.lime_text import LimeTextExplainer
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')

# Load the Logistic Regression (BoW) model
lrBOW = joblib.load('lrBOW_model.pkl')

# Load the Logistic Regression (TF-IDF) model
lr_tfidf = joblib.load('lr_tfidf_model.pkl')

# Load the Naive Bayes (BoW) model
cnb_bow = joblib.load('cnb_bow_model.pkl')

# Load the Naive Bayes (TF-IDF) model
cnb_tfidf = joblib.load('cnb_tfidf_model.pkl')

svm = joblib.load('svm_model.pkl')

features = joblib.load('features.pkl')

vectorizer = joblib.load('vectorizer.pkl')

token_to_index_mapping = {t:i for t, i in zip(features, range(len(features)))}
token_to_index_mapping

def extract_email_body(message):
    if message.is_multipart():
        for part in message.walk():
            type_content = part.get_content_maintype()
            if type_content == 'text':
                message = part
                break
        else:
            return ''
    body = message.get_payload(decode=False)
    return body

def read_email_from_string(s):
    message = email.message_from_string(s)
    return message

def remove_html(s):
    soup = BeautifulSoup(s, 'lxml')
    for sp in soup(['script', 'style', 'head', 'meta', 'noscript', 'http']):
        sp.decompose()
    s = ' '.join(soup.stripped_strings)
    return s


def preprocess_text(text):
    # Extract email body
    body = extract_email_body(read_email_from_string(text))
    body = remove_html(body)

    # Tokenize the text
    words = nltk.word_tokenize(body)

    # Remove punctuation and convert to lower case
    words = [word.lower() for word in words if word.isalpha()]

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    # Lemmatize the words
    lemmatizer = WordNetLemmatizer()
    specific_words = ['enron', 'e', 'hou', 'subject', 'http']
    words = [word for word in words if word not in specific_words]

    # Remove singular character tokens
    words = [word for word in words if len(word) > 1]

    words = [lemmatizer.lemmatize(word) for word in words]
    words = [word for word in words]

    return ' '.join(words)


def message_to_count_vector(message):
    count_vector = np.zeros(len(features))

    processed_list_of_tokens = nltk.word_tokenize(message)

    for token in processed_list_of_tokens:
      if token not in features:
        continue
      index = token_to_index_mapping[token]
      count_vector[index] += 1

    return count_vector

# Prediction functions
def predict_lrbow(email):
    preprocessed_email = preprocess_text(email)
    email_vector = message_to_count_vector(preprocessed_email).reshape(1, -1)
    return lrBOW.predict(email_vector)

def predict_lrtfidf(email):
    preprocessed_email = preprocess_text(email)
    email_vector = vectorizer.transform([preprocessed_email])
    return lr_tfidf.predict(email_vector)

def predict_cnbbow(email):
    preprocessed_email = preprocess_text(email)
    email_vector = message_to_count_vector(preprocessed_email).reshape(1, -1)
    return cnb_bow.predict(email_vector)

def predict_cnbtfidf(email):
    preprocessed_email = preprocess_text(email)
    email_vector = vectorizer.transform([preprocessed_email])
    return cnb_tfidf.predict(email_vector)

def predict_svm(email):
    preprocessed_email = preprocess_text(email)
    email_vector = vectorizer.transform([preprocessed_email])
    return svm.predict(email_vector)

# Function to predict probabilities using the SVM model

def predict_proba(texts):
    text_vectors = vectorizer.transform(texts)
    # SVM does not have predict_proba by default, so use decision_function
    decision_scores = svm.decision_function(text_vectors)
    
    # Ensure the decision_scores is a 2D array
    if len(decision_scores.shape) == 1:
        decision_scores = decision_scores.reshape(-1, 1)
    
    # Convert decision scores to two-class probabilities
    probas = np.hstack([1 - decision_scores, decision_scores])
    probas = np.exp(probas) / np.sum(np.exp(probas), axis=1, keepdims=True)
    return probas

# Create a LIME explainer
explainer = LimeTextExplainer(class_names=['Not Spam', 'Spam'])

