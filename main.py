# Fake news detection library
from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
import os
# from sklearn.feature_extraction.text import TfidfVectorizer
# import numpy as np
import re
import string
import nltk
# from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from heapq import nlargest
# nltk.download('stopwords')

# Topic modelling
# import matplotlib.pyplot as plt
# from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn.decomposition import LatentDirichletAllocation


app = Flask(__name__)
CORS(app)

# FakeNews Detection Model
model = pickle.load(open('./model/model.pkl', 'rb'))
vectorization = pickle.load(open('./model/vector.pkl', 'rb'))

def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('\\W', ' ', text)
    text = re.sub('https?://\S+\www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

def output_label(n):
    if n == 0:
        return "Fake News"
    elif n == 1:
        return "Real News"


# Text summarize
def nltk_extractive_summarization(text, num_sentences):
    # Tokenize the text into sentences
    sentences = sent_tokenize(text)

    # Initialize stopword remover using Sastrawi
    stopword_factory = StopWordRemoverFactory()
    stop_words = stopword_factory.get_stop_words()

    # Calculate word frequency
    word_freq = {}
    for sentence in sentences:
        for word in nltk.word_tokenize(sentence.lower()):
            if word not in stop_words:
                if word not in word_freq:
                    word_freq[word] = 1
                else:
                    word_freq[word] += 1

    # Calculate sentence scores based on word frequency
    sentence_scores = {}
    for i, sentence in enumerate(sentences):
        for word in nltk.word_tokenize(sentence.lower()):
            if word in word_freq:
                if i not in sentence_scores:
                    sentence_scores[i] = word_freq[word]
                else:
                    sentence_scores[i] += word_freq[word]

    # Get the top N sentences with highest scores
    top_sentences = nlargest(num_sentences, sentence_scores, key=sentence_scores.get)

    # Sort the top sentences in their original order
    summary = ' '.join([sentences[j] for j in sorted(top_sentences)])

    return summary

@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.get_json()
    text = data['text']
    num_sentences = data.get('num_sentences')
    
    summary = nltk_extractive_summarization(text, num_sentences)
    
    response = {
        'summary': summary,
        'sentences': num_sentences
    }
    
    return jsonify(response)

# Fake News
@app.route('/predict', methods=['POST'])
def predict():
    news = request.json['news']
    result = manual_testing(news)
    
    return jsonify(result)

def manual_testing(news):
    test_news = {"test": [news]}
    new_def_test = pd.DataFrame(test_news)
    new_def_test['text'] = new_def_test['test'].apply(wordopt)
    new_x_test = new_def_test['text']
    new_xv_test = vectorization.transform(new_x_test)
    pred_LR = model.predict(new_xv_test)

    return {"prediction": output_label(pred_LR[0])}

# Topic modelling
@app.route('/topic-modeling', methods=['POST'])
def topic_modeling():
    data = request.get_json()
    text = data['text']
    num_topics = data['num_topics']
    result = nltk_topic_modeling(text, num_topics)
    return jsonify(result)

def nltk_topic_modeling(text, num_topics, stopwords_file='./stopwords_indo_topic.xlsx'):
    # Tokenize the text into sentences
    sentences = sent_tokenize(text)

    # Initialize stopword remover using Sastrawi
    stopword_factory = StopWordRemoverFactory()
    stop_words = stopword_factory.get_stop_words()

    # Load additional stopwords from Excel file
    stopwords_df = pd.read_excel(stopwords_file)
    additional_stop_words = list(stopwords_df['Kata'])
    stop_words.extend(additional_stop_words)

    # Initialize CountVectorizer
    vectorizer = CountVectorizer(stop_words=stop_words)

    # Create document-term matrix
    X = vectorizer.fit_transform(sentences)

    # Generate topic model using LDA
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(X)

    # Get the most important words for each topic
    feature_names = vectorizer.get_feature_names_out()
    topic_keywords = []
    for topic_idx, topic in enumerate(lda.components_):
        top_words_indices = topic.argsort()[:-6:-1]  # Get top 5 words for each topic
        top_words = [feature_names[i] for i in top_words_indices]
        topic_keywords.append(top_words)

    return {
        'topic_keywords': topic_keywords,
    }

# Home
@app.route('/')
def index():

    return jsonify({"message": "Welcome to Fake News Detection API"})

if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))
