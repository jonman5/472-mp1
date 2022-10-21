import re
import sys
import gzip
import json
import csv
import pandas as pd
import contractions

import matplotlib.pyplot as plt
import seaborn
from matplotlib import rcParams
import seaborn as sns
import itertools
import datetime

import pprint
import warnings
import joblib

import nltk
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import metrics
from sklearn.neural_network import MLPClassifier

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
# stopwords = nltk.corpus.stopwords.words('english)')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.tree  import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV


from time import time
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder

# CONSTANTS
MNB = "Multinomial Naive Bayes"
DT = "Decision Tree"
MLP = "Multi Layer Perceptron"
CLASSIFIER_SENTIMENT = "sentiment"
CLASSIFIER_EMOTION = "emotion"
PERFORMANCE_FILE_PATH = "../../Task 2/performance.txt"
TRAIN_CSV_FILE_PATH = "../../Task 2/train.csv"
MAIN_DATA_FILE_PATH = "../../Task 1/goemotions.json.gz"


with gzip.open(MAIN_DATA_FILE_PATH, 'rt') as zipfile:
    raw_data = json.load(zipfile)

# 2.1 process the dataset, extract tokens, and frequencies and display number of tokens
corpus = []
for line in raw_data:
    corpus.append(line[0])

vectorizer = CountVectorizer()
vectorizer.fit(corpus)
# print("The size of vocabulary is ", vectorizer.vocabulary_.__sizeof__())

# DATA PREPARATION
header = ['text', 'emotion', 'sentiment']
data = []
for line in raw_data:
    data_row = line[0], line[1], line[2]
    data.append(data_row)

# place data into a .csv file
with open(TRAIN_CSV_FILE_PATH, 'w', encoding='UTF-8') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    for d in data:
        writer.writerow(d)

# read data from .csv
df = pd.read_csv(TRAIN_CSV_FILE_PATH)
df = df.applymap((lambda x: x.strip() if isinstance(x, str) else x))

# DATA PROCESSING
# Removing empty rows from csv
df.dropna(axis=0, how='all', inplace=True)

# remove rows that contain null values
df = df[df['text'].notna()]

# apply contractions
df['text'] = df['text'].apply(lambda x: [contractions.fix(word) for word in x.split(' ')])
df['text'] = [' '.join(map(str, ln)) for ln in df['text']]

# noise cleaning
df['text'] = df['text'].str.lower()
df['text'] = df['text'].apply(lambda x: re.sub(r'[^\w\d\s\']+', '', x))

# nlkt tokenization
# df['text_tokenized'] = df['text'].apply(word_tokenize)

# encode labels columns
labelEncoder_sentiment = LabelEncoder()
labelEncoder_emotion = LabelEncoder()
df['encoded_label_sentiment'] = labelEncoder_sentiment.fit_transform(df['sentiment'])
df['encoded_label_emotion'] = labelEncoder_sentiment.fit_transform(df['emotion'])

# Extract label columns
targets_sentiment = df['encoded_label_sentiment'].values
targets_emotion = df['encoded_label_emotion'].values

# Remove label columns to get only text data
df.drop('sentiment', axis=1, inplace=True)
df.drop('emotion', axis=1, inplace=True)
df.drop('encoded_label_sentiment', axis=1, inplace=True)
df.drop('encoded_label_emotion', axis=1, inplace=True)

transformer = TfidfTransformer(smooth_idf=False)
count_vectorizer = CountVectorizer(ngram_range=(1, 2))

# fit data to the count vectorizer
df_counts = count_vectorizer.fit_transform(df['text'].values)

def write_to_performance_file(filename, model, classifier_task, c_matrix, c_report):
    with open(filename, 'a', encoding='UTF-8') as file:
        file.write(model)
        file.write("\n")
        file.write("Classification task: " + classifier_task)
        file.write("\n")
        file.write("\nConfusion Matrix\n")
        file.write(str(c_matrix))
        file.write("\n\nClassification Report\n")
        file.write(str(c_report))


# ============= Top MNB ============================
# For Sentiment
X_train, X_test, y_train, y_test = train_test_split(df_counts, targets_sentiment, test_size=0.2, random_state=0)
classifier = MultinomialNB()
parameters = {'alpha': (0.5, 0.0, 1.0, 0.8)}

grid_search = GridSearchCV(classifier, parameters, scoring='f1_weighted')
top_classifier = grid_search.fit(X_train, y_train)
print("********Top MNB********")
joblib.dump(top_classifier, "../trained/TOP_MNB_sentiment_trained.joblib")
print(MLP + " for classifier " + CLASSIFIER_SENTIMENT + " trained")
print("Best Parameters: ", top_classifier.best_params_)

y_preds = top_classifier.predict(X_test)
confusion_matrix = metrics.confusion_matrix(y_test, y_preds)
cl_report = metrics.classification_report(y_test, y_preds)

write_to_performance_file(PERFORMANCE_FILE_PATH, "Top-" + MNB, CLASSIFIER_SENTIMENT, confusion_matrix, cl_report)

# For Emotion
X_train, X_test, y_train, y_test = train_test_split(df_counts, targets_sentiment, test_size=0.99, random_state=0)
classifier = MultinomialNB()

grid_search = GridSearchCV(classifier, parameters, scoring='f1_weighted')
top_classifier = grid_search.fit(X_train, y_train)
joblib.dump(top_classifier, "../trained/TOP_MNB_emotion_trained.joblib")
print(MLP + " for classifier " + CLASSIFIER_EMOTION + " trained")
print("Best Parameters: ", top_classifier.best_params_)

y_preds = top_classifier.predict(X_test)
confusion_matrix = metrics.confusion_matrix(y_test, y_preds)
cl_report = metrics.classification_report(y_test, y_preds)

write_to_performance_file(PERFORMANCE_FILE_PATH, "Top-" + MNB, CLASSIFIER_EMOTION, confusion_matrix, cl_report)
