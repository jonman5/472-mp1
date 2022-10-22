import gzip
import json

import gensim.downloader as api
from nltk.tokenize import word_tokenize
from sklearn import metrics
from gensim.models import KeyedVectors, Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class Helper:
    def __init__(self, embedding_model_name):
        self.w2model = api.load(embedding_model_name)
        self.posts, self.sentiments, self.emotions = [], [], []
        self.encoded_emotions, self.encoded_sentiments = [], []
        self.average_embeddings = []
        self.hit_rates = []
        self.tokenized_posts = []

    def extract_posts_emotions_sentiments(self):
        json_data = gzip.open('goemotions.json.gz', 'r')
        raw_data = json.load(json_data)
        self.posts = [x[0] for x in raw_data]
        self.emotions = [x[1] for x in raw_data]
        self.sentiments = [x[2] for x in raw_data]

    def get_posts(self):
        if len(self.posts) == 0:
            self.extract_posts_emotions_sentiments()
        return self.posts

    def get_emotions(self):
        if len(self.emotions) == 0:
            self.extract_posts_emotions_sentiments()
        return self.emotions

    def get_sentiments(self):
        if len(self.sentiments) == 0:
            self.extract_posts_emotions_sentiments()
        return self.sentiments

    def tokenize_posts(self, p):
        results = [word_tokenize(t) for t in p]
        self.tokenized_posts = results

    def get_tokenized_posts(self):
        return self.tokenized_posts

    def compute_embeddings_and_hit_rates(self):
        model = self.w2model

        # Loop through the tokenized posts and append embeddings to 'average_embeddings'
        # As well as hit rates to 'hit_rates'
        for tokenized_post in self.tokenized_posts:
            skipped_words = 0
            hit_rate = 0

            # Get mean word embeddings vector from pre-trained model from tokenized post
            self.average_embeddings.append(model.get_mean_vector(tokenized_post))

            # Calculate hit rate for the tokenized post
            for word in tokenized_post:
                if not model.__contains__(word):
                    skipped_words += 1
            try:
                no_of_words = len(tokenized_post)
                hit_rate = ((no_of_words - skipped_words) / no_of_words) * 100
            except ZeroDivisionError:
                post_average = 0
            self.hit_rates.append(hit_rate)

    def encode_emotions_sentiments(self):
        # encode emotions and sentiments
        label_encoder_emotion = LabelEncoder()
        label_encoder_sentiment = LabelEncoder()
        self.encoded_emotions = label_encoder_emotion.fit_transform(self.emotions)
        self.encoded_sentiments = label_encoder_sentiment.fit_transform(self.sentiments)

    def get_train_test_data(self, classification, test_size):
        self.encode_emotions_sentiments()
        if classification == "emotions":
            return train_test_split(self.average_embeddings, self.encoded_emotions, test_size=test_size, random_state=0)
        elif classification == "sentiments":
            return train_test_split(self.average_embeddings, self.encoded_sentiments, test_size=test_size, random_state=0)

    def get_hit_rates(self):
        return self.hit_rates

    def get_embedding_scores(self):
        return self.average_embeddings

    def display_nr_tokens(self):
        no_tokens = 0
        for tokenized_post in self.tokenized_posts:
            no_tokens += len(tokenized_post)
        print("Total Number of tokens in all posts: ", no_tokens)

    def display_embeddings_test(self):
        embeddings = self.w2model.get_vector('computer')  # get numpy vector of a word
        print(embeddings)
