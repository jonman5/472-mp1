import gzip
import json

import gensim.downloader as api
from nltk.tokenize import word_tokenize
from gensim.models import KeyedVectors, Word2Vec


class Part3:
    def __init__(self):
        self.w2model = api.load("word2vec-google-news-300")
        self.posts_average_emb_scores = []
        self.hit_rates = []
        self.tokenized_posts = []

    def get_posts(self):
        json_data = gzip.open('../Task 1/goemotions.json.gz', 'r')
        raw_data = json.load(json_data)
        posts = [x[0] for x in raw_data]
        return posts

    def tokenize_posts(self, p):
        results = [word_tokenize(t) for t in p]
        self.tokenized_posts = results

    def tokenize_posts_partial_for_debugging(self, p, nr_posts):
        posts = p[:nr_posts]
        results = [word_tokenize(t) for t in posts]
        self.tokenized_posts = results

    def get_tokenized_posts(self):
        return self.tokenized_posts

    def compute_embeddings_and_hit_rates(self):
        model = self.w2model
        for tp in self.tokenized_posts:
            sum_embeddings = 0
            nr_words = 0
            missed_words = 0
            hit_rate = 0
            for w in tp:
                try:
                    tp = model.most_similar(w)[0]
                    vl = tp[1]
                    sum_embeddings += vl
                    nr_words += 1
                except KeyError:
                    missed_words += 1
                    continue
            try:
                post_average = sum_embeddings / nr_words
                hit_rate = ((nr_words - missed_words) / nr_words) * 100
            except ZeroDivisionError:
                post_average = 0
            self.posts_average_emb_scores.append(post_average)
            self.hit_rates.append(hit_rate)

    def compute_embeddings_and_hit_rates_partial_for_debugging(self, nr_posts):
        model = self.w2model
        for tokenized_post in self.tokenized_posts[:nr_posts]:
            nr_words = 0
            missed_words = 0
            hit_rate = 0

            try:
                mean_embeddings = model.get_mean_vector(tokenized_post)
                # tokenized_posts = model.most_similar(w)[0]
                # vl = tokenized_posts[1]
                # sum_embeddings += vl
                # nr_words += 1
            except KeyError:
                missed_words += 1
                continue
            # try:
            #     post_average = sum_embeddings / nr_words
            #     hit_rate = ((nr_words - missed_words) / nr_words) * 100
            # except ZeroDivisionError:
            #     post_average = 0
            self.posts_average_emb_scores.append(mean_embeddings)
            self.hit_rates.append(hit_rate)

    def get_hit_rates(self):
        return self.hit_rates

    def get_embedding_scores(self):
        return self.posts_average_emb_scores

    def display_nr_tokens(self):
        print("Total Number of tokens in all posts: ", len(self.tokenized_posts))

    def display_embeddings_test(self):
        embeddings = self.w2model.get_vector('computer')  # get numpy vector of a word
        print(embeddings)


def main():
    p3 = Part3()
    debugging_nr_posts = 3
    #p3.tokenize_posts(p3.get_posts())
    p3.tokenize_posts_partial_for_debugging(p3.get_posts(), debugging_nr_posts)
    p3.display_nr_tokens()
    p3.compute_embeddings_and_hit_rates_partial_for_debugging(debugging_nr_posts)
    print(p3.get_embedding_scores())
    print(p3.get_hit_rates())
    # p3.display_embeddings_test()


if __name__ == '__main__':
    main()
