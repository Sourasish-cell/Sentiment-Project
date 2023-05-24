import nltk
#nltk.download('averaged_perceptron_tagger')
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import time
analyzer = SentimentIntensityAnalyzer()
threshold = 0.5

ls = analyzer.polarity_scores("VADER Sentiment looks interesting, I have high hopes!")
print(ls)


pos_count = 0
pos_correct  = 0
with open("Positive.txt","r") as f:
    for line in f.read().split('\n'):
        vs = analyzer.polarity_scores(line)
        if not vs['neg'] > 0.1:
            if vs['pos']-vs['neg'] >= 0:
                pos_correct += 1
            pos_count += 1

neg_count = 0
neg_correct = 0

with open("Negative.txt","r") as f:
    for line in f.read().split('\n'):
        vs = analyzer.polarity_scores(line)
        if not vs['pos'] > 0.1:
            if vs['pos']-vs['neg'] >= 0:
                neg_correct += 1
            neg_count += 1

print("Positive Accuracy = {}% via {} samples".format(pos_correct/pos_count*100.0, pos_count))
print("Negative Accuracy = {}% via {} samples".format(neg_correct/neg_count*100.0, neg_count))
