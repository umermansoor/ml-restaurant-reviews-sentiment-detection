from __future__ import division
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import csv
import os

__current_dir__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

# Open the training file
reader = csv.reader(open(os.path.join(__current_dir__, "data/yelp_labelled.txt")), delimiter="\t")

# Separate the training data into reviews and their labels for classifier
categories = ['0', '1'] # 0 => Unhappy; 1 => Happy
my_list = list(reader)
reviews = [i[0] for i in my_list]
labels = [int(i[1]) for i in my_list]

# Tokenize and filter stop words. Builds a dictionary of words
count_vect = CountVectorizer()
words_dictionary_x = count_vect.fit_transform(reviews)

# Obtain term frequencies to avoid discrepancies with longer reviews that use some words many times
tf_transformer = TfidfTransformer(use_idf=True).fit(words_dictionary_x)
words_dictionary_tf_x = tf_transformer.transform(words_dictionary_x)

# Use Naive-Bayes for classification
clf = MultinomialNB().fit(words_dictionary_tf_x, labels)

## Test our positive reviews from the file
total_samples = 0
mistakes = 0
reader = csv.reader(open(os.path.join(__current_dir__, "data/positive_reviews.txt")), delimiter="\t")
positive_reviews =  [i[0] for i in list(reader)]
positive_words_x = count_vect.transform(list(positive_reviews))
positive_words_tf_x = tf_transformer.transform(positive_words_x)
predicted = clf.predict(positive_words_tf_x)

for doc, category in zip(positive_reviews, predicted):
    print('%s => %r' % (':)' if categories[category] == '1' else ':@',
                             doc[:120] + '...' if  len(doc) > 120 else doc ))
    total_samples += 1
    if categories[category] == '0':
        mistakes += 1

## Test our negative reviews from the file. TODO: Ugh: code repetition. Move to a function.
reader = csv.reader(open(os.path.join(__current_dir__, "data/negative_reviews.txt")), delimiter="\t")
negative_reviews =  [i[0] for i in list(reader)]
negative_words_x = count_vect.transform(list(negative_reviews))
negative_words_tf_x = tf_transformer.transform(negative_words_x)
predicted = clf.predict(negative_words_tf_x)

for doc, category in zip(negative_reviews, predicted):
    print('%s => %r' % (':)' if categories[category] == '1' else ':@',
                        doc[:120] + '...' if  len(doc) > 120 else doc ))
    total_samples += 1
    if categories[category] == '1':
        mistakes += 1

print('Success Percentage: %.2f' % ( (total_samples - mistakes)/total_samples))
