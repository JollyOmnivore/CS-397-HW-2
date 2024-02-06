import pandas as pd
import nltk

from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import  accuracy_score, precision_score, recall_score, f1_score #needed to add precision_score, recall_score, f1_score

"""
Seperates out the data to train on (X) and the labels we wish to classify (y).
"""
spam_df = pd.read_csv("SPAM.csv")
spam_y = spam_df["Category"] # what we are trying to classify!
spam_X = spam_df["Message"] # What we are training our model on -> convert to BoW vector

"""
Perform vectorization and label encoding
"""
cv = CountVectorizer()
spamXVectorized = cv.fit_transform(spam_X)
le = LabelEncoder()
spamYEncoded = le.fit_transform(spam_y)

"""
Create a 5-fold cross validation experiment using Naive Bayes.
Calculate the average precision, accuracy, recall, and f1-scores across the five folds.
"""

KF = KFold(n_splits=5)
accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []


for train_index, test_index in KF.split(spamXVectorized): #loop created to iterate through Folds 

    X_train, X_test = spamXVectorized[train_index], spamXVectorized[test_index]
    y_train, y_test = spamYEncoded[train_index], spamYEncoded[test_index]


    nb = MultinomialNB()
    nb.fit(X_train, y_train)


    predictions = nb.predict(X_test)
    accuracy_scores.append(accuracy_score(y_test, predictions))
    precision_scores.append(precision_score(y_test, predictions))
    recall_scores.append(recall_score(y_test, predictions))
    f1_scores.append(f1_score(y_test, predictions))


avg_accuracy = sum(accuracy_scores) / len(accuracy_scores)
avg_precision = sum(precision_scores) / len(precision_scores)
avg_recall = sum(recall_scores) / len(recall_scores)
avg_f1 = sum(f1_scores) / len(f1_scores)

print(f"Average Accuracy: {avg_accuracy}")
print(f"Average Precision: {avg_precision}")
print(f"Average Recall: {avg_recall}")
print(f"Average F1-Score: {avg_f1}")
"""
Q1 Precision score was the worst because there is many more instances of nonspam than spam in the training data

Q2 The accuracy score was the highest on average this also come down to the unbalanced dataset and this causes to just assume most are ham rather than spam

Q3 From the data and scores the model has gotten precision is more important this will lower the number of false Positives and precision is a more true test of a models capability to detect spam

"""
