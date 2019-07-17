'''
Build a spam classifier using Naive Bayes
Data source: https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection
'''

# read tab-separated file using pandas
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

df = pd.read_table('smsspamcollection/SMSSpamCollection', sep='\t', header=None, names=['label', 'sms_message'])  # tab separated, no header, assigns column names


# print the first 5 rows of your data
df.head(5)

# preprocess your data and convert your label into a binary classification to 1 or 0
df['label'] = df.label.map({'ham':0, 'spam':1})  # maps a 0 or 1 to a string input in your dataframe
df.shape   # gives you the number of rows and columns in the data


# split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['sms_message'], 
                                                    df['label'], 
                                                    random_state=1)

print('Number of rows in the total set: {}'.format(df.shape[0]))
print('Number of rows in the training set: {}'.format(X_train.shape[0]))
print('Number of rows in the test set: {}'.format(X_test.shape[0]))


# start using Bag of Words to count the frequency of words in text and create an instance of CountVectorizer
count_vector = CountVectorizer()
print(count_vector) # shows the parameter values automatically included with CountVectorizer

# fit CountVectorizer to your dataset
training_data = count_vector.fit_transform(X_train)  # fits the training data and then return the matrix
testing_data = count_vector.transform(X_test)  # transforms testing data and returns the matrix. Note we are not fitting the testing data into the CountVectorizer()
count_vector.get_feature_names() # returns the feature names for the dataset that make up the vocabulary in the df


# use sklearn.naive_bayes to make predictions
naive_bayes = MultinomialNB() # call the method
naive_bayes.fit(training_data, y_train) # train the classifier on the training set
predictions = naive_bayes.predict(testing_data)  # make your predictions on the test data and store them in a new variable


# evaluate model’s precision, recall, and accuracy
print('Accuracy score: ', format(accuracy_score(y_test, predictions)))
print('Precision score: ', format(precision_score(y_test, predictions)))
print('Recall score: ', format(recall_score(y_test, predictions)))
print('F1 score: ', format(f1_score(y_test, predictions)))
