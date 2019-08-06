'''
Build a spam classifier using Naive Bayes
Data source: https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection
'''

# Read tab-separated file using pandas
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

df = pd.read_table('smsspamcollection/SMSSpamCollection', sep='\t', header=None, names=['label', 'sms_message'])  # Tab separated, no header, assigns column names


# Print the first 5 rows
df.head(5)

# Preprocess the data and convert labels into a binary classification
df['label'] = df.label.map({'ham':0, 'spam':1})  

# Show the number of rows and columns in the data
df.shape  


# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['sms_message'], 
                                                    df['label'], 
                                                    random_state=1)

print('Number of rows in the total set: {}'.format(df.shape[0]))
print('Number of rows in the training set: {}'.format(X_train.shape[0]))
print('Number of rows in the test set: {}'.format(X_test.shape[0]))


# Start using Bag of Words to count the frequency of words in text and create an instance of CountVectorizer
count_vector = CountVectorizer()

# Show the parameter values automatically included with CountVectorizer
print(count_vector) 

# Fit CountVectorizer and return the matrix
training_data = count_vector.fit_transform(X_train) 
testing_data = count_vector.transform(X_test) 

# Return the feature names for the dataset that make up the vocabulary in the df
count_vector.get_feature_names()


# Use Naive Bayes to make predictions
naive_bayes = MultinomialNB() 

# Train the classifier on the training set
naive_bayes.fit(training_data, y_train) 

# Make predictions on the test data and store them in a new variable
predictions = naive_bayes.predict(testing_data)  

# Evaluate model’s precision, recall, F1 score, and accuracy
print('Accuracy score: ', format(accuracy_score(y_test, predictions)))
print('Precision score: ', format(precision_score(y_test, predictions)))
print('Recall score: ', format(recall_score(y_test, predictions)))
print('F1 score: ', format(f1_score(y_test, predictions)))
