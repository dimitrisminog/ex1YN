import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')

import tensorflow as tf
# synarthsh_error
def custom_error(y_true, y_pred):
    
    error = tf.where(y_pred <= y_true, y_true - y_pred, y_pred - y_true)
    return error


script_path = os.path.abspath(__file__)

print("Script path:", script_path)

# diavasma arxeiou
dataframe = pd.read_csv(r"C:\\python\\iphi2802.csv", header=0, sep='\t', encoding='utf-8', engine='python')

# afairesh shmeiwn stixhs
dataframe['text'] = dataframe['text'].str.replace(r'[.,\[\]\(\)\-]', '', regex=True)
#print(dataframe)

all_words = ' '.join(dataframe['text']).split()
#print(all_words)
# counter gia na vrw thn syxnothta
word_counts = Counter(all_words)

# epilogh twn 100 pio syxnwn
most_common_words = [word for word, _ in word_counts.most_common(1000)]
#print(most_common_words)

# vectorizer me vocabulary mono gia tis 1000
vectorizer = TfidfVectorizer(vocabulary=most_common_words)

# ypologismos se olo to text
tfidf_matrix = vectorizer.fit_transform([' '.join(all_words)])

# epilogh mono twn 1000
selected_indices = [vectorizer.vocabulary_[word] for word in most_common_words]
tfidf_values = tfidf_matrix[:, selected_indices]
#print(tfidf_values)



dense_tfidf_values = tfidf_values.toarray()
#print(dense_tfidf_values)
dense_tfidf_values=dense_tfidf_values.reshape(-1,1)
#print(dense_tfidf_values)
#  MinMaxScaler kanonikopoihsh
scaler = MinMaxScaler(feature_range=(0, 1))
nrmlzd_tfidf_values = scaler.fit_transform(dense_tfidf_values)
#print(normalized_tfidf_values)
# pairnw ta antistoixa dates mono twn 1000 lexewn
spes_words = dataframe.iloc[:nrmlzd_tfidf_values.shape[0]]

ypologismos meshs timhs
labels_min_max = spes_words[['date_min', 'date_max']].mean(axis=1).values

# xwrismos dedomenw se syola ekpaideyshs
X_train, X_test, y_train, y_test = train_test_split(nrmlzd_tfidf_values, labels_min_max, test_size=0.2, random_state=42)
def create_model(hidden_units, learning_rate=0.001):
    inputs = Input(shape=(X_train.shape[1],))  # Define input shape
    x = Dense(hidden_units, activation='relu')(inputs)
    predictions = Dense(1, activation='linear')(x)
    model = Model(inputs=inputs, outputs=predictions)
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss=custom_error, optimizer=optimizer)
    return model


optimizer = Adam(learning_rate=0.001)

 # ypologismos rmse gai 5 fold cv
kf = KFold(n_splits=5)
rmse_scores = []

for train_index, test_index in kf.split(nrmlzd_tfidf_values):
    X_train, X_test = nrmlzd_tfidf_values[train_index], nrmlzd_tfidf_values[test_index]
    y_train, y_test = labels_min_max[train_index], labels_min_max[test_index]

    #orismos modelou
    model = create_model(hidden_units=64)  # Experiment with the number of hidden units
    #ekpaideysh
    history=model.fit(X_train, y_train, epochs=100, batch_size=128, verbose=0)
    #Plot convergence (loss) over epochs
    #plt.plot(history.history['loss'])
    #plt.xlabel('Epoch')
    #plt.ylabel('Loss')
    #plt.title('Convergence Plot')
    #plt.show()
    
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    rmse_scores.append(rmse)
average_rmse = np.mean(rmse_scores)
print("Average RMSE:", average_rmse)
