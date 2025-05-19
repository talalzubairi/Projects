import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import streamlit as st

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Embedding
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

# Load your dataset
data = pd.read_csv('ChatGPT.csv')

# Mapping labels to integers
label_mapping = {'good': 0, 'bad': 1, 'neutral': 2}
data['labels'] = data['labels'].map(label_mapping)

# Extract tweets (features) and labels (target)
tweets = data['tweets'].values
labels = data['labels'].values

### SVM MODEL ###

# Split the data into training and test sets
X_train_svm, X_test_svm, y_train_svm, y_test_svm = train_test_split(
    tweets, labels, test_size=0.2, random_state=42
)

# Vectorize the text data using TF-IDF
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')
X_train_tfidf = tfidf.fit_transform(X_train_svm)
X_test_tfidf = tfidf.transform(X_test_svm)

# Train the SVM model
svm = SVC(kernel='linear', probability=True, random_state=42)
svm.fit(X_train_tfidf, y_train_svm)

# Predict on test data
y_pred_svm = svm.predict(X_test_tfidf)

# Evaluate SVM model
svm_accuracy = accuracy_score(y_test_svm, y_pred_svm)
print("\nSVM Classification Report:")
print(classification_report(y_test_svm, y_pred_svm, target_names=label_mapping.keys()))

print("SVM Confusion Matrix:")
print(confusion_matrix(y_test_svm, y_pred_svm))

### GRU MODEL ###

# Tokenization and padding
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(tweets)
x_data = tokenizer.texts_to_sequences(tweets)
max_words = 400
x_data = sequence.pad_sequences(x_data, maxlen=max_words)

# Convert labels to categorical format
y_data = to_categorical(labels, num_classes=3)

# Split data into training, validation, and test sets
x_train_gru, x_temp, y_train_gru, y_temp = train_test_split(
    x_data, y_data, test_size=0.3, random_state=42
)
x_valid_gru, x_test_gru, y_valid_gru, y_test_gru = train_test_split(
    x_temp, y_temp, test_size=0.5, random_state=42
)

# Build the GRU model
embd_len = 32
gru_model = Sequential(name="GRU_Model")
gru_model.add(Embedding(5000, embd_len, input_length=max_words))
gru_model.add(GRU(128, activation='tanh', return_sequences=False))
gru_model.add(Dense(3, activation='softmax'))

gru_model.compile(loss="categorical_crossentropy", optimizer=Adam(), metrics=['accuracy'])

# Train the GRU model
history_gru = gru_model.fit(
    x_train_gru, y_train_gru, batch_size=64, epochs=5, validation_data=(x_valid_gru, y_valid_gru), verbose=1
)

# Evaluate GRU model
gru_score = gru_model.evaluate(x_test_gru, y_test_gru, verbose=0)
print("\nGRU Model Test Accuracy:", gru_score[1])

### PERFORMANCE COMPARISON ###

# Plotting Training and Validation Accuracy for GRU
plt.figure(figsize=(10, 6))
plt.plot(history_gru.history['accuracy'], label='Training Accuracy (GRU)', color='blue')
plt.plot(history_gru.history['val_accuracy'], label='Validation Accuracy (GRU)', color='orange')
plt.title('GRU Model Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# Bar plot for SVM vs GRU Test Accuracy
plt.figure(figsize=(6, 4))
models = ['SVM', 'GRU']
accuracy_scores = [svm_accuracy, gru_score[1]]
plt.bar(models, accuracy_scores, color=['blue', 'green'])
plt.title('Model Test Accuracy Comparison')
plt.ylabel('Accuracy')
plt.show()

# Confusion Matrix for GRU
y_pred_gru = gru_model.predict(x_test_gru)
y_pred_gru = np.argmax(y_pred_gru, axis=1)
y_test_gru_labels = np.argmax(y_test_gru, axis=1)

print("\nGRU Confusion Matrix:")
print(confusion_matrix(y_test_gru_labels, y_pred_gru))
print("\nGRU Classification Report:")
print(classification_report(y_test_gru_labels, y_pred_gru, target_names=label_mapping.keys()))

# Conclusion on performance
print(f"SVM Test Accuracy: {svm_accuracy:.2f}")
print(f"GRU Test Accuracy: {gru_score[1]:.2f}")


# User Input for Prediction
st.subheader("Predict Sentiment for New Input")
user_input = st.text_input("Enter a sentence:")
if user_input:
    input_seq = tokenizer.texts_to_sequences([user_input])
    input_pad = pad_sequences(input_seq, maxlen=400)
    pred_gru = np.argmax(gru_model.predict(input_pad), axis=1)[0]
    st.write(f"GRU Model Prediction: {label_mapping[pred_gru]}")

    pred_svm = svm.predict(tfidf.transform([user_input]))[0]
    st.write(f"SVM Model Prediction: {label_mapping[pred_svm]}")

# Save the trained model
gru_model.save("gru_model.h5")
print("GRU model saved as 'gru_model.h5'")