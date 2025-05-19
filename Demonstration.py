from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Load the saved GRU model
loaded_gru_model = load_model("gru_model.h5")
# Recompile the model with the appropriate optimizer, loss, and metrics
loaded_gru_model.compile(optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])

# Define the tokenizer and parameters
tokenizer = Tokenizer(num_words=5000)
max_words = 400

# Sample tweets for testing
sample_tweets = [
    "It is such a pleasure to replace Google and Wikipedia with ChatGPT. Saves so much time and lets your curiosity run wild.",
    "can kind of already do this with chatgpt video voice mode it can look at the thing and tell you what it is and how to use it",
    "If you use chatgpt for your diet plan you’re burning more rainforests than you will EVER burn calories!!",
]

# Preprocess the sample tweets
sample_sequences = tokenizer.texts_to_sequences(sample_tweets)
sample_padded = pad_sequences(sample_sequences, maxlen=max_words)

# Make predictions
predictions = loaded_gru_model.predict(sample_padded)
predicted_classes = np.argmax(predictions, axis=1)

# Decode and print results
class_labels = {0: "good", 1: "bad", 2: "neutral"}

for i, tweet in enumerate(sample_tweets):
    print(f"Tweet: \"{tweet}\"")
    print(f"Predicted Sentiment: {class_labels[predicted_classes[i]]}\n")
