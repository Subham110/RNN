import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, LSTM, GRU, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split

# 1. Sample Data Preparation
texts = [
    "I love machine learning",
    "Deep learning is amazing",
    "RNNs are powerful for sequences",
    "I enjoy learning AI",
    "Neural networks can learn patterns",
    "AI is the future",
    "I dislike bad predictions",
    "This model works well",
    "Sometimes models overfit",
    "Data preprocessing is important"
]

labels = [1, 1, 1, 1, 1, 1, 0, 1, 0, 1]  # Binary classification labels

# Tokenize text
tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

# Pad sequences to same length
max_len = max(len(seq) for seq in sequences)
X = pad_sequences(sequences, maxlen=max_len, padding='post')

# Convert labels to numpy array
y = np.array(labels)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Build RNN Model
model = Sequential([
    Embedding(input_dim=1000, output_dim=16, input_length=max_len),
    LSTM(32, return_sequences=False),  # You can replace with SimpleRNN or GRU
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')  
])


# 3. Compile Model
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])


# 4. Train Model
history = model.fit(X_train, y_train,validation_data=(X_test, y_test),epochs=10,batch_size=2,verbose=1)

# 5. Evaluate Model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {accuracy:.4f}")


# 6. Make Predictions
sample_text = ["I love AI"]
sample_seq = tokenizer.texts_to_sequences(sample_text)
sample_pad = pad_sequences(sample_seq, maxlen=max_len, padding='post')
prediction = model.predict(sample_pad)
print(f"Prediction (probability): {prediction[0][0]:.4f}")
print("Class:", "Positive" if prediction[0][0] > 0.5 else "Negative")
