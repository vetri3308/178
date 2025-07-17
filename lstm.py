!pip install datasets
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import numpy as np
from datasets import load_dataset
import matplotlib.pyplot as plt



# Load SNLI dataset
snli = load_dataset("snli")

# Debug: Print unique labels
print("Unique labels in SNLI dataset:", set(snli['train']['label']))

# Filter valid labels (0: entailment, 1: contradiction, 2: neutral, -1 is invalid)
valid_labels = {0: "entailment", 1: "contradiction", 2: "neutral"}

filtered_data = [
    (p, h, l) for p, h, l in zip(snli['train']['premise'], snli['train']['hypothesis'], snli['train']['label'])
    if l in valid_labels and p is not None and h is not None
]

# Ensure valid data exists
if not filtered_data:
    raise ValueError("❌ No valid data found! Check if dataset is correctly loaded.")

# Unzip data
premises, hypotheses, labels = zip(*filtered_data)

# Tokenize and preprocess data
def preprocess_texts(premises, hypotheses, vocab_size=10000, max_len=50):
    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(premises + hypotheses)

    premise_seq = tokenizer.texts_to_sequences(premises)
    hypothesis_seq = tokenizer.texts_to_sequences(hypotheses)

    premise_padded = pad_sequences(premise_seq, maxlen=max_len, padding='post')
    hypothesis_padded = pad_sequences(hypothesis_seq, maxlen=max_len, padding='post')

    return premise_padded, hypothesis_padded, tokenizer

# Preprocess data
premise_padded, hypothesis_padded, tokenizer = preprocess_texts(list(premises), list(hypotheses))

# Combine premise and hypothesis as input features
X = np.concatenate((premise_padded, hypothesis_padded), axis=1)
y = np.array(labels)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and train LSTM model
def build_lstm_model(vocab_size, embedding_dim=100, max_len=50):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim))
    model.add(LSTM(128, return_sequences=True))
    model.add(LSTM(64))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(3, activation='softmax')) # 3 classes: entailment, contradiction, neutral


    model.build(input_shape=(None,max_len * 2))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

vocab_size = 10000
model = build_lstm_model(vocab_size)
model.summary()

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3, batch_size=32, steps_per_epoch=500)

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"LSTM Model Accuracy: {accuracy * 100:.2f}%")

# 🔹 *User Input Prediction Function*
def predict_snli(model, tokenizer, premise, hypothesis, max_len=50):
    premise_seq = tokenizer.texts_to_sequences([premise])
    hypothesis_seq = tokenizer.texts_to_sequences([hypothesis])

    premise_padded = pad_sequences(premise_seq, maxlen=max_len, padding='post')
    hypothesis_padded = pad_sequences(hypothesis_seq, maxlen=max_len, padding='post')

    X_input = np.concatenate((premise_padded, hypothesis_padded), axis=1)

    prediction = model.predict(X_input)
    predicted_label = np.argmax(prediction, axis=1)[0]

    label_map = {0: "Entailment", 1: "Contradiction", 2: "Neutral"}

    print("\n💡 *Prediction Result* 💡")
    print(f"📝 Premise: {premise}")
    print(f"📝 Hypothesis: {hypothesis}")
    print(f"✅ Prediction: {label_map[predicted_label]} (Confidence: {max(prediction[0]) * 100:.2f}%)")
   🔹 *User Input*
user_premise = input("Enter a premise sentence: ")
user_hypothesis = input("Enter a hypothesis sentence: ")

predict_snli(model, tokenizer, user_premise, user_hypothesis)


# Define EarlyStopping callback
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3) # Stop if val_loss doesn't improve for 3 epochs

# Train and capture history
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stop]
)

# Plot training & validation accuracy values
plt.figure(figsize=(8,5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy over Epochs')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()
import matplotlib.pyplot as plt

def plot_accuracy(history):
    plt.figure(figsize=(8,5))
    plt.plot(history.history['accuracy'], label='Training Accuracy', marker='o')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='o')
    plt.title('Model Accuracy over Epochs')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

# --- Usage ---
# After your model.fit(), pass the 'history' to this function:

# Example:
#history = model.fit(...)
plot_accuracy(history)
