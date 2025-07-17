import pennylane as qml
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from datasets import load_dataset
from sklearn.model_selection import train_test_split

# Setup for debugging custom layers
tf.config.run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()

# Load SNLI dataset (subset)
snli = load_dataset("snli")
valid_labels = {0: "entailment", 1: "contradiction", 2: "neutral"}

data = [
    (p, h, l)
    for p, h, l in zip(snli["train"]["premise"], snli["train"]["hypothesis"], snli["train"]["label"])
    if l in valid_labels and p and h
][:2000]  # Use more samples to improve training

premises, hypotheses, labels = zip(*data)

# Tokenization
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=2000, oov_token="<OOV>")
tokenizer.fit_on_texts(premises + hypotheses)

# Sequence preparation
max_len = 16
premise_seq = tokenizer.texts_to_sequences(premises)
hypothesis_seq = tokenizer.texts_to_sequences(hypotheses)

premise_pad = tf.keras.preprocessing.sequence.pad_sequences(premise_seq, maxlen=max_len, padding='post')
hypothesis_pad = tf.keras.preprocessing.sequence.pad_sequences(hypothesis_seq, maxlen=max_len, padding='post')

X = np.concatenate([premise_pad, hypothesis_pad], axis=1)  # shape: (samples, 32)
y = np.array(labels)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Quantum circuit setup
n_qubits = 2
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="tf")
def quantum_circuit(inputs, weights):
    qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.templates.BasicEntanglerLayers(weights, wires=range(n_qubits))
    return qml.math.stack([qml.expval(qml.PauliZ(i)) for i in range(n_qubits)])

# Custom quantum layer
class QuantumLayer(tf.keras.layers.Layer):
    def _init_(self, n_qubits):
        super()._init_()
        self.n_qubits = n_qubits
        self.weight_shape = (2, n_qubits)
        self.weights_q = self.add_weight(
            name="weights_q",
            shape=self.weight_shape,
            initializer="random_normal",
            trainable=True
        )
        self.dense = tf.keras.layers.Dense(n_qubits)

    def call(self, inputs):
        inputs = self.dense(inputs)
        inputs = tf.math.l2_normalize(inputs, axis=-1)

        batch_size = tf.shape(inputs)[0]
        weights_batch = tf.repeat(tf.expand_dims(self.weights_q, 0), repeats=batch_size, axis=0)

        def circuit_with_weights(x):
            return quantum_circuit(x[0], x[1])

        return tf.map_fn(
            circuit_with_weights,
            (inputs, weights_batch),
            fn_output_signature=tf.TensorSpec(shape=(self.n_qubits,), dtype=tf.float32)
        )

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.n_qubits)

# Hybrid Quantum-Classical Model
input_layer = tf.keras.Input(shape=(32,))
x = layers.Embedding(input_dim=2000, output_dim=8)(input_layer)
x = layers.Bidirectional(tf.keras.layers.LSTM(8))(x)
x = layers.Dense(4, activation="relu")(x)
x = QuantumLayer(n_qubits=2)(x)
output_layer = layers.Dense(3, activation="softmax")(x)

model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

# Early stopping
early_stop = tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)

# Training
model.fit(X_train, y_train, epochs=10, batch_size=8, validation_data=(X_test, y_test), callbacks=[early_stop])

# Prediction function
def predict_quantum(premise, hypothesis):
    p_seq = tokenizer.texts_to_sequences([premise])
    h_seq = tokenizer.texts_to_sequences([hypothesis])
    p_pad = tf.keras.preprocessing.sequence.pad_sequences(p_seq, maxlen=max_len, padding='post')
    h_pad = tf.keras.preprocessing.sequence.pad_sequences(h_seq, maxlen=max_len, padding='post')
    x_input = np.concatenate([p_pad, h_pad], axis=1)

    pred = model.predict(x_input)
    label_map = {0: "Entailment", 1: "Contradiction", 2: "Neutral"}
    label = np.argmax(pred, axis=1)[0]
    confidence = np.max(pred)
    print(f"\nPremise: {premise}")
    print(f"Hypothesis: {hypothesis}")
    print(f"Prediction: {label_map[label]} ({confidence * 100:.2f}%)")

# Test it
user_premise = input("Enter a premise: ")
user_hypothesis = input("Enter a hypothesis: ")
predict_quantum(user_premise, user_hypothesis)
