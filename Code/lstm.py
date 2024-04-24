import pandas as pd
from keras.src.layers import Dropout
from keras.src.utils import to_categorical
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import LSTM, Embedding, Dense, Bidirectional
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences

# Read the dataset
df = pd.read_csv("ChChSe-Decagon_polypharmacy/filteredData.csv")

# Extract features and target variable
smiles = df['C1 SMILES'] + ' ' + df['C2 SMILES']
side_effects = df["Side Effect Name"]

# Tokenize concatenated SMILES strings
tokenizer = Tokenizer()
tokenizer.fit_on_texts(smiles)
X_seq = tokenizer.texts_to_sequences(smiles)

# Pad sequences to ensure equal length
max_length = max(map(len, X_seq))
X_padded = pad_sequences(X_seq, maxlen=max_length, padding='post')

# Encode side effects
label_encoder = LabelEncoder()
side_effects_encoded = label_encoder.fit_transform(side_effects)

# One-hot encode the target variable
y_encoded = to_categorical(side_effects_encoded, num_classes=len(label_encoder.classes_))

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_padded, y_encoded, test_size=0.2, random_state=42)

# Build LSTM model
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 300
num_classes = len(label_encoder.classes_)

# Create a Bidirectional LSTM
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim))
model.add(Bidirectional(LSTM(units=128, return_sequences=True)))
model.add(Dropout(0.1))
model.add(Bidirectional(LSTM(units=64)))
model.add(Dropout(0.1))
model.add(Dense(num_classes, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=128, validation_split=0.1)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Loss: {loss}, Accuracy: {accuracy}")

y_pred = model.predict(X_test).argmax(axis=1)
precision = precision_score(y_test.argmax(axis=1), y_pred, average='macro')
recall = recall_score(y_test.argmax(axis=1), y_pred, average='macro')
f1 = f1_score(y_test.argmax(axis=1), y_pred, average='macro')
conf_matrix = confusion_matrix(y_test.argmax(axis=1), y_pred)

print(f"Precision: {precision}, Recall: {recall}, F1-score: {f1}")
print("Confusion Matrix:")
print(conf_matrix)
