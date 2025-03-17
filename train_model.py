import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from utils.preprocessing import clean_text
import pickle
import numpy as np

# ======================
# 1. Data Preparation
# ======================

# Load dataset with error handling
try:
    # Read compressed CSV (automatically decompresses .zip)
    df = pd.read_csv("C:\\Users\\KIIT\\Desktop\\dataimdb_reviews.csv.zip", compression='zip')
    
    # Convert sentiment labels to numerical values
    df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})
    
except Exception as e:
    print(f"Error loading data: {str(e)}")
    exit()

# ======================
# 2. Text Preprocessing
# ======================

# Clean text data
df['cleaned_review'] = df['review'].apply(clean_text)

# ======================
# 3. Tokenization
# ======================

# Initialize and fit tokenizer
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(df['cleaned_review'])

# Convert text to sequences
sequences = tokenizer.texts_to_sequences(df['cleaned_review'])
padded_sequences = pad_sequences(sequences, maxlen=200, padding='post', truncating='post')

# ======================
# 4. Data Splitting
# ======================

# Convert labels to numpy array with proper dtype
labels = np.array(df['sentiment'], dtype=np.int32)

X_train, X_test, y_train, y_test = train_test_split(
    padded_sequences, 
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels  # Maintain class balance
)

# ======================
# 5. Model Architecture
# ======================

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(
        input_dim=10000, 
        output_dim=128,
        input_length=200
    ),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# ======================
# 6. Model Compilation
# ======================

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy', 
             tf.keras.metrics.Precision(),
             tf.keras.metrics.Recall()]
)

# ======================
# 7. Model Training
# ======================

# Add early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    epochs=10,  # Increased epochs for better convergence
    validation_split=0.2,
    batch_size=64,
    callbacks=[early_stopping],
    verbose=2
)

# ======================
# 8. Model Evaluation
# ======================

print("\nModel Evaluation:")
test_loss, test_acc, test_precision, test_recall = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall: {test_recall:.4f}")

# ======================
# 9. Save Artifacts
# ======================

# Create model directory if not exists
os.makedirs('model', exist_ok=True)

# Save model and tokenizer
model.save('model/model.h5')
with open('model/tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

# Save training history for analysis
pd.DataFrame(history.history).to_csv('model/training_history.csv', index=False)

print("\nTraining completed and model artifacts saved!")