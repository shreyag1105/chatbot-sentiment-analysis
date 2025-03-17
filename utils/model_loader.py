import tensorflow as tf
import pickle

def load_model(model_path):
    return tf.keras.models.load_model(model_path)

def load_tokenizer(tokenizer_path):
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    return tokenizer