import os,warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore", message=".*oneDNN custom operations.*")
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from transformers import AutoTokenizer

#Ingeniera de caracteristicas

def get_max_len(X):
    max_len = 0
    for x in X['news_headline']:
        if len(str(x).split()) > max_len:
            max_len = len(str(x).split())
    return max_len


class Features:
    def __init__(self, data, embedding_matrix, tokenizer="keras"):
        data_stopwords = data
        X = data_stopwords[['news_headline']]
        y = data_stopwords[['reliable']]

        # Dividimos en train y test (90% / 10%)
        X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=0.1, shuffle=True, random_state=42)

        # Dividimos train en train y validation (80% / 20%)
        X_trn, X_val, y_trn, y_val = train_test_split(X_trn, y_trn, test_size=0.2, shuffle=True, random_state=42)

        # Guardamos referencias
        self.y_trn = y_trn.to_numpy()
        self.y_val = y_val.to_numpy()
        self.y_tst = y_tst.to_numpy()

        # Guardamos la matriz de embeddings
        self.matriz_embedding = embedding_matrix.to_numpy()

        self.max_len = 500
        print("Max Len: ", self.max_len)

        # Tokenización y padding
        if tokenizer=="keras":
            tokenizer = Tokenizer(num_words=self.max_len, oov_token='<OOV>')
            tokenizer.fit_on_texts(X['news_headline'].tolist())

            # Entrenamiento
            X_trn_sequences = tokenizer.texts_to_sequences(X_trn['news_headline'].tolist())
            self.X_trn_padded = pad_sequences(X_trn_sequences, maxlen=self.max_len, padding='post', truncating='post')


            # Validación
            X_val_sequences = tokenizer.texts_to_sequences(X_val['news_headline'].tolist())
            self.X_val_padded = pad_sequences(X_val_sequences, maxlen=self.max_len, padding='post', truncating='post')

            # Test
            X_tst_sequences = tokenizer.texts_to_sequences(X_tst['news_headline'].tolist())
            self.X_tst_padded = pad_sequences(X_tst_sequences, maxlen=self.max_len, padding='post', truncating='post')
        if tokenizer=="transformer":
            tokenizer = AutoTokenizer.from_pretrained("tukx/fake-news-classificator")

            # Entrenamiento
            print("1.")
            enc_trn = tokenizer(
                X_trn['news_headline'].tolist(),
                padding='max_length',
                truncation=True,
                max_length=self.max_len,
                return_tensors='np'  # también puede ser 'pt' o 'tf'
            )
            self.X_trn_padded = enc_trn['input_ids']
            print("2.")
            # Validación
            enc_val = tokenizer(
                X_val['news_headline'].tolist(),
                padding='max_length',
                truncation=True,
                max_length=self.max_len,
                return_tensors='np'
            )
            self.X_val_padded = enc_val['input_ids']
            print("3.")
            # Test
            enc_tst = tokenizer(
                X_tst['news_headline'].tolist(),
                padding='max_length',
                truncation=True,
                max_length=self.max_len,
                return_tensors='np'
            )
            self.X_tst_padded = enc_tst['input_ids']
            print("Finalizó Features Interno.")
            print(self.X_trn_padded)


