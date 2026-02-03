import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from transformers import AutoTokenizer

#Divison del dataset
def make_embedding_matrix(glove_path, processed_data_path, embedding_matrix_path, tokenizer="keras"):
    print("Creando matriz de embeddings...")
    data = pd.read_csv(processed_data_path)
    embeddings_index = {}
    with open(glove_path, encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            #print(word)
            vector = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = vector
    #Dividimos el dataset
    embedding_dim = 100
    X = data[['news_headline']]


    #Tokenizacion y Padding
    embedding_matrix = None # Lo defino
    if (tokenizer=="keras"):
        max_len = max([len(str(text).split()) for text in X['news_headline']])
        tokenizer = Tokenizer(num_words=max_len,oov_token='<OOV>')
        tokenizer.fit_on_texts(X['news_headline'].tolist())

        vocab_size = len(tokenizer.word_index) + 1  # +1 por el padding
        embedding_matrix = np.zeros((vocab_size, embedding_dim))

        for word, i in tokenizer.word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
            else:
                # Palabra no encontrada -> vector aleatorio
                embedding_matrix[i] = np.random.normal(size=(embedding_dim,))
    if (tokenizer=="transformer"):
        tokenizer = AutoTokenizer.from_pretrained("tukx/fake-news-classificator")
        vocab = tokenizer.get_vocab()
        vocab_size = len(vocab) # 0 se reserva para el padding
        embedding_matrix = np.zeros((vocab_size, embedding_dim))
        

        # vocab["palabra"] = índice
        for w in vocab:
            embedding_vector = embeddings_index.get(w)
            if embedding_vector is not None:
                embedding_matrix[vocab[w]] = embedding_vector
            else:
                if w == '[PAD]': # Si es el padding queda la fila de 0's
                    continue
                else:
                    embedding_matrix[vocab[w]] = np.random.normal(size=(embedding_dim,))
        #pass

    np.savetxt(embedding_matrix_path, embedding_matrix, delimiter=',')