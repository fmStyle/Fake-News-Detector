import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import re
import unicodedata
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import random

def preprocess_data(dataset_path: str, dataset_save_path: str):
    print("Preprocesando dataset...")
    data = pd.read_csv(dataset_path)

    # Hay un paso que aparentemente es agarrar el 50% del dataset para acelerar el proceso.
    # Dejo la línea comentada para posiblemente trabajarlo en un futuro, pero por simplicidad voy a trabajar con todos los datos
    #data = data.sample(n=108341, random_state=1)
    #data = data.sample(n=100, random_state=1)

    processed_data = []

    # Se reemplazan los datos vacíos o nulos con fillna()
    data[['reliable']] = data[['reliable']].fillna(value=0)
    data[['news_headline']] = data[['news_headline']].fillna('')


    # Si después de la decodificación sigue medio mal el texto, esto devuelve True
    def has_weird_chars(text):
        # permite letras, números, espacios, signos comunes y caracteres acentuados válidos
        pattern = r"^[a-zA-Z0-9\s\.,;:'\"!?¡¿\-\(\)áéíóúÁÉÍÓÚñÑüÜ]*$"
        return not bool(re.match(pattern, text))

    # Esto es para detectar textos incompletos.
    def has_ellipsis(text):
        return '...' in text
    cont = 0
    for i in range(0, data.shape[0]):
        txt = data.iloc[i, 0]

        soup = BeautifulSoup(txt, features="html.parser")
        txt = soup.get_text()

        # La siguiente línea es para eliminar elementos entre corchetes, incluidos los corchetes
        txt = re.sub(r'\[.*?\]', '', txt)

        # Idem, pero con urls
        txt = re.sub(r'http[s]?://\S+', '', txt)

        # Limpiar texto
        txt = re.sub(r'"+', '"', txt)  # Limpiar comillas múltiples
        txt = re.sub(r'\s+', ' ', txt)  # Espacios múltiples

        # Si finalmente tiene caracteres raros o puntos suspensivos, seteamos el texto como vacío.
        if (has_weird_chars(txt)):
            txt = ""
        if (has_ellipsis(txt)):
            txt = ""

        # Si el texto resultante es vacío, no lo contamos.
        if data.iloc[i, 1] == 1:
            cont+=1
        if txt == "":
            continue
        
        # Las primeras 44000 noticias reales no las contamos
        if cont < 54000 and data.iloc[i, 1] == 1:
            continue
        # Le sacamos las stopwords
        stop_words = set(stopwords.words('english'))
        tokens = word_tokenize(txt.lower())
    
        # Remove stopwords
        filtered_tokens = [word for word in tokens if word not in stop_words]

        txt_nostopwords = ""
        for f in filtered_tokens:
            txt_nostopwords = txt_nostopwords + f + " "
        
        processed_data.append({'news_headline': txt, 'reliable': data.iloc[i, 1]})
        if (txt_nostopwords == " " or txt_nostopwords == "" or txt_nostopwords == "  "):
            continue

    random.shuffle(processed_data)

    df_stopwords = pd.DataFrame(processed_data)
        
    df_stopwords.to_csv(dataset_save_path, 
            index=False,
            encoding='utf-8',     # Codificación
            header=True)          # Incluir nombres de columnas

