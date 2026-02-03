import pandas as pd
import os
import numpy as np
from preprocesamiento import preprocess_data
from matriz_embedding import make_embedding_matrix
from features import Features
from red import CNN
from training import train
from testing import test
from transformers import Trainer, TrainingArguments
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, DataCollatorWithPadding
import evaluate
from datasets import load_dataset

glove_path = 'glove.twitter.27B.100d.txt'
dataset_path = 'SherLockFakenewsNetOriginal.csv'
#Trabajamos con stopwords, ya que obtuvimos mejores resultados con ellas.
dataset_preprocessed_path = 'SherLockFakenewsProcessedWithStopWords.csv'
embedding_matrix_path = 'embedding_matrixKWithStopwords'

tokenizer = "transformer"

#---------------------------------------------------CNN---------------------------------------------------#
cnn_path = 'cnn.pytorch'

# Verificaciones de existencia
if not os.path.exists(dataset_preprocessed_path):
    preprocess_data(dataset_path, dataset_preprocessed_path)

if not os.path.exists(embedding_matrix_path):
    make_embedding_matrix(glove_path, dataset_preprocessed_path, embedding_matrix_path, tokenizer)

# Finalmente cargamos el dataset
dataset = pd.read_csv(dataset_preprocessed_path)
embedding_matrix = pd.read_csv(embedding_matrix_path)
print("Dataset y matriz de embeddings cargadas correctamente.")

print("Comenzando Features")
features = Features(dataset, embedding_matrix, tokenizer)
print("Finalizó Features")
# Si el modelo no existe, lo entrenamos
if not os.path.exists(cnn_path):
    print("No existe el modelo. Se procede a entrenarlo")
    train(features, CNN, cnn_path)

print("Resultados de la CNN:")
test(features, CNN, cnn_path)

#---------------------------------------------------TRANSFORMER---------------------------------------------------#
# Cargando el transformer

model = AutoModelForSequenceClassification.from_pretrained("tukx/fake-news-classificator")
ds = load_dataset("tukx/processed_fake_news")
tokenizer = AutoTokenizer.from_pretrained("tukx/fake-news-classificator")
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True) # El truncar funciona porque está dentro del tokenizer
tokenized_ds = ds.map(preprocess_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    metrics = accuracy.compute(predictions=predictions, references=labels)
    metrics.update(f1.compute(predictions=predictions, references=labels))
    return metrics
training_args = TrainingArguments(
    output_dir=r"TRABAJO CREATIVO\transformer\Modelo",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["test"],
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)
print("Resultados de Transformer: ")
results = trainer.evaluate()
print(results)







