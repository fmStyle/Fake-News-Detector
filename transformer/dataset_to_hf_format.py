import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
import json


data_stopwords = pd.read_csv('TRABAJO CREATIVO\SherLockFakenewsProcessedWithStopWords.csv')

X = data_stopwords[['news_headline']]
y = data_stopwords[['reliable']]
X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=0.2, shuffle=True)

X_trn = X_trn['news_headline'].tolist()
X_tst = X_tst['news_headline'].tolist()

y_trn = y_trn['reliable'].tolist()
y_tst = y_tst['reliable'].tolist()



dataset = {"train": [], "test": []}
for i in range(0, len(X_trn)):
    dataset["train"].append({"text": X_trn[i], "label": int(y_trn[i])})
for i in range(0, len(X_tst)):
    dataset["test"].append({"text": X_tst[i], "label": int(y_tst[i])})

def gen_trn():
    for i in range(0, len(X_trn)):
        yield {"text": X_trn[i], "label": int(y_trn[i])}

def gen_tst():
    for i in range(0, len(X_tst)):
        yield {"text": X_tst[i], "label": int(y_tst[i])}

ds_train = Dataset.from_generator(gen_trn)
print(ds_train['label'])
ds_test = Dataset.from_generator(gen_tst)


dataset_dict = DatasetDict({"train": ds_train, "test": ds_test})

dataset_dict.save_to_disk(r"TRABAJO CREATIVO\transformer\dataset.hf")  # Save dataset
