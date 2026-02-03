import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

data_nostopwords = pd.read_csv('TRABAJO CREATIVO\SherLockFakenewsProcessedWithStopWords.csv')

fake_news = 0
real_news = 0

for i in range(0, data_nostopwords.shape[0]):
    if data_nostopwords.iloc[i, 1] == 0:
        fake_news +=1
    else:
        real_news +=1

print('Fake News: ' + str(fake_news))
print('Real News: ' + str(real_news))