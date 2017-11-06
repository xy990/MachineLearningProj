"""Filter Out Non-English Annotations"""

import os
import pandas as pd
from glob import glob
import zipfile
from langdetect import detect

with zipfile.ZipFile('/home/llq205/annotated.zip', 'r') as f:
    directories = [item for item in f.namelist() if item.endswith('.csv')]

df = pd.DataFrame(columns=("text", "cmp_code", "eu_code"))

zf = zipfile.ZipFile('/home/llq205/annotated.zip') 

for i in range(len(directories)):        
    data = pd.read_csv(zf.open(directories[i]))
    data.columns = ["text", "cmp_code", "eu_code"]
    data = data.dropna(subset = ['text'], how = 'all')
    data = data.reset_index(drop = True)
    print("i",i, directories[i])
    count = 0
    for j in range(len(data)):
        try:
            if detect(data.ix[:,0][j]) =='en':
#             print("Detecting position:", j)
                count += 1
#             df = df.append(data.ix[j,])
        except LangDetectException:
            count += 1
            pass
    if count == len(data):
        df = df.append(data)
        print("Appended:", i)
df = df.reset_index()
df.to_csv("/home/llq205/final_eng.csv")
