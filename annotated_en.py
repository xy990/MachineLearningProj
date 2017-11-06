import re
import pandas as pd
import enchant
from collections import Counter
import os
from glob import glob

d = enchant.Dict("en_US")
df = pd.DataFrame()
directory = '/home/ym1495/annotated/'
os.chdir(directory)
csvfiles = glob('*csv')
for csvname in csvfiles:
    #fill nan with 0 as in 'unknown'
    annotated = pd.read_csv(directory+csvname).fillna(value = 0)
    #rename column names that don't match
    annotated = annotated.rename(index = str,columns={"content":"text"})
    #not all data has 'eu_code' column, drop them
    if annotated.shape[1]>2:
        annotated = annotated.drop('eu_code',1)
    for i in range(len(annotated)):
        row = annotated.iloc[i]
        words = row['text']
        if type(words)!=str:
            continue
        else:
            words = re.sub(r'[\W\d+]',' ',words)
            words = words.split()
            if len(words)>0:
                en = Counter(d.check(w)==True for w in words)[True]
                frac = float(en)/float(len(words))
                if frac>=0.8:
                    print(frac)
                    print(words)
                    df = df.append(row,ignore_index=True)
                else:
                    continue
df.to_csv('annotated.csv', sep = '\t')
