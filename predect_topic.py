import pickle
import re
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import os
from sklearn.utils import shuffle
from sklearn.svm import LinearSVC
from glob import glob
from zipfile import ZipFile
from collections import Counter
import shutil
import scipy
from pandas import Series

total = []
cl_clf = pickle.load(open('/home/ym1495/cl_clf.p','rb'))
os.chdir('/home/ym1495/1990')
zipfiles = glob('*zip')

for zfname in zipfiles:
    print(zfname)
    zfile = ZipFile(zfname)
    year = zfname.split('/')[-1][:-4]

    members = zfile.namelist()
    threshold = len(members) / 200
    c = 0
    for fname in members:
        if fname.endswith('.p')==False:
            continue
        elif fname.endswith('dis/.p')==True:
            continue
        else:
            docid = fname.split('/')[-1][:-2]
            caseid = docid.split('-')[0]
            para_count = pickle.load(zfile.open(fname,'r'))
            if para_count.shape[0]!=0:
                topics = cl_clf.predict(para_count)
                probability = cl_clf.predict_proba(para_count)
                for i in range(len(topics)):
                        total.append({(str(year),caseid,fname.split('/')[-1].split('-')[-1][:-2],str(i),str(topics[i])):np.asarray(probability[i])})
pickle.dump(total,open('/home/ym1495/topic_1990.p','wb'))
