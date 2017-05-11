
# coding: utf-8

# In[175]:

import gensim.models as g

LabeledSentence = gensim.models.doc2vec.LabeledSentence
import pandas as pd
import numpy as np
import random
from sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle
from sklearn.svm import LinearSVC
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier

data = pd.read_csv("new_annotated.csv", encoding="latin-1",sep = '\t')
data = data[['cmp_code', 'text']]
data.loc[data['cmp_code'] =='H'] = '999'
data = shuffle(data)


# In[176]:

X = data['text']
Y = data['cmp_code']


# In[177]:

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size = .75)


# In[178]:

def cleanText(corpus):
    punctuation = """.,?!:;(){}[]"""
    corpus = [z.lower().replace('\n','') for z in corpus]
    corpus = [z.replace('<br />', ' ') for z in corpus]

    #treat punctuation as individual words
    for c in punctuation:
        corpus = [z.replace(c, ' %s '%c) for z in corpus]
    corpus = [z.split() for z in corpus]
    return corpus


# In[179]:

X_train = cleanText(X_train)
X_test = cleanText(X_test)


# In[180]:

def labelizeReviews(reviews, label_type):
    labelized =[]
    for i in range(len(reviews)):
        

        result = reviews[i]
        label = '%s_%s'%(label_type,i)
        labelized.append(LabeledSentence(result,[label]))
    return labelized

X_train = labelizeReviews(X_train, 'TRAIN')
X_test = labelizeReviews(X_test, 'TEST')


# In[181]:



size = 400

#instantiate our DM and DBOW models
model_dm = gensim.models.Doc2Vec(min_count=1, window=10, size=size, sample=1e-3, negative=5, workers=3)
model_dbow = gensim.models.Doc2Vec(min_count=1, window=10, size=size, sample=1e-3, negative=5, dm=0, workers=3)


# In[182]:

#build vocab over all reviews
#model_dm.build_vocab(X_train)
#model_dm.build_vocab(np.concatenate((X_train, X_test)))
model_dm.build_vocab(X_train + X_test)
model_dbow.build_vocab(X_train + X_test)


# In[186]:

epochs = 10
for i in range(epochs):
    model_dm.train(X_train, total_examples=model_dm.corpus_count)
    model_dbow.train(X_train,total_examples=model_dm.corpus_count)
model_dm.save('dm.doc2vec')
model_dbow.save('dbow.doc2vec')


# In[187]:

model_dbow =gensim.models.Doc2Vec.load('dbow.doc2vec')
model_dm = gensim.models.Doc2Vec.load('dm.doc2vec')


# In[198]:

train_arrays = np.zeros((len(X_train), 400))
train_labels = np.zeros(len(X_train))
test_arrays = np.zeros((len(X_test), 400))
test_labels = np.zeros(len(X_test))


# In[199]:

for i in range(train_arrays.shape[0]):
    train_arrays[i] = model_dm.docvecs[i]
    train_labels[i] = list(Y_train)[i]
    


# In[200]:

for i in range(test_arrays.shape[0]):
    test_arrays[i] = model_dm.docvecs[i]
    test_labels[i] = list(Y_test)[i]


# In[201]:


classifier = OneVsRestClassifier(LogisticRegression())
classifier.fit(train_arrays,train_labels )
predicted = classifier.predict(test_arrays)


# In[202]:

score = classifier.score(test_arrays, test_labels)
np.save('lrscore.npy',score)


# In[204]:


classifier1 = OneVsRestClassifier(RandomForestClassifier())
classifier1.fit(train_arrays,train_labels )
predicted = classifier1.predict(test_arrays)


# In[205]:

score1 = classifier1.score(test_arrays, test_labels)
np.save('lrscore.npy',score1)

