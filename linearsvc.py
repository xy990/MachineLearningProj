import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle
from sklearn.svm import LinearSVC
from sklearn.grid_search import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

data = pd.read_csv("new_annotated.csv", encoding="latin-1",sep = '\t')
data = data[['cmp_code', 'text']]
data = shuffle(data)
X = data['text']
Y = data['cmp_code']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size = .75)





tfidf_vectorizer = TfidfVectorizer( stop_words='english', use_idf=True, ngram_range=(1,3))
tfidf_vectorizer.fit(X_train)
X_train_tfidf = tfidf_vectorizer.transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

np.save('svctarget.npy',Y_test)

lin_clf = LinearSVC()
param_grid = {'C':[0.1,1, 10], 'class_weight':[None,'balanced'] }
CV_clf = GridSearchCV(estimator=lin_clf, param_grid=param_grid)
CV_clf.fit(X_train_tfidf, Y_train)

print(CV_clf.best_params_)
#{'C': 1, 'class_weight': 'balanced'}


lin_clf2 = LinearSVC(C =1, class_weight='balanced')

d2 =lin_clf2.fit(X_train_tfidf, Y_train).predict(X_test_tfidf)

np.save('svcpredict2.npy',d2)

