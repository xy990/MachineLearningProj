import pickle
import re
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import os
from glob import glob
from zipfile import ZipFile
from collections import Counter
import shutil

vocab = pickle.load(open('/Users/muriel820/Downloads/ml-project/finalfeature.p','rb'))
vocab = set(vocab)
for w in vocab:
    vocab.remove(w)
    vocab.add(re.sub(r'_',' ',w))
n = len(vocab)
count_vectorizer=CountVectorizer(binary=False,vocabulary = vocab,ngram_range=(1,4))

os.chdir('/Users/muriel820/Downloads/ml-project/cleaned_1880')
zipfiles = glob('*zip')

for zfname in zipfiles:        
    print(zfname)
    zfile = ZipFile(zfname)    
    year = zfname.split('/')[-1][:-4]
 
    members = zfile.namelist()        
    threshold = len(members) / 200    
    docfreqs = Counter()

    zip_name = '/Users/muriel820/Downloads/ml-project/cleaned_1880/frequency_count_by_paragraph/' + zfname[0:4]+'/'
    zip_name_2 = '/Users/muriel820/Downloads/ml-project/cleaned_1880/frequency_count_by_doc/' + zfname[0:4]+'/'
    #maj ='maj'
    os.makedirs(zip_name)
    #os.chdir(zip_name)
    os.makedirs(zip_name+'maj/')
    os.makedirs(zip_name+'con/')
    os.makedirs(zip_name+'dis/')
    os.makedirs(zip_name+'condis/')
    os.makedirs(zip_name_2)
    os.makedirs(zip_name_2+'maj/')
    os.makedirs(zip_name_2+'con/')
    os.makedirs(zip_name_2+'dis/')
    os.makedirs(zip_name_2+'condis/')
    all_ = []
    
    for fname in members:
        if fname.endswith('-maj.p'):
            docid = fname.split('/')[-1][:-2]                     
            text = pickle.load(zfile.open(fname,'r'))
            para_count = count_vectorizer.transform(text)
            name = docid + '.p'        
            pickle.dump(para_count, open(zip_name+'maj/'+name, "wb"))
            doc_count = scipy.sparse.csr_matrix((1, n))
            for i in range(para_count.shape[0]):
                doc_count = doc_count + para_count[i]
            pickle.dump(doc_count, open(zip_name_2+'maj/'+name, "wb"))
            all_.append((docid.split('-')[0],doc_count))
        
        elif fname.endswith('.p')==False:
            continue 
        elif fname.endswith('dis/.p')==True:
            continue
        else:
            optype = fname.split('-')[-1][:-2]
            docid = fname.split('/')[-1][:-2]
            text = pickle.load(zfile.open(fname,'r'))

            if len(optype) ==7:        
                para_count = count_vectorizer.transform(text)
                name = docid + '.p'        
                pickle.dump(para_count, open(zip_name+'condis/'+name, "wb"))
                doc_count = scipy.sparse.csr_matrix((1, n))
                for i in range(para_count.shape[0]):
                    doc_count = doc_count + para_count[i]
                pickle.dump(doc_count, open(zip_name_2+'condis/'+name, "wb"))
                all_.append((docid.split('-')[0],doc_count))

            elif len(optype) == 4 and optype[0] =='c':
                para_count = count_vectorizer.transform(text)
                name = docid + '.p'        
                pickle.dump(para_count, open(zip_name+'con/'+name, "wb"))
                doc_count = scipy.sparse.csr_matrix((1, n))
                for i in range(para_count.shape[0]):
                    doc_count = doc_count + para_count[i]
                pickle.dump(doc_count, open(zip_name_2+'con/'+name, "wb"))
                all_.append((docid.split('-')[0],doc_count))

            else:
                para_count = count_vectorizer.transform(text)
                name = docid + '.p'        
                pickle.dump(para_count, open(zip_name+'dis/'+name, "wb"))
                doc_count = scipy.sparse.csr_matrix((1, n))
                for i in range(para_count.shape[0]):
                    doc_count = doc_count + para_count[i]
                pickle.dump(doc_count, open(zip_name_2+'dis/'+name, "wb"))
                all_.append((docid.split('-')[0],doc_count))
    all_ = dict(all_)
    pickle.dump(all_,open('all.p', "wb"))
    shutil.make_archive(zip_name, 'zip', zip_name)
    shutil.rmtree(zip_name,ignore_errors=True, onerror=None)
    shutil.make_archive(zip_name_2, 'zip', zip_name_2)
    shutil.rmtree(zip_name_2,ignore_errors=True, onerror=None)
