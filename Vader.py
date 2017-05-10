
# coding: utf-8

# In[1]:

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import tokenize
import numpy as np
from collections import Counter
import os
from glob import glob
from zipfile import ZipFile
import pickle
import re
import shutil
def case_sentmetrics(doc):
    sid = SentimentIntensityAnalyzer()
    case_res =[]
    for para in doc:
        para_list = tokenize.sent_tokenize(para)
        para_res= Counter()
        count =0
        for sent in para_list:
            #print(sent)
            ss = Counter(sid.polarity_scores(sent))
            para_res += ss
            count +=1 
        try:
            s = para_res['compound']/count
        except:
            pass

        
        case_res.append(s)
    return case_res


# In[2]:

def case_sentmetrics1(doc):
    sid = SentimentIntensityAnalyzer()
    case_res =[]
    for para in doc:
        ss = sid.polarity_scores(para)
        s =ss['compound']
        case_res.append(s)
    return case_res


# In[4]:

os.chdir('/home/xy990/cleaned_Mar_28')
zipfiles = glob('*zip')

for zfname in zipfiles:        
    print(zfname)
    zfile = ZipFile(zfname)    
    year = zfname.split('/')[-1][:-4]
 
    members = zfile.namelist()        
    

    zip_name = '/home/xy990/ParaSentimentMetrics/' + zfname[0:4]+'/'
    #zip_name2 = 'C:/Users/sherryyang/Desktop/1003pro/NoParaSentimentMetrics/' + zfname[0:4]+'/'
    #maj ='maj'
    os.makedirs(zip_name)
    #os.chdir(zip_name)
    os.makedirs(zip_name+'maj/')
    os.makedirs(zip_name+'cons/')
    os.makedirs(zip_name+'dis/')
    os.makedirs(zip_name+'condis/')
    
#     os.makedirs(zip_name2)
#     #os.chdir(zip_name)
#     os.makedirs(zip_name2+'maj/')
#     os.makedirs(zip_name2+'cons/')
#     os.makedirs(zip_name2+'dis/')
#     os.makedirs(zip_name2+'condis/')
    
    for fname in members:
        # "maj" means this is the majority opinion
        if fname.endswith('-maj.p'):
            docid = fname.split('/')[-1][:-4]                     
            text = pickle.load(zfile.open(fname,'r'))
            #p1 = case_sentmetrics1(text)
            p = case_sentmetrics(text)
            name = docid + '.p'        
            pickle.dump(p, open(zip_name+'maj/'+name, "wb"))
            #pickle.dump(p1, open(zip_name2+'maj/'+name, "wb"))
        
        elif fname.endswith('.p')==False:
            continue 
        elif fname.endswith('dis/.p')==True:
            continue
            
        else:
            optype = fname.split('-')[-1][:-2]
            docid = fname.split('/')[-1][:-2]
            text = pickle.load(zfile.open(fname,'r'))
            if len(optype) ==7:        
                #para_count1 = case_sentmetrics1(text)
                para_count = case_sentmetrics(text)
                name = docid + '.p'        
                #pickle.dump(para_count1, open(zip_name2+'condis/'+name, "wb"))
                pickle.dump(para_count, open(zip_name+'condis/'+name, "wb"))
                
                

            elif len(optype) == 4 and optype[0] =='c':
                #para_count1 = case_sentmetrics1(text)
                para_count = case_sentmetrics(text)
                name = docid + '.p'        
                pickle.dump(para_count, open(zip_name+'cons/'+name, "wb"))
                #pickle.dump(para_count1, open(zip_name2+'cons/'+name, "wb"))
                

            else:
                #para_count1 = case_sentmetrics1(text)
                para_count = case_sentmetrics(text)
                name = docid + '.p'        
                pickle.dump(para_count, open(zip_name+'dis/'+name, "wb"))
                #pickle.dump(para_count1, open(zip_name2+'dis/'+name, "wb"))
               
        
    shutil.make_archive(zip_name, 'zip', zip_name)
    shutil.rmtree(zip_name,ignore_errors=True, onerror=None)
#     shutil.make_archive(zip_name2, 'zip', zip_name2)
#     shutil.rmtree(zip_name2,ignore_errors=True, onerror=None)

