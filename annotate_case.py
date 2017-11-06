"""Extract Annotations in Cases"""

import os
from glob import glob
from zipfile import ZipFile
from collections import Counter
import pickle
import re
import shutil

def sum_count(cList):
    #
    #input: list of Counter
    #output: sum of paragraph Counters
    #
    agg = sum(cList, Counter())
    return agg


os.chdir('/home/llq205/clean_annotate')
zipfiles = glob('*zip')

for zfname in zipfiles:        
    print(zfname)
    zfile = ZipFile(zfname)    
    year = zfname.split('/')[-1][:-4]
 
    members = zfile.namelist()        
    threshold = len(members) / 200    
    docfreqs = Counter()

    zip_name = '/home/llq205/clean_annotate_case/' + zfname[0:4]+'/'
    #maj ='maj'
    os.makedirs(zip_name)
    #os.chdir(zip_name)
    os.makedirs(zip_name+'maj/')
    os.makedirs(zip_name+'con/')
    os.makedirs(zip_name+'dis/')
    os.makedirs(zip_name+'condis/')
    
    for fname in members:
        # "maj" means this is the majority opinion
        #print(fname)
        if fname.endswith('-maj.p'):
            docid = fname.split('/')[-1][:-2]                     
            text = pickle.load(zfile.open(fname,'r'))
            d = sum_count(text)
            name = docid + '.p'        
            pickle.dump(d, open(zip_name+'maj/'+name, "wb"))
        
        elif fname.endswith('.p')==False:
            continue 
        elif fname.endswith('dis/.p')==True:
            continue
        else:
            optype = fname.split('-')[-1][:-2]
            docid = fname.split('/')[-1][:-2]


            if len(optype) ==7:        
                text = pickle.load(zfile.open(fname,'r'))
                d = sum_count(text)
                name = docid + '.p'        
                pickle.dump(d, open(zip_name+'condis/'+name, "wb"))


            elif len(optype) == 4 and optype[0] =='c':
                text = pickle.load(zfile.open(fname,'r'))
                d = sum_count(text)
                name = docid + '.p'        
                pickle.dump(d, open(zip_name+'con/'+name, "wb"))

            else:
                text = pickle.load(zfile.open(fname,'r'))
                d = sum_count(text)
                name = docid + '.p'        
                pickle.dump(d, open(zip_name+'dis/'+name, "wb"))
    
    shutil.make_archive(zip_name, 'zip', zip_name)
    shutil.rmtree(zip_name,ignore_errors=True, onerror=None)
