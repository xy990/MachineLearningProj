import os
from glob import glob
from zipfile import ZipFile
from collections import Counter
import pickle
import re
import shutil

def cite_count(paragraph):
    #
    #input: paragraph(string), regex(processed regex form of a single vocabulary)
    #output: count(int)
    #
    a = re.findall(r'[0-9]+\sF.[0-9]+d\s[0-9]+', paragraph)
    citecount = Counter(a)
    return citecount


def cite_count_dict(text):
    #
    #input: text@list of strings, vocab_all@list of regex, vocab_id@list of keys
    #output: list_of_dict@list of dictionaries, each dictionary corresponds to one paragraph
    #
    list_of_dict = []
    for i in text:
        d = cite_count(i)
        list_of_dict.append(d)
    return list_of_dict


os.chdir('/home/llq205/cleaned_Mar_28')
zipfiles = glob('*zip')

for zfname in zipfiles:        
    print(zfname)
    zfile = ZipFile(zfname)    
    year = zfname.split('/')[-1][:-4]
 
    members = zfile.namelist()        
    threshold = len(members) / 200    
    docfreqs = Counter()

    zip_name = '/home/llq205/clean_annotate/' + zfname[0:4]+'/'
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
            d = cite_count_dict(text)
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
                d = cite_count_dict(text)
                name = docid + '.p'        
                pickle.dump(d, open(zip_name+'condis/'+name, "wb"))


            elif len(optype) == 4 and optype[0] =='c':
                text = pickle.load(zfile.open(fname,'r'))
                d = cite_count_dict(text)
                name = docid + '.p'        
                pickle.dump(d, open(zip_name+'con/'+name, "wb"))

            else:
                text = pickle.load(zfile.open(fname,'r'))
                d = cite_count_dict(text)
                name = docid + '.p'        
                pickle.dump(d, open(zip_name+'dis/'+name, "wb"))
    
    shutil.make_archive(zip_name, 'zip', zip_name)
    shutil.rmtree(zip_name,ignore_errors=True, onerror=None)
