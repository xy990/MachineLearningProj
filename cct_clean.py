import os
from glob import glob
from zipfile import ZipFile
from collections import Counter
import pickle
import re
import shutil

def paragraph_list(text):
    #eliminate page number in format of [*123]
    text = re.sub(r"\[\*\d+\]", "", text)
    text = re.sub(r'\[(\w)\]', r'\1', text)
    p = []
    para = text.split('\n')
    for i in range(len(para)):
        para[i].strip()
        if para[i]!='' and para[i]!=' ' and para[i]!='  ':
            #group incorrect breaklined back to the form of paragraphs
            if para[i][-1]!='.' and i+1<len(para):
                para[i+1] = para[i] +' ' +para[i+1]
            else:
                p.append(para[i])
    return p

os.chdir('/Users/muriel820/Downloads/ml-project/cleaned')
zipfiles = glob('*zip')

for zfname in zipfiles:        
    print(zfname)
    zfile = ZipFile(zfname)    
    year = zfname.split('/')[-1][:-4]
 
    members = zfile.namelist()        
    threshold = len(members) / 200    
    docfreqs = Counter()

    zip_name = '/Users/muriel820/Downloads/ml-project/cleaned/' + zfname[0:4]+'/'
    #maj ='maj'
    os.makedirs(zip_name)
    #os.chdir(zip_name)
    os.makedirs(zip_name+'maj/')
    os.makedirs(zip_name+'con/')
    os.makedirs(zip_name+'dis/')
    os.makedirs(zip_name+'condis/')
    
    for fname in members:
        # "maj" means this is the majority opinion
        if fname.endswith('-maj.txt'):
            docid = fname.split('/')[-1][:-4]                     
            text = zfile.open(fname).read().decode()
            p = paragraph_list(text)
            name = docid + '.p'        
            pickle.dump(p, open(zip_name+'maj/'+name, "wb"))
        
        elif fname.endswith('pre-header.txt'):
            continue 
            
        else:
            optype = fname.split('-')[-1][:-4]
            docid = fname.split('/')[-1][:-4]


            if len(optype) ==7:        
                text = zfile.open(fname).read().decode() 
                p = paragraph_list(text)
                name =docid +'.p'
                pickle.dump(p, open(zip_name+'condis/' + name, "wb"))


            elif len(optype) == 4 and optype[0] =='c':
                text = zfile.open(fname).read().decode() 
                p = paragraph_list(text)
                name = docid +'.p'
                pickle.dump(p, open(zip_name+'con/' + name, "wb"))

            else:
                text = zfile.open(fname).read().decode() 
                p = paragraph_list(text)
                name =docid +'.p'
                pickle.dump(p, open(zip_name+'dis/' + name, "wb"))
        
    shutil.make_archive(zip_name, 'zip', zip_name)
    shutil.rmtree(zip_name,ignore_errors=True, onerror=None)
        # featurize

