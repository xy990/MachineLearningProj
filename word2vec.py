import gensim
import os
from glob import glob
from zipfile import ZipFile
from collections import Counter
import pickle
import re
os.chdir('cleaned_Mar_28')
zipfiles = glob('*zip')
#all_str =" "
#year_str =" "
all_str = []
for zfname in zipfiles:
    zfile = ZipFile(zfname)
    members = zfile.namelist()
    for fname in members:
        if fname.endswith('/'):
            continue
        elif fname == 'dis/.p':
            continue            
        else:
            sent = pickle.load(zfile.open(fname,'r'))
            for line in sent:
                line = re.sub(r'[^\w]',' ',line)
                
                all_str.append(line.split())
                
            #sent_str = " ".join(sent)
            #year_str += sent_str

pickle.dump(all_str, open( "w2vall.p", "wb" ))
model = gensim.models.Word2Vec(all_str, size =300, window =5, min_count = 10)
model.save("w2v.bin")
