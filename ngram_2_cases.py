import os
from glob import glob
from zipfile import ZipFile
from collections import Counter
import pickle
import re

os.chdir('/home/llq205/clean_feature1')
zipfiles = glob('*zip')


for zfname in zipfiles:
    print(zfname)
    zfile = ZipFile(zfname)    
    year = zfname.split('/')[-1][:-4]
 
    members = zfile.namelist()        
    year_count = Counter()

    for fname in members:
#         print(fname)
        if fname.endswith('-maj.p'):
            each = pickle.load(zfile.open(fname, 'r'))
            if each != [] and each[0] != Counter():
                d = Counter({k:1 for k,v in each[0].items()})
                year_count += d
        
        elif fname.endswith('.p')==False:
            continue 
        elif fname.endswith('dis/.p')==True:
            continue
        else:
            optype = fname.split('-')[-1][:-2]
            docid = fname.split('/')[-1][:-2]

            if len(optype) ==7:        
                each = pickle.load(zfile.open(fname, 'r'))
                if each != [] and each[0] != Counter():
                    d = Counter({k:1 for k,v in each[0].items()})
                    year_count += d

            elif len(optype) == 4 and optype[0] =='c':
                each = pickle.load(zfile.open(fname, 'r'))
                if each != [] and each[0] != Counter():
                    d = Counter({k:1 for k,v in each[0].items()})
                    year_count += d

            else:
                each = pickle.load(zfile.open(fname, 'r'))
                if each != [] and each[0] != Counter():
                    d = Counter({k:1 for k,v in each[0].items()})
                    year_count += d
                
    c1 = Counter(x for x in year_count.elements() if year_count[x] >= 2)
    name = zfname[0:4]+"_dict"+".p"
    pickle.dump(c1, open('/home/llq205/2cases/' + name,"wb"))
#     print(c1)
