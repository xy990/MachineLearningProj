
# coding: utf-8

# In[ ]:

import os
from glob import glob
from zipfile import ZipFile
from collections import Counter
import pickle
import re
from numpy import prod
from collections import Counter
from nltk import sent_tokenize,ngrams,PorterStemmer,SnowballStemmer,WordNetLemmatizer
from nltk.tag import perceptron
tagger = perceptron.PerceptronTagger()
porter = PorterStemmer()
snowball = SnowballStemmer('english')
lemmatizer = WordNetLemmatizer()

def word_normalize(word,stemmer=None):
    w = word.lower()
    if stemmer == 'porter':
        w = porter.stem(w)
    elif stemmer == 'snowball':
        w = snowball.stem(w)
    elif stemmer == 'lemma':
        w = lemmatizer.lemmatize(w)
    return w
    
# Normalize Penn tags
tagdict = {'NN':'N',
            'NNS':'N',
            'NNP':'N',
            'NNPS':'N',
            'JJ':'A',
            'JJR':'A',
            'JJS':'A',
            'VBG':'A',
            'RB':'A', # adverbs treated as adjectives
            'DT':'D',
            'IN':'P',
            'TO':'P',
            'VB':'V',
            'VBD':'V',
            'VBN':'V',
            'VBP':'V',
            'VBZ':'V',
            
            'MD': 'V', # modals treated as verbs
            'RP': 'V', # particles treated as verbs
            'CC': 'C'}

# Allowed sequences of tag patterns (from Ash 2016)
tagpatterns = {'A','N','J',
           'AN','NN', 'VN', 'VV', 'NV',
            'VP',                                    
            'NNN','AAN','ANN','NAN','NPN',
            'VAN','VNN', 'AVN', 'VVN',
            'VPN','ANV','NVV','VDN', 'VVV', 'NNV',
            'VVP','VAV','VVN',
            'NCN','VCV', 'ACA',  
            'PAN',
            'NCVN','ANNN','NNNN','NPNN', 'AANN' 'ANNN','ANPN','NNPN','NPAN', 
            'ACAN', 'NCNN', 'NNCN', 'ANCN', 'NCAN',
            'PDAN', 'PNPN',
            'VDNN', 'VDAN','VVDN'}

def tagsentence(sent,stemmer=None,vocab=None):
    # convert to one-letter tags if applicable, 
    # replace with none if word not in vocab
    # replace with none if tag not in tagdict
    tagwords = []
    for x in tagger.tag(sent):
        if (vocab is None or x[0] in vocab) and x[1] in tagdict:
            normword = word_normalize(x[0],stemmer=stemmer)
            normtag = tagdict[x[1]]
            tagwords.append((normword,normtag))
        else:
            tagwords.append(None)
    return tagwords

def gmean(phrase, termfreqs):
    """geometric mean association."""
    n = len(phrase)
    p = [termfreqs[w] for w in phrase.split('_')]
    pg = termfreqs[phrase]    
    return pg / (prod(p) ** (1/n))

def train_phraser(documents, max_phrase_length=4, stemmer=None, vocab=None,
                            min_doc_freq=None, min_gmean=None):    
    # take documents and get POS-gram dictionary
    
    numdocs = len(documents)
    if min_doc_freq is None:
        min_doc_freq = round(numdocs / 200) + 1

    docfreqs = Counter()
    termfreqs = Counter()        
    
    for document in documents:
        docgrams = set()
        # split into sentences
        sentences = sent_tokenize(document)
        for sentence in sentences:     
            # split into words and get POS tags
            words = sentence.split()
            tagwords = tagsentence(words,stemmer,vocab)        
            for n in range(1,max_phrase_length+1):            
                rawgrams = ngrams(tagwords,n)
                for rawgram in rawgrams:
                    # skip grams that have words not in vocab
                    if None in rawgram:
                        continue
                    gramtags = ''.join([x[1][0] for x in rawgram])
                    if gramtags in tagpatterns:
                         # if tag sequence is allowed, add to counter
                        gram = '_'.join([x[0] for x in rawgram])
                        termfreqs[gram] += 1
                        docgrams.add(gram)                                    
        docfreqs.update(docgrams)
        
    # filter vocabulary based on document frequency and make gram ids
    gram2id = {}
    id2gram = {}
       
    i = 0
    for (phrase,v) in docfreqs.most_common():   
        if v < min_doc_freq:
            break      
        if min_gmean is not None:
            # check geometric mean association
            n = v.count('_') + 1
            if len(n) >= 2:
                gscore = gmean(phrase,termfreqs) 
                if gscore[n] < min_gmean[n]:
                    continue
        gram2id[phrase] = i
        id2gram[i] = phrase
        i += 1
    
    return gram2id, id2gram


# In[ ]:

def apply_phraser(words, gram2id, max_phrase_length=4):
    """"apply phraser method to sentence."
         Input should be list of lower-case (stemmed) words"""
    sentlength = len(words)
    skip = 0
    new_s = []
    for i in range(sentlength):
        if skip > 0:
            skip -= 1
            continue
        if words[i] is None:
            continue
        for n in reversed(range(1,max_phrase_length+1)):
            if i+n > sentlength:
                continue
            gram = words[i:i+n]
            if None in gram:
                continue
            gram_word = '_'.join(gram)    
            if gram_word in gram2id:
                new_s.append(gram2id[gram_word])
                skip = n-1
                break
    return new_s


# In[ ]:


# import zipfile as zips

os.chdir('cleaned')
zipfiles = glob('*zip')


for zfname in zipfiles:        
    
    zfile = ZipFile(zfname)    
    year = zfname.split('/')[-1][:-4]
 
    members = zfile.namelist()        
    threshold = len(members) / 200    
    docfreqs = Counter()

    #z = ZipFile(zfname[0:4] + " (1)" + ".zip", "w")
    year_c =Counter()
    for fname in members:
        
        # "maj" means this is the majority opinion
        if not fname.endswith('-maj.txt'):
            continue

        #docid = fname.split('/')[-1][:-4]                     
        text = zfile.open(fname).read().decode() 
        text = re.sub("[\\[].*?[\\]]", "", text)
        text_list = text.split(' ')
        phrase_id = train_phraser(text_list)[0]
        for k in phrase_id:
            phrase_id[k] = 1
        phrase_id =Counter(phrase_id)
        year_c +=phrase_id
    c1 = Counter(x for x in year_c.elements() if year_c[x]>=2)
    name = zfname[0:4]+"feature"+".p"
    pickle.dump(c1, open(name,"wb"))

directory = "C:/Users/sherryyang/Desktop/1003 project/feature"
os.chdir(directory)
fnames = os.listdir(directory)
overal = list()
total_10 = Counter()


for i in range(0,len(fnames)-9):
    for j in range(i, i+10):
        f = pickle.load(open(fnames[j],"rb"))
        for b in f:
            f[b] =1
        total_10 +=f
    overal.append(Counter(x for x in total_10.elements() if total_10[x]>=10))
#print(overal)
final =Counter()
for m in range(len(overal)):
    final += overal[m]
for k in final:
    final[k] =1
pickle.dump(final, open("finalfeature.p", "wb"))
    


# In[ ]:




