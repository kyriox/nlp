import scipy.sparse as sp
from nltk.stem.porter import PorterStemmer
import nltk
import numpy as np
from collections import Counter
from matplotlib import pyplot as plt
import re
# Utilice el lematizador y stemmer de su preferencia 
# usaremos la variable derivador y podra tomar los valores de "lemma", "stem" o None

stopwords = nltk.corpus.stopwords.words('english')
remove_punct=re.compile(r'[^\w\s]')

class Tokenizer:
    def __init__(self,corpus=[],ngrams=[],case_folding=True, derivador=None, remove_stop_words=False, user_replace=False, link_replace=False, remove_punctuation=True):
        self.ngrams=ngrams
        self.vocabulario={}
        self.vocabulario_index=[]
        stemers={'stem': PorterStemmer()}
        self.derivador=stemers.get(derivador)
        self.case_folding=case_folding
        self.remove_stop_words=remove_stop_words
        self.remove_punctuation=remove_punctuation
        if len(corpus)==0:
            self._construye_vocabulario(corpus)
            
    def _remove_punctuation(self,texts):
        norm_texts=[]
        for doc in texts:
           norm_texts.append(remove_punct.sub(string=doc, repl=""))
        return norm_texts

    def _case_folding(self,texts):
        norm_texts=[]
        for doc in texts:
           norm_texts.append(doc.lower())
        return norm_texts
    
    def _ngrams(self, doc):
        ngramas=[]
        for n in self.ngrams:
            for i in range(0,len(doc)-(n-1)):
                ngramas.append(' '.join(doc[i:i+n]))
        return ngramas    
    
    def _remove_stop_words(self,texts):
        norm_texts=[]
        for doc in texts:
           norm_texts.append([w for w in doc if w not in stopwords])
        return norm_texts
    
    def _stem(self, texts):
        norm_text=[]
        for doc in texts:
            #print(doc)
            docn=[self.derivador.stem(w) for w in doc]
            norm_text.append(docn)
        return norm_text
   
    
    def _construye_vocabulario(self,corpus):
        ## Aplicar normalizaciones al corpus
        ## construir tabla de vocabulario
        self.vocabulario={}
        vocabulario_index=[]
        norm_corpus=self._normalize(corpus)
        for doc in norm_corpus:
            for token in doc:
                if token not in vocabulario_index:
                    vocabulario_index.append(token)
                id_token=vocabulario_index.index(token)
                self.vocabulario[token]=self.vocabulario.get(token,0)+1
        self.vocabulario_index=np.array(vocabulario_index)
                
    def _normalize(self,corpus):
        if self.remove_punctuation:
            corpus=self._remove_punctuation(corpus)
        if self.case_folding:
            corpus=self._case_folding(corpus)
        norm_corpus=[text.split() for text in corpus]
        if self.remove_stop_words:
            norm_corpus=self._remove_stop_words(norm_corpus)
        if self.derivador is not None:
            norm_corpus=self._stem(norm_corpus)
        if  self.ngrams:
            norm_corpus=[doc+self._ngrams(doc) for doc in norm_corpus]
        return norm_corpus
            
    def fit(self, corpus):
           self._construye_vocabulario(corpus) 
    
    def tokenize(self, docs):
        if type(docs)==str:
            return self._normalize([docs])[0]
        return self._normalize(docs)
        
            
    def transform(self, texts):
        norm_texts=self._normalize(texts)
        vectores=[]
        N=len(self.vocabulario_index)
        for doc in norm_texts:
            #C=[self.vocabulario_index.index(token) for token in doc if token in self.vocabulario_index]
            C=np.argwhere(np.isin(self.vocabulario_index, doc)).flatten()
            #print("XXXXXX", doc,np.isin(self.vocabulario_index, doc))
            nt=len(C)
            X=[1 for i in range(nt)]
            R=[0 for i in range(nt)]
            vectores.append(sp.csr_matrix((X,(R,C)), shape=(1,N)))
        return sp.vstack(vectores)
        ## Aplicar normalizaciones al texto
        ## Tokenizar el texto utilizando el vocabulario 
        #(decidir que hacer cuando ocurren palabras que no se encuentres en el vocabulario)
        ## regresar lista de tokens
    
    def _tf_transform(self, texts):
        norm_texts=self._normalize(texts)
        vectores=[]
        N=len(self.vocabulario_index)
        for doc in norm_texts:
            C=[np.argwhere(self.vocabulario_index==token)[0][0] for token in doc
               if len(np.argwhere(self.vocabulario_index==token))]
            nt=len(C)
            X=[1 for i in range(nt)]
            R=[0 for i in range(nt)]
            vectores.append(sp.csr_matrix((X,(R,C)), shape=(1,N)))
        return sp.vstack(vectores)
    
    def plot_hist(self,top=20):
        z=Counter(self.vocabulario)
        labels=[w  for w,h in z.most_common(top)]
        plt.bar(range(top),[h  for w,h in z.most_common(top)])
        plt.xticks(ticks=range(top),labels=labels,rotation='vertical');
        
    
    
