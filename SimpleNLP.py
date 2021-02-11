import scipy.sparse as sp
# Utilice el lematizador y stemmer de su preferencia 
# usaremos la variable derivador y podra tomar los valores de "lemma", "stem" o None
class Tokenizer:
    def __init__(self,corpus=[],ngrams=[],case_folding=True, derivador=None):
        self.ngrams=ngrams
        self.vocabulario={}
        self.vocabulario_index=[]
        if corpus:
            self._construye_vocabulario(corpus)

    def _case_folding(self,texts):
        if type(texts)==str:
            return [str(texts).lower().split()]
        return [str(text).lower().split() for text in texts]
    
    def _construye_vocabulario(self,corpus):
        ## Aplicar normalizaciones al corpus
        ## construir tabla de vocabulario
        self.vocabulario={}
        self.vocabulario_index=[]
        norm_corpus=self._case_folding(corpus)
        for doc in norm_corpus:
            for token in doc:
                if token not in self.vocabulario_index:
                    self.vocabulario_index.append(token)
                id_token=self.vocabulario_index.index(token)
                self.vocabulario[id_token]=self.vocabulario.get(id_token,0)+1
  
    def fit(self, corpus):
           self._construye_vocabulario(corpus) 
            
    def transform(self, texts):
        norm_texts=self._case_folding(texts)
        vectores=[]
        N=len(self.vocabulario_index)
        for doc in norm_texts:
            C=[self.vocabulario_index.index(token) for token in doc if token in self.vocabulario_index]
            nt=len(C)
            X=[1 for i in range(nt)]
            R=[0 for i in range(nt)]
            vectores.append(sp.csr_matrix((X,(R,C)), shape=(1,N)))
        return sp.vstack(vectores)
        ## Aplicar normalizaciones al texto
        ## Tokenizar el texto utilizando el vocabulario 
        #(decidir que hacer cuando ocurren palabras que no se encuentres en el vocabulario)
        ## regresar lista de tokens
        pass
    
