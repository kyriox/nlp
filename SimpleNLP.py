import scipy.sparse as sp
# Utilice el lematizador y stemmer de su preferencia 
# usaremos la variable derivador y podra tomar los valores de "lemma", "stem" o None
class Tokenizer:
    def __init__(self,corpus=[],ngrams=[],case_folding=True, derivador=None):
        self.ngrams.append(1)
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
        pass
  
    def fit(self, corpus):
           self._construye_vocabulario(corpus) 
            
    def transform(self, corpus):
        ## Aplicar normalizaciones al texto
        ## Tokenizar el texto utilizando el vocabulario 
        #(decidir que hacer cuando ocurren palabras que no se encuentres en el vocabulario)
        ## regresar lista de tokens
        pass
    
