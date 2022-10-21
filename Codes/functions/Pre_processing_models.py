import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection._search import GridSearchCV
from sklearn.model_selection._split import StratifiedKFold
from sklearn import svm

from sklearn.model_selection import train_test_split, cross_validate, cross_val_score
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif, SelectFdr

from sklearn.metrics import f1_score, precision_score, recall_score

class Pre_processing_models:
    def __init__(self, df) -> None:
        self.df = df
        pass

    def run_graves(self):
        #self.encoding()
        self.one_hot_encoding ('sexo')
        #self.one_hot_encoding ('classificacao')
        #self.one_hot_encoding ('evolucao')
        self.df['idade'] = self.NormalizeData( self.df['idade'] )



    def NormalizeData(self , data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    def encoding (self):
        def categorizing_sexo(sexo):
            if sexo == 'Masculino':
                sexo = 1
            elif sexo == 'Feminino':
                sexo = 0
            elif sexo == 'Ignorado':
                sexo = 0

            return sexo

        def categorizing_classificacao(classificacao):
            if classificacao == 'DESCARTADO':
                classificacao = 0
            elif classificacao == 'INCONCLUSIVO':
                classificacao = 1
            elif classificacao == 'CONFIRMADO':
                classificacao = 2
            elif classificacao == 'EM ANÁLISE':
                classificacao == 3
            elif classificacao == 'NÃO INFORMADO':
                classificacao = 4

            return classificacao
        
        def categorizing_evolucao(evolucao):
            if evolucao == 'ISOLAMENTO DOMICILIAR':
                evolucao = 0
            elif evolucao == 'INTERNADO LEITO DE ISOLAMENTO':
                evolucao = 1
            elif evolucao == 'RECUPERADO':
                evolucao = 2
            elif evolucao == 'ÓBITO':
                evolucao = 3
            elif evolucao == 'INTERNADO UTI':
                evolucao = 4
            elif evolucao == 'EM ANÁLISE':
                evolucao = 5
            elif evolucao == 'NÃO INFORMADO':
                evolucao = 6

            return evolucao


        self.df['classificacao_final'] = self.df['classificacao_final'].apply(categorizing_classificacao)
        self.df['sexo'] = self.df['sexo'].apply(categorizing_sexo)         
        self.df['evolucao'] = self.df['evolucao'].apply(categorizing_evolucao)

    def one_hot_encoding(self, features):
        self.df = pd.get_dummies(self.df, columns = [features])

    