import pandas as pd
from functions.Pre_processing_Casos_Graves import *
import matplotlib.pyplot as plt
import seaborn as sns


class analises:
    def __init__(self): 
        pass

    def run_graves(self, df):
        #criando os datasets
        self.df = df
        self.df2 = self.creating_data_symptoms()
        self.df3 = self.creating_data_dieases()
    
    def run_leves(self, df):
        self.df = df
        self.df2 = self.creating_data_symptoms()

    #Criando Dataset com o número de aparições dos sintomas e doenças: 

    def creating_data_symptoms(self):
        count_symptoms = []
        perc_list = []

        for symptom in list_symptoms: 
            count = len(self.df[self.df[symptom] == 1])
            perc = (count/len(self.df))*100
            
            perc_list.append(perc)
            count_symptoms.append(count)
        
        self.df2 = pd.DataFrame(list_symptoms,columns=["Sintoma"])
        self.df2["Count"] = count_symptoms
        self.df2['percentage'] = perc_list

        return self.df2 

    def creating_data_dieases(self):
        count_dieases = []
        perc_list_2 = []

        for dieases in list_dieases:
            count = len(self.df[self.df[dieases] == 1])
            perc_2 = (count/len(self.df))*100
            
            perc_list_2.append(perc_2)
            count_dieases.append(count)

        self.df3 = pd.DataFrame(list_dieases, columns=["Doencas_preexistentes"])
        self.df3['Count'] = count_dieases
        self.df3['percentage'] = perc_list_2

        return self.df3

    #funções de gráficos
    def barplot(self, x_axis):
        print('-'*40)
        print(self.df.groupby(x_axis).death.value_counts())

        self.df[self.df[x_axis] == 1].groupby(x_axis).death.value_counts().plot.bar(x = x_axis, y='val', rot=0, color = ['green' , 'blue'])
    
    def barplot_age(self):

        total_survived = self.df[self.df['death'] == 0]
        total_not_survived = self.df[self.df['death'] == 1]

        plt.figure(figsize=[15,5])
        plt.subplot(111)
        sns.distplot(total_survived['idade'].dropna().values, kde=False, color='blue', label='Survived')
        sns.distplot(total_not_survived['idade'].dropna().values, kde=False, color='red', axlabel='Age', label='Not Survived')
        plt.legend()

    def plot_data_doencas(self):
        print('Análises gráficas quantitativa das doenças')
        sns.barplot(data = self.df3.sort_values('Count', ascending=False), y="Doencas_preexistentes", x="Count")

    def plot_data_sintomas(self):
        print('Análises gráficas quantitativa dos sintomas')
        sns.barplot(data = self.df2.sort_values('Count', ascending=False), y="Sintoma", x="Count")