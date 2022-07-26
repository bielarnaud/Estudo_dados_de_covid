#from importlib.resources import path
import pandas as pd
import numpy as np
import glob
from datetime import datetime

# Repositório de armazenamento dos dados dos casos leves
path = r'C:\Users\gabri\Desktop\Git - Tcc\Estudo_dados_de_covid\Data\Casos Leves'

#Listas das features
list_features = ['sexo','data_notificacao','idade','data_inicio_sintomas','sintomas','outros_sintomas','classificacao_final','evolucao_caso',
                 'resultado_final','etnia','raca_cor','profissional_saude','cbo','bairro','ds','municipio_residencia']

columns_to_drop = ['etnia','cbo','municipio_residencia','evolucao_caso','raca_cor','ds','profissional_saude','resultaado_final']

columns_geral = ['sexo','data_notificacao','idade','data_inicio_sintomas','bairro','classificacao_final', 'resultado_final']
columns_symptoms = ['sintomas','outros_sintomas']

#Categorias dos sintomas
list_symptoms =["SYMPTOM_COUGH",'SYMPTOM_COLD','SYMPTOM_AIR_INSUFFICIENCY','SYMPTOM_FEVER','SYMPTOM_LOW_OXYGEN_SATURATION','SYMPTOM_BREATHING_CONDITION','SYMPTOM_TORACIC_APERTURE','SYMPTOM_THROAT_CONDITION',
                'SYMPTOM_HEADACHE','SYMPTOM_BODY_PAIN','SYMPTOM_DIARRHEA','SYMPTOM_RUNNY_NOSE','SYMPTOM_NOSE_CONGESTION','SYMPTOM_WEAKNESS','SYMPTOM_ANOSMIA_OR_HYPOSMIA','SYMPTOM_NAUSEA','SYMPTOM_LACK_OF_APPETITE',
                'SYMPTOM_ABDOMINAL_PAIN','SYMPTOM_CONSCIOUSNESS_DEGRADATION']

MAIN_SYMPTOMS = {
    'SYMPTOM_COUGH': ["TOSSE", "HEMOPTISE", "TOSSE PRODUTIVA", "GRIPADA"],
    'SYMPTOM_COLD': ["ESPIRROS", "ESPIRRO", "ESPIRRANDO", "ESPIROS", "ESPIRO", "GRIPE", "GRIPAL", "SINTOMAS GRIPAIS", "SINDROME GRIPAL", "SINTOMAS DE GRIPE"],
    'SYMPTOM_AIR_INSUFFICIENCY': ["DISPNEIA", "FALTA DE AR", "TAQUIPNEIA", "TAQUIDISPNEIA", "INSUFICIENCIA RESPIRATORIA"],
    'SYMPTOM_FEVER': ["FEBRE"],
    'SYMPTOM_LOW_OXYGEN_SATURATION': ["SATURACAO O2 < 95", "DESSATURACAO"],
    'SYMPTOM_BREATHING_CONDITION': ["DESCONFORTO RESPIRATORIO", "BATIMENTO ASA DE NARIZ", "TIRAGEM INTERCOSTAL", "DESCONFORTO TORACICO"],
    'SYMPTOM_TORACIC_APERTURE': ["APERTO TORACICO", "APERTO NO PEITO", "DOR TORAXICA", "DOR DORACICA", "DOR TORACICA", "DOR NO PEITO"],
    'SYMPTOM_THROAT_CONDITION': ["DOR DE GARGANTA", "ODINOFAGIA"],
    'SYMPTOM_HEADACHE': ["CEFALEIA", "DOR DE CABECA", "DORES NA CABECA"],
    'SYMPTOM_BODY_PAIN': ["MIALGIA", "DOR MUSCULAR", "DOR NO CORPO", "DORES NO CORPO", "ALGIA", "DOR LOMBAR", "DOR NA REGIAO LOMBAR", "DOR NAS COSTAS", "QUEDA DO ESTADO GERAL", "QUEDA DO ESTADO EM GERAL", "QUEDA DE ESTADO GERAL", "DORSALGIA", "LOMBALGIA", "DOR EM MMII", "DORES EM MMII", "DOR", "ARTRALGIA", "DORES NAS COSTAS", "NO CORPO", "DOR NAS ARTICULACOES", "DORES MUSCULARES", "DOR CORPO", ],
    'SYMPTOM_DIARRHEA': ["DIARREIA"],
    'SYMPTOM_RUNNY_NOSE': ["CORIZA", "RINORREIA", "SECRECAO"],
    'SYMPTOM_NOSE_CONGESTION': ["CONGESTAO NASAL", "OBSTRUCAO NASAL"],
    'SYMPTOM_WEAKNESS': ["ASTENIA", "FRAQUEZA", "CANSACO/FADIGA", "FADIGA", "CANSACO", "FADIGA/CANSACO", "ADINAMIA", "MOLEZA", "MOLEZA NO CORPO", "CORPO MOLE", "MAL ESTAR", "INDISPOSICAO"],
    'SYMPTOM_ANOSMIA_OR_HYPOSMIA': ["ALTERACAO/PERDA DE OLFATO E/OU PALADAR", "DISTURBIOS GUSTATIVOS", "DISTURBIOS OLFATIVOS", "PERDA DO OLFATO", "PERDA DE OLFATO", "OLFATO", "PERDA OLFATO", "PERDA DO OLFATO/ MIALGIA", "PERDA DO PALADAR", "PERDA DE PALADAR", "DO PALADAR", "PALADAR", "PERDA D PALADAR", "SEM PALADAR", "ERDA DO PALADAR", "AGEUSIA", "AUGESIA", "AGUESIA", "DISGEUSIA", "PERDA DE OLFATO (CHEIRO) E/OU PALADAR (GOSTO)", "FALTA DE PALADAR", "SEM OFATO"],
    'SYMPTOM_NAUSEA': ["NAUSEA", "NAUSEAS", "ENJOO", "VOMITO", "VOMITOS", "EMESE", "ANSIA DE VOMITO", "TONTURA"],
    'SYMPTOM_LACK_OF_APPETITE': ["INAPTENCIA", "FALTA DE APETITE", "ANOREXIA", "HIPOREXIA", "DIMINUICAO DO APETITE"],
    'SYMPTOM_ABDOMINAL_PAIN': ["DOR ABDOMINAL", "DOR EPIGASTRICA"],
    'SYMPTOM_CONSCIOUSNESS_DEGRADATION': ["RNC", "SONOLENCIA", "DESORIENTACAO", "PROSTRACAO", "PROSTACAO", "REBAIXAMENTO DO NIVEL DE CONSCIENCIA", "SINCOPE", "DESORIENTADO", "CONFUSAO", "AFASIA", "CONFUSAO MENTAL", "REBAIXAMENTO DO NIVEL DE CONCIENCIA", "PERDA PONDERAL", "PERDA DA CONSCIENCIA", "REBAIXAMENTO NIVEL DE CONSCIENCIA"],
}


class Pre_Processing_Casos_Leves:
    def __init__(self):
        self.df = None
        self.df2 = None
        self.path = None
        self.df_temp = None 


    def run(self,columns_symptoms,columns_to_drop,path):
        #pre-processing
        self.merge(path)
        self.Fillna(columns_symptoms)
        self.Drop(columns_to_drop)
        self.Split_symbols(columns_symptoms[0],columns_symptoms[1])
        self.age_adjustment()
        self.types_adjustment()
        self.encoding()

        # feature engineering
        self.add_symptoms_columns()
        self.categorizing_symptoms()

        #Creating news datasets
        self.creating_data_symptoms()

    #Concatenando os datasets
    def merge (self,path):
        self.path = path
        filenames = glob.glob(path + "/*.csv")

        li= []

        for filename in filenames:
            self.df_temp = pd.read_csv(filename, index_col=None, header=0,sep=';',decimal=',')
            li.append(self.df_temp)

        self.df = pd.concat(li, axis=0, ignore_index=True)
    
    def age_adjustment (self):
        list_months = ['IGN', '0 meses', '1 meses', '1 mês', '1 mes', '2 meses','3 meses','4 meses','5 meses', '6 meses', '7 meses','8 meses', 
                        '9 meses','10 meses', '11 meses', '12 meses']

        def if_in_list_months(idade):
            if idade in list_months:
                idade = 0
            else:
                idade = idade

            return idade

        self.df['idade'] = self.df['idade'].apply(if_in_list_months)

    def types_adjustment(self):
        self.df['data_inicio_sintomas'] = self.df['data_inicio_sintomas'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
        self.df['data_notificacao'] = self.df['data_notificacao'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
        self.df['idade'] = self.df['idade'].astype(int)

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
            if classificacao == 'Descartado':
                classificacao = 0
            elif classificacao == 'Confirmado Laboratorial':
                classificacao = 1
            elif classificacao == 'Síndrome Gripal Não Especificada':
                classificacao = 2
            elif classificacao == 'Confirmado Clínico-Epidemiológico':
                classificacao = 3
            elif classificacao == 'Confirmação Laboratorial':
                classificacao = 4
            elif classificacao == 'Confirmado Clínico-Imagem':
                classificacao = 5
            elif classificacao == 'Confirmado por Critério Clínico':
                classificacao = 6
            elif classificacao == 'NÃO INFORMADO':
                classificacao = 7

            return classificacao
        
        def categorizing_resultado(resultado):
            if resultado == 'Negativo':
                resultado = 1
            elif resultado == 'Positivo':
                resultado = 2
            elif resultado == 'Inconclusivo ou indeterminado':
                resultado = 3
            elif resultado == 'NÃO INFORMADO':
                resultado = 4

            return resultado


        self.df['classificacao_final'] = self.df['classificacao_final'].apply(categorizing_classificacao)
        self.df['resultado_final'] = self.df['resultado_final'].apply(categorizing_resultado)
        self.df['sexo'] = self.df['sexo'].apply(categorizing_sexo)         

    #Ajustando os dados faltantes e Dropando as colunas não utilizadas
    def Fillna(self,columns_symptoms):
        #Tratando dados features gerais
        self.df['sexo'].fillna((self.df['sexo'].describe().top), inplace = True)
        self.df['idade'].fillna((self.df['idade'].mode()[0]), inplace = True)
        self.df['bairro'].fillna((self.df['bairro'].describe().top), inplace = True)
        self.df['classificacao_final'].fillna('NÃO INFORMADO', inplace = True)
        self.df['resultado_final'].fillna('NÃO INFORMADO', inplace = True)

        #tratando a features das doenças e dos sintomas
        for column in columns_symptoms:
            self.df[column].fillna( '' , inplace = True )
    
    def Drop(self,columns_to_drop):
        for column in columns_to_drop:
            self.df.drop(columns=[column], inplace = True)

    #Ajustando as strings de sintomas
    def Split_symbols(self,column_1,column_2):
        symbols = ", |,| / | \+ |\+| E |;"
        self.split_and_trim(column_2, symbols)
        self.split_and_trim(column_1, symbols)

    def split_and_trim(self, column, separator):
        self.df[column] = self.df[column].str.split(separator)
        self.df[column] = self.df[column].apply(lambda x: [item.strip().upper() for item in x])
        self.df[column] = self.df[column].apply(lambda x: [item for item in x if item != ""])
        self.df[column] = self.df[column].apply(lambda x: [item.replace(".", "") for item in x])
        self.df[column] = self.df[column].apply(lambda x: sorted(x))
    
    #Categorizando os sintomas
        #As colunas seram criadas com ZEROS e posteriormente categorizadas
            # 0 -> Não apresentou o sintoma
            # 1 -> Apresentou o sintoma

    def add_symptoms_columns(self):
        for column, value in MAIN_SYMPTOMS.items():
            self.df[column] = 0
    
    def categorizing_symptoms(self):
        for i in range(len(self.df)):
            for column,value in MAIN_SYMPTOMS.items():
                for symptom in value:
                    if (symptom in self.df['sintomas'][i]) or (symptom in self.df['outros_sintomas'][i]):
                        self.df[column][i] = 1
    
    #Criando Dataset com o número de aparições dos sintomas: 

    def creating_data_symptoms(self):
        count_symptoms = []

        for symptom in list_symptoms: 
            count = len(self.df[self.df[symptom]==1])
            count_symptoms.append(count)
        
        self.df2 = pd.DataFrame(list_symptoms,columns=["Sintoma"])
        self.df2["Count"] = count_symptoms
    
