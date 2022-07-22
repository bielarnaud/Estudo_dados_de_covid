import pandas as pd
import numpy as np
import glob
from datetime import datetime

# Repositório de armazenamento dos dados dos casos graves
path = r'C:\Users\gabri\Desktop\Git - Tcc\Estudo_dados_de_covid\Data\Casos Graves'

#Listas das features
list_features = ['data_notificacao','sexo','raca','etnia','idade','municipio_residencia','bairro','distrito_sanitario','data_inicio_sintomas',
                'sintomas','outros_sintomas','doencas_preexistentes','outras_doencas_preexistentes','profissional_saude','categoria_profissional',
                'classificacao_final','evolucao','data_obito']

columns_to_drop = ['etnia','municipio_residencia','raca','profissional_saude','distrito_sanitario','categoria_profissional']
columns_geral = ['sexo','data_notificacao','idade','data_inicio_sintomas','classificacao_final','bairro','evolucao','data_obito']
columns_symptoms = ['sintomas','outros_sintomas']
columns_dieases = ['outras_doencas_preexistentes ','doencas_preexistentes']

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

#Categorias das doenças pré-existentes

list_dieases = ['DISEASE_HEART_OR_VASCULAR','DISEASE_DIABETES','DISEASE_HYPERTENSION','DISEASE_RESPIRATORY','DISEASE_OBESITY','DISEASE_KIDNEY','DISEASE_IMMUNOSUPPRESSION',
                'DISEASE_TABAGIST','DISEASE_ETILISM','DISEASE_LIVER','DISEASE_NEUROLOGIC']

MAIN_DISEASES = {
    'DISEASE_HEART_OR_VASCULAR': ["DOENCAS CARDIACAS OU VASCULARES", "DOENCAS CARDIACAS CRONICAS", "CARDIOPATIA", "CARDIOPATA", "DISLIPIDEMIA", "ICC", "DOENCA CARDIOVASCULAR CRONICA", "DAC", "DOENCA CARDIOVASCULAR", "TVP", "TROMBOSE"],
    'DISEASE_DIABETES': ["DIABETES", "DM"],
    'DISEASE_HYPERTENSION': ["HAS", "HIPERTENSAO ARTERIAL", "HIPERTENSAO", "HIPERTENSO", "HIPERTENSA", "HIPERTENSAO ARTERIAL SISTEMICA", "PORTADORA DE HIPERTENSAO ARTERIAL SISTEMICA", "HIPERTENSSO", "PACIENTE HIPERTENSO", "HIPERTENCAO", "INCLUINDO HIPERTENSAO"],
    'DISEASE_RESPIRATORY': ["DOENCAS RESPIRATORIAS CRONICAS DESCOMPENSADAS", "DOENCAS RESPIRATORIAS CRONICAS", "DOENCA PULMONAR CRONICA", "ASMA", "ASMATICA", "ASMATICO", "DPOC", "TUBERCULOSE", "PNEUMOPATIA CRONICA", "TUBERCULOSE PULMONAR", "OUTRA PNEUMOPATIA CRONICA", "PNEUMONIA CRONICA", "CANCER DE PULMAO"],
    'DISEASE_OBESITY': ["OBESIDADE", "SOBREPESO/OBESIDADE"],
    'DISEASE_KIDNEY': ["DOENCAS RENAIS CRONICAS", "DRC", "DOENCA RENAL CRONICA", "RENAL CRONICA", "DOENCAS RENAIS CRONICAS EM ESTAGIO AVANCADO (GRAUS 3", "4 OU 5)"],
    'DISEASE_IMMUNOSUPPRESSION': ["IMUNOSSUPRESSAO", "IMUNODEFICIENCIA", "IMUNODEPRESSAO"],
    'DISEASE_TABAGIST': ["TABAGISTA", "EX TABAGISTA", "TABAGISMO", "EX-TABAGISTA", "FUMANTE", "EX-FUMANTE", "EX-TABAGISMO", "EX- TABAGISTA", "EX FUMANTE", "HISTORICO DE TABAGISMO", "EX TABAGISMO"],
    'DISEASE_ETILISM': ["ETILISTA", "ETILISMO", "EX ETILISTA", "EX-ETILISTA", "ETILISTA CRONICO", "ETILISMO CRONICO", "ALCOOLISMO", "ETILISTA CRONICO"],
    'DISEASE_LIVER': ["DOENCA HEPATICA CRONICA"],
    'DISEASE_NEUROLOGIC': ["DOENCA NEUROLOGICA CRONICA", "DOENCA NEUROLOGICA", "ALZHEIMER", "PARKINSON", "DEPRESSAO", "EPILEPSIA", "DEMENCIA", "ESQUIZOFRENIA", "SINDROME DEMENCIAL", "TRANSTORNO MENTAL", "DOENCAS NEUROLOGICAS", "SINDROME DE DOWN", "DISTURBIO PSIQUIATRICO", "ANSIEDADE", "MAL DE ALZHEIMER", "TRANSTORNO PSIQUIATRICO"],
}


#CRIANDO A CLASS
class Pre_Processing_Casos_Graves:
    def __init__(self):
        #self.csv = pd.read_csv(r"C:\Users\gabri\Desktop\TCC\Data\vacinados.csv",index_col=None, header=0,sep=';',decimal=',')
        self.df = None #Dataset completo
        self.df2 = None #Dataset com o número de aparições de cada sintoma
        self.df3 = None #Dataser com o número de aparições de cada doença
        self.path = None
        self.df_temp = None 

    #Função para rodar todas as outras na ordem certa!! Teoricamente só usando a run ja faz o pré processamento!!
    def run(self,columns_geral,columns_symptoms,columns_to_drop,columns_dieases,path):
        self.merge(path)
        self.Fillna(columns_geral,columns_symptoms,columns_dieases)
        self.Drop(columns_to_drop)
        self.Split_symbols(columns_symptoms[0],columns_symptoms[1],columns_dieases[0],columns_dieases[1])
        self.add_symptoms_columns()
        self.add_dieases_columns()
        self.categorizing_symptoms()
        self.categorizing_dieases()
        self.creating_data_symptoms()
        self.creating_data_dieases()

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

        for i in range(len(self.df)):
            if self.df['idade'][i] in list_months:
                self.df['idade'][i] = 0
            else:
                self.df['idade'][i] = int(self.df['idade'][i])

    def encoding_feature_sexo (self):
        def categorizing_sexo(sexo):
            if sexo == 'Masculino':
                sexo = 1
            elif sexo == 'Feminino':
                sexo = 0
            elif sexo == 'Ignorado':
                sexo = 0

            return sexo
        
        self.df['sexo'] = self.df['sexo'].apply(categorizing_sexo)
    
    def encoding_feature_classificacao_final(self):
        def categorizing_classificacao(classificacao):
            if classificacao == 'DESCARTADO':
                classificacao = 0
            elif classificacao == 'INCONCLUSIVO':
                classificacao = 1
            elif classificacao == 'CONFIRMADO':
                classificacao = 2
            elif classificacao == 'EM ANÁLISE':
                classificacao == 3

            return classificacao
        
        self.df['classificacao_final'] = self.df['classificacao_final'].apply(categorizing_classificacao)
    
    def encoding_feature_evolucao(self):
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
            elif evolucao == 'NÃO INFORMADO':
                evolucao = 5

            return evolucao
        
        self.df['evolucao'] = self.df['evolucao'].apply(categorizing_evolucao)


    
    def types_adjustment(self):
        self.df['data_inicio_sintomas'] = self.df['data_inicio_sintomas'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
        self.df['data_notificacao'] = self.df['data_notificacao'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
        self.df['idade'] = self.df['idade'].astype(int)


    #Ajustando os dados faltantes e Dropando as colunas não utilizadas
    def Fillna(self,columns_symptoms,columns_dieases):
        #Tratando dados features gerais
        self.df['sexo'] = self.df['sexo'].fillna(self.df['sexo'].describe().top, inplace =True)
        self.df['idade'] = self.df['idade'].fillna(self.df['idade'].mode(), inplace =True)
        self.df['bairro'] = self.df['bairro'].fillna(self.df['bairro'].describe().top, inplace =True)
        self.df['evolucao'] = self.df['evolucao'].fillna("NÃO INFORMADO", inplace = True)

        #Tratando dados features de data
        self.df['data_obito'].fillna('', inplace = True)

        #tratando a features das doenças e dos sintomas
        for column in columns_symptoms:
            self.df[column].fillna( '' , inplace = True )
        for column in columns_dieases:
            self.df[column].fillna( '' , inplace = True )

    def Drop(self,columns_to_drop):
        for column in columns_to_drop:
            self.df.drop(columns=[column], inplace = True)

    #Ajustando as strings de sintomas
    def Split_symbols(self,column_1,column_2,column_3,column_4):
        symbols = ", |,| / | \+ |\+| E |;"
        self.split_and_trim(column_1, symbols)
        self.split_and_trim(column_2, symbols)
        self.split_and_trim(column_3, symbols)
        self.split_and_trim(column_4, symbols)

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
    
    #Categorizando as doenças preexistentes
    def add_dieases_columns(self):
        for column, value in MAIN_DISEASES.items():
            self.df[column] = 0

    def categorizing_dieases(self):
        for i in range(len(self.df)):
            for column,value in MAIN_DISEASES.items():
                for dieases in value:
                    if (dieases in self.df['doencas_preexistentes'][i]) or (dieases in self.df['outras_doencas_preexistentes '][i]):
                        self.df[column][i] = 1
    
    #Criando Dataset com o número de aparições dos sintomas e doenças: 

    def creating_data_symptoms(self):
        count_symptoms = []

        for symptom in list_symptoms: 
            count = len(self.df[self.df[symptom] == 1])
            count_symptoms.append(count)
        
        self.df2 = pd.DataFrame(list_symptoms,columns=["Sintoma"])
        self.df2["Count"] = count_symptoms

    def creating_data_dieases(self):
        count_dieases = []

        for dieases in list_dieases:
            count = len(self.df[self.df[dieases] == 1])
            count_dieases.append(count)

        self.df3 = pd.DataFrame(list_dieases, columns=["Doencas_preexistentes"])
        self.df3['Count'] = count_dieases