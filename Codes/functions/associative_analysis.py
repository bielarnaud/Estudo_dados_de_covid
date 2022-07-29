import pandas as pd
import numpy as np

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


'''
Associando as colunas de sintomas, doenças preexistentes e mortes para ver a relação entre elas.
'''
columns_to_associete = ['DISEASE_HEART_OR_VASCULAR','DISEASE_DIABETES','DISEASE_HYPERTENSION','DISEASE_RESPIRATORY','DISEASE_OBESITY',
                        'DISEASE_KIDNEY','DISEASE_IMMUNOSUPPRESSION','DISEASE_TABAGIST','DISEASE_ETILISM','DISEASE_LIVER','DISEASE_NEUROLOGIC',
                        'SYMPTOM_COUGH','SYMPTOM_COLD','SYMPTOM_AIR_INSUFFICIENCY','SYMPTOM_FEVER','SYMPTOM_LOW_OXYGEN_SATURATION',
                        'SYMPTOM_BREATHING_CONDITION','SYMPTOM_TORACIC_APERTURE','SYMPTOM_THROAT_CONDITION','SYMPTOM_HEADACHE','SYMPTOM_BODY_PAIN',
                        'SYMPTOM_DIARRHEA','SYMPTOM_RUNNY_NOSE','SYMPTOM_NOSE_CONGESTION','SYMPTOM_WEAKNESS','SYMPTOM_ANOSMIA_OR_HYPOSMIA',
                        'SYMPTOM_NAUSEA','SYMPTOM_LACK_OF_APPETITE','SYMPTOM_ABDOMINAL_PAIN','SYMPTOM_CONSCIOUSNESS_DEGRADATION']

class associative_analysis:
    def __init__(self,df):
        self.df = df
        self.df1 = self.df[columns_to_associete]
        self.df2 = self.df[self.df['death'] == 1]

    def run(self):
        self.frequent_itemsets1 = apriori(self.df1, min_support=0.1, use_colnames=True)
        self.rules1 = association_rules(self.frequent_itemsets1, metric="lift", min_threshold=1)

        self.frequent_itemsets2 = apriori(self.df2, min_support=0.1, use_colnames=True)
        self.rules2 = association_rules(self.frequent_itemsets2, metric="lift", min_threshold=1)

        self.rules1[(self.rules1['confidence'] >= 0.7)].sort_values('lift', ascending=False)
        print('-'*60)
        self.rules2[(self.rules2['confidence'] >= 0.7)].sort_values('lift', ascending=False)
        