import pandas as pd
pd.options.mode.chained_assignment = None
from sklearn.metrics.pairwise import cosine_similarity
import warnings
from sas7bdat import SAS7BDAT
import platform
import numpy as np

# Graph Creation
from networkx.algorithms import bipartite
import networkx as nx

def read_data(path):

    full_dataset = pd.read_sas(path, encoding="latin-1")
    new_data = full_dataset[['RRID', 'STUDY', 'AGE', 'BMI', 'HIGHBP', 'HI_CHOL', 'FADIAB', 'MODIAB',
                             'SMOKENOW', 'DIABETIC', 'FABP', 'MCORRSYS', 'MCORRDIA', 'MMSE_SCORE', 'GENDER']]
    new_data = new_data[new_data['STUDY'] == 'BASELINE']
    new_data = new_data.drop_duplicates(subset='RRID', keep="last")
    new_data = new_data.dropna()
    new_data = new_data.reset_index()
    new_data['HYPERLIPIDEMIA'] = new_data['HI_CHOL']
    del new_data['index']
    
    return new_data, full_dataset



def categorize_data(data):

    conditions_obese = [
    (data['BMI'] <= 3),
    (data['BMI'] > 3)
]
    # 0.0: No Obese, 1.0: Obese
    values_obese = [0, 1]
    # Create New Column and fill with the values based on the conditions
    data['OBESE'] = np.select(conditions_obese, values_obese)


    # Add HYPERTENSION Condition to Dataset based on MCORRSYS and MCORRDIA
    conditions_hypertension = [
    (data['MCORRSYS'] <= 130) & (data['MCORRDIA'] <= 80), # both conditions must be true
    (data['MCORRSYS'] > 130) | (data['MCORRDIA'] > 80)    # either condition must be true
    ]
    # 0.0: No Hypertension, 1.0: Hypertension
    values_hypertension = [0, 1]
    # Create New Column and fill with the values based on the conditions
    data['HYPERTENSION'] = np.select(conditions_hypertension, values_hypertension)



    # Add COGNITIVE_IMPAIRMENT Condition to Dataset based on MMSE_SCORE
    conditions_cognitive_impairment = [
    (data['MMSE_SCORE'] >= 19),
    (data['MMSE_SCORE'] < 19)
    ]
    # 0.0: No Cognitive Impairment, 1.0: Cognitive Impairment
    values_cognitive_impairment = [0, 1]
    # Create New Column and fill with the values based on the conditions
    data['COGNITIVE_IMPAIRMENT'] = np.select(conditions_cognitive_impairment, values_cognitive_impairment)
    
    ###############################################################################################
    

    ###############################################################################################
    diab_positive = data.loc[data['DIABETIC'] == 1]
    diab_negative = data.loc[data['DIABETIC'] == 2]
    diab_positive = diab_positive[0:300]
    diab_negative = diab_negative[0:300]
    data = pd.concat([diab_positive, diab_negative], axis = 0)
    data = data.reset_index()
    #data = data.sample(frac = 1).reset_index()
    del data['index']
    data['RRID'] = data.index + 1
    data['DIABETIC'].replace(to_replace = 2, value = 0, inplace = True)
    ###############################################################################################


   
    data['BMI'].replace(to_replace = 1, value = 0, inplace = True)
    data['BMI'].replace(to_replace = 2, value = 1, inplace = True)
    data['BMI'].replace(to_replace = 3, value = 2, inplace = True)
    data['BMI'].replace(to_replace = 4, value = 3, inplace = True)
    data['BMI'].replace(to_replace = 5, value = 4, inplace = True)
    data['HIGHBP'].replace(to_replace = 2, value = 0, inplace = True)
    data['HI_CHOL'].replace(to_replace = 2, value = 0, inplace = True)
    data['SMOKENOW'].replace(to_replace = 1, value = 0, inplace = True)
    data['SMOKENOW'].replace(to_replace = 2, value = 1, inplace = True)
    data['SMOKENOW'].replace(to_replace = 3, value = 2, inplace = True)
    data['FADIAB'].replace(to_replace = 1, value = 0, inplace = True)
    data['FADIAB'].replace(to_replace = 2, value = 1, inplace = True)
    data['FADIAB'].replace(to_replace = 3, value = 2, inplace = True)
    data['FADIAB'].replace(to_replace = 4, value = 3, inplace = True)
    data['MODIAB'].replace(to_replace = 1, value = 0, inplace = True)
    data['MODIAB'].replace(to_replace = 2, value = 1, inplace = True)
    data['MODIAB'].replace(to_replace = 3, value = 2, inplace = True)
    data['MODIAB'].replace(to_replace = 4, value = 3, inplace = True)
    data['FABP'].replace(to_replace = 1, value = 0, inplace = True)
    data['FABP'].replace(to_replace = 2, value = 1, inplace = True)
    data['FABP'].replace(to_replace = 3, value = 2, inplace = True)
    data['HYPERLIPIDEMIA'] = data['HI_CHOL']
    data['HYPERTENSION'] = data['HIGHBP']


    del data['STUDY']
    data = data.astype(int)

    ###############################################################################################
    #Create different Categories for ranges of age.
    conditions = [
    (data['AGE'] > 18) & (data['AGE']  <= 30),
    (data['AGE'] > 30) & (data['AGE']  <= 40),
    (data['AGE'] > 40) & (data['AGE']  <= 50),
    (data['AGE'] > 50) & (data['AGE']  <= 60),
    (data['AGE'] > 60) & (data['AGE']  <= 70),
    (data['AGE'] > 70) & (data['AGE']  <= 80),
    (data['AGE'] > 80) & (data['AGE']  <= 87),
    ]

    values = [0, 1, 2, 3, 4, 5, 6]
    # Add Column 'AGE_CATEGORY based on conditions
    data['AGE_CATEGORY'] = np.select(conditions, values)
    ###############################################################################################

    # # Create different Categories based on Chronin Conditions
    conditions = [
    (data['DIABETIC'] == 0.0) & (data['OBESE'] == 0.0) & (data['HYPERLIPIDEMIA'] == 0.0) & (data['HYPERTENSION'] == 0.0) & (data['COGNITIVE_IMPAIRMENT'] == 0.0), #'No Diabetic, No Obese, No Hyperlipidemia, No Hypertension'
    (data['DIABETIC'] == 1.0) & (data['OBESE'] == 0.0) & (data['HYPERLIPIDEMIA'] == 0.0) & (data['HYPERTENSION'] == 0.0) & (data['COGNITIVE_IMPAIRMENT'] == 0.0), #'Diabetic, No Obese, No Hyperlipidemia, No Hypertension'
    (data['DIABETIC'] == 0.0) & (data['OBESE'] == 1.0) & (data['HYPERLIPIDEMIA'] == 0.0) & (data['HYPERTENSION'] == 0.0) & (data['COGNITIVE_IMPAIRMENT'] == 0.0), #'No Diabetic, Obese, No Hyperlipidemia, No Hypertension'
    (data['DIABETIC'] == 0.0) & (data['OBESE'] == 0.0) & (data['HYPERLIPIDEMIA'] == 1.0) & (data['HYPERTENSION'] == 0.0) & (data['COGNITIVE_IMPAIRMENT'] == 0.0), #'No Diabetic, No Obese, Hyperlipidemia, No Hypertension'

    (data['DIABETIC'] == 0.0) & (data['OBESE'] == 0.0) & (data['HYPERLIPIDEMIA'] == 0.0) & (data['HYPERTENSION'] == 1.0) & (data['COGNITIVE_IMPAIRMENT'] == 0.0), #'No Diabetic, No Obese, No Hyperlipidemia, Hypertension'
    (data['DIABETIC'] == 1.0) & (data['OBESE'] == 1.0) & (data['HYPERLIPIDEMIA'] == 0.0) & (data['HYPERTENSION'] == 0.0) & (data['COGNITIVE_IMPAIRMENT'] == 0.0), #'Diabetic, Obese, No Hyperlipidemia, No Hypertension'
    (data['DIABETIC'] == 1.0) & (data['OBESE'] == 0.0) & (data['HYPERLIPIDEMIA'] == 1.0) & (data['HYPERTENSION'] == 0.0) & (data['COGNITIVE_IMPAIRMENT'] == 0.0), #'Diabetic, No Obese, Hyperlipidemia, No Hypertension'
    (data['DIABETIC'] == 1.0) & (data['OBESE'] == 0.0) & (data['HYPERLIPIDEMIA'] == 0.0) & (data['HYPERTENSION'] == 1.0) & (data['COGNITIVE_IMPAIRMENT'] == 0.0), #'Diabetic, No Obese, No Hyperlipidemia, Hypertension'

    (data['DIABETIC'] == 0.0) & (data['OBESE'] == 1.0) & (data['HYPERLIPIDEMIA'] == 1.0) & (data['HYPERTENSION'] == 0.0) & (data['COGNITIVE_IMPAIRMENT'] == 0.0), #'No Diabetic, Obese, Hyperlipidemia, No Hypertension'
    (data['DIABETIC'] == 0.0) & (data['OBESE'] == 1.0) & (data['HYPERLIPIDEMIA'] == 0.0) & (data['HYPERTENSION'] == 1.0) & (data['COGNITIVE_IMPAIRMENT'] == 0.0), #'No Diabetic, Obese, No Hyperlipidemia, Hypertension'
    (data['DIABETIC'] == 0.0) & (data['OBESE'] == 0.0) & (data['HYPERLIPIDEMIA'] == 1.0) & (data['HYPERTENSION'] == 1.0) & (data['COGNITIVE_IMPAIRMENT'] == 0.0), #'No Diabetic, No Obese, Hyperlipidemia, Hypertension'
    (data['DIABETIC'] == 1.0) & (data['OBESE'] == 1.0) & (data['HYPERLIPIDEMIA'] == 1.0) & (data['HYPERTENSION'] == 0.0) & (data['COGNITIVE_IMPAIRMENT'] == 0.0), #'Diabetic, Obese, Hyperlipidemia, No Hypertension'

    (data['DIABETIC'] == 1.0) & (data['OBESE'] == 1.0) & (data['HYPERLIPIDEMIA'] == 0.0) & (data['HYPERTENSION'] == 1.0) & (data['COGNITIVE_IMPAIRMENT'] == 0.0), #'Diabetic, Obese, No Hyperlipidemia, Hypertension'
    (data['DIABETIC'] == 1.0) & (data['OBESE'] == 0.0) & (data['HYPERLIPIDEMIA'] == 1.0) & (data['HYPERTENSION'] == 1.0) & (data['COGNITIVE_IMPAIRMENT'] == 0.0), #'Diabetic, No Obese, Hyperlipidemia, Hypertension'
    (data['DIABETIC'] == 0.0) & (data['OBESE'] == 1.0) & (data['HYPERLIPIDEMIA'] == 1.0) & (data['HYPERTENSION'] == 1.0) & (data['COGNITIVE_IMPAIRMENT'] == 0.0), #'No Diabetic, Obese, Hyperlipidemia, Hypertension'
    (data['DIABETIC'] == 1.0) & (data['OBESE'] == 1.0) & (data['HYPERLIPIDEMIA'] == 1.0) & (data['HYPERTENSION'] == 1.0) & (data['COGNITIVE_IMPAIRMENT'] == 0.0), #'Diabetic, Obese, Hyperlipidemia, Hypertension'





    (data['DIABETIC'] == 0.0) & (data['OBESE'] == 0.0) & (data['HYPERLIPIDEMIA'] == 0.0) & (data['HYPERTENSION'] == 0.0) & (data['COGNITIVE_IMPAIRMENT'] == 1.0), #'No Diabetic, No Obese, No Hyperlipidemia, No Hypertension'
    (data['DIABETIC'] == 1.0) & (data['OBESE'] == 0.0) & (data['HYPERLIPIDEMIA'] == 0.0) & (data['HYPERTENSION'] == 0.0) & (data['COGNITIVE_IMPAIRMENT'] == 1.0), #'Diabetic, No Obese, No Hyperlipidemia, No Hypertension'
    (data['DIABETIC'] == 0.0) & (data['OBESE'] == 1.0) & (data['HYPERLIPIDEMIA'] == 0.0) & (data['HYPERTENSION'] == 0.0) & (data['COGNITIVE_IMPAIRMENT'] == 1.0), #'No Diabetic, Obese, No Hyperlipidemia, No Hypertension'
    (data['DIABETIC'] == 0.0) & (data['OBESE'] == 0.0) & (data['HYPERLIPIDEMIA'] == 1.0) & (data['HYPERTENSION'] == 0.0) & (data['COGNITIVE_IMPAIRMENT'] == 1.0), #'No Diabetic, No Obese, Hyperlipidemia, No Hypertension'

    (data['DIABETIC'] == 0.0) & (data['OBESE'] == 0.0) & (data['HYPERLIPIDEMIA'] == 0.0) & (data['HYPERTENSION'] == 1.0) & (data['COGNITIVE_IMPAIRMENT'] == 1.0), #'No Diabetic, No Obese, No Hyperlipidemia, Hypertension'
    (data['DIABETIC'] == 1.0) & (data['OBESE'] == 1.0) & (data['HYPERLIPIDEMIA'] == 0.0) & (data['HYPERTENSION'] == 0.0) & (data['COGNITIVE_IMPAIRMENT'] == 1.0), #'Diabetic, Obese, No Hyperlipidemia, No Hypertension'
    (data['DIABETIC'] == 1.0) & (data['OBESE'] == 0.0) & (data['HYPERLIPIDEMIA'] == 1.0) & (data['HYPERTENSION'] == 0.0) & (data['COGNITIVE_IMPAIRMENT'] == 1.0), #'Diabetic, No Obese, Hyperlipidemia, No Hypertension'
    (data['DIABETIC'] == 1.0) & (data['OBESE'] == 0.0) & (data['HYPERLIPIDEMIA'] == 0.0) & (data['HYPERTENSION'] == 1.0) & (data['COGNITIVE_IMPAIRMENT'] == 1.0), #'Diabetic, No Obese, No Hyperlipidemia, Hypertension'

    (data['DIABETIC'] == 0.0) & (data['OBESE'] == 1.0) & (data['HYPERLIPIDEMIA'] == 1.0) & (data['HYPERTENSION'] == 0.0) & (data['COGNITIVE_IMPAIRMENT'] == 1.0), #'No Diabetic, Obese, Hyperlipidemia, No Hypertension'
    (data['DIABETIC'] == 0.0) & (data['OBESE'] == 1.0) & (data['HYPERLIPIDEMIA'] == 0.0) & (data['HYPERTENSION'] == 1.0) & (data['COGNITIVE_IMPAIRMENT'] == 1.0), #'No Diabetic, Obese, No Hyperlipidemia, Hypertension'
    (data['DIABETIC'] == 0.0) & (data['OBESE'] == 0.0) & (data['HYPERLIPIDEMIA'] == 1.0) & (data['HYPERTENSION'] == 1.0) & (data['COGNITIVE_IMPAIRMENT'] == 1.0), #'No Diabetic, No Obese, Hyperlipidemia, Hypertension'
    (data['DIABETIC'] == 1.0) & (data['OBESE'] == 1.0) & (data['HYPERLIPIDEMIA'] == 1.0) & (data['HYPERTENSION'] == 0.0) & (data['COGNITIVE_IMPAIRMENT'] == 1.0), #'Diabetic, Obese, Hyperlipidemia, No Hypertension'

    (data['DIABETIC'] == 1.0) & (data['OBESE'] == 1.0) & (data['HYPERLIPIDEMIA'] == 0.0) & (data['HYPERTENSION'] == 1.0) & (data['COGNITIVE_IMPAIRMENT'] == 1.0), #'Diabetic, Obese, No Hyperlipidemia, Hypertension'
    (data['DIABETIC'] == 1.0) & (data['OBESE'] == 0.0) & (data['HYPERLIPIDEMIA'] == 1.0) & (data['HYPERTENSION'] == 1.0) & (data['COGNITIVE_IMPAIRMENT'] == 1.0), #'Diabetic, No Obese, Hyperlipidemia, Hypertension'
    (data['DIABETIC'] == 0.0) & (data['OBESE'] == 1.0) & (data['HYPERLIPIDEMIA'] == 1.0) & (data['HYPERTENSION'] == 1.0) & (data['COGNITIVE_IMPAIRMENT'] == 1.0), #'No Diabetic, Obese, Hyperlipidemia, Hypertension'
    (data['DIABETIC'] == 1.0) & (data['OBESE'] == 1.0) & (data['HYPERLIPIDEMIA'] == 1.0) & (data['HYPERTENSION'] == 1.0) & (data['COGNITIVE_IMPAIRMENT'] == 1.0) #'Diabetic, Obese, Hyperlipidemia, Hypertension'

    ]

    # 0: 'No Diabetic, No Obese, No Hyperlipidemia'
    # 1: 'No Diabetic, Obese, No Hyperlipidemia'
    # 2: 'No Diabetic, No Obese, Hyperlipidemia'
    # 3: 'No Diabetic, Obese, Hyperlipidemia'
    # 4: 'Diabetic, No Obese, No Hyperlipidemia'
    # 5: 'Diabetic, Obese, No Hyperlipidemia'
    # 6: 'Diabetic, No Obese, Hyperlipidemia'
    # 7: 'Diabetic, Obese, Hyperlipidemia'
    values = ['No Diabetic, No Obese, No Hyperlipidemia, No Hypertension, No Cognitive_Imp', 'Diabetic, No Obese, No Hyperlipidemia, No Hypertension, No Cognitive_Imp', 'No Diabetic, Obese, No Hyperlipidemia, No Hypertension, No Cognitive_Imp', 'No Diabetic, No Obese, Hyperlipidemia, No Hypertension, No Cognitive_Imp',
    'No Diabetic, No Obese, No Hyperlipidemia, Hypertension, No Cognitive_Imp', 'Diabetic, Obese, No Hyperlipidemia, No Hypertension, No Cognitive_Imp', 'Diabetic, No Obese, Hyperlipidemia, No Hypertension, No Cognitive_Imp', 'Diabetic, No Obese, No Hyperlipidemia, Hypertension, No Cognitive_Imp',
    'No Diabetic, Obese, Hyperlipidemia, No Hypertension, No Cognitive_Imp', 'No Diabetic, Obese, No Hyperlipidemia, Hypertension, No Cognitive_Imp', 'No Diabetic, No Obese, Hyperlipidemia, Hypertension, No Cognitive_Imp', 'Diabetic, Obese, Hyperlipidemia, No Hypertension, No Cognitive_Imp',
    'Diabetic, Obese, No Hyperlipidemia, Hypertension, No Cognitive_Imp', 'Diabetic, No Obese, Hyperlipidemia, Hypertension, No Cognitive_Imp', 'No Diabetic, Obese, Hyperlipidemia, Hypertension, No Cognitive_Imp', 'Diabetic, Obese, Hyperlipidemia, Hypertension, No Cognitive_Imp',
    'No Diabetic, No Obese, No Hyperlipidemia, No Hypertension, Cognitive_Imp', 'Diabetic, No Obese, No Hyperlipidemia, No Hypertension, Cognitive_Imp', 'No Diabetic, Obese, No Hyperlipidemia, No Hypertension, Cognitive_Imp', 'No Diabetic, No Obese, Hyperlipidemia, No Hypertension, Cognitive_Imp',
    'No Diabetic, No Obese, No Hyperlipidemia, Hypertension, Cognitive_Imp', 'Diabetic, Obese, No Hyperlipidemia, No Hypertension, Cognitive_Imp', 'Diabetic, No Obese, Hyperlipidemia, No Hypertension, Cognitive_Imp', 'Diabetic, No Obese, No Hyperlipidemia, Hypertension, Cognitive_Imp',
    'No Diabetic, Obese, Hyperlipidemia, No Hypertension, Cognitive_Imp', 'No Diabetic, Obese, No Hyperlipidemia, Hypertension, Cognitive_Imp', 'No Diabetic, No Obese, Hyperlipidemia, Hypertension, Cognitive_Imp', 'Diabetic, Obese, Hyperlipidemia, No Hypertension, Cognitive_Imp',
    'Diabetic, Obese, No Hyperlipidemia, Hypertension, Cognitive_Imp', 'Diabetic, No Obese, Hyperlipidemia, Hypertension, Cognitive_Imp', 'No Diabetic, Obese, Hyperlipidemia, Hypertension, Cognitive_Imp', 'Diabetic, Obese, Hyperlipidemia, Hypertension, Cognitive_Imp']
    #values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]


    # Add Column 'Category' based on conditions
    data['CATEGORY'] = np.select(conditions, values)
    ###############################################################################################

    data = data[['RRID', 'AGE', 'AGE_CATEGORY', 'BMI', 'HIGHBP', 'HI_CHOL', 'FADIAB', 'MODIAB', 
              'SMOKENOW','FABP', 'MCORRSYS', 'MCORRDIA', 'MMSE_SCORE', 'COGNITIVE_IMPAIRMENT','GENDER', 'DIABETIC', 'OBESE', 'HYPERLIPIDEMIA', 'HYPERTENSION', 'CATEGORY']]

    conditions_features = data[['AGE_CATEGORY', 'BMI', 'HIGHBP', 'HI_CHOL', 'FADIAB', 'MODIAB', 'SMOKENOW', 'FABP', 'GENDER']]
    patients = data[['RRID', 'AGE']]

    return patients, conditions_features, data


def e_distance(conditions_features, patients):
    from sklearn.metrics.pairwise import euclidean_distances

    
    dist = euclidean_distances(conditions_features, conditions_features)
    #similarity_p = cosine_similarity(conditions_features)
    index = list(conditions_features.index.values)
    ids = patients['RRID'].tolist()
    warnings.filterwarnings('ignore')
    df = []
    for i in range(len(index)):  
        for j in range(1+i,len(index)): 
            temp = {'PatientX':ids[i],'PatientY':ids[j],'Euclidean_Dist':dist[i,j]}   
            df.append(temp)

    df = pd.DataFrame(df)
    df = df.sort_values(by=['PatientX', 'Euclidean_Dist'],ascending = [True, False])
    df = df.reset_index()
    del df['index']
    
    return df


def relationships(conditions_features, patients):
    
    relationships = []
    i = 0
    for age_cat, bmi, highbp, highch, fdia, mdia, smk, fabp, in zip(conditions_features['AGE_CATEGORY'], conditions_features['BMI'], conditions_features['HIGHBP'],
                                                       conditions_features['HI_CHOL'], conditions_features['FADIAB'], conditions_features['MODIAB'],
                                                       conditions_features['SMOKENOW'], conditions_features['FABP']):

        if bmi == 0:
            tmp = [patients['RRID'][i], 'UNDERWEIGHT']
            relationships.append(tuple(tmp))
        elif bmi == 1:
            tmp = [patients['RRID'][i], 'NORMAL']
            relationships.append(tuple(tmp))
        elif bmi == 2:
            tmp = [patients['RRID'][i], 'OVERWEIGHT']
            relationships.append(tuple(tmp))
        elif bmi == 3:
            tmp = [patients['RRID'][i], 'OBESE']
            relationships.append(tuple(tmp))
        else:
            tmp = [patients['RRID'][i], 'MORBIDLY_OBESE']
            relationships.append(tuple(tmp))


        if highbp:
            tmp = [patients['RRID'][i], 'HIGHBP']
            relationships.append(tuple(tmp))
        else:
          tmp = [patients['RRID'][i], 'NO_HIGHBP']
          relationships.append(tuple(tmp))


        if highch:
            tmp = [patients['RRID'][i], 'HI_CHOL']
            relationships.append(tuple(tmp))
        else:
          tmp = [patients['RRID'][i], 'NO_HI_CHOL']
          relationships.append(tuple(tmp))


        if smk == 0:
            tmp = [patients['RRID'][i], 'SMOKENOW']
            relationships.append(tuple(tmp))
        elif smk == 1:
            tmp = [patients['RRID'][i], 'NO_SMOKENOW']
            relationships.append(tuple(tmp))
        else:
            tmp = [patients['RRID'][i], 'OTHER']
            relationships.append(tuple(tmp))


        if mdia == 0:
            tmp = [patients['RRID'][i], 'M_YES']
            relationships.append(tuple(tmp))
        elif mdia == 1:
            tmp = [patients['RRID'][i], 'M_BORDERLINE']
            relationships.append(tuple(tmp))
        else:
            tmp = [patients['RRID'][i], 'M_NO']
            relationships.append(tuple(tmp))


        if fdia == 0:
            tmp = [patients['RRID'][i], 'F_YES']
            relationships.append(tuple(tmp))
        elif fdia == 1:
            tmp = [patients['RRID'][i], 'F_BORDERLINE']
            relationships.append(tuple(tmp))
        else:
            tmp = [patients['RRID'][i], 'F_NO']
            relationships.append(tuple(tmp))


        if age_cat == 0:
            tmp = [patients['RRID'][i],'AGE_CAT_0']
            relationships.append(tuple(tmp))
        elif age_cat == 1:
            tmp = [patients['RRID'][i],'AGE_CAT_1']
            relationships.append(tuple(tmp))
        elif age_cat == 2:
            tmp = [patients['RRID'][i],'AGE_CAT_2']
            relationships.append(tuple(tmp))
        elif age_cat == 3:
            tmp = [patients['RRID'][i],'AGE_CAT_3']
            relationships.append(tuple(tmp))
        elif age_cat == 4:
            tmp = [patients['RRID'][i],'AGE_CAT_4']
            relationships.append(tuple(tmp))
        elif age_cat == 5:
            tmp = [patients['RRID'][i],'AGE_CAT_5']
            relationships.append(tuple(tmp))
        else:
            tmp = [patients['RRID'][i],'AGE_CAT_6']
            relationships.append(tuple(tmp))



        if fabp == 0:
            tmp = [patients['RRID'][i], 'F_HIGH_BLOOD']
            relationships.append(tuple(tmp))
        elif fabp == 1:
            tmp = [patients['RRID'][i], 'F_NO_HIGH_BLOOD']
            relationships.append(tuple(tmp))
        else:
            fabp = [patients['RRID'][i], 'F_HIGH_BLOOD_OTHER']
            relationships.append(tuple(tmp))

        i+=1


    return relationships

def bipartite(patients, relationships):
    
    patients_node = patients['RRID'].tolist()
    condition_node = [ 'AGE_CAT_0', 'AGE_CAT_1', 'AGE_CAT_2', 'AGE_CAT_3', 'AGE_CAT_4', 'AGE_CAT_5',
                        'AGE_CAT_6', 'HIGHBP', 'NO_HIGHBP', 'HI_CHOL', 'NO_HI_CHOL', 'SMOKENOW', 'NO_SMOKENOW',
                         'OTHER', 'M_YES', 'M_BORDERLINE', 'M_NO', 'F_YES', 'F_BORDERLINE', 'F_NO', 'UNDERWEIGHT',
                        'NORMAL', 'OVERWEIGHT', 'OBESE', 'MORBIDLY_OBESE', 'F_HIGH_BLOOD', 'F_NO_HIGH_BLOOD', 'HIGH_BLOOD_OTHER']
    #Bipartite Graph Creation
    G = nx.Graph()
    # Add nodes with the node attribute "bipartite"
    G.add_nodes_from(patients_node, bipartite=0)
    G.add_nodes_from(condition_node, bipartite=1)
    # Add edges only between nodes of opposite node sets
    G.add_edges_from(relationships)
    
    return G, patients_node, condition_node
    

    
def graph_projection(G, patients_node, df, category):

    
    PG = nx.bipartite.projection.projected_graph(G, patients_node)
    A = nx.convert_matrix.to_numpy_array(PG)
    weight = df['Euclidean_Dist']

    
    node_list = list(PG.nodes())
    edge_list = list(PG.edges())
    count_dict1 = { k1:v1 for k1,v1 in zip(node_list,category)}
    count_dict2 = { k2:v2 for k2,v2 in zip(node_list,A)}

    edges_dict = {}
    for pat_x,pat_y,sim in zip(df['PatientX'], df['PatientY'], weight):
        tmp = tuple((pat_x,pat_y))
        edges_dict[tmp] = sim
        
                
    nx.set_node_attributes(PG, count_dict1, 'y')
    nx.set_node_attributes(PG, count_dict2, 'feature')
    nx.set_edge_attributes(PG, edges_dict, 'weight' )
    for node in PG.nodes:
        del PG.nodes[node]['bipartite']

    return PG, node_list

def simplify_graph(PG, simplify, distance):

    if simplify:
      nodes = PG.nodes()
      edges = list(PG.edges(nodes, data = True))
      e1 = list(filter(lambda x: x[2]['weight'] <= distance, edges))
      e1_new = []
      for entry in e1:
        e1_new.append(entry[0:2])
      

              
      old_edges = list(PG.edges)
      new_edges = e1_new
      list1 = new_edges
      list2 = old_edges
      difference = list(set(list2).difference(set(list1)))
      
      PG.remove_edges_from(difference)
    
    else:

      new_edges = list(PG.edges)

    return PG, new_edges
