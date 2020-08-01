

import pandas as pd
import numpy as np
from sklearn.feature_extraction import text

data = pd.read_excel("training_data_CC.xlsx",usecols= [0,1])
target = data["Domain"].unique().tolist()


data['category_id'] = data['Domain'].factorize()[0]


category_id_df = data[['Domain', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'Domain']].values)


from sklearn.feature_selection import chi2
from sklearn.feature_extraction.text import TfidfVectorizer

stops = text.ENGLISH_STOP_WORDS.difference(["AI", "ai"])

tdf_ob = TfidfVectorizer(sublinear_tf=True, min_df=2, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words= stops, token_pattern = r"(?u)c\+{2}|\b\w+\b")
features = tdf_ob.fit_transform(data["Input_Events"].values)
labels = data["Domain"]


#importing pretrained model.trained in data_preprocessing.py
import pickle
model =pickle.load(open('model.sav', 'rb'))


N=2
for domain, category_id in sorted(category_to_id.items()):
    indices = np.argsort(model.coef_[category_id])
    feature_names = np.array(tdf_ob.get_feature_names())[indices]
    unigrams = [v for v in reversed(feature_names) if len(v.split(' ')) == 1][:N]
    bigrams = [v for v in reversed(feature_names) if len(v.split(' ')) == 2][:N]

#query solver on employee data
def sol_que(event_dom, event_type, employees):
    return employees.query("Domain == '" + event_dom + "' and (Event1 == '" + event_type +"' or Event2 == '" + event_type + "')")


#matching algorithm
def predict(events, employees):
        recommend = []
        pred = model.predict(tdf_ob.transform(events))  
        for text, predicted in zip(events, pred):
            print('"{}"'.format(text))
            print("  - Predicted as: '{}'".format(predicted))
            print("")
        for prediction in pred.tolist():
            domain, event_type = prediction.split(".")
            if domain == 'Artificial_Intelligence':
                recommend_to = sol_que('Artificial Intelligence', event_type, employees)
                
            elif domain == 'WebDev':
                recommend_to = sol_que('Web Development', event_type, employees)
                
            elif domain == 'Mobile_Applications':
                recommend_to = sol_que('Mobile Applications', event_type, employees)
                
            elif domain == 'Finance':
                recommend_to = sol_que('Finance', event_type, employees)
            
            elif domain == 'ML':
                recommend_to = sol_que('Machine Learning', event_type, employees)
                
            elif domain == 'CC':
                recommend_to =sol_que('Cloud Computing', event_type, employees)
                
            elif domain == 'Higher_Education':
                recommend_to = sol_que('Higher Education', event_type, employees)
                
            elif domain == 'DevOps':
                recommend_to =sol_que('Development Processes', event_type, employees)
                
            elif domain == 'Software_Architecture':
                recommend_to = sol_que('Software Architecture', event_type, employees)
                
            elif domain == 'Data_Science':
                recommend_to = sol_que('Data Science', event_type, employees)
                
            elif domain == 'Cpp':
                recommend_to = sol_que('C++', event_type, employees)
                
            elif domain == 'None':
                recommend_to = employees.query("Event1 == '" + event_type + "' or Event2 == '" + event_type + "'")
                
            else:
                recommend_to = sol_que(domain, event_type, employees)
                
            recommend.append(", ".join(recommend_to['Name'].values))
            
        return recommend


def get_predictions(path):
    employees = pd.read_csv("CCMLEmployeeData.csv")
    to_pred_events = pd.read_csv(path, encoding= 'unicode_escape')
    recommendations = predict(to_pred_events.Events, employees)
    to_pred_events['Employees'] = recommendations
    to_pred_events.to_excel('result.xlsx', index=False)


#Input filename or pathname of csv file to get your predictions. For eg :  input.csv
path_for_input=input("Enter path or name of input csv file")
get_predictions(path_for_input)






