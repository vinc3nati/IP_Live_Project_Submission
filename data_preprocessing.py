

import pandas as pd
import numpy as np

from sklearn.feature_selection import chi2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text


data = pd.read_excel("training_data_CC.xlsx",usecols= [0,1])
target = data["Domain"].unique().tolist()


data['category_id'] = data['Domain'].factorize()[0]

category_id_df = data[['Domain', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'Domain']].values)

stops = text.ENGLISH_STOP_WORDS.difference(["AI", "ai"])

tdf_ob = TfidfVectorizer(sublinear_tf=True, min_df=2, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words= stops, token_pattern = r"(?u)c\+{2}|\b\w+\b")
features = tdf_ob.fit_transform(data["Input_Events"].values)
labels = data["Domain"]


for _, category_id in sorted(category_to_id.items()):
    features_chi2 = chi2(features, labels == category_id)
    indices = np.argsort(features_chi2[0])
    feature_names = np.array(tdf_ob.get_feature_names())[indices]
    unigrams = [f for f in feature_names if len(f.split(' ')) == 1]
    bigrams = [f for f in feature_names if len(f.split(' ')) == 2]



from sklearn.svm import LinearSVC
model = LinearSVC()
from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest, indices_train, indices_test = train_test_split(features, labels, data.index, test_size=0.01, random_state=42)


model.fit(Xtrain, ytrain)
y_pred = model.predict(Xtest)

model = LinearSVC()
model.fit(features, labels)


import pickle
pickle.dump(model, open('model.sav', 'wb'))




