import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2,f_classif,f_regression


data = pd.read_csv('')
y = data['']
X = data.drop('', axis=1, inplace=True)
mylabel = LabelEncoder()
data["smoker"] = mylabel.fit_transform(data["smoker"])
# print(data_df.head())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

mymodel = LogisticRegression()
mymodel = GaussianNB()
mymodel = MultinomialNB()
mymodel = Lasso()

mymodel.fit(X_train, y_train)

X, y = make_classification(n_samples=2000, n_features=20, n_informative=7, n_redundant=13, flip_y=0.3, random_state=1)
scores = cross_val_score(mymodel, X, y, scoring='accuracy', cv=20, n_jobs=-1)
print(scores)
print(np.mean(scores))  # საშუალო
print(np.std(scores))  # სტანდარტული გადახრა


pipe = Pipeline([('scale',StandardScaler()),('PCA',PCA(n_components=2)),('classifier',LogisticRegression())])
pipe.fit(X_train, y_train)
print(pipe.score(X_train, y_train))

selector =SelectKBest(score_func=chi2,k=2)
selector.fit(X,y)
# selected = selector.fit_transform(X, y)
print(selector.get_feature_names_out())

X =X.iloc[:,selector.get_support()]
# selected = selector.fit_transform(X, y)
# print(selector.get_feature_names_out())
# parameters = {'scaler': [MinMaxScaler(), StandardScaler(), MaxAbsScaler()], 'selector__threshold': [0, 0.001, 0.01],
#               'classifier__n_estimators': [50, 60, 90, 100, 120]}
pipe = Pipeline(steps=[('selector', SelectKBest(score_func=f_classif)), ('algo', AdaBoostClassifier())])
parameters = {"selector__k": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 12, 13]}
hybrid = GridSearchCV(pipe, parameters, scoring='accuracy', cv=2, n_jobs=-1)
hybrid.fit(X, y)
print(hybrid.best_params_)
print(hybrid.best_params_,hybrid.best_score_)



# over = SMOTE()
# X, y = over.fit_resample(X, y)
# under = RandomUnderSampler()
# X, y = under.fit_resample(X, y)
# pipe = Pipeline(steps=[("over",SMOTE(sampling_strategy=0.2)), ("under", RandomUnderSampler(sampling_strategy=0.5))])
# X, y = pipe.fit_resample(X,y)
# print(X.shape)
# print(Counter(y))
