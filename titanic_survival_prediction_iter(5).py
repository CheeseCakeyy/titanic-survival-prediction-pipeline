#This is iteration 5 of survival prediction on titanic dataset
#This iteration will cover ensembling of all three models that were trained so far 

import pandas as pd 
from feature_creation_titanic_survival import feature_creation
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,accuracy_score


train_path = "data/train.csv"
train_df = pd.read_csv(train_path)

print(train_df.head())
df = feature_creation(train_df)
df.info()
#not dropping the featuer 'Title' this time, cuz one of the models that is Logistic Regression was able to gain info from that feature even though other models failed 

#seperating label and features
X = df.drop('Survived',axis=1)
y = df['Survived']

#splitting data into train/validation
X_train,X_validate,y_train,y_validate = train_test_split(X,y,stratify=y,test_size=0.2,random_state=42)

#----------------
'''Data-Preprocessing'''
#----------------
#since we are ensembling models, different models will require different preprocessing 
#some things will remain common th are as follows:
num_cols = X.select_dtypes('float64').columns 
count_cols = ['SibSp','Parch','Family_size']

#1# Logistic regression preprocessing:
lr_cat_cols = ['Pclass','Sex','Embarked','Travelling_alone','Title']

#2# RandomForest and XGBoost
rf_cat_cols = ['Pclass','Sex','Embarked','Travelling_alone',]

#Pipes:
count_pipe = Pipeline([
    ('impute',SimpleImputer(strategy='constant',fill_value=0))
])
cat_pipe = Pipeline([
    ('impute',SimpleImputer(strategy='most_frequent')),
    ('encoder',OneHotEncoder(drop='first',handle_unknown='ignore'))
])
lr_num_pipe = Pipeline([
    ('impute',SimpleImputer(strategy='median')),
    ('scaler',StandardScaler())
])
rf_num_pipe = Pipeline([
    ('impute',SimpleImputer(strategy='median'))
])

#Preprocessors
lr_preprocessor = ColumnTransformer(
    transformers=[
        ('num',lr_num_pipe,num_cols),
        ('cat',cat_pipe,lr_cat_cols),
        ('count',count_pipe,count_cols)
    ],
    remainder='drop'
)

rf_xgb_preprocessor = ColumnTransformer(
    transformers=[
        ('num',rf_num_pipe,num_cols),
        ('cat',cat_pipe,rf_cat_cols),
        ('count',count_pipe,count_cols)
    ],
    remainder='drop'
)


#-------------
'''Ensembling'''
#-------------
'''Ensembling = using multiple models together to make one final prediction, instead of trusting a single model.
Because different models make different mistakes.'''

#tuned LR model from iter 2
LR_pipeline = Pipeline([
        ('prep',lr_preprocessor),
        ('model',LogisticRegression(
            class_weight= None,
            solver='liblinear', 
            max_iter=250,
            C=0.1, 
            penalty='l2', 
            random_state=42,
            ))
        ])

#tuned RF model from iter 3 
RF_pipeline = Pipeline([
        ('prep',rf_xgb_preprocessor),
        ('model',RandomForestClassifier(
            n_estimators=200,
            n_jobs=-1,
            random_state=42,
            criterion='entropy',
            max_depth=8,
            min_samples_leaf=2
            ))
        ])

#tuned XGB model from iter 4
XGB_pipeline = Pipeline([
        ('prep',rf_xgb_preprocessor),
        ('model',XGBClassifier(
            n_estimators = 300,
            max_depth = 2,
            learning_rate = 0.03,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1
            ))
        ])


#training on train data:
LR_pipeline.fit(X_train,y_train)
RF_pipeline.fit(X_train,y_train)
XGB_pipeline.fit(X_train,y_train)

#predicting the y on validation set just to get the idea of the weights to be selected for the final ensemble 
y_pred_lr = LR_pipeline.predict(X_validate)
y_pred_rf = RF_pipeline.predict(X_validate)
y_pred_xgb = XGB_pipeline.predict(X_validate)

#calculating there f1 scores to decide weigths (why f1-score:Class imbalance (more non-survivors than survivors), A dumb model that predicts: everyone dead already gets a 62% accuracy so not trusting accuracy on this)
print(f1_score(y_validate,y_pred_lr)) #0.7384615384615385
print(f1_score(y_validate,y_pred_rf)) #0.7317073170731707
print(f1_score(y_validate,y_pred_xgb)) #0.7008547008547008
'''total = 0.7384615385 + 0.7317073171 + 0.7008547009 ≈ 2.170
weights:
w_lr = 0.7384615385 / 2.170 ≈ 0.34
w_rf = 0.7317073171 / 2.170 ≈ 0.34
w_xgb = 0.7008547009 / 2.170 ≈ 0.32

Although Logistic Regression achieved the highest standalone F1-score, Random Forest and XGBoost captured complementary non-linear patterns. 
Therefore, I used a weighted probability ensemble with a slight bias toward
Logistic Regression, based on validation F1-scores, and froze the weights prior to evaluation.

final weights: LR  → 0.40
               RF  → 0.33
               XGB → 0.27

'''

#calculating probabilities of prediction from each model for values in validation set
LR_prob = LR_pipeline.predict_proba(X_validate)[:,1] #this line says: calculate probability of each class in target for every X and then select the probability that the passenger survived 
RF_prob = RF_pipeline.predict_proba(X_validate)[:,1]
XGB_prob = XGB_pipeline.predict_proba(X_validate)[:,1]

final_prob = (
    (0.40 * LR_prob) + (0.33 * RF_prob) + (0.27 * XGB_prob)
)

final_y_pred = (final_prob > 0.5).astype(int)

print(accuracy_score(y_validate,final_y_pred)) #0.8268
print(f1_score(y_validate,final_y_pred)) #0.739


#--------------
'''Submission CSV1 iter 5'''
#--------------
test_path = "data/test.csv"
test_df = pd.read_csv(test_path)

X_test = feature_creation(test_df)
print(X_test.head())

#test
LR_pipeline.fit(X,y)
RF_pipeline.fit(X,y)
XGB_pipeline.fit(X,y)

LR_prob = LR_pipeline.predict_proba(X_test)[:,1] 
RF_prob = RF_pipeline.predict_proba(X_test)[:,1]
XGB_prob = XGB_pipeline.predict_proba(X_test)[:,1]

y_test_prob = (
    (0.40 * LR_prob) + (0.33 * RF_prob) + (0.27 * XGB_prob)
)

y_test_pred = (y_test_prob > 0.5).astype(int)

submission = pd.DataFrame({
    'PassengerId' : test_df['PassengerId'],
    'Survived' : y_test_pred
})

submission.to_csv('submission_iter5.csv',index=False) #0.78708 same score as randomforest alone
