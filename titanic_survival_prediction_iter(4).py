#iteration 4 will cover usage of gradient boosting algorithm for better results 

import pandas as pd 
from feature_creation_titanic_survival import feature_creation
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold,cross_val_score


train_path = "C:/Users/Adwait Tagalpallewar/Desktop/datasets/titanic/train.csv"
train_df = pd.read_csv(train_path)

print(train_df.head())

df = feature_creation(train_df)
df = df.drop('Title',axis=1)
print(df.head().to_string())

#seperating target and features 
X = df.drop('Survived',axis=1)
y = df['Survived']

#----------------
'''Preprocessing'''
#----------------
num_cols = X.select_dtypes(include='float64').columns
cat_cols = ['Pclass','Sex','Embarked','Travelling_alone',]
count_cols = ['SibSp','Parch','Family_size']

num_pipe = Pipeline([
    ('impute',SimpleImputer(strategy='median')),
])
cat_pipe = Pipeline([
    ('impute',SimpleImputer(strategy='most_frequent')),
    ('encode',OneHotEncoder(drop='first',handle_unknown='ignore')) 
])
count_pipe = Pipeline([
    ('impute',SimpleImputer(strategy='constant',fill_value=0))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num',num_pipe,num_cols),
        ('cat',cat_pipe,cat_cols),
        ('count',count_pipe,count_cols)
    ],
    remainder='drop'
)

#---------------
'''Baseline XGBoostClassifier'''
#---------------
xgb_model = XGBClassifier(
    n_estimators = 300,
    max_depth = 2,
    learning_rate = 0.03,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)

XGB_pipeline = Pipeline([
    ('prep',preprocessor),
    ('model',xgb_model)
])

cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)

score = cross_val_score(
    XGB_pipeline,
    X,
    y,
    cv=cv,
    scoring='accuracy',
    n_jobs=-1
)
print(score)
print(score.mean(),score.std()) #0.8282656455966354 0.01330238920347726


'''Gradient boosting achieved the best bias–variance trade-off, outperforming both Logistic Regression and Random Forest 
in cross-validation by effectively capturing non-linear interactions while maintaining generalization.
Rise in Recall and Precision was also observed making the precision recall trade-off better than logistic regression and randomforest'''


#-------------------
'''Submission CSV-1 iter(4)'''
#-------------------
test_path = "C:/Users/Adwait Tagalpallewar/Desktop/datasets/titanic/test.csv"
test_df = pd.read_csv(test_path)

X_test = feature_creation(test_df)
X_test = X_test.drop('Title',axis=1)

#model and testing
Final_pipeline = Pipeline([
        ('prep',preprocessor),
        ('model',xgb_model)
        ])

Final_pipeline.fit(X,y)
y_pred = Final_pipeline.predict(X_test)

submission = pd.DataFrame({
    "PassengerId" : test_df['PassengerId'],
    "Survived" : y_pred
})

submission.to_csv("Submission1_iter(4).csv",index=False) #0.7751 

'''Although title-based features improved cross-validation scores, they consistently reduced test performance
 for tree-based models due to high variance and redundancy. Removing the feature improved generalization across both 
 Random Forest from iter(3) and XGBoost'''

'''Gradient boosting initially outperformed Random Forest in cross-validation, 
feature simplification reduced data complexity, allowing Random Forest to generalize better 
and achieve the highest test performance.'''
