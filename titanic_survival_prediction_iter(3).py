#This is iteration 3 of titanic survival prediction 
#Iteration 3 will mostly focus on learning from non-linear relationships between features which our logistic regression model wasnt able to capture 

import pandas as pd 
import matplotlib.pyplot as plt 
from feature_creation_titanic_survival import feature_creation
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
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

#-------------------
'''Preprocessing'''
#-------------------
num_cols = X.select_dtypes(include='float64').columns
cat_cols = ['Pclass','Sex','Embarked','Travelling_alone',]
count_cols = ['SibSp','Parch','Family_size']

num_pipe = Pipeline([
    ('impute',SimpleImputer(strategy='median')),
])
cat_pipe_rf = Pipeline([
    ('impute',SimpleImputer(strategy='most_frequent')),
    ('encode',OneHotEncoder(drop='first',handle_unknown='ignore')) 
])
count_pipe = Pipeline([
    ('impute',SimpleImputer(strategy='constant',fill_value=0))
])

preprocessor_RF = ColumnTransformer(
    transformers=[
        ('num',num_pipe,num_cols),
        ('cat',cat_pipe_rf,cat_cols),
        ('count',count_pipe,count_cols)
    ],
    remainder='drop'
)
#In next iteration if theres any, i'll try to automate this preprocessing uder a single function as well to save time and lines of code 


#----------------
'''RandomForest baseline'''
#----------------
RF_pipeline = Pipeline([
    ('prep',preprocessor_RF),
    ('model',RandomForestClassifier(
        n_estimators=300,
        n_jobs=-1,
        random_state=42,
        criterion='entropy'
    ))
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

score = cross_val_score(
    RF_pipeline,
    X,
    y,
    cv = cv,
    scoring='accuracy',
    n_jobs=-1
)

print(score)
print(score.mean(),score.std()) #0.8069487163392128  0.012242394504493466; similar to the CV results of baseline logistic regression 


#---------------
'''Hypertuning RandomForest'''
#---------------

#Tuning n_estimators  
values = [200,250,300,400,500]
for i in values:
    RF_pipeline = Pipeline([
        ('prep',preprocessor_RF),
        ('model',RandomForestClassifier(
            n_estimators=i,
            n_jobs=-1,
            random_state=42,
            criterion='entropy'
            ))
        ])
    
    score = cross_val_score(
        RF_pipeline,
        X,
        y,
        cv = cv,
        scoring='accuracy',
        n_jobs=-1
    )
    plt.scatter(score.mean(),score.std(),label=i)

plt.xlabel('mean')
plt.ylabel('std')
plt.legend()
plt.show() #for n_estimators =200:  0.8092021844203126 0.010667905626154212

'''Increasing the number of trees beyond 200 did not yield performance improvements, indicating that variance reduction had saturated 
and further increases provided diminishing returns.'''

#Tuning max_depth
values = range(5,16)
for i in values:
    RF_pipeline = Pipeline([
        ('prep',preprocessor_RF),
        ('model',RandomForestClassifier(
            n_estimators=200,
            n_jobs=-1,
            random_state=42,
            criterion='entropy',
            max_depth=i
            ))
        ])
    
    score = cross_val_score(
        RF_pipeline,
        X,
        y,
        cv = cv,
        scoring='accuracy',
        n_jobs=-1
    )
    print(f'for max_depth={i}: ',score.mean(),score.std())
    plt.scatter(score.mean(),score.std(),label=i)

plt.xlabel('mean')
plt.ylabel('std')
plt.legend()
plt.show() #for max_depth=8:  0.835019772770071 0.009066153473789296
'''Model performance improved substantially after increasing tree depth, peaking around depths 8–10,
 beyond which performance degraded due to overfitting. This indicates that the Random Forest successfully 
 captured non-linear interactions present in the data'''
'''Without a fixed depth, trees in the Random Forest learned inconsistent hierarchical patterns due to sampling randomness;
 constraining the depth aligned their decision structures, reducing effective underfitting'''


#-------------------
'''Submission CSV-1 iter(3)'''
#-------------------
test_path = "C:/Users/Adwait Tagalpallewar/Desktop/datasets/titanic/test.csv"
test_df = pd.read_csv(test_path)

X_test = feature_creation(test_df)
X_test = X_test.drop('Title',axis=1)

#model and testing
RF_pipeline = Pipeline([
        ('prep',preprocessor_RF),
        ('model',RandomForestClassifier(
            n_estimators=200,
            n_jobs=-1,
            random_state=42,
            criterion='entropy',
            max_depth=8,
            min_samples_leaf=2
            ))
        ])

RF_pipeline.fit(X,y)
y_pred = RF_pipeline.predict(X_test)

submission = pd.DataFrame({
    "PassengerId" : test_df['PassengerId'],
    "Survived" : y_pred
})

submission.to_csv("Submission1_iter(3).csv",index=False) #0.78708, heigher than that of iter(2) results

'''Title was a high-variance feature that RF (the final virdict is written in iter(4)) 
Iteration 4 will cover trial of xgboost to see if it can perform better'''