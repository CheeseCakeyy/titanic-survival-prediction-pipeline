#The dataset used belongs to a competition 'Titanic - Machine Learning from Disaster'
#Goal is to predict the survived passengers from the given features 
#Each row represents a passenger and there information 

import pandas as pd 
import matplotlib.pyplot as plt 
from collections import Counter
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.model_selection import train_test_split,StratifiedKFold,cross_val_score
from sklearn.pipeline import Pipeline 
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix


train_path = "C:/Users/Adwait Tagalpallewar/Desktop/datasets/titanic/train.csv"
df = pd.read_csv(train_path)

print(df.head())
print(df.columns)
df.info()
print(df.isna().sum()) #Age=177,Cabin=687,Embarked=2; the column cabin has more than half of the values null so imputing it can be dangerous 
print(df.describe().to_string())

df['Pclass'] = df['Pclass'].astype('category')

#class distribution of target
counts = Counter(df['Survived'])

plt.bar(counts.keys(),counts.values(),color='blue',width=0.2)
plt.xlabel('Survival')
plt.ylabel('Counts')
plt.show() #only 342 out of 891 passengers survived, its imbalanced 

#Feature distribution 
df.hist(figsize=(12,4))
plt.tight_layout()
plt.show()

useless_cols = ['PassengerId','Name','Cabin','Ticket']
df = df.drop(columns = useless_cols)

print(df.select_dtypes('number').corr()) #no strong linear relationships with target

#seperatig target and features 
X = df.drop('Survived',axis=1)
y = df['Survived']

#splitting data into train/validate spit
X_train,X_validate,y_train,y_validate = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)


#-------------------
'''Preprocessing'''
#-------------------
num_cols = X.select_dtypes(include='float64').columns
cat_cols = ['Pclass','Sex','Embarked']
count_cols = ['SibSp','Parch']

num_pipe = Pipeline([
    ('impute',SimpleImputer(strategy='median')),
    ('scaler',StandardScaler())
])
cat_pipe_lr = Pipeline([
    ('impute',SimpleImputer(strategy='most_frequent')),
    ('encode',OneHotEncoder(drop='first',handle_unknown='ignore')) #dropping first cuz it helps logistic regression to be mathematically well-posed
])
cat_pipe_rf = Pipeline([
    ('impute',SimpleImputer(strategy='most_frequent')),
    ('encode',OneHotEncoder(drop=None,handle_unknown='ignore')) #no drop first since dropping can sometimes slightly hurt RandomForests performance
])
count_pipe = Pipeline([
    ('impute',SimpleImputer(strategy='constant',fill_value=0))
])

preprocessor_LR = ColumnTransformer(
    transformers=[
        ('num',num_pipe,num_cols),
        ('cat',cat_pipe_lr,cat_cols),
        ('count',count_pipe,count_cols)
    ],
    remainder='drop'
)
preprocessor_RF = ColumnTransformer(
    transformers=[
        ('num',num_pipe,num_cols),
        ('cat',cat_pipe_rf,cat_cols),
        ('count',count_pipe,count_cols)
    ],
    remainder='drop'
)

#-------------------
'''Testing on 3 baseline Models'''
#-------------------
LR_pipeline = Pipeline([
    ('prep',preprocessor_LR),
    ('model',LogisticRegression(
        max_iter=250,
        random_state=42,
        n_jobs=-1
        ))
])

DT_pipeline = Pipeline([
    ('prep',preprocessor_RF),
    ('model',DecisionTreeClassifier(
        criterion='entropy',
        random_state=42
        ))
])

RF_pipeline = Pipeline([
    ('prep',preprocessor_RF),
    ('model',RandomForestClassifier(
        n_estimators=300,
        n_jobs=-1,
        random_state=42
        ))
])


#----------------
'''Cross-Validation'''
#----------------

cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)

score_lr = cross_val_score(
    LR_pipeline,
    X_train,
    y_train,
    cv=cv,
    scoring='accuracy',
    n_jobs=-1
)

score_dt = cross_val_score(
    DT_pipeline,
    X_train,
    y_train,
    cv=cv,
    scoring='accuracy',
    n_jobs=-1
)

score_rf = cross_val_score(
    RF_pipeline,
    X_train,
    y_train,
    cv=cv,
    scoring='accuracy',
    n_jobs=-1
)

print('LR CV mean,std: ', score_lr.mean(),' ',score_lr.std()) #0.8019797104304146   0.014154894890856475
print('DT CV mean,std: ', score_dt.mean(),' ',score_dt.std()) #0.7878656554712893   0.02010365123723594
print('RF CV mean,std: ', score_rf.mean(),' ',score_rf.std()) #0.7935388555106866   0.015780253665075163

'''as the cross validation results suggest, logistic regression performed best with avg accuracy of 80.19 and std of 0.01 across 5 folds
so it makes more sense to hypertune it over the other two before trying gradient boosting model'''

#---------------------
'''Hypertuning Logistic regression model''' 
#---------------------

#hypertuning penalty and C(multiplier)
values = ['l1','l2'] #l1(lasso): makes the useless features coff 0, and l2(ridge): make the useless feature coff small not 0

for i in values:
    for C in [0.01,0.1,1,10]:
        LR_pipeline = Pipeline([
        ('prep',preprocessor_LR),
        ('model',LogisticRegression(
            solver='liblinear', #A solver is the algorithm used to find the best model parameters
            max_iter=250,
            C=C, #C controls the strength of the panelty; total loss = log loss + (1/C) * penalty
            penalty=i, #regularization
            random_state=42,
            ))
        ])
        score = cross_val_score(
        LR_pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1
        )
        print(score.mean(),' ',score.std())
        plt.scatter(score.mean(),score.std(),label=(i,C))

plt.xlabel('mean')
plt.ylabel('std')
plt.legend()
plt.show() #l2 with C=0.1 increases the avg_accuracy from 80.1 to ~80.9 also decreases the std to half of baseline std 

# hypertuning class_weight (since this dataset is imbalanced)
# Class weighting can improve recall / F1 significantly
class_weights = [None,'balanced']
for i in class_weights:
        LR_pipeline = Pipeline([
        ('prep',preprocessor_LR),
        ('model',LogisticRegression(
            class_weight=i,
            solver='liblinear', 
            max_iter=250,
            C=0.1, 
            penalty='l2', 
            random_state=42,
            ))
        ])
        score = cross_val_score(
        LR_pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring='accuracy', #was also evaluated on f1 and recall 
        n_jobs=-1
        )
        print(score.mean(),' ',score.std())
        plt.scatter(score.mean(),score.std(),label=(i))

plt.xlabel('mean')
plt.ylabel('std')
plt.legend()
plt.show() 
'''changes in class_weight increased the recall for value= balanced but the accuracy went down by ~1.5% also no significant gain in f1
Conclusion for class_weigth: Class weighting does NOT make the model better — it makes it behave differently.'''

#hypertuning max_iter (to make sure model isnt underfit)
values = [200,300,500]

for i in values:
        LR_pipeline = Pipeline([
        ('prep',preprocessor_LR),
        ('model',LogisticRegression(
            class_weight=None,
            solver='liblinear', 
            max_iter=i,
            C=0.1, 
            penalty='l2', 
            random_state=42,
            ))
        ])
        score = cross_val_score(
        LR_pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1
        )
        print(score.mean(),' ',score.std())
        plt.scatter(score.mean(),score.std(),label=(i))

plt.xlabel('mean')
plt.ylabel('std')
plt.legend()
plt.show() #all values give same results therefore concluding the model doesnt remain an underfit 

'''Conclusion for hyperparameter tuning:
Once regularization and class weighting were optimized, further tuning showed no meaningful improvement in cross-validation performance,
signaling the natural performance limit of Logistic Regression for this feature set.'''


#------------
'''validating on validation set'''
#------------
#Tuned LR model
LR_pipeline = Pipeline([
        ('prep',preprocessor_LR),
        ('model',LogisticRegression(
            class_weight= None,
            solver='liblinear', 
            max_iter=250,
            C=0.1, 
            penalty='l2', 
            random_state=42,
            ))
        ])

LR_pipeline.fit(X_train,y_train)
y_pred = LR_pipeline.predict(X_validate)
print('accuracy_score of tuned model on validation set: ',accuracy_score(y_validate,y_pred), precision_score(y_validate,y_pred),recall_score(y_validate,y_pred)) 
print(confusion_matrix(y_validate,y_pred)) 

'''0.8044, minimal mistakes on validation set, though we were unable to find a perfect treadoff between precision and recall,
that was because the target classes were imbalanced so recall dropped'''
'''I tried changing the class_weight to balanced for validation set and came to the conclusion:
While using class_weight='balanced' improved recall and validation accuracy on a held-out split, cross-validation results did not show consistent improvement, 
indicating that the gain was split-dependent rather than a robust generalization improvement.'''


#concluding the first iteration here, applying the gained model on the test data and making the first submission csv
#-------------------
'''Submission CSV-1'''
#-------------------
test_path = "C:/Users/Adwait Tagalpallewar/Desktop/datasets/titanic/test.csv"
test_df = pd.read_csv(test_path)

print(test_df.head())
print(test_df.isna().sum())
test_df.info()

test_df['Pclass'] = test_df['Pclass'].astype('category')
X_test = test_df.drop(columns = useless_cols)

#model and testing
LR_pipeline = Pipeline([
        ('prep',preprocessor_LR),
        ('model',LogisticRegression(
            class_weight= None,
            solver='liblinear', 
            max_iter=250,
            C=0.1, 
            penalty='l2', 
            random_state=42,
            ))
        ])

LR_pipeline.fit(X,y)
y_pred = LR_pipeline.predict(X_test)

submission = pd.DataFrame({
    "PassengerId" : test_df['PassengerId'],
    "Survived" : y_pred
})

submission.to_csv("Submission1_iter(1).csv",index=False) #0.7511;rank=12000/13804; got a baseline to work on 

'''Although cross-validation accuracy was approximately 80%, the initial Kaggle submission achieved 75% accuracy. 
This gap is expected due to distributional differences between the training data and Kaggle’s hidden test set, 
as well as the absence of domain-specific feature engineering such as title extraction and family-based features.'''

'''Iteration 2 will cover domain specific feature extraction and we'll compare a tunedd nonlinear model to see if that gets us anywhere'''