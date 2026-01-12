#Iteration 2 of titanic survival prediction 
#goals: increase prediction accuracy through feature engineering creating new doamin specific features
#comapre the results with the baseline restuls established in iteration 1 
#also to try a model that could catch non linear interactions in the data (RF)

import pandas as pd 
from collections import Counter
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score,StratifiedKFold


train_path = "C:/Users/Adwait Tagalpallewar/Desktop/datasets/titanic/train.csv"
train_df = pd.read_csv(train_path)

print(train_df.head())
print(train_df.columns)
print(train_df.isna().sum()) #Age=177,Cabin=687,Embarked=2; the column cabin has more than half of the values null so imputing it can be dangerous 


#----------------
'''Featuer Engineering'''
#----------------
#automating this process by wrapping it in a function 
def feature_creation(df):

    df['Pclass'] = df['Pclass'].astype('category')

    #After observing the data, the values in column Name have Title = ['Mr.','Master.','Miss.','Mrs.'] 
    df['Title'] = df['Name'].str.split(',').str[1].str.split('.').str[0].str.strip()
    # print(df['Title'].to_string())
    # counts = Counter(df['Title'])
    # print(counts.keys())
    # print(counts) #Some titles are rarely seen so replacing them with 'Rare' and some are miss spelled so they will be replaced too by corrosponding correct Title
    df['Title'] = df['Title'].replace(
        ['Don', 'Rev', 'Dr',  'Major', 'Lady', 'Sir', 'Col', 
        'Capt', 'the Countess', 'Jonkheer'],
        'Rare'
        )

    df['Title'] = df['Title'].replace({
        'Mlle': 'Miss',
        'Ms': 'Miss',
        'Mme': 'Mrs'
    })

    #Parch + SibSp + 1 can be written as family_size also the family_size == 1 can show that the passenger was travelling alone 
    df['Family_size'] = df['Parch'] + df['SibSp'] + 1
    df['Travelling_alone'] = (df['Family_size'] == 1).astype(int)

    #removing useless redundent features 
    useless_cols = ['PassengerId','Name','Ticket','Cabin']
    df = df.drop(columns = useless_cols)

    return df

df = feature_creation(train_df)

#seperating target and features 
X = df.drop('Survived',axis=1)
y = df['Survived']


#-------------------
'''Preprocessing'''
#-------------------
num_cols = X.select_dtypes(include='float64').columns
cat_cols = ['Pclass','Sex','Embarked','Travelling_alone','Title']
count_cols = ['SibSp','Parch','Family_size']

num_pipe = Pipeline([
    ('impute',SimpleImputer(strategy='median')),
    ('scaler',StandardScaler())
])
cat_pipe_lr = Pipeline([
    ('impute',SimpleImputer(strategy='most_frequent')),
    ('encode',OneHotEncoder(drop='first',handle_unknown='ignore')) 
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


#---------------
'''Using the same tuned LR model from last iteration to see if theres any gain'''
#---------------
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

cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)

score = cross_val_score(
    LR_pipeline,
    X,
    y,
    cv=cv,
    scoring='accuracy',
    n_jobs=-1
)
print(score)
print('CV mean_score,std across 5 folds: ', score.mean(), score.std()) # 0.8114619295712762  0.01850814460080681
'''Not a significant gain but a positive gain'''


#-------------------
'''Submission CSV-1 iter(2)'''
#-------------------
test_path = "C:/Users/Adwait Tagalpallewar/Desktop/datasets/titanic/test.csv"
test_df = pd.read_csv(test_path)

X_test = feature_creation(test_df)

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

submission.to_csv("Submission1_iter(2).csv",index=False) #0.77990; rank =4317/13804

'''Reached the ceiling of logistic regression on the given set of features, for further gain
we'll have to introduce interaction features since the model isnt able to capture non linear interactions between features 
It will be time consuming to produce new features, so iteration 3 will cover comparision with tree based models which 
capture non linear interaction '''