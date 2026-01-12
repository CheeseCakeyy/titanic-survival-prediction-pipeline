#this is the feature creation function from iteration 2 
#had to write it in a new file for clean and safe imports in newer versions 


def feature_creation(df):

    df['Pclass'] = df['Pclass'].astype('category')

    df['Title'] = df['Name'].str.split(',').str[1].str.split('.').str[0].str.strip()
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

    df['Family_size'] = df['Parch'] + df['SibSp'] + 1
    df['Travelling_alone'] = (df['Family_size'] == 1).astype(int)

    useless_cols = ['PassengerId','Name','Ticket','Cabin']
    df = df.drop(columns = useless_cols)

    return df
