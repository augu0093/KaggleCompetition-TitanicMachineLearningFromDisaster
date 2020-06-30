'''
Import data function.
@AugustSemrau
'''

import pandas as pd
from sklearn.preprocessing import LabelEncoder

def dataLoader(test=False, return_X_y=True, ageNAN="median"):

    # Loading both training and test set

    train_path = "./data/train.csv"
    test_path = "./data/test.csv"
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    num_train_rows = df_train.shape[0]
    num_test_rows = df_test.shape[0]

    # First separate Survived label from training data
    X = df_train[[col for col in df_train if col not in ['Survived']]]
    y = df_train[['Survived']]

    # For preprocessing, both sets are combined
    df_all = pd.concat([X, df_test])

    # Data Cleaning
    # Removing name column
    df_all = df_all.drop(columns=['Name'])
    # Fixing NAN's
    # First replacing NAN values for age
    if ageNAN == "median":
        ageReplacement = df_all["Age"].median()
    elif ageNAN == "mean":
        ageReplacement = df_all["Age"].mean()
    df_all["Age"].fillna(ageReplacement, inplace=True)
    # Secondly replacing NAN values for Cabin with "BD" as these are assumed to traveling Below Deck
    df_all["Cabin"].fillna("BD", inplace=True)
    # Thirdly replacing NAN values for Embarked with most frequent which is Southampton "S"
    df_all["Embarked"].fillna("S", inplace=True)
    # Finally replacing NAN value for Fare with mean
    df_all['Fare'].fillna(df_all['Fare'].mean(), inplace=True)

    # Now all categorical variables need to be label encoded
    # Get list of categorical variables
    s = (df_all.dtypes == 'object')
    objectCols = list(s[s].index)
    # Encode all categorical variable columns
    label_df = df_all.copy()
    labelEncoder = LabelEncoder()
    for col in objectCols:
        label_df[col] = labelEncoder.fit_transform(df_all[col])
    df_all = label_df

    # Now the two datasets are separated again
    X = df_all[:num_train_rows]
    df_test = df_all[-num_test_rows:]


    # Either the test set is returned
    if test:
        return df_test

    # Else training set is, return data and target values separately
    elif not test:
        if return_X_y:
            return X, y





X, y = dataLoader()
print(X)
print(y)
print(X.dtypes)


testData = dataLoader(test=True)

print(testData.isna().sum())
print(testData)