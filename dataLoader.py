'''
Import data function.
@AugustSemrau
'''

import pandas as pd


def dataLoader(test=False, return_X_y=True, ageNAN="median"):

    # Loading either training or test set
    if not test:
        path = "./data/train.csv"
    elif test:
        path = "./data/test.csv"
    df = pd.read_csv(path)

    # Data Cleaning
    # First replacing NAN values for age
    if ageNAN == "median":
        ageReplacement = df["Age"].median()
    elif ageNAN == "mean":
        ageReplacement = df["Age"].mean()
    df["Age"].fillna(ageReplacement, inplace=True)

    # Secondly replacing NAN values for Cabin with "BD" as these are assumed to traveling Below Deck
    df["Cabin"].fillna("BD", inplace=True)

    # Thirdly replacing NAN values for Embarked with most frequent which is Southampton "S"
    df["Embarked"].fillna("S", inplace=True)

    # If test data, return dataframe now
    if test:
        return df

    # Else if training data, return data and target values seperately
    elif not test:
        if return_X_y:
            # First rearange so Survived comes last
            X = df[[col for col in df if col not in ['Survived']]]
            y = df[['Survived']]
            return X, y


X, y = dataLoader()
print(X.head())
print(y.head())



testData = dataLoader(test=True)

print(testData.isna().sum())