'''
Import data function.
@AugustSemrau
'''

import pandas as pd


def dataLoader(test=False, ageNAN="median"):

    # Loading either training or test set
    if not test:
        path = "./data/train.csv"
    elif test:
        path = "./data/test.csv.csv"
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

    return df



data = dataLoader()
print(data)
print(len(data))

print(data.isna().sum())

# print(sum(data["Embarked"] == "S"))
# print(sum(data["Embarked"] == "Q"))
# print(sum(data["Embarked"] == "C"))