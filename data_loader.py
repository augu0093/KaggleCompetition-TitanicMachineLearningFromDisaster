'''
Import data function.
@AugustSemrau
'''

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def dataLoader(test=False, optimize_set=False, ageNAN="median"):

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
        age_replacement = df_all["Age"].median()
    elif ageNAN == "mean":
        age_replacement = df_all["Age"].mean()
    df_all["Age"].fillna(age_replacement, inplace=True)
    # Secondly replacing NAN values for Cabin with "BD" as these are assumed to traveling Below Deck
    df_all["Cabin"].fillna("BD", inplace=True)
    # Thirdly replacing NAN values for Embarked with most frequent which is Southampton "S"
    df_all["Embarked"].fillna("S", inplace=True)
    # Finally replacing NAN value for Fare with mean
    df_all['Fare'].fillna(df_all['Fare'].mean(), inplace=True)

    # Will also round Fare cost to natural number
    df_all = df_all.round({'Fare': 0})

    # Now all categorical variables need to be label encoded
    # Get list of categorical variables
    s = (df_all.dtypes == 'object')
    object_cols = list(s[s].index)
    # Encode all categorical variable columns
    label_df = df_all.copy()
    labelEncoder = LabelEncoder()
    for col in object_cols:
        label_df[col] = labelEncoder.fit_transform(df_all[col])
    df_all = label_df

    # Now the two data sets are separated again
    X = df_all[:num_train_rows]
    df_test = df_all[-num_test_rows:]


    # The test set is returned
    if test:
        return df_test

    # Training set is returned, either split or not
    elif not test:
        if optimize_set:
            return train_test_split(X, y, test_size=0.2, random_state=0)
        else:
            return X, y





def csvSaver(predictions, type):

    # Get passenger ID's from test set
    test_path = "./data/test.csv"
    df_test = pd.read_csv(test_path)
    passenger_id = df_test['PassengerId']

    df = pd.DataFrame(list(zip(passenger_id, predictions)), columns=['PassengerId', 'Survived'], index=None)
    # predictions_csv = df.to_csv(index=False)
    output_filename = 'predictions/predictions_{}.csv'.format(type)
    return df.to_csv(output_filename, index=False)


# print(csvSaver(type="Logistic", predictions=[0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0]))

# X, y = dataLoader()
# print(X)
# print(y)
# print(X.dtypes)
#
#
# testData = dataLoader(test=True)
#
# print(testData.isna().sum())
# print(testData)