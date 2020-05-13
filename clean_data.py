import warnings
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

pd.options.mode.chained_assignment = None


def tidy_data(data):

    data = data.copy()
    if "MATURITY_DATE" in data:
        data["MATURITY_YEAR"] = data.MATURITY_DATE//100
        data["MATURITY_MON"] = data.MATURITY_DATE % 100
        data.drop("MATURITY_DATE", inplace=True, axis=1)
    if "FIRST_PAYMENT_DATE" in data:
        data["FIRST_PAYMENT_YEAR"] = data.FIRST_PAYMENT_DATE//100
        data["FIRST_PAYMENT_MON"] = data.FIRST_PAYMENT_DATE % 100
        data.drop("FIRST_PAYMENT_DATE", inplace=True, axis=1)

    data["DELINQUENT"] = data["DELINQUENT"] * 1
    return data


def clean_data(train_data, test_data):
    fill_numeric(train_data, test_data)
    fill_cat(train_data, test_data)


def get_train_test_split_for_ml(data, split_year, return_encoder=False, pre_process=False):
    """
    Get a train and test set based on the spilt year. The function does not modify the dataframe
    and returns a copy.


    Arguments:

    - data {pd.Dataframe} -- Datafame with all of the data, that you want to split

    - split_year {int} -- year that you want to split on

    Keyword Arguments:

    - return_encoder {bool} -- return the One hot encoder that was trained on the train
    and test (default: {False})

    - pre_process {bool} -- clean and fill the data (default: {False})

    Returns:

    - [tuple] -- return a (train dataframe, test dataframe, encoder or None) tuple.
    """
    data = data.copy()
    # create train test spilt
    train_data = data.query(f"FIRST_PAYMENT_YEAR<= {split_year}")
    test_data = data.query(f"FIRST_PAYMENT_YEAR >  {split_year}")

    # fill missing data based on train data for train and test
    if pre_process:
        clean_data(train_data, test_data)

    # separate train numerical and categorical
    filled_numeric_train_data = (train_data.select_dtypes(
        ["number", "bool"])).apply(pd.to_numeric)
    filled_cat_train_data = train_data.select_dtypes(["category", "object"])

    # create a one encoder based on train data
    encoder = OneHotEncoder(handle_unknown='ignore')
    encoder.fit(filled_cat_train_data)

    # convert all categorical data to dummies for train
    train_dummy_vars = encoder.transform(filled_cat_train_data).toarray()
    train_dummy_df = pd.DataFrame(
        train_dummy_vars, index=filled_numeric_train_data.index, columns=encoder.get_feature_names())
    joined_train = filled_numeric_train_data.join(train_dummy_df)

    # separate test numerical and categorical
    filled_numeric_test_data = (test_data.select_dtypes(
        ["number", "bool"])).apply(pd.to_numeric)
    filled_cat_test_data = test_data.select_dtypes(["category", "object"])

    # convert all categorical data to dummies for test
    test_dummy_vars = encoder.transform(filled_cat_test_data).toarray()
    test_dummy_df = pd.DataFrame(
        test_dummy_vars, index=filled_numeric_test_data.index, columns=encoder.get_feature_names())
    joined_test = filled_numeric_test_data.join(test_dummy_df)

    return (joined_train, joined_test, encoder if return_encoder else None)


def fill_cat(train, test):

    # fill NUMBER_OF_BORROWERS
    fill = train.NUMBER_OF_BORROWERS.mode()[0]
    train["Missing_NUMBER_OF_BORROWERS"] = train.NUMBER_OF_BORROWERS.isna()
    train.NUMBER_OF_BORROWERS.fillna(fill, inplace=True)
    if test is not None:
        test["Missing_NUMBER_OF_BORROWERS"] = test.NUMBER_OF_BORROWERS.isna()
        test.NUMBER_OF_BORROWERS.fillna(fill, inplace=True)

    # fill FIRST_TIME_HOMEBUYER_FLAG
    fill = train.FIRST_TIME_HOMEBUYER_FLAG.mode()[0]
    train["Missing_FIRST_TIME_HOMEBUYER_FLAG"] = train.FIRST_TIME_HOMEBUYER_FLAG.isna()
    train.FIRST_TIME_HOMEBUYER_FLAG.fillna(fill, inplace=True)
    if test is not None:
        test["Missing_FIRST_TIME_HOMEBUYER_FLAG"] = test.FIRST_TIME_HOMEBUYER_FLAG.isna()
        test.FIRST_TIME_HOMEBUYER_FLAG.fillna(fill, inplace=True)

    # fill METROPOLITAN_STATISTICAL_AREA
    fill = train.METROPOLITAN_STATISTICAL_AREA.mode()[0]
    train["Missing_METROPOLITAN_STATISTICAL_AREA"] = train.METROPOLITAN_STATISTICAL_AREA.isna()
    train.METROPOLITAN_STATISTICAL_AREA.fillna(fill, inplace=True)
    if test is not None:
        test["Missing_METROPOLITAN_STATISTICAL_AREA"] = test.METROPOLITAN_STATISTICAL_AREA.isna()
        test.METROPOLITAN_STATISTICAL_AREA.fillna(fill, inplace=True)

    # fill PREPAYMENT_PENALTY_MORTGAGE_FLAG
    fill = train.PREPAYMENT_PENALTY_MORTGAGE_FLAG.mode()[0]
    train["Missing_PREPAYMENT_PENALTY_MORTGAGE_FLAG"] = train.PREPAYMENT_PENALTY_MORTGAGE_FLAG.isna()
    train.PREPAYMENT_PENALTY_MORTGAGE_FLAG.fillna(fill, inplace=True)
    if test is not None:
        test["Missing_PREPAYMENT_PENALTY_MORTGAGE_FLAG"] = test.PREPAYMENT_PENALTY_MORTGAGE_FLAG.isna()
        test.PREPAYMENT_PENALTY_MORTGAGE_FLAG.fillna(fill, inplace=True)

    # fill POSTAL_CODE
    fill = train.POSTAL_CODE.mode()[0]
    train["Missing_POSTAL_CODE"] = train.POSTAL_CODE.isna()
    train.POSTAL_CODE.fillna(fill, inplace=True)
    if test is not None:
        test["Missing_POSTAL_CODE"] = test.POSTAL_CODE.isna()
        test.POSTAL_CODE.fillna(fill, inplace=True)

    # fill PROPERTY_TYPE
    fill = train.PROPERTY_TYPE.mode()[0]
    train["Missing_PROPERTY_TYPE"] = train.PROPERTY_TYPE.isna()
    train.PROPERTY_TYPE.fillna(fill, inplace=True)
    if test is not None:
        test["Missing_PROPERTY_TYPE"] = test.PROPERTY_TYPE.isna()
        test.PROPERTY_TYPE.fillna(fill, inplace=True)

    # fill PROPERTY_TYPE
    fill = train.NUMBER_OF_UNITS.mode()[0]
    train["Missing_NUMBER_OF_UNITS"] = train.NUMBER_OF_UNITS.isna()
    train.NUMBER_OF_UNITS.fillna(fill, inplace=True)
    if test is not None:
        test["Missing_NUMBER_OF_UNITS"] = test.NUMBER_OF_UNITS.isna()
        test.NUMBER_OF_UNITS.fillna(fill, inplace=True)


def fill_numeric(train, test):

    # fill CREDIT_SCORE
    fill = train.CREDIT_SCORE.median()
    train["Missing_CREDIT_SCORE"] = train.CREDIT_SCORE.isna()
    train.CREDIT_SCORE.fillna(fill, inplace=True)
    if test is not None:
        test["Missing_CREDIT_SCORE"] = test.CREDIT_SCORE.isna()
        test.CREDIT_SCORE.fillna(fill, inplace=True)

    # fill MORTGAGE_INSURANCE_PERCENTAGE
    fill = train.MORTGAGE_INSURANCE_PERCENTAGE.median()
    train["Missing_MORTGAGE_INSURANCE_PERCENTAGE"] = train.MORTGAGE_INSURANCE_PERCENTAGE.isna()
    train.MORTGAGE_INSURANCE_PERCENTAGE.fillna(fill, inplace=True)
    if test is not None:
        test["Missing_MORTGAGE_INSURANCE_PERCENTAGE"] = test.MORTGAGE_INSURANCE_PERCENTAGE.isna()
        test.MORTGAGE_INSURANCE_PERCENTAGE.fillna(fill, inplace=True)

    # fill ORIGINAL_COMBINED_LOAN_TO_VALUE
    fill = train.ORIGINAL_COMBINED_LOAN_TO_VALUE.mean()
    train["Missing_ORIGINAL_COMBINED_LOAN_TO_VALUE"] = train.ORIGINAL_COMBINED_LOAN_TO_VALUE.isna()
    train.ORIGINAL_COMBINED_LOAN_TO_VALUE.fillna(fill, inplace=True)
    if test is not None:
        test["Missing_ORIGINAL_COMBINED_LOAN_TO_VALUE"] = test.ORIGINAL_COMBINED_LOAN_TO_VALUE.isna()
        test.ORIGINAL_COMBINED_LOAN_TO_VALUE.fillna(fill, inplace=True)

    # fill ORIGINAL_DEBT_TO_INCOME_RATIO
    fill = train.ORIGINAL_DEBT_TO_INCOME_RATIO.mean()
    train["Missing_ORIGINAL_DEBT_TO_INCOME_RATIO"] = train.ORIGINAL_DEBT_TO_INCOME_RATIO.isna()
    train.ORIGINAL_DEBT_TO_INCOME_RATIO.fillna(fill, inplace=True)
    if test is not None:
        test["Missing_ORIGINAL_DEBT_TO_INCOME_RATIO"] = test.ORIGINAL_DEBT_TO_INCOME_RATIO.isna()
        test.ORIGINAL_DEBT_TO_INCOME_RATIO.fillna(fill, inplace=True)

    # fill ORIGINAL_LOAN_TO_VALUE
    fill = train.ORIGINAL_LOAN_TO_VALUE.mean()
    train["Missing_ORIGINAL_LOAN_TO_VALUE"] = train.ORIGINAL_LOAN_TO_VALUE.isna()
    train.ORIGINAL_LOAN_TO_VALUE.fillna(fill, inplace=True)
    if test is not None:
        test["Missing_ORIGINAL_LOAN_TO_VALUE"] = test.ORIGINAL_LOAN_TO_VALUE.isna()
        test.ORIGINAL_LOAN_TO_VALUE.fillna(fill, inplace=True)
