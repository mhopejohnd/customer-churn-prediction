# preprocess - prepare for modeling
# clean 
''' remove identifier columns,
    convert categorical variables to numerical, 
    handle missing values    '''

# import necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

def csv_to_dataframe(file_path: str) -> pd.DataFrame:
    """ Read a CSV file into a DataFrame """
    return pd.read_csv(file_path)

def preprocess_data(csv_file_path: str):
    """ Split the data into training and testing sets and preprocess it for modeling """

    df = csv_to_dataframe(csv_file_path)           # read the CSV file into a DataFrame
    df = df.copy()                                 # create a copy of the DataFrame to avoid modifying the original

    # remove identifier columns
    df = df.drop(columns=['customerID'], inplace=False)  

    # define target variable
    y = df['Churn'].map({'Yes': 1, 'No': 0})
    X = df.drop(columns=['Churn'], errors='ignore')

    if y.isna().any():
        raise ValueError("Target variable 'Churn' contains missing values.")


    #convert TotalCharges to numerical
    if "TotalCharges" in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')


    return df

# remove identifier columns
# convert categorical variables to numerical
# handle missing values 