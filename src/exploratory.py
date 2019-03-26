#This file is meant to collect general information about the data set

import os
import subprocess
import pandas as pd

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from matplotlib.pyplot import *

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', -1)

def get_data():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    # print(dir_path)
    file_path=dir_path+"/../BlackFriday.csv"
    if os.path.exists(file_path):
        print("Reading csv file...")
        df = pd.read_csv(file_path, index_col=0)
    else:
        exit("BlackFriday.csv not found, exiting...")
    return df

if __name__ == '__main__':
    raw_df=get_data()
    num = raw_df.shape[0]
    print("Processing dataframe for information... \n")
    print("Number of rows:" +str(num))
    print("Number of columns:" +str(raw_df.shape[1]))
    print("Columns: "+ str(list(raw_df.columns.values)))
    print("\n")
    print("Columns containing na values?: \n" + str(raw_df.isna().any()))
    print("\n")

    print("Average age: " + str(raw_df["Age"].mode()))
    female_count = len(raw_df[raw_df["Gender"]=='F'])
    male_count = len(raw_df[raw_df["Gender"]=='M'])
    print("Count of females: " + str(female_count))
    print("Count of males: " + str(male_count))
    print("Percent female: " + str(float(female_count)/num))
    print("Percent male: " + str(float(male_count)/num))

    print("Mode purchase amount in cents: " + str(raw_df["Purchase"].mode()))
    print("Average purchase amount in cents: " + str(raw_df["Purchase"].mean()))
    print("Average purchase amount by gender: " + str(raw_df.groupby('Gender')['Purchase'].mean()))
    print("\n")

    print("Average purchase amount by age: " + str(raw_df.groupby('Age')['Purchase'].mean()))
    print("\n")

    print("Average purchase amount by occupation: " + str(raw_df.groupby('Occupation')['Purchase'].mean()))
    print("\n")

    print("Average purchase amount by city category: " + str(raw_df.groupby('City_Category')['Purchase'].mean()))
    print("\n")

    print("How many unique items for each category?")
    for col in raw_df.columns:
        print('{} unique element: {}'.format(col,raw_df[col].nunique()))
