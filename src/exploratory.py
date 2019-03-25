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

    print("Average age: " + str(raw_df["Age"].mode()))
    female_count = len(raw_df[raw_df["Gender"]=='F'])
    male_count = len(raw_df[raw_df["Gender"]=='M'])
    print("Count of females: " + str(female_count))
    print("Count of males: " + str(male_count))
    print("Percent female: " + str(float(female_count)/num))
    print("Percent male: " + str(float(male_count)/num))

    print("Average purchase amount in cents: " + str(raw_df["Purchase"].mode()))
