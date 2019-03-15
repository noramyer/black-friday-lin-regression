from __future__ import print_function

import os
import subprocess

import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', -1)

def get_data():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    # print(dir_path)
    file_path=dir_path+"/../BlackFriday.csv"
    if os.path.exists(file_path):
        print("Reading csv file")
        df = pd.read_csv(file_path, index_col=0)
    else:
        exit("BlackFriday.csv not found, exiting...")
    return df

def onehot_encode(df, target_column):
    """
        Self explanatory, keeps the purchase column at the end.
    :param df:
    :param target_column:
    :return:
    """
    encoded=pd.get_dummies(df[target_column], prefix=target_column,  prefix_sep='=')
    replaced = pd.concat([df,encoded],axis=1).drop([target_column],axis=1)
    return replaced.reindex(columns=(list([a for a in replaced.columns if a != 'Purchase']) + ['Purchase'] ))

def label_encode(df, target_column):
    """
        Replace <target_column> column in df with integers.
    """
    df_mod = df.copy()
    targets = sorted(df_mod[target_column].unique())
    map_to_int = {name: n for n, name in enumerate(targets)}
    df_mod[target_column] = df_mod[target_column].replace(map_to_int)

    return df_mod


def preprocess(raw_df):
    """
        Ordinal data encoded using ordered-label encoding and categorical using one-hot.
        Missing data is replaced with median for now, can look at estimators if need be.
    :param raw_df:
    :return:
    """

    df=onehot_encode(raw_df, "Gender")
    df=onehot_encode(df, "City_Category")

    # maintains the order of age groups while labeling
    df=label_encode(df,"Age")
    # should try one-hot for this one as well
    df=label_encode(df,"Stay_In_Current_City_Years")

    df['Product_Category_1'].fillna(df['Product_Category_1'].median(), inplace=True)
    df['Product_Category_2'].fillna(df['Product_Category_2'].median(), inplace=True)
    df['Product_Category_3'].fillna(df['Product_Category_3'].median(), inplace=True)



    return df

def visualize_tree(tree, feature_names):
    """Create tree png using graphviz.

    Args
    ----
    tree -- scikit-learn DecsisionTree.
    feature_names -- list of feature names.
    """
    with open("dt1.dot", 'w') as f:
        export_graphviz(tree, out_file=f,
                        feature_names=feature_names)

    command = ["dot", "-Tpng", "dt1.dot", "-o", "dt.png"]
    try:
        subprocess.check_call(command)
    except:
        exit("Could not run dot, ie graphviz, to "
             "produce visualization")

if __name__ == '__main__':
    raw_df=get_data()

    df=preprocess(raw_df)
    features=list(df.columns[1:-1])
    X=df[features]
    y=df["Purchase"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.33)


    dt = DecisionTreeClassifier(min_samples_split=20, random_state=99)
    dt.fit(X_train, y_train)
    visualize_tree(dt, features)
