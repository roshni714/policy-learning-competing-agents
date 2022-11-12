import pandas as pd


def generate_codebook():
    df = pd.read_excel("data/NELS_Variable_List.xlsx", skiprows=2)
    key = list(df["VARIABLE"])
    label = list(df["LABEL"])
    dic = dict(zip(key, label))
    return dic


def impute_values(df, to_replace_dic, replacement_vals_dic):
    for key in to_replace_dic:
        for val in to_replace_dic[key]:
            df[key].replace(
                to_replace=val, value=replacement_vals_dic[key], inplace=True
            )
