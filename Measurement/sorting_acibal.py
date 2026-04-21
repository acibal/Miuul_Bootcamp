import pandas as pd
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("datasets/product_sorting.csv")

def preprocess_sorting_data(dataframe):
    dataframe = dataframe.rename(columns={'commment_count': 'comment_count'})
    dataframe["purchase_count_scaled"] = MinMaxScaler(feature_range=(1, 5)).fit(dataframe[["purchase_count"]]).transform(dataframe[["purchase_count"]])
    dataframe["comment_count_scaled"] = MinMaxScaler(feature_range=(1, 5)).fit(dataframe[["comment_count"]]).transform(dataframe[["comment_count"]])
    return dataframe

df = preprocess_sorting_data(df)

def weighted_sorting_score(dataframe, w1=32, w2=26, w3=42):
    return (dataframe["comment_count_scaled"] * w1 / 100 +
            dataframe["purchase_count_scaled"] * w2 / 100 +
            dataframe["rating"] * w3 / 100)

df["weighted_sorting_score"] = weighted_sorting_score(df)

def bayesian_average_rating(n, confidence=0.95):
    if sum(n) == 0:
        return 0
    K = len(n)
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    N = sum(n)
    first_part = 0.0
    second_part = 0.0
    for k, n_k in enumerate(n):
        first_part += (k + 1) * (n[k] + 1) / (N + K)
        second_part += (k + 1) * (k + 1) * (n[k] + 1) / (N + K)
    score = first_part - z * math.sqrt((second_part - first_part * first_part) / (N + K + 1))
    return score

df["bar_score"] = df.apply(lambda x: bayesian_average_rating(x[["1_point", "2_point", "3_point", "4_point", "5_point"]]), axis=1)

def hybrid_sorting_score(dataframe, bar_w=60, wss_w=40):
    bar_score = dataframe.apply(lambda x: bayesian_average_rating(x[["1_point",
                                                                     "2_point",
                                                                     "3_point",
                                                                     "4_point",
                                                                     "5_point"]]), axis=1)
    wss_score = weighted_sorting_score(dataframe)

    return bar_score*bar_w/100 + wss_score*wss_w/100

df["hybrid_sorting_score"] = hybrid_sorting_score(df)

df.head()