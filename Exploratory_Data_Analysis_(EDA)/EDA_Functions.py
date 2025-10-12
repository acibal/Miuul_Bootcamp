
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import datetime as dt
import warnings
warnings.simplefilter(action="ignore")

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
df = sns.load_dataset("titanic")
df.head()

##############################################################
######################   Ön Bilgi   ##########################
##############################################################

def check_df(dataframe, head=5, tail=5):
    print("#################### Shape ####################")
    print(dataframe.shape)
    print("#################### Types ####################")
    print(dataframe.dtypes)
    print("#################### Head ####################")
    print(dataframe.head(head))
    print("#################### Tail ####################")
    print(dataframe.tail(tail))
    print("#################### NA ####################")
    print(dataframe.isnull().sum())
    print("#################### Quantiles ####################")
    print(dataframe.describe([0.10, 0.25, 0.50, 0.75, 0.90]).T)

check_df(df)


##############################################################
######################   Outliers   ##########################
##############################################################

def outlier_thresholds(dataframe, col_name):
    quartile1 = dataframe[col_name].quantile(0.05)
    quartile3 = dataframe[col_name].quantile(0.95)
    interquartile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquartile_range
    low_limit = quartile1 - 1.5 * interquartile_range
    return low_limit, up_limit


def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[((dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit))].shape[0] > 0:
        return True
    else:
        return False


def grab_outlier(dataframe, col_name, index=False):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[((dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit))].shape[0] > 5:
        print(dataframe[((dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit))].head(10))
    else:
        print(dataframe[((dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit))])
    if index:
        outlier_index = dataframe[((dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit))].index
        return outlier_index


def remove_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    df_without_outlier = dataframe[~((dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit))]
    return df_without_outlier


def replace_with_thresholds(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    dataframe.loc[(dataframe[col_name] < low_limit), col_name] = low_limit
    dataframe.loc[(dataframe[col_name] > up_limit), col_name] = up_limit


# LOF (Local Outlier Factor) (notnull & num_cols) (scores 1> or -1<)

clf = LocalOutlierFactor(n_neighbors=20)

clf.fit_predict(dataframe[num_cols])

df_scores = clf.negative_outlier_factor_

scores = pd.DataFrame(np.sort(df_scores))
scores.plot(stacked=True, xlim=[0, 20], style='.-')
plt.show()

th = np.sort(df_scores)[3] # selecting from elbow
dataframe[df_scores < th] # outliers **
# ** çok farklı bir kullanım. içeriye numpy array bir true-false yaptık ve o gözükmeyen indexlerden seçim yaptı.
dataframe[df_scores < th].shape # shape or outliers
dataframe.describe().T # finding out why they are outliers

dataframe.drop(axis=0, labels=dataframe[df_scores < th].index, inplace=True)


##############################################################
###################   Missing Values   #######################
##############################################################


def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=["n_miss", "ratio"])
    print(missing_df, end="\n")
    if na_name:
        return na_columns


# examining missing values
msno.bar(dataframe)
plt.show()

msno.matrix(dataframe)
plt.show()

msno.heatmap(dataframe)
plt.show()


# examining missing values
def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=["n_miss", "ratio"])
    print(missing_df, end="\n")
    if na_name:
        return na_columns

na_columns = missing_values_table(dataframe)

def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()
    for col in na_columns:
        temp_df[col + "_NA_FLAG"] = np.where(temp_df[col].isnull(), 1, 0)
    na_flags = temp_df.loc[:, temp_df.columns.str.comtains('_NA_')].columns
    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")



# remove
dataframe.dropna(inplace=True)


# basic imputation
dataframe[col_name].fillna(dataframe[col_name].mean())
dataframe[col_name].fillna(dataframe[col_name].median))
dataframe[col_name].fillna(0))
dataframe[col_name].fillna("missing"))


def quick_missing_imp(dataframe, num_method="median", cat_length=20, target="target"):
    variables_with_na = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]  # Eksik değere sahip olan değişkenler listelenir

    temp_target = dataframe[target] # bu target değişkenini işlem sonunda eski haline getirmek tutuyoruz.

    print("# BEFORE")
    print(dataframe[variables_with_na].isnull().sum(), "\n\n")  # Uygulama öncesi değişkenlerin eksik değerlerinin sayısı

    # değişken object ve sınıf sayısı cat_lengthe eşit veya altındaysa boş değerleri mode ile doldur
    dataframe = dataframe.apply(lambda x: x.fillna(x.mode()[0]) if (str(x.dtypes) in ["category", "object", "bool"] and len(x.unique()) <= cat_length) else x, axis=0)

    # num_method mean ise tipi object olmayan değişkenlerin boş değerleri ortalama ile dolduruluyor
    if num_method == "mean":
        dataframe = dataframe.apply(lambda x: x.fillna(x.mean()) if x.dtypes in ["int", "float"] else x, axis=0)
    # num_method median ise tipi object olmayan değişkenlerin boş değerleri ortalama ile dolduruluyor
    elif num_method == "median":
        dataframe = dataframe.apply(lambda x: x.fillna(x.median()) if x.dtypes in ["int", "float"] else x, axis=0)

    dataframe[target] = temp_target # target değişkeninin orjinal halini tekrar dataframe'e tanıtıyoruz.

    print("# AFTER \n Imputation method is 'MODE' for categorical variables!")
    print(" Imputation method is '" + num_method.upper() + "' for numeric variables! \n")
    print(dataframe[variables_with_na].isnull().sum(), "\n\n")

    return dataframe

df = quick_missing_imp(df, num_method="median", cat_length=17)


# for num_cols
dataframe = dataframe.apply(lambda x: x.fillna(x.mean()) if str(x.dtypes) not in ["category", "object", "bool"] else x, axis=0)
# for cat_cols
dataframe = dataframe.apply(lambda x: x.fillna(x.mode()[0]) if (str(x.dtypes) in ["category", "object", "bool"] and len(x.unique()) <=20) else x, axis=0)


# categorical breakdown
dataframe[col_name_1].fillna(dataframe.groupby(colname_2)[col_name_1].transform("mean"))


# machine learning filling (advance)
# use grab_col_names
cat_cols, num_cols, cat_but_car = grab_col_names(dataframe)
# encoding cat_cols
dataframe_2 = pd.get_dummies(dataframe[cat_cols + num_cols], drop_first=True)
# scaling all cols
scaler = MinMaxScaler()
dataframe_2 = pd.DataFrame(scaler.fit_transform(dataframe_2), columns=dataframe_2.columns)
# applying KNN
imputer = KNNImputer(n_neighbors=5)
dataframe_2 = pd.DataFrame(imputer.fit_transform(dataframe_2), columns=dataframe_2.columns)
# unscaling all cols
dataframe_2 = pd.DataFrame(scaler.inverse_transform(dataframe_2), columns=dataframe_2.columns)
# looking at filling values
dataframe[col_name_KNN] = dataframe_2[[col_name]]
dataframe.loc[dataframe[col_name].isnull(), [col_name, col_name_KNN]]
# or looking at all of them at the same time
dataframe.loc[dataframe[col_name].isnull()]
dataframe.loc[dataframe.isnull()]


##############################################################
#################   Kategorilere Ayırma   ####################
##############################################################

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir

    Parameters
    ----------
    dataframe: dataframe
        Değişken isimleri alınmak istenen dataframe'dir.
    cat_th: int, float (optional)
        Numerik fakat kategorik olan değişkenler için sınıf eşik değeri
    car_th: int, float (optional)
        Kategorik fakat kardinal olan değişkenler için sınıf eşik değeri

    Returns
    -------
    cat_cols: list
        Kategorik değişken listesi
    num_cols: list
        Numerik değişken listesi
    cat_but_car: list
        Kategorik görünümlü kardinal değişken listesi

    Notes
    _______
    cat_cols + num_cols + cat_but_car = toplam değişken sayısı
    num_but_cat  cat_cols'un içerisnde.
    Return olan 3 liste toplamı, toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car
    """
    cat_cols = [col for col in dataframe.columns if str(dataframe[col].dtypes) in ["category", "object", "bool"]]
    num_but_cat = [col for col in dataframe.columns if (dataframe[col].nunique() < cat_th) & (dataframe[col].dtypes in ["int", "float"])]
    cat_but_car = [col for col in dataframe.columns if (dataframe[col].nunique() > car_th) & (str(dataframe[col].dtypes) in ["category", "object"])]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if (dataframe[col].dtypes in ["int", "float"]) & (col not in cat_cols)]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")

    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

# num_cols = [col for col in num_cols if col not in "Id"]
# date tipindekileri de kontrol et

##############################################################
####################  Devamı Aşağısı #########################
##############################################################

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("########################################")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe) # for döngüsünde bunda bool tipini değiştirmek gerekiyor. matplotlip ile gerekmiyor.
        plt.show(block=True)

#### devamında;

for col in cat_cols:
    if df[col].dtypes == "bool":
        df[col] = df[col].astype(int)
        cat_summary(df, col, plot=True)
    else:
        cat_summary(df, col, plot=True)

def num_summary(dataframe, col_name, plot=False):
    quantiles = [0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.98,0.99]
    print(dataframe[col_name].describe(quantiles).T)

    if plot:
        dataframe[col_name].hist()
        plt.xlabel(col_name)
        plt.title(col_name)
        plt.show(block=True)

### devamında;

for col in num_cols:
    num_summary(df, col, plot=True)

### Hedef değişken analizi


# Bağımlı değişkenin incelenmesi (target sayısal)
df[target].hist(bins=100)
plt.show()

# Bağımlı değişkenin logaritmasının incelenmesi (target sayısal)
np.log1p(df[target]).hist(bins=50)
plt.show()

# Bağımlı değişkenin incelenmesi (target kategorik)
sns.countplot(x=df[target], data=df)
plt.show()


## Kategorik değişken
def target_summary_with_cat(dataframe, categorical_col, target):
    print(categorical_col)
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean(),
                        "Count": dataframe[categorical_col].value_counts(),
                        "Ratio": 100 * dataframe[categorical_col].value_counts() / len(dataframe)}), end="\n\n\n")

### devamında;

for col in cat_cols:
    target_summary_with_cat(df, col, "survived")

## Sayısal değişken

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

## devamında;

for col in num_cols:
    target_summary_with_num(df,"survived", col)




# Korelasyon Analizi

# korelasyon analizi sadece sayısal değişkenlerde yapılır.
# önce sayısal değişkenler seçilir.
# sonra df[num_cols] ile yeni sayısal değişkenlerin olduğu dataframe oluşturulur.

cat_cols, num_cols, cat_but_car = grab_col_names(df)

# ve korelasyon analizine devam edilir.

#**************************************

corr_df = df[num_cols]

"""bu adımı yapmadan önce df'in yedeğini al! 
Orjinal df'i değitiriyorsun korelasyon için.
Orjinalinden sayısal yüksek korelasyonu bulup siliniyor ama
aynı zamanda tüm kategorikler de df'den gidiyor
"""
#**************************************

# Only looking high corr variables with target
dataframe.corrwith(dataframe[target]).sort_values(ascending=False)


def high_correlated_cols(dataframe, plot=False, corr_th=0.90):
    corr = dataframe.corr()
    corr_matrix = corr.abs()
    upper_triangle_matrix = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={"figure.figsize":  (15, 15)})
        sns.heatmap(corr, cmap="RdBu")
        plt.show(block=True)
    return drop_list

drop_list = high_correlated_cols(df)

# yüksel korelasyonları silerek grafiği görme
high_correlated_cols(df.drop(drop_list, axis=1), plot=True)


##############################################################
#######################   Encoding   #########################
##############################################################

# Label Encoding & Binary Encoding (for binary or ordinal cat.)

# Be aware of NaN's
# Alphabetic order 0-1...

# **** bağımlı değişken hiç bir encoding'e sokulmaz

# len(df[col].unique() ile yapılırsa ve eksik değer varsa sayı 3 çıkar
# df[col].nunique() ile eksik değerler sayılmaz ama label encoderdan geçirirken eksik değerleri de 2 olarak doldurur.
# ya 2'nin eksik değer olduğunu bileceğiz ya da öncesinde eksik değerlere müdahale edeceğiz.

binary_cols = [col for col in dataframe.columns if dataframe[col].dtype not in [int, float]
               and dataframe[col].nunique() == 2]


def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


le.inverse_transform([0, 1])


# Rare Encoding

def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")


def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()
    rare_columns = [col for col in temp_df.columns if str(temp_df[col].dtypes) in ["category", "object", "bool"]
                    and (temp_df[col].value_counts() / len (temp_df) < rare_perc).any(axis=None)]
    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), "Rare", temp_df[var])
    return temp_df


# One Hot Encoding (for all)

# Be aware of getdummies (drop first)
# If NaN matters, use dummy_na=True

ohe_cols = [col for col in dataframe.columns if 10 >= dataframe[col].nunique() > 2]

def one_hot_encoder(dataframe, ohe_cols, drop_first=True, dtype=int, dummy_na=True):
    dataframe = pd.get_dummies(data=dataframe, columns=cat_cols, drop_first=drop_first, dtype=dtype, dummy_na=dummy_na)
    return dataframe

useless_cols = [col for col in dataframe.columns if dataframe[col].nunique() == 2 and
                (df[col].values_counts() / len(dataframe) < 0.01).any(axis=None)]

dataframe.drop(useless_cols, axis=1, inplace=True)

# Feature Scaling

# StandardScaler: z = (x - ort.) / std      (mean = 0, std = 1)

ss = StandardScaler()
dataframe[col_name_ss] = ss.fit_transform(dataframe[[col_name]])  #fit_transform'a dataframe vermek gerekiyor [[col_name]]

# RobustScaler: robust = (x - median) / iqr      (outliers do not affect)

rs = RobustScaler()
dataframe[col_name_rs] = rs.fit_transform(dataframe[[col_name]])  #fit_transform'a dataframe vermek gerekiyor [[col_name]]

# MinMaxScaler (0-1)

mms = MinMaxScaler(feature_range=(0,1))  #feature_range=(0,1) zaten öntanımlı değeri yazılmasa da olur.
dataframe[col_name_mms] = mms.fit_transform(dataframe[[col_name]])  #fit_transform'a dataframe vermek gerekiyor [[col_name]]

# Numeric to Cat. (Binning)

dataframe[col_name_qcut] = pd.qcut(dataframe[col_name], 5) # labels=["a", "b", "c", "d", "e"]


##############################################################
##################   Feature Extraction   ####################
##############################################################

# Binary Feature

dataframe[col_name_bool] = dataframe[col_name].notnull().astype("int")

dataframe.groupby(col_name_bool).agg({target_col:"mean"})

from statsmodels.stats.proportion import proportions_ztest
test_stat, pvalue = proportions_ztest(count=[dataframe.loc[dataframe[col_name_bool] == 1, target_col].sum(),
                                             dataframe.loc[dataframe[col_name_bool] == 0, target_col].sum()],
                                      nobs=[dataframe.loc[dataframe[col_name_bool] == 1, target_col].shape[0],
                                             dataframe.loc[dataframe[col_name_bool] == 0, target_col].shape[0]])

print("Test Stat = %.4f, p-value = %.4f" % (test_stat, pvalue))

# Letter Count

dataframe[col_name_count] = dataframe[col_name].str.len()

# Word Count

dataframe[col_name_word_count] = dataframe[col_name].apply(lambda x: len(str(x).split(" ")))

# Capturing Specific Structures

dataframe[col_name_Dr] = dataframe[col_name].apply(lambda x: len([x for x in x.split() if x.startswith("Dr")]))  #ismi Dr ile başlayanlar da doktormuş gibi gelecek.

dataframe.groupby(col_name_Dr).agg({target_col: ["mean", "count"]})

# Regex Feature

dataframe["New_Title"] = dataframe[col_name].str.extract(' ([A-Za-z]+)\.', expand=False)

# Date Feature

dataframe['Timestamp'] = pd.to_datetime(dataframe['Timestamp'], format='%Y-%m-%d %H:%M:%S')

dataframe["year"] = dataframe["Timestamp"].dt.year

dataframe["month"] = dataframe["Timestamp"].dt.month

dataframe["year_diff"] = date.today().year - dataframe["Timestamp"].dt.year

dataframe["month_diff"] = (date.today().year - dataframe["Timestamp"].dt.year) * 12 + date.today().month - dataframe["Timestamp"].dt.month

dataframe["day_name"] = dataframe["Timestamp"].dt.day_name()

# Feature Interaction

dataframe.columns = [col.upper() for col in dataframe.columns]




