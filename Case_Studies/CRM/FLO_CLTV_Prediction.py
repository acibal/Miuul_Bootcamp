##############################################################
# BG-NBD ve Gamma-Gamma ile CLTV Prediction
##############################################################

###############################################################
# İş Problemi (Business Problem)
###############################################################
# FLO satış ve pazarlama faaliyetleri için roadmap belirlemek istemektedir.
# Şirketin orta uzun vadeli plan yapabilmesi için var olan müşterilerin gelecekte şirkete sağlayacakları potansiyel değerin tahmin edilmesi gerekmektedir.


###############################################################
# Veri Seti Hikayesi
###############################################################

# Veri seti son alışverişlerini 2020 - 2021 yıllarında OmniChannel(hem online hem offline alışveriş yapan) olarak yapan müşterilerin geçmiş alışveriş davranışlarından
# elde edilen bilgilerden oluşmaktadır.

# master_id: Eşsiz müşteri numarası
# order_channel : Alışveriş yapılan platforma ait hangi kanalın kullanıldığı (Android, ios, Desktop, Mobile, Offline)
# last_order_channel : En son alışverişin yapıldığı kanal
# first_order_date : Müşterinin yaptığı ilk alışveriş tarihi
# last_order_date : Müşterinin yaptığı son alışveriş tarihi
# last_order_date_online : Muşterinin online platformda yaptığı son alışveriş tarihi
# last_order_date_offline : Muşterinin offline platformda yaptığı son alışveriş tarihi
# order_num_total_ever_online : Müşterinin online platformda yaptığı toplam alışveriş sayısı
# order_num_total_ever_offline : Müşterinin offline'da yaptığı toplam alışveriş sayısı
# customer_value_total_ever_offline : Müşterinin offline alışverişlerinde ödediği toplam ücret
# customer_value_total_ever_online : Müşterinin online alışverişlerinde ödediği toplam ücret
# interested_in_categories_12 : Müşterinin son 12 ayda alışveriş yaptığı kategorilerin listesi


###############################################################
# GÖREVLER
###############################################################
# GÖREV 1: Veriyi Hazırlama
           # 1. flo_data_20K.csv verisini okuyunuz.Dataframe’in kopyasını oluşturunuz.
           # 2. Aykırı değerleri baskılamak için gerekli olan outlier_thresholds ve replace_with_thresholds fonksiyonlarını tanımlayınız.
           # Not: cltv hesaplanırken frequency değerleri integer olması gerekmektedir.Bu nedenle alt ve üst limitlerini round() ile yuvarlayınız.
           # 3. "order_num_total_ever_online","order_num_total_ever_offline","customer_value_total_ever_offline","customer_value_total_ever_online" değişkenlerinin
           # aykırı değerleri varsa baskılayanız.
           # 4. Omnichannel müşterilerin hem online'dan hemde offline platformlardan alışveriş yaptığını ifade etmektedir. Herbir müşterinin toplam
           # alışveriş sayısı ve harcaması için yeni değişkenler oluşturun.
           # 5. Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz.

# GÖREV 2: CLTV Veri Yapısının Oluşturulması
           # 1.Veri setindeki en son alışverişin yapıldığı tarihten 2 gün sonrasını analiz tarihi olarak alınız.
           # 2.customer_id, recency_cltv_weekly, T_weekly, frequency ve monetary_cltv_avg değerlerinin yer aldığı yeni bir cltv dataframe'i oluşturunuz.
           # Monetary değeri satın alma başına ortalama değer olarak, recency ve tenure değerleri ise haftalık cinsten ifade edilecek.


# GÖREV 3: BG/NBD, Gamma-Gamma Modellerinin Kurulması, CLTV'nin hesaplanması
           # 1. BG/NBD modelini fit ediniz.
                # a. 3 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_3_month olarak cltv dataframe'ine ekleyiniz.
                # b. 6 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_6_month olarak cltv dataframe'ine ekleyiniz.
           # 2. Gamma-Gamma modelini fit ediniz. Müşterilerin ortalama bırakacakları değeri tahminleyip exp_average_value olarak cltv dataframe'ine ekleyiniz.
           # 3. 6 aylık CLTV hesaplayınız ve cltv ismiyle dataframe'e ekleyiniz.
                # b. Cltv değeri en yüksek 20 kişiyi gözlemleyiniz.

# GÖREV 4: CLTV'ye Göre Segmentlerin Oluşturulması
           # 1. 6 aylık tüm müşterilerinizi 4 gruba (segmente) ayırınız ve grup isimlerini veri setine ekleyiniz. cltv_segment ismi ile dataframe'e ekleyiniz.
           # 2. 4 grup içerisinden seçeceğiniz 2 grup için yönetime kısa kısa 6 aylık aksiyon önerilerinde bulununuz

# BONUS: Tüm süreci fonksiyonlaştırınız.


###############################################################
# GÖREV 1: Veriyi Hazırlama
###############################################################


# 1. OmniChannel.csv verisini okuyunuz.Dataframe’in kopyasını oluşturunuz.

import pandas as pd
df_ = pd.read_csv("Data_Science_BC/W3_CRM/Case_Studies/FLOCLTVPrediction/flo_data_20k.csv")
df = df_.copy()

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 20)
pd.set_option("display.width", 400)
pd.set_option("display.float_format", lambda x: "%.4f" % x)

# 2. Aykırı değerleri baskılamak için gerekli olan outlier_thresholds ve replace_with_thresholds fonksiyonlarını tanımlayınız.
# Not: cltv hesaplanırken frequency değerleri integer olması gerekmektedir.Bu nedenle alt ve üst limitlerini round() ile yuvarlayınız.

def on_hazirlik(dataframe):

    date_col_list = [col for col in dataframe.columns if "date" in col]
    dataframe[date_col_list] = dataframe[date_col_list].astype("datetime64")

    dataframe["order_num_total_ever_all"] = dataframe.order_num_total_ever_offline + dataframe.order_num_total_ever_online
    dataframe["customer_value_total_ever_all"] = dataframe.customer_value_total_ever_offline + dataframe.customer_value_total_ever_online

    def outlier_thresholds(dataframe, variable):
        quartile1 = dataframe[variable].quantile(0.01)
        quartile3 = dataframe[variable].quantile(0.99)
        interquartile_range = quartile3 - quartile1
        up_limit = quartile3 + 1.5 * interquartile_range
        low_limit = quartile1 - 1.5 * interquartile_range
        return low_limit, up_limit

    def replace_with_thresholds(dataframe, variable):
        low_limit, up_limit = outlier_thresholds(dataframe, variable)
        dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit.round()
        dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit.round()

    def grab_col_names(dataframe, cat_th=10, car_th=20):
        """
        Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir
        Parameters
        ----------
        dataframe: dataframe
            Değişken isimleri alınmak istenen dataframe'dir.
        cat_th: int, float
            Numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, float
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
        num_but_cat = [col for col in dataframe.columns if
                       (dataframe[col].nunique() < cat_th) & (dataframe[col].dtypes in ["int", "float"])]
        cat_but_car = [col for col in dataframe.columns if
                       (dataframe[col].nunique() > car_th) & (str(dataframe[col].dtypes) in ["category", "object"])]
        cat_cols = cat_cols + num_but_cat
        cat_cols = [col for col in cat_cols if col not in cat_but_car]

        num_cols = [col for col in dataframe.columns if
                    (dataframe[col].dtypes in ["int", "float"]) & (col not in cat_cols)]

        print(f"Observations: {dataframe.shape[0]}")
        print(f"Variables: {dataframe.shape[1]}")
        print(f"cat_cols: {len(cat_cols)}")
        print(f"num_cols: {len(num_cols)}")
        print(f"cat_but_car: {len(cat_but_car)}")
        print(f"num_but_cat: {len(num_but_cat)}")

        return cat_cols, num_cols, cat_but_car

    cat_cols, num_cols, cat_but_car = grab_col_names(dataframe)

    for i in num_cols:
            replace_with_thresholds(df, i)

    return df

on_hazirlik(df)

df.describe().T
df.head()
df.info()

# 3. "order_num_total_ever_online","order_num_total_ever_offline","customer_value_total_ever_offline","customer_value_total_ever_online" değişkenlerinin
#aykırı değerleri varsa baskılayanız.

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquartile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquartile_range
    low_limit = quartile1 - 1.5 * interquartile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit.round()
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit.round()

def grab_col_names(dataframe, cat_th=10, car_th=20):
        """
        Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir
        Parameters
        ----------
        dataframe: dataframe
            Değişken isimleri alınmak istenen dataframe'dir.
        cat_th: int, float
            Numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, float
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
        num_but_cat = [col for col in dataframe.columns if
                       (dataframe[col].nunique() < cat_th) & (dataframe[col].dtypes in ["int", "float"])]
        cat_but_car = [col for col in dataframe.columns if
                       (dataframe[col].nunique() > car_th) & (str(dataframe[col].dtypes) in ["category", "object"])]
        cat_cols = cat_cols + num_but_cat
        cat_cols = [col for col in cat_cols if col not in cat_but_car]

        num_cols = [col for col in dataframe.columns if
                    (dataframe[col].dtypes in ["int", "float"]) & (col not in cat_cols)]

        print(f"Observations: {dataframe.shape[0]}")
        print(f"Variables: {dataframe.shape[1]}")
        print(f"cat_cols: {len(cat_cols)}")
        print(f"num_cols: {len(num_cols)}")
        print(f"cat_but_car: {len(cat_but_car)}")
        print(f"num_but_cat: {len(num_but_cat)}")

        return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

for i in num_cols:
        replace_with_thresholds(df, i)

df.describe().T

# 4. Omnichannel müşterilerin hem online'dan hemde offline platformlardan alışveriş yaptığını ifade etmektedir.
# Herbir müşterinin toplam alışveriş sayısı ve harcaması için yeni değişkenler oluşturun.

df["order_num_total_ever_all"] = df.order_num_total_ever_offline + df.order_num_total_ever_online
df["customer_value_total_ever_all"] = df.customer_value_total_ever_offline + df.customer_value_total_ever_online
df.head()

# 5. Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz.

date_col_list = [col for col in df.columns if "date" in col]
df[date_col_list] = df[date_col_list].astype("datetime64")

###############################################################
# GÖREV 2: CLTV Veri Yapısının Oluşturulması
###############################################################

# 1.Veri setindeki en son alışverişin yapıldığı tarihten 2 gün sonrasını analiz tarihi olarak alınız.

import datetime as dt
df.head()
df.last_order_date.max()
today_date = dt.datetime(2021, 6, 1)
type(today_date)

# 2.customer_id, recency_cltv_weekly, T_weekly, frequency ve monetary_cltv_avg değerlerinin yer aldığı yeni bir cltv dataframe'i oluşturunuz.

df.head()

df["recency"] = df.last_order_date - df.first_order_date
df["T"] = today_date - df.first_order_date
df["frequency"] = df.order_num_total_ever_all
df["monetary"] = df.customer_value_total_ever_all
df.head()

df.recency = df.recency.astype("str")
df["T"] = df["T"].astype("str")
df.recency = df.recency.apply(lambda x: x.replace("days",""))
df["T"] = df["T"].apply(lambda x: x.replace("days",""))
df.recency = df.recency.astype("int")
df["T"] = df["T"].astype("int")
df.info()
df.head()



cltv_df = df[["master_id", "recency", "T", "frequency", "monetary"]]
cltv_df.head()
cltv_df.index = cltv_df.master_id
cltv_df.drop("master_id", axis=1, inplace=True) # 1 kere yap!! uyarı veriyor ama çalışıyor.

cltv_df.monetary = cltv_df.monetary / cltv_df.frequency # 1 kere yap!! uyarı veriyor ama işlem yapıyor
cltv_df = cltv_df[(cltv_df.frequency > 1)]
cltv_df.recency = cltv_df.recency / 7
cltv_df["T"] = cltv_df["T"] / 7
cltv_df.head()
cltv_df.describe().T

###############################################################
# GÖREV 3: BG/NBD, Gamma-Gamma Modellerinin Kurulması, 6 aylık CLTV'nin hesaplanması
###############################################################

# 1. BG/NBD modelini kurunuz.

import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv_df.frequency,
        cltv_df.recency,
        cltv_df["T"])

# 3 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_3_month olarak cltv dataframe'ine ekleyiniz.

cltv_df["exp_sales_3_month"] = bgf.predict(4 * 3,
                                                cltv_df.frequency,
                                                cltv_df.recency,
                                                cltv_df["T"])
cltv_df.head()

# 6 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_6_month olarak cltv dataframe'ine ekleyiniz.

cltv_df["exp_sales_6_month"] = bgf.predict(4 * 6,
                                                cltv_df.frequency,
                                                cltv_df.recency,
                                                cltv_df["T"])
cltv_df.head()

# 3. ve 6.aydaki en çok satın alım gerçekleştirecek 10 kişiyi inceleyeniz.

bgf.predict(4 * 3,
            cltv_df.frequency,
            cltv_df.recency,
            cltv_df["T"]).sort_values(ascending=False).head(10)

bgf.predict(4 * 6,
            cltv_df.frequency,
            cltv_df.recency,
            cltv_df["T"]).sort_values(ascending=False).head(10)

plot_period_transactions(bgf)
plt.show()

# 2.  Gamma-Gamma modelini fit ediniz. Müşterilerin ortalama bırakacakları değeri tahminleyip exp_average_value olarak cltv dataframe'ine ekleyiniz.

ggf = GammaGammaFitter(penalizer_coef=0.01)

cltv_df.frequency = cltv_df.frequency.astype("int")
cltv_df.monetary = cltv_df.monetary.astype("int")

ggf.fit(cltv_df.frequency, cltv_df.monetary)

cltv_df.head()

cltv_df["expected_average_profit"] = ggf.conditional_expected_average_profit(cltv_df.frequency,
                                        cltv_df.monetary)


# 3. 6 aylık CLTV hesaplayınız ve cltv ismiyle dataframe'e ekleyiniz.

cltv_3 = ggf.customer_lifetime_value(bgf,
                                   cltv_df.frequency,
                                   cltv_df.recency,
                                   cltv_df["T"],
                                   cltv_df.monetary,
                                   time=3,   # aylık
                                   freq="W",   # T'nin frekans bilgisi
                                   discount_rate=0.01)


cltv_6 = ggf.customer_lifetime_value(bgf,
                                   cltv_df.frequency,
                                   cltv_df.recency,
                                   cltv_df["T"],
                                   cltv_df.monetary,
                                   time=6,   # aylık
                                   freq="W",   # T'nin frekans bilgisi
                                   discount_rate=0.01)

cltv_3 = cltv_3.reset_index()
cltv_6 = cltv_6.reset_index()

cltv_final = cltv_df.merge(cltv_3, on="master_id", how="left")
cltv_final = cltv_final.merge(cltv_6, on="master_id", how="left")


# CLTV değeri en yüksek 20 kişiyi gözlemleyiniz.

cltv_final.sort_values("clv_x", ascending=False).head(10)
cltv_final.sort_values("clv_y", ascending=False).head(10)


###############################################################
# GÖREV 4: CLTV'ye Göre Segmentlerin Oluşturulması
###############################################################

# 1. 6 aylık CLTV'ye göre tüm müşterilerinizi 4 gruba (segmente) ayırınız ve grup isimlerini veri setine ekleyiniz.
# cltv_segment ismi ile atayınız.

cltv_final["cltv_segment"] = pd.qcut(cltv_final.clv_y, 4, ["D", "C", "B", "A"])
cltv_final.head()

cltv_final.sort_values("clv_y", ascending=False).head()

# 2. Segmentlerin recency, frequnecy ve monetary ortalamalarını inceleyiniz.

cltv_final.groupby("cltv_segment").agg(["count","mean","sum"])






