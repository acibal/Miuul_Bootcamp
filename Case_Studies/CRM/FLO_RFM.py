
###############################################################
# RFM ile Müşteri Segmentasyonu (Customer Segmentation with RFM)
###############################################################

###############################################################
# İş Problemi (Business Problem)
###############################################################
# FLO müşterilerini segmentlere ayırıp bu segmentlere göre pazarlama stratejileri belirlemek istiyor.
# Buna yönelik olarak müşterilerin davranışları tanımlanacak ve bu davranış öbeklenmelerine göre gruplar oluşturulacak..

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

# GÖREV 1: Veriyi Anlama (Data Understanding) ve Hazırlama
           # 1. flo_data_20K.csv verisini okuyunuz.
           # 2. Veri setinde
                     # a. İlk 10 gözlem,
                     # b. Değişken isimleri,
                     # c. Betimsel istatistik,
                     # d. Boş değer,
                     # e. Değişken tipleri, incelemesi yapınız.
           # 3. Omnichannel müşterilerin hem online'dan hemde offline platformlardan alışveriş yaptığını ifade etmektedir. Herbir müşterinin toplam
           # alışveriş sayısı ve harcaması için yeni değişkenler oluşturun.
           # 4. Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz.
           # 5. Alışveriş kanallarındaki müşteri sayısının, ortalama alınan ürün sayısının ve ortalama harcamaların dağılımına bakınız.
           # 6. En fazla kazancı getiren ilk 10 müşteriyi sıralayınız.
           # 7. En fazla siparişi veren ilk 10 müşteriyi sıralayınız.
           # 8. Veri ön hazırlık sürecini fonksiyonlaştırınız.

# GÖREV 2: RFM Metriklerinin Hesaplanması

# GÖREV 3: RF ve RFM Skorlarının Hesaplanması

# GÖREV 4: RF Skorlarının Segment Olarak Tanımlanması

# GÖREV 5: Aksiyon zamanı!
           # 1. Segmentlerin recency, frequnecy ve monetary ortalamalarını inceleyiniz.
           # 2. RFM analizi yardımı ile 2 case için ilgili profildeki müşterileri bulun ve müşteri id'lerini csv ye kaydediniz.
                   # a. FLO bünyesine yeni bir kadın ayakkabı markası dahil ediyor. Dahil ettiği markanın ürün fiyatları genel müşteri tercihlerinin üstünde. Bu nedenle markanın
                   # tanıtımı ve ürün satışları için ilgilenecek profildeki müşterilerle özel olarak iletişime geçeilmek isteniliyor. Sadık müşterilerinden(champions,loyal_customers),
                   # ortalama 250 TL üzeri ve kadın kategorisinden alışveriş yapan kişiler özel olarak iletişim kuralacak müşteriler. Bu müşterilerin id numaralarını csv dosyasına
                   # yeni_marka_hedef_müşteri_id.cvs olarak kaydediniz.
                   # b. Erkek ve Çoçuk ürünlerinde %40'a yakın indirim planlanmaktadır. Bu indirimle ilgili kategorilerle ilgilenen geçmişte iyi müşteri olan ama uzun süredir
                   # alışveriş yapmayan kaybedilmemesi gereken müşteriler, uykuda olanlar ve yeni gelen müşteriler özel olarak hedef alınmak isteniliyor. Uygun profildeki müşterilerin id'lerini csv dosyasına indirim_hedef_müşteri_ids.csv
                   # olarak kaydediniz.


# GÖREV 6: Tüm süreci fonksiyonlaştırınız.

###############################################################
# GÖREV 1: Veriyi  Hazırlama ve Anlama (Data Understanding)
###############################################################

import pandas as pd
df_ = pd.read_csv("Data_Science_BC/W3_CRM/Case_Studies/FLOCLTVPrediction/flo_data_20k.csv")
df = df_.copy()

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 20)
pd.set_option("display.width", 400)
pd.set_option("display.float_format", lambda x: "%.4f" % x)

# 2. Veri setinde
        # a. İlk 10 gözlem,
        # b. Değişken isimleri,
        # c. Boyut,
        # d. Betimsel istatistik,
        # e. Boş değer,
        # f. Değişken tipleri, incelemesi yapınız.

df.head(10)
df.info()
df.shape
df.columns
df.describe().T
df.isnull().sum()
df.info()

# 3. Omnichannel müşterilerin hem online'dan hemde offline platformlardan alışveriş yaptığını ifade etmektedir.
# Herbir müşterinin toplam alışveriş sayısı ve harcaması için yeni değişkenler oluşturunuz.

df["order_num_total_ever_all"] = df.order_num_total_ever_offline + df.order_num_total_ever_online
df["customer_value_total_ever_all"] = df.customer_value_total_ever_offline + df.customer_value_total_ever_online

# 4. Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz.


# df["last_order_date"] = df["last_order_date"].apply(pd.to_datetime)

df.info()
import datetime as dt
df.head()
df.last_order_date.max()
today_date = dt.datetime(2021, 6, 1)
type(today_date)

date_col_list = [col for col in df.columns if "date" in col]
df[date_col_list] = df[date_col_list].astype("datetime64")
df.info()

# 5. Alışveriş kanallarındaki müşteri sayısının, toplam alınan ürün sayısı ve toplam harcamaların dağılımına bakınız. 


df.groupby("order_channel").agg({"master_id": "count",
                                 "order_num_total_ever_all": "sum",
                                 "customer_value_total_ever_all": "sum"})

# 6. En fazla kazancı getiren ilk 10 müşteriyi sıralayınız.

df.sort_values("customer_value_total_ever_all", ascending=False)["customer_value_total_ever_all"].head(10)

# 7. En fazla siparişi veren ilk 10 müşteriyi sıralayınız.

(df.sort_values("order_num_total_ever_all", ascending=False)["order_num_total_ever_all"].head(10))

# 8. Veri ön hazırlık sürecini fonksiyonlaştırınız.

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
        dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
        dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

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

###############################################################
# GÖREV 2: RFM Metriklerinin Hesaplanması
###############################################################

# Veri setindeki en son alışverişin yapıldığı tarihten 2 gün sonrasını analiz tarihi

# customer_id, recency, frequnecy ve monetary değerlerinin yer aldığı yeni bir rfm dataframe

# recency, frequency, monetary
df.last_order_date.max()
import datetime as dt
today_date = dt.datetime(2021, 6, 1)
type(today_date)

rfm = df.groupby("master_id").agg({"last_order_date": lambda date: (today_date - date.max()).days,
                             "order_num_total_ever_all": lambda x: x,
                             "customer_value_total_ever_all": lambda x: x})

rfm.columns = ["recency", "frequency", "monetary"]

rfm.head()
rfm.describe().T
rfm.shape

###############################################################
# GÖREV 3: RF ve RFM Skorlarının Hesaplanması (Calculating RF and RFM Scores)
###############################################################

#  Recency, Frequency ve Monetary metriklerini qcut yardımı ile 1-5 arasında skorlara çevrilmesi ve
# Bu skorları recency_score, frequency_score ve monetary_score olarak kaydedilmesi

rfm["recency_score"] = pd.qcut(rfm.recency, 5, [5, 4, 3, 2, 1])

rfm["frequency_score"] = pd.qcut(rfm.frequency.rank(method="first"), 5, [1, 2, 3, 4, 5])

rfm["monetary_score"] = pd.qcut(rfm.monetary, 5, [1, 2, 3 ,4 ,5])

rfm.head()

rfm["rfm_score"] = rfm.recency_score.astype(str) + rfm.frequency_score.astype(str)

# recency_score ve frequency_score’u tek bir değişken olarak ifade edilmesi ve RF_SCORE olarak kaydedilmesi


###############################################################
# GÖREV 4: RF Skorlarının Segment Olarak Tanımlanması
###############################################################

# Oluşturulan RFM skorların daha açıklanabilir olması için segment tanımlama ve  tanımlanan seg_map yardımı ile RF_SCORE'u segmentlere çevirme

seg_map = {
    r"[1-2][1-2]": "hibernating",
    r"[1-2][3-4]": "at_risk",
    r"[1-2]5": "cant_loose",
    r"3[1-2]": "about_to_sleep",
    r"33": "need_attention",
    r"[3-4][4-5]": "loyal_customers",
    r"41": "promising",
    r"51": "new_customers",
    r"[4-5][2-3]": "potential_loyalists",
    r"5[4-5]": "champions"
}

rfm["segment"] = rfm.rfm_score.replace(seg_map, regex=True)

rfm.head()

###############################################################
# GÖREV 5: Aksiyon zamanı!
###############################################################

# 1. Segmentlerin recency, frequnecy ve monetary ortalamalarını inceleyiniz.

rfm.groupby("segment").agg({"recency":["count", "mean", "sum"],
                           "frequency": ["count", "mean", "sum"],
                           "monetary": ["count", "mean", "sum"]})

# 2. RFM analizi yardımı ile 2 case için ilgili profildeki müşterileri bulunuz ve müşteri id'lerini csv ye kaydediniz.

# a. FLO bünyesine yeni bir kadın ayakkabı markası dahil ediyor. Dahil ettiği markanın ürün fiyatları genel müşteri tercihlerinin üstünde. Bu nedenle markanın
# tanıtımı ve ürün satışları için ilgilenecek profildeki müşterilerle özel olarak iletişime geçeilmek isteniliyor. Bu müşterilerin sadık  ve
# kadın kategorisinden alışveriş yapan kişiler olması planlandı. Müşterilerin id numaralarını csv dosyasına yeni_marka_hedef_müşteri_id.cvs
# olarak kaydediniz.

df["customer_value_average_ever_all"] = df.customer_value_total_ever_all / df.order_num_total_ever_all

rfm["average_price"] = rfm.monetary / rfm.frequency
rfm.average_price.mean()

yeni_marka_df = rfm[((rfm.segment == "champions") | (rfm.segment == "loyal_customers")) & (rfm.average_price > 250)]

# or

yeni_marka_df = rfm[((rfm.segment.str.contains("champions | loyal_customers")) & (rfm.average_price > 250)]

yeni_marka_df = yeni_marka_df.reset_index()
master_id_df = yeni_marka_df.master_id
master_id_df.to_csv("yeni_marka_hedef_musteri_id.cvs")

# b. Erkek ve Çoçuk ürünlerinde %40'a yakın indirim planlanmaktadır. Bu indirimle ilgili kategorilerle ilgilenen geçmişte iyi müşterilerden olan ama uzun süredir
# alışveriş yapmayan ve yeni gelen müşteriler özel olarak hedef alınmak isteniliyor. Uygun profildeki müşterilerin id'lerini csv dosyasına indirim_hedef_müşteri_ids.csv
# olarak kaydediniz.

indirim_df = rfm[(rfm.segment == "cant_loose") | (rfm.segment == "about_to_sleep") | (rfm.segment == "new_customers")]

# or
indirim_df = rfm[(rfm.segment.str.contains("cant_loose | about_to_sleep | new_customers")]

indirim_df = indirim_df.reset_index()
indirim_id_df = indirim_df.master_id
indirim_id_df.to_csv("indirim_hedef_musteri_ids")

