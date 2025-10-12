
###################################################
# PROJE: Rating Product & Sorting Reviews in Amazon
###################################################

###################################################
# İş Problemi
###################################################

# E-ticaretteki en önemli problemlerden bir tanesi ürünlere satış sonrası verilen puanların doğru şekilde hesaplanmasıdır.
# Bu problemin çözümü e-ticaret sitesi için daha fazla müşteri memnuniyeti sağlamak, satıcılar için ürünün öne çıkması ve satın
# alanlar için sorunsuz bir alışveriş deneyimi demektir. Bir diğer problem ise ürünlere verilen yorumların doğru bir şekilde sıralanması
# olarak karşımıza çıkmaktadır. Yanıltıcı yorumların öne çıkması ürünün satışını doğrudan etkileyeceğinden dolayı hem maddi kayıp
# hem de müşteri kaybına neden olacaktır. Bu 2 temel problemin çözümünde e-ticaret sitesi ve satıcılar satışlarını arttırırken müşteriler
# ise satın alma yolculuğunu sorunsuz olarak tamamlayacaktır.

###################################################
# Veri Seti Hikayesi
###################################################

# Amazon ürün verilerini içeren bu veri seti ürün kategorileri ile çeşitli metadataları içermektedir.
# Elektronik kategorisindeki en fazla yorum alan ürünün kullanıcı puanları ve yorumları vardır.

# Değişkenler:
# reviewerID: Kullanıcı ID’si
# asin: Ürün ID’si
# reviewerName: Kullanıcı Adı
# helpful: Faydalı değerlendirme derecesi
# reviewText: Değerlendirme
# overall: Ürün rating’i
# summary: Değerlendirme özeti
# unixReviewTime: Değerlendirme zamanı
# reviewTime: Değerlendirme zamanı Raw
# day_diff: Değerlendirmeden itibaren geçen gün sayısı
# helpful_yes: Değerlendirmenin faydalı bulunma sayısı
# total_vote: Değerlendirmeye verilen oy sayısı

import pandas as pd
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 500)
pd.set_option("display.expand_frame_repr", False)
pd.set_option("display.float_format", lambda x: "%.5f" % x)

###################################################
# GÖREV 1: Average Rating'i Güncel Yorumlara Göre Hesaplayınız ve Var Olan Average Rating ile Kıyaslayınız.
###################################################

# Paylaşılan veri setinde kullanıcılar bir ürüne puanlar vermiş ve yorumlar yapmıştır.
# Bu görevde amacımız verilen puanları tarihe göre ağırlıklandırarak değerlendirmek.
# İlk ortalama puan ile elde edilecek tarihe göre ağırlıklı puanın karşılaştırılması gerekmektedir.


###################################################
# Adım 1: Veri Setini Okutunuz ve Ürünün Ortalama Puanını Hesaplayınız.
###################################################

df = pd.read_csv("Data_Science_BC/W4_Measurement/Case_Studies/Rating Product&SortingReviewsinAmazon/amazon_review.csv")
df.head()
df.nunique()
df["total_vote"].max()
df["helpful_yes"].max()
df.shape
df.describe().T
df.overall.mean()

###################################################
# Adım 2: Tarihe Göre Ağırlıklı Puan Ortalamasını Hesaplayınız.
###################################################

df.reviewTime.max()
df[df["reviewTime"] == "2014-12-07"]
# current_date = 2014-12-08
df.describe().T

df.info()
def time_based_weighted_average(dataframe, w1=25, w2=22, w3=20, w4= 18, w5=15):
    return dataframe.loc[df["day_diff"] <= 30, "overall"].mean() * w1 / 100 + \
        dataframe.loc[(dataframe["day_diff"] > 30) & (dataframe["day_diff"] <= 90), "overall"].mean() * w2 / 100 + \
        dataframe.loc[(dataframe["day_diff"] > 90) & (dataframe["day_diff"] <= 360), "overall"].mean() * w3 / 100 + \
        dataframe.loc[(dataframe["day_diff"] > 360) & (dataframe["day_diff"] <= 600), "overall"].mean() * w4 / 100 + \
        dataframe.loc[dataframe["day_diff"] > 600, "overall"].mean() * w5 / 100

time_based_weighted_average(df)

###################################################
# Adım 3: Ağırlıklandırılmış puanlamada her bir zaman diliminin ortalamasını karşılaştırıp yorumlayınız.
###################################################

df.loc[df["day_diff"] <= 30, "overall"].mean()
df.loc[(df["day_diff"] > 30) & (df["day_diff"] <= 90), "overall"].mean()
df.loc[(df["day_diff"] > 90) & (df["day_diff"] <= 360), "overall"].mean()
df.loc[(df["day_diff"] > 360) & (df["day_diff"] <= 600), "overall"].mean()
df.loc[df["day_diff"] > 600, "overall"].mean()

###################################################
# Görev 2: Ürün için Ürün Detay Sayfasında Görüntülenecek 20 Review'i Belirleyiniz.
###################################################


###################################################
# Adım 1. helpful_no Değişkenini Üretiniz
###################################################

# Not:
# total_vote bir yoruma verilen toplam up-down sayısıdır.
# up, helpful demektir.
# veri setinde helpful_no değişkeni yoktur, var olan değişkenler üzerinden üretilmesi gerekmektedir.

df.head()
df["helpful_no"] = df["total_vote"] - df["helpful_yes"]

###################################################
# Adım 2. score_pos_neg_diff, score_average_rating ve wilson_lower_bound Skorlarını Hesaplayıp Veriye Ekleyiniz
###################################################

def score_up_down_diff(up, down):
    return up - down

def score_average_rating(up, down):
    if up + down == 0:
        return 0
    return up / (up + down)

def wilson_lower_bound(up, down, confidence=0.95):
    """
    Wilson Lower Bound Score hesapla

    - Bernoulli parametresi p için hesaplanacak güven aralığının alt sınırı WLB skoru olarak kabul edilir.
    - Hesaplanacak skor ürün sıralaması için kullanılır.
    - Not:
    Eğer skorlar 1-5 arasıdaysa 1-3 negatif, 4-5 pozitif olarak işaretlenir ve bernoulli'ye uygun hale getirilebilir.
    Bu beraberinde bazı problemleri de getirir. Bu sebeple bayesian average rating yapmak gerekir.

    Parameters
    ----------
    up: int
        up count
    down: int
        down count
    confidence: float
        confidence

    Returns
    -------
    wilson score: float

    """
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)



# score_pos_neg_diff
df["score_pos_neg_diff"] = df.apply(lambda x: score_up_down_diff(x["helpful_yes"], x["helpful_no"]), axis=1)

# score_average_rating
df["score_average_rating"] = df.apply(lambda x: score_average_rating(x["helpful_yes"], x["helpful_no"]), axis=1)

# wilson_lower_bound
df["wilson_lower_bound"] = df.apply(lambda x: wilson_lower_bound(x["helpful_yes"], x["helpful_no"]), axis=1)


##################################################
# Adım 3. 20 Yorumu Belirleyiniz ve Sonuçları Yorumlayınız.
###################################################

df.sort_values("wilson_lower_bound", ascending=False).head(20)



##################### alternatif çözüm ##########################


df["score_pos_neg_diff2"] = score_up_down_diff(df["helpful_yes"], df["helpful_no"])

import numpy as np

def score_average_rating_vectorized(up, down):
    # Oyların toplamı 0 olan yerler için bir mask oluştur
    zero_mask = (up + down) == 0
    # Vektörleştirilmiş bölme işlemi
    ratings = np.where(zero_mask, 0, up / (up + down))
    return ratings

df["score_average_rating_vectorized"] = score_average_rating_vectorized(df["helpful_yes"], df["helpful_no"])

def wilson_lower_bound_vectorized(up, down, confidence=0.95):
    """
    Wilson Lower Bound Score hesapla

    - Bernoulli parametresi p için hesaplanacak güven aralığının alt sınırı WLB skoru olarak kabul edilir.
    - Hesaplanacak skor ürün sıralaması için kullanılır.
    - Not:
    Eğer skorlar 1-5 arasıdaysa 1-3 negatif, 4-5 pozitif olarak işaretlenir ve bernoulli'ye uygun hale getirilebilir.
    Bu beraberinde bazı problemleri de getirir. Bu sebeple bayesian average rating yapmak gerekir.

    Parameters
    ----------
    up: int
        up count
    down: int
        down count
    confidence: float
        confidence

    Returns
    -------
    wilson score: float

    """
    # Oyların toplamı 0 olan yerler için bir mask oluştur
    zero_mask = (up + down) == 0
    # Vektörleştirilmiş bölme işlemi
    rating = np.where(zero_mask, 0, ((1.0 * up / (up + down)) + (st.norm.ppf(1 - (1 - confidence) / 2)) * (st.norm.ppf(1 - (1 - confidence) / 2)) / (2 * (up + down)) - (st.norm.ppf(1 - (1 - confidence) / 2)) * math.sqrt(((1.0 * up / (up + down)) * (1 - (1.0 * up / (up + down))) + (st.norm.ppf(1 - (1 - confidence) / 2)) * (st.norm.ppf(1 - (1 - confidence) / 2)) / (4 * (up + down))) / (up + down))) / (1 + (st.norm.ppf(1 - (1 - confidence) / 2)) * (st.norm.ppf(1 - (1 - confidence) / 2)) / (up + down)))
    return rating

# df["wilson_lower_bound_vectorized"] = wilson_lower_bound_vectorized(df["helpful_yes"], df["helpful_no"])

df.head(20)
df["score_pos_neg_diff"].value_counts()
df["score_pos_neg_diff2"].value_counts()

df["score_average_rating"].value_counts()
df["score_average_rating_vectorized"].value_counts()



###########################################

=R2/86400000+DATE(1970;1;1)

# UnixTime Formülü

###########################################




######################### ÇÖZÜM #######################

def score_average_rating(up, down):
    if up + down == 0:
        return 0
    return up / (up + down)

df["score_average_rating"] = df.apply(lambda x: score_average_rating(x["helpful_yes"], x["helpful_no"]), axis=1)

################### ALTERNATİF ÇÖZÜM ##################

def score_average_rating_vectorized(up, down):
    # Oyların toplamı 0 olan yerler için bir mask oluştur
    zero_mask = (up + down) == 0
    # Vektörleştirilmiş bölme işlemi
    ratings = np.where(zero_mask, 0, up / (up + down))
    return ratings

df["score_average_rating_vectorized"] = score_average_rating_vectorized(df["helpful_yes"], df["helpful_no"])

# fonksiyonu direkt olarak iki pandas Serisi'ne uygulamaya çalışınca sorun oluyor. Fonksiyonumuz  için tekil değerler üzerinde çalışacak şekilde tasarlandığı için.
# Ancak df["helpful_yes"] ve df["helpful_no"] pandas Serileri olduğundan sorun yaşıyoruz. bu yüzden de lambda kullanılıyor zaten
# Bu nedenle, fonksiyonumuzu doğrudan bu iki Seri'ye uygulamaya çalıştığınızda if up + down == 0: satırı, iki Seri'nin toplamının sıfır olup olmadığını kontrl etmeye çalışır.
# ve bu da  "ValueError" hatasına neden oluyor.
# Bu sorunu çözmek için, fonksiyonunun her bir satırda ayrı ayrı çalışacak şekilde uygulamak için apply() fonksiyonunu ve bir lambda fonksiyonunu kullanıyoruz.
# Gelelim biz ne yaptık?
# fonksiyonu vektörleştirilmiş bir yapıda yazarak doğrudan pandas Serilerine uygulayabiliyoruz. Bu, NumPy'ın vektörleştirme özelliklerinden yararlanarak yapılır.
