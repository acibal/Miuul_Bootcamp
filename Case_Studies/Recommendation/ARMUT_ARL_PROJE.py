
#########################
# İş Problemi
#########################

# Türkiye’nin en büyük online hizmet platformu olan Armut, hizmet verenler ile hizmet almak isteyenleri buluşturmaktadır.
# Bilgisayarın veya akıllı telefonunun üzerinden birkaç dokunuşla temizlik, tadilat, nakliyat gibi hizmetlere kolayca
# ulaşılmasını sağlamaktadır.
# Hizmet alan kullanıcıları ve bu kullanıcıların almış oldukları servis ve kategorileri içeren veri setini kullanarak
# Association Rule Learning ile ürün tavsiye sistemi oluşturulmak istenmektedir.


#########################
# Veri Seti
#########################
#Veri seti müşterilerin aldıkları servislerden ve bu servislerin kategorilerinden oluşmaktadır.
# Alınan her hizmetin tarih ve saat bilgisini içermektedir.

# UserId: Müşteri numarası
# ServiceId: Her kategoriye ait anonimleştirilmiş servislerdir. (Örnek : Temizlik kategorisi altında koltuk yıkama servisi)
# Bir ServiceId farklı kategoriler altında bulanabilir ve farklı kategoriler altında farklı servisleri ifade eder.
# (Örnek: CategoryId’si 7 ServiceId’si 4 olan hizmet petek temizliği iken CategoryId’si 2 ServiceId’si 4 olan hizmet mobilya montaj)
# CategoryId: Anonimleştirilmiş kategorilerdir. (Örnek : Temizlik, nakliyat, tadilat kategorisi)
# CreateDate: Hizmetin satın alındığı tarih

# !pip install mlxtend
import pandas as pd
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
# çıktının tek bir satırda olmasını sağlar.
pd.set_option('display.expand_frame_repr', False)
from mlxtend.frequent_patterns import apriori, association_rules


#########################
# GÖREV 1: Veriyi Hazırlama
#########################

# Adım 1: armut_data.csv dosyasınız okutunuz.

df_ = pd.read_csv("Data_Science_BC/W5_Recommendation/Case_Studies/ArmutARL-221114-234936/armut_data.csv")
df = df_.copy()
df.head()
df.shape

# Adım 2: ServisID her bir CategoryID özelinde farklı bir hizmeti temsil etmektedir.
# ServiceID ve CategoryID'yi "_" ile birleştirerek hizmetleri temsil edecek yeni bir değişken oluşturunuz.

df.ServiceId = df.ServiceId.astype(str)
df.CategoryId = df.CategoryId.astype(str)
df["Hizmet"] = df[["ServiceId", "CategoryId"]].apply(lambda x: "_".join(x), axis=1)
df.head()
df.ServiceId = df.ServiceId.astype(int)
df.CategoryId = df.CategoryId.astype(int)

################### alternatifi (kısa) #####################

df["Hizmet"] = df[["ServiceId", "CategoryId"]].apply(lambda x: "_".join(x.astype(str)), axis=1)

############################################################

# Adım 3: Veri seti hizmetlerin alındığı tarih ve saatten oluşmaktadır, herhangi bir sepet tanımı (fatura vb. ) bulunmamaktadır.
# Association Rule Learning uygulayabilmek için bir sepet (fatura vb.) tanımı oluşturulması gerekmektedir.
# Burada sepet tanımı her bir müşterinin aylık aldığı hizmetlerdir. Örneğin; 7256 id'li müşteri 2017'in 8.ayında aldığı 9_4, 46_4 hizmetleri bir sepeti;
# 2017’in 10.ayında aldığı  9_4, 38_4  hizmetleri başka bir sepeti ifade etmektedir. Sepetleri unique bir ID ile tanımlanması gerekmektedir.
# Bunun için öncelikle sadece yıl ve ay içeren yeni bir date değişkeni oluşturunuz. UserID ve yeni oluşturduğunuz date değişkenini "_"
# ile birleştirirek ID adında yeni bir değişkene atayınız.

df.shape
df.describe().T
df.head()
df.CreateDate.dtype

df["New_Date"] = df["CreateDate"].apply(lambda x: x[0:7])
df.UserId = df.UserId.astype(str)
df.UserId = df.UserId.astype(int)

##################### alternatifi (kısa) #########################

from datetime import datetime
df["CreateDate"] = pd.to_datetime(df["CreateDate"]) # Datetime sütununu datetime veri türüne çevirme
df["New_Date"] = df["CreateDate"].dt.strftime("%Y-%m")

##################################################################

df["SepetID"] = df[["UserId", "New_Date"]].apply(lambda x: "_".join(x.astype(str)), axis=1)
df.head()

#########################
# GÖREV 2: Birliktelik Kuralları Üretiniz
#########################

# Adım 1: Aşağıdaki gibi sepet hizmet pivot table’i oluşturunuz.

# Hizmet         0_8  10_9  11_11  12_7  13_11  14_7  15_1  16_8  17_5  18_4..
# SepetID
# 0_2017-08        0     0      0     0      0     0     0     0     0     0..
# 0_2017-09        0     0      0     0      0     0     0     0     0     0..
# 0_2018-01        0     0      0     0      0     0     0     0     0     0..
# 0_2018-04        0     0      0     0      0     1     0     0     0     0..
# 10000_2017-08    0     0      0     0      0     0     0     0     0     0..

def create_sepetid_hizmet_df(dataframe, id=False):
    if id:
        return dataframe.groupby(['SepetID', "Hizmet"])['UserId'].count().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)
    else:
        return dataframe.groupby(['SepetID', "Hizmet"])['UserId'].count().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)

spt_hzmt_df = create_sepetid_hizmet_df(df, id=True)
spt_hzmt_df


# Adım 2: Birliktelik kurallarını oluşturunuz.

frequent_itemsets = apriori(spt_hzmt_df,
                            min_support=0.01,
                            use_colnames=True)

frequent_itemsets.sort_values("support", ascending=False)

rules = association_rules(frequent_itemsets,
                          metric="lift",
                          min_threshold=0.01)

rules[(rules["support"] > 0.01) & (rules["confidence"] > 0.05) & (rules["lift"] > 3)].sort_values("confidence", ascending=False)

#Adım 3: arl_recommender fonksiyonunu kullanarak en son 2_0 hizmetini alan bir kullanıcıya hizmet önerisinde bulununuz.

df.head()

def arl_recommender(rules_df, product_id, rec_count=1):
    sorted_rules = rules_df.sort_values("confidence", ascending=False)
    recommendation_list = []
    for i, product in enumerate(sorted_rules["antecedents"]):
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])

    return recommendation_list[0:rec_count]


arl_recommender(rules, "2_0", 1)
arl_recommender(rules, "2_0", 2)
arl_recommender(rules, "2_0", 3)

