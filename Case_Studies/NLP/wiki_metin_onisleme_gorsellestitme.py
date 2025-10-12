#################################################
# WIKI 1 - Metin Ã¶niÅŸleme ve GÃ¶rselleÅŸtirme (NLP - Text Preprocessing & Text Visualization)
#################################################

###################f##############################
# Problemin TanÄ±mÄ±
#################################################
# Wikipedia Ã¶rnek datasÄ±ndan metin Ã¶n iÅŸleme, temizleme iÅŸlemleri gerÃ§ekleÅŸtirip, gÃ¶rselleÅŸtirme Ã§alÄ±ÅŸmalarÄ± yapmak.

#################################################
# Veri Seti Hikayesi
#################################################
# Wikipedia datasÄ±ndan alÄ±nmÄ±ÅŸ metinleri iÃ§ermektedir.

#################################################
# Gerekli KÃ¼tÃ¼phaneler ve ayarlar



import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords
from textblob import Word, TextBlob
from warnings import filterwarnings


filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.set_option('display.width', 200)

# DatayÄ± okumak
df = pd.read_csv("NLP/Case_Study_2/wiki-221126-161428/wiki_data.csv", index_col=0)
df.head()
df = df[:2000]

df.head()
df.shape

#################################################
# GÃ¶revler:
#################################################

# GÃ¶rev 1: Metindeki Ã¶n iÅŸleme iÅŸlemlerini gerÃ§ekleÅŸtirecek bir fonksiyon yazÄ±nÄ±z.
# â€¢	BÃ¼yÃ¼k kÃ¼Ã§Ã¼k harf dÃ¶nÃ¼ÅŸÃ¼mÃ¼nÃ¼ yapÄ±nÄ±z.
# â€¢	Noktalama iÅŸaretlerini Ã§Ä±karÄ±nÄ±z.
# â€¢	Numerik ifadeleri Ã§Ä±karÄ±nÄ±z.


def clean_text(text):
    # Normalizing Case Folding
    text = text.str.lower()
    # Punctuations
    text = text.str.replace(r'[^\w\s]', '',regex=True)
    text = text.str.replace(r"\n", '', regex=True)
    # Numbers
    text = text.str.replace(r'\d', '', regex=True)
    return text

df["text"] = clean_text(df["text"])

df.head()



# GÃ¶rev 2: Metin iÃ§inde Ã¶znitelik Ã§Ä±karÄ±mÄ± yaparken Ã¶nemli olmayan kelimeleriÃ§Ä±karacak fonksiyon yazÄ±nÄ±z.
# import nltk
# nltk.download('stopwords')
def remove_stopwords(text):
    stop_words = stopwords.words('English')
    text = text.apply(lambda x: " ".join(x for x in str(x).split() if x not in stop_words))
    return text

df["text"] = remove_stopwords(df["text"])




# GÃ¶rev 3: Metinde az tekrarlayan kelimeleri bulunuz.

pd.Series(' '.join(df['text']).split()).value_counts()[-1000:]



# GÃ¶rev 4: Metinde az tekrarlayan kelimeleri metin iÃ§erisinden Ã§Ä±kartÄ±nÄ±z. (Ä°pucu: lambda fonksiyonunu kullanÄ±nÄ±z.)

sil = pd.Series(' '.join(df['text']).split()).value_counts()[-1000:]
df['text'] = df['text'].apply(lambda x: " ".join(x for x in x.split() if x not in sil))




# GÃ¶rev 5: Metinleri tokenize edip sonuÃ§larÄ± gÃ¶zlemleyiniz.
# import nltk
df["text"].apply(lambda x: TextBlob(x).words)


# GÃ¶rev 6: Lemmatization iÅŸlemini yapÄ±nÄ±z.
# ran, runs, running -> run (normalleÅŸtirme)
# nltk.download('wordnet')
df['text'] = df['text'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

df.head()

# GÃ¶rev 7: Metindeki terimlerin frekanslarÄ±nÄ± hesaplayÄ±nÄ±z. (Ä°pucu: Barplot grafiÄŸi iÃ§in gerekli)

tf = df["text"].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index() # kodu gÃ¼ncellemek gerekecek

tf.head()

# GÃ¶rev 8: Barplot grafiÄŸini oluÅŸturunuz.

# SÃ¼tunlarÄ±n isimlendirilmesi
tf.columns = ["words", "tf"]
# 5000'den fazla geÃ§en kelimelerin gÃ¶rselleÅŸtirilmesi
tf[tf["tf"] > 2000].plot.bar(x="words", y="tf")
plt.show()

# Kelimeleri WordCloud ile gÃ¶rselleÅŸtiriniz.

# kelimeleri birleÅŸtirdik
text = " ".join(i for i in df["text"])

# wordcloud gÃ¶rselleÅŸtirmenin Ã¶zelliklerini belirliyoruz
wordcloud = WordCloud(max_font_size=50,
max_words=100,
background_color="black").generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

# GÃ¶rev 9: TÃ¼m aÅŸamalarÄ± tek bir fonksiyon olarak yazÄ±nÄ±z.
# â€¢	Metin Ã¶n iÅŸleme iÅŸlemlerini gerÃ§ekleÅŸtiriniz.
# â€¢	GÃ¶rselleÅŸtirme iÅŸlemlerini fonksiyona argÃ¼man olarak ekleyiniz.
# â€¢	Fonksiyonu aÃ§Ä±klayan 'docstring' yazÄ±nÄ±z.

df = pd.read_csv("NLP/Case_Study_2/wiki-221126-161428/wiki_data.csv", index_col=0)


def wiki_preprocess(text, Barplot=False, Wordcloud=False):
    """
    Textler Ã¼zerinde Ã¶n iÅŸleme iÅŸlemleri yapar.

    :param text: DataFrame'deki textlerin olduÄŸu deÄŸiÅŸken
    :param Barplot: Barplot gÃ¶rselleÅŸtirme
    :param Wordcloud: Wordcloud gÃ¶rselleÅŸtirme
    :return: text


    Example:
            wiki_preprocess(dataframe[col_name])

    """
    # Normalizing Case Folding
    text = text.str.lower()
    # Punctuations
    text = text.str.replace(r'[^\w\s]', '', regex=True)
    text = text.str.replace(r"\n", '', regex=True)
    # Numbers
    text = text.str.replace(r'\d', '', regex=True)
    # Stopwords
    sw = stopwords.words('English')
    text = text.apply(lambda x: " ".join(x for x in str(x).split() if x not in sw))
    # Rarewords / Custom Words
    sil = pd.Series(' '.join(text).split()).value_counts()[-1000:]
    text = text.apply(lambda x: " ".join(x for x in x.split() if x not in sil))


    if Barplot:
        # Terim FrekanslarÄ±nÄ±n HesaplanmasÄ±
        tf = text.apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()
        # SÃ¼tunlarÄ±n isimlendirilmesi
        tf.columns = ["words", "tf"]
        # 5000'den fazla geÃ§en kelimelerin gÃ¶rselleÅŸtirilmesi
        tf[tf["tf"] > 2000].plot.bar(x="words", y="tf")
        plt.show()

    if Wordcloud:
        # Kelimeleri birleÅŸtirdik
        text = " ".join(i for i in text)
        # wordcloud gÃ¶rselleÅŸtirmenin Ã¶zelliklerini belirliyoruz
        wordcloud = WordCloud(max_font_size=50,
                              max_words=100,
                              background_color="white").generate(text)
        plt.figure()
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.show()

    return text

wiki_preprocess(df["text"])

wiki_preprocess(df["text"], True, True)