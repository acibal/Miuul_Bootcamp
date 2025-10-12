##################################################
# Smoothing Methods (Holt-Winters)
##################################################

import itertools
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.tsa.api as smt

warnings.filterwarnings('ignore')

############################
# Veri Seti
############################

# Atmospheric CO2 from Continuous Air Samples at Mauna Loa Observatory, Hawaii, U.S.A.
# Period of Record: March 1958 - December 2001

data = sm.datasets.co2.load_pandas()
y = data.data

y = y['co2'].resample('MS').mean()

y.isnull().sum()

y = y.fillna(y.bfill())

# ffill() önceki değer bfill() sonraki değer

y.plot(figsize=(15, 6))
plt.show()

############################
# Holdout
############################

train = y[:'1997-12-01']
len(train)  # 478 ay

# 1998'ilk ayından 2001'in sonuna kadar test set.
test = y['1998-01-01':]
len(test)  # 48 ay

##################################################
# Zaman Serisi Yapısal Analizi
##################################################

# Durağanlık Testi (Dickey-Fuller Testi)

def is_stationary(y):

    # "HO: Non-stationary"
    # "H1: Stationary"

    p_value = sm.tsa.stattools.adfuller(y)[1]
    if p_value < 0.05:
        print(F"Result: Stationary (H0: non-stationary, p-value: {round(p_value, 3)})")
    else:
        print(F"Result: Non-Stationary (H0: non-stationary, p-value: {round(p_value, 3)})")

is_stationary(y)

# Zaman Serisi Bileşenleri ve Durağanlık Testi
def ts_decompose(y, model="additive", stationary=False):
    result = seasonal_decompose(y, model=model)
    fig, axes = plt.subplots(4, 1, sharex=True, sharey=False)
    fig.set_figheight(10)
    fig.set_figwidth(15)

    axes[0].set_title("Decomposition for " + model + " model")
    axes[0].plot(y, 'k', label='Original ' + model)
    axes[0].legend(loc='upper left')

    axes[1].plot(result.trend, label='Trend')
    axes[1].legend(loc='upper left')

    axes[2].plot(result.seasonal, 'g', label='Seasonality & Mean: ' + str(round(result.seasonal.mean(), 4)))
    axes[2].legend(loc='upper left')

    axes[3].plot(result.resid, 'r', label='Residuals & Mean: ' + str(round(result.resid.mean(), 4)))
    axes[3].legend(loc='upper left')
    plt.show(block=True)

    if stationary:
        is_stationary(y)

ts_decompose(y, stationary=True)

##################################################
# Single Exponential Smoothing
##################################################

# SES = Level

ses_model = SimpleExpSmoothing(train).fit(smoothing_level=0.5) # smooting_level'ı aslında girmiyoruz model kendi en uygununa bulup kuruyor. Tek bileşenli olduğu görünsün diye girdik.

y_pred = ses_model.forecast(48)
# SES'de predict değil forecest var.
# Test testinde 48 ay olduğu için burada da 48 ya tahmin et parametresini giriyoruz.

mean_absolute_error(test, y_pred)
# mae yerine (mse) mean squared error ya da (rmse) root mean squared error da kullanılabilir.

train.plot(title="Single Exponential Smoothing")
test.plot()
y_pred.plot()
plt.show(block=True)


train["1985":].plot(title="Single Exponential Smoothing")
test.plot()
y_pred.plot()
plt.show(block=True)

def plot_co2(train, test, y_pred, title):
    mae = mean_absolute_error(test, y_pred)
    train["1985":].plot(legend=True, label="TRAIN", title=f"{title}, MAE: {round(mae,2)}")
    test.plot(legend=True, label="TEST", figsize=(6, 4))
    y_pred.plot(legend=True, label="PREDICTION")
    plt.show(block=True)

# bu fonksiyondaki 1985 daha yakından tahminleri görmek için yazdık. Genellenemez bir girdi. Veri setine göre değiştir ya da kaldır.

plot_co2(train, test, y_pred, "Single Exponential Smoothing")

ses_model.params

############################
# Hyperparameter Optimization
############################

def ses_optimizer(train, alphas, step=48):

    best_alpha, best_mae = None, float("inf")

    for alpha in alphas:
        ses_model = SimpleExpSmoothing(train).fit(smoothing_level=alpha)
        y_pred = ses_model.forecast(step)
        mae = mean_absolute_error(test, y_pred)

        if mae < best_mae:
            best_alpha, best_mae = alpha, mae

        print("alpha:", round(alpha, 2), "mae:", round(mae, 4))
    print("best_alpha:", round(best_alpha, 2), "best_mae:", round(best_mae, 4))
    return best_alpha, best_mae

alphas = np.arange(0.5, 1, 0.01)
# 0.1'den 1'e kadar da denenebilir. Ama biliyoruz ki alfanın 0.5'ten büyük olması daha mantıklı.

# yt_sapka = a * yt-1 + (1-a)* (yt_-1)_sapka

ses_optimizer(train, alphas)

best_alpha, best_mae = ses_optimizer(train, alphas)

############################
# Final SES Model
############################

ses_model = SimpleExpSmoothing(train).fit(smoothing_level=best_alpha)
y_pred = ses_model.forecast(48)

plot_co2(train, test, y_pred, "Single Exponential Smoothing")


##################################################
# Double Exponential Smoothing (DES)
##################################################

# DES: Level (SES) + Trend
# toplamsal ve çarpımsal modeller:
# mevsimsellik ve artık bileşenleri trendten bağımsızsa toplamsal seri değilse çarpımsal seri
# grafikten mevsimsellik ve artıklar sıfır etrafında dağılıyorsa, trendten bağımsızdır ve toplamsaldır.
# artıklar = error'lar
# eğer grafiklerden yorumlamak istemezsek iki modeli de kurup düşük hatalı modeli seçiyoruz.
# y(t) = Level + Trend + Seasonality + Noise
# y(t) = Level * Trend * Seasonality * Noise


ts_decompose(y) # burdan grafikleriden görebiliriz.

des_model = ExponentialSmoothing(train, trend="add").fit(smoothing_level=0.5,
                                                         smoothing_trend=0.5)
# smoothing_trend = smoothing_slope aynı şeydir.

y_pred = des_model.forecast(48)

plot_co2(train, test, y_pred, "Double Exponential Smoothing")

############################
# Hyperparameter Optimization
############################

def des_optimizer(train, alphas, betas, step=48):
    best_alpha, best_beta, best_mae = None, None, float("inf")
    for alpha in alphas:
        for beta in betas:
            des_model = ExponentialSmoothing(train, trend="add").fit(smoothing_level=alpha, smoothing_slope=beta)
            y_pred = des_model.forecast(step)
            mae = mean_absolute_error(test, y_pred)
            if mae < best_mae:
                best_alpha, best_beta, best_mae = alpha, beta, mae
            print("alpha:", round(alpha, 2), "beta:", round(beta, 2), "mae:", round(mae, 4))
    print("best_alpha:", round(best_alpha, 2), "best_beta:", round(best_beta, 2), "best_mae:", round(best_mae, 4))
    return best_alpha, best_beta, best_mae

alphas = np.arange(0.01, 1, 0.10)
betas = np.arange(0.01, 1, 0.10)

best_alpha, best_beta, best_mae = des_optimizer(train, alphas, betas)

############################
# Final DES Model
############################

final_des_model = ExponentialSmoothing(train, trend="add").fit(smoothing_level=best_alpha,
                                                               smoothing_slope=best_beta)
# trend="add" additive mevsimsellik ve artıklar trendten bağımsız olduğu için. (toplamsal seri)
# trend="mul" multiplicative
# final_des_model = ExponentialSmoothing(train, trend="mul").fit(smoothing_level=best_alpha,
#                                                                smoothing_slope=best_beta)
# iki türlü de hataya bakılabilir eğer görsel yorumuna güvenmiyorsan.

y_pred = final_des_model.forecast(48)

plot_co2(train, test, y_pred, "Double Exponential Smoothing")

##################################################
# Triple Exponential Smoothing (Holt-Winters)
##################################################

# TES = SES + DES + Mevsimsellik

tes_model = ExponentialSmoothing(train,
                                 trend="add",
                                 seasonal="add",
                                 seasonal_periods=12).fit(smoothing_level=0.5,
                                                          smoothing_slope=0.5,
                                                          smoothing_seasonal=0.5)
# burda seasonal_periods'un 12 olduğunu görselden anlıyoruz.
# yani mevsimsellik yıllık periodlarla tekrar ediyor.

y_pred = tes_model.forecast(48)
plot_co2(train, test, y_pred, "Triple Exponential Smoothing")

############################
# Hyperparameter Optimization
############################

alphas = betas = gammas = np.arange(0.10, 1, 0.10)

abg = list(itertools.product(alphas, betas, gammas))

def tes_optimizer(train, abg, step=48):
    best_alpha, best_beta, best_gamma, best_mae = None, None, None, float("inf")
    for comb in abg:
        tes_model = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=12).\
            fit(smoothing_level=comb[0], smoothing_slope=comb[1], smoothing_seasonal=comb[2])
        y_pred = tes_model.forecast(step)
        mae = mean_absolute_error(test, y_pred)
        if mae < best_mae:
            best_alpha, best_beta, best_gamma, best_mae = comb[0], comb[1], comb[2], mae
        print([round(comb[0], 2), round(comb[1], 2), round(comb[2], 2), round(mae, 2)])

    print("best_alpha:", round(best_alpha, 2), "best_beta:", round(best_beta, 2), "best_gamma:", round(best_gamma, 2),
          "best_mae:", round(best_mae, 4))

    return best_alpha, best_beta, best_gamma, best_mae

best_alpha, best_beta, best_gamma, best_mae = tes_optimizer(train, abg)


############################
# Final TES Model
############################

final_tes_model = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=12).\
            fit(smoothing_level=best_alpha, smoothing_trend=best_beta, smoothing_seasonal=best_gamma)

y_pred = final_tes_model.forecast(48)

plot_co2(train, test, y_pred, "Triple Exponential Smoothing")


""" Veri seti zenginse sadece train-test olarak değil, 
train-validation-test olarak 3'e bölebiliriz. Eğitimi train'de, 
parametre ayarlaması validation'da, test etmeyi de test setinde yapabiliriz"""