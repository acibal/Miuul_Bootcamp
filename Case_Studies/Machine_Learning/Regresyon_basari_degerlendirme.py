
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.float_format', lambda x: '%.2f' % x)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score

Liste = [[5, 600], [7, 900], [3, 550], [3, 500], [2, 400], [7, 950], [3, 540], [10, 1200], [6, 900], [4, 550], [8, 1100], [1, 460], [1, 400], [9, 1000], [1, 380]]
# sözlükle daha kolaymış

######################################################
# Simple Linear Regression with OLS Using Scikit-Learn
######################################################

df = pd.DataFrame(Liste, columns= ["deneyim_yili", "maas"])
df.head()

X = df[["deneyim_yili"]]
y = df[["maas"]]

##########################
# Model
##########################

reg_model = LinearRegression().fit(X, y)

# y_hat = b + w*maas

# sabit (b - bias)
reg_model.intercept_[0]
# 274.3560

# maas'ın katsayısı (w1)
reg_model.coef_[0][0]
# 90.2094

##########################
# Tahmin
##########################

# Modelin Görselleştirilmesi
g = sns.regplot(x=X, y=y, scatter_kws={'color': 'b', 's': 9},
                ci=False, color="r")

g.set_title(f"Model Denklemi: Sales = {round(reg_model.intercept_[0], 2)} + TV*{round(reg_model.coef_[0][0], 2)}")
g.set_ylabel("Maas")
g.set_xlabel("Deneyim_yili")
plt.xlim(0, 11)
plt.ylim(bottom=300)
plt.show()


##########################
# Tahmin Başarısı
##########################

# MSE
y_pred = reg_model.predict(X)
mean_squared_error(y, y_pred)
# 4437.8499
y.mean()
y.std()

# RMSE
np.sqrt(mean_squared_error(y, y_pred))
# 66.6172

# MAE
mean_absolute_error(y, y_pred)
# 54.3204

# R-KARE
reg_model.score(X, y)
# 0.9396















