import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split

# Homework Basic Q1 load boston from remote
from sklearn.datasets import load_boston
data = load_boston()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.25, random_state=0)

# Refactor code : feature selection for each feature in one figure
j=0
fig, axes = plt.subplots(4, 3, figsize=(10,9))
for index, feature_name in enumerate(data.feature_names):
    if (j > 2):
        break
    axes[index % 4][j].scatter(data.data[:, index], data.target)
    axes[index % 4][j].set_ylabel('Price', size=9)
    axes[index % 4][j].set_xlabel(feature_name, size= 9)
    if (index % 4  == 3):
        j = j + 1
plt.tight_layout()


# Homework Basic Q2 & Advanced Q1,2 use KNeighborsRegressor and LinearRegression
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

# Training a model using two algorithms and comparing the results
for Model in [LinearRegression, KNeighborsRegressor]:
    model = Model()
    model.fit(X_train, y_train)
    predicted_values = model.predict(X_test)
    print(f"Printing RMSE error for {Model}: {np.sqrt(metrics.mean_squared_error(y_test, predicted_values))}")


#homework Reach Q1 compare multiple algorithms
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn import metrics

clf1 = LinearRegression()
clf2 = Ridge(alpha=1.0)
clf3 = Lasso(alpha=0.1)
clf4 = SVR(gamma='scale', C=1.0, epsilon=0.2)
alg = ["LinearRegression","Ridge","Lasso","SVR"]
clf = [clf1,clf2,clf3,clf4]
for i in [0,1,2,3]:
    clf[i].fit(X_train, y_train)
    predicted = clf[i].predict(X_test)
    expected = y_test
    score = clf[i].score( X_test, y_test)
    plt.figure(figsize=(4, 3))
    plt.scatter(expected, predicted)
    plt.plot([0, 50], [0, 50], '--k')
    plt.axis('tight')
    plt.xlabel('True price ($1000s)')
    plt.ylabel('Predicted price ($1000s)')

    rms = round(np.sqrt(np.mean((predicted - expected) ** 2)),3)
    rms_str = str(list(np.reshape(np.asarray(rms), (1, np.size(rms)))[0]))[1:-1]
    plt.title(alg[i] + " RMS: "+ rms_str)
    plt.tight_layout()
    print(alg[i] +" Score:", score)
    print(alg[i] + f" MAE error(avg abs residual): {metrics.mean_absolute_error(y_test, predicted)}")
    print(alg[i] + f" MSE error: {metrics.mean_squared_error(y_test, predicted)}")
    print(alg[i] + f" RMSE error: {np.sqrt(metrics.mean_squared_error(y_test, predicted))}")

plt.show()
























# home_price = df["MEDV"]
# tax_price = df[["TAX","MEDV","DIS"]]
# zn_price = df["NOX"]
# crim = df["CRIM"]
# rm_crim_b = df[["RM","CRIM","B"]]
#
# #multiple plots in same figure
# fig, axes = plt.subplots(3, 2, figsize=(10,9))
#
# #label color
#
# axes[0][0].plot(home_price, label="Price")
# axes[0][1].plot(tax_price)
# axes[1][0].scatter(df['TAX'], df['RM'], c = df["RAD"], s=df['MEDV']**0.8, alpha=0.7)
#
# #multiple plots in same axes
# axes[1][1].plot(home_price,df["RM"],label="room")
# axes[1][1].plot(home_price,crim,label="crim")
# axes[1][1].plot(home_price,df["B"], label="block")
#
# axes[2][0].plot(df)
# axes[2][1].plot(home_price, df)
# #title xlable ylabel legend for single plot
# axes[0][0].set_title("Home Price")
# axes[0][1].set_title("Tax Price & DIS")
# axes[1][0].set_title("TAX & Rooms ")
# axes[1][1].set_title("RM CRIM B to PRICE")
#
# axes[2][0].set_title("All figures ")
# axes[2][1].set_title("PRICE to All figure")
#
# axes[0][0].set_xlabel('')
# axes[0][0].set_ylabel('PRICE')
# axes[0][1].set_xlabel('')
# axes[0][1].set_ylabel('Tax Cost DIS')
# axes[1][0].set_xlabel('')
# axes[1][0].set_ylabel('ROOM')
# axes[1][1].set_xlabel('')
# axes[1][1].set_ylabel('room-crim-b')
#
# axes[1][1].legend()
# plt.show()

