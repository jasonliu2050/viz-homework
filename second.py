import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
df = pd.read_csv('housing.data', sep='\s+', header=None)
# Setting columns to dataset
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

home_price = df["MEDV"]
tax_price = df[["TAX","MEDV","DIS"]]
zn_price = df["NOX"]
crim = df["CRIM"]
rm_crim_b = df[["RM","CRIM","B"]]

#multiple plots in same figure
fig, axes = plt.subplots(3, 2, figsize=(10,9))

#label color
axes[0][0].plot(home_price, label="Price")
axes[0][1].plot(tax_price)
axes[1][0].scatter(df['TAX'], df['RM'], color='green', marker='s', alpha=0.2, s=df['MEDV'] ** 1.1)
#multiple plots in same axes
axes[1][1].plot(home_price,df["RM"],label="room")
axes[1][1].plot(home_price,crim,label="crim")
axes[1][1].plot(home_price,df["B"], label="block")

axes[2][0].plot(df)
axes[2][1].plot(home_price, df)
#title xlable ylabel legend for single plot
axes[0][0].set_title("Home Price")
axes[0][1].set_title("Tax Price & DIS")
axes[1][0].set_title("TAX & Rooms ")
axes[1][1].set_title("RM CRIM B to PRICE")

axes[2][0].set_title("All figures ")
axes[2][1].set_title("PRICE to All figure")

axes[0][0].set_xlabel('')
axes[0][0].set_ylabel('PRICE')
axes[0][1].set_xlabel('')
axes[0][1].set_ylabel('Tax Cost DIS')
axes[1][0].set_xlabel('')
axes[1][0].set_ylabel('ROOM')
axes[1][1].set_xlabel('')
axes[1][1].set_ylabel('room-crim-b')

axes[1][1].legend()
plt.show()