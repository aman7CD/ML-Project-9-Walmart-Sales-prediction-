
## Importing the Dependecies 

import numpy as np

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

from sklearn.metrics import r2_score

from sklearn.preprocessing import LabelEncoder




## Data Colection and Preprocessing

data = pd.read_csv("/kaggle/input/bigmart-sales-data/Train.csv")

data.isnull().sum()

data.head()

data["Item_Weight"].fillna(data["Item_Weight"].mean(), inplace=True)
data["Outlet_Size"].fillna(data["Outlet_Size"].mode(), inplace=True)




## Converting data from text to numerical

encoder = LabelEncoder()
data["Item_Identifier"] = encoder.fit_transform(data["Item_Identifier"])
data["Item_Fat_Content"] = encoder.fit_transform(data["Item_Fat_Content"])
data["Item_Type"] = encoder.fit_transform(data["Item_Type"])
data["Outlet_Identifier"] = encoder.fit_transform(data["Outlet_Identifier"])
data["Outlet_Size"] = encoder.fit_transform(data["Outlet_Size"])
data["Outlet_Location_Type"] = encoder.fit_transform(data["Outlet_Location_Type"])
data["Outlet_Type"] = encoder.fit_transform(data["Outlet_Type"])




## Splitting the Data
x = data.drop(["Item_Outlet_Sales"],axis=1)

y = data["Item_Outlet_Sales"]

xtn,xtt,ytn,ytt = train_test_split(x,y, test_size=0.1, random_state=2, )
x_plot = xtt["Item_MRP"]




## Training The Model

model_1 = XGBRegressor()
model_2 = LinearRegression()

model_1.fit(xtn,ytn)
model_2.fit(xtn,ytn)




## Model Evaluation Through r2 Score and MSE

y_pred_1 = model_1.predict(xtt)
y_pred_2 = model_2.predict(xtt)

r2score_1 =r2_score(ytt,y_pred_1)
r2score_2 = r2_score(ytt,y_pred_2)


print(f"The r2score of XGBoost is {r2score_1} ")
print(f"The r2score of LinearRegression is {r2score_2} ")

plt.scatter(x_plot,ytt)
plt.scatter(x_plot,y_pred_1)
plt.scatter(x_plot,y_pred_2)
plt.xlabel("Item_MRP")
plt.ylabel("Total Sales")
plt.legend(["sales","sales_pred_by_xgb","sales_pred_by_LR"])
plt.show()
