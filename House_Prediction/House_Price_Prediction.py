import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.datasets
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics

from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()
print(housing)
house_price_dataframe=pd.DataFrame(housing.data,columns=housing.feature_names)
print(house_price_dataframe.head())

#Adding the targated "Price" to the DataFrame
house_price_dataframe['price']=housing.target
print(house_price_dataframe.head())

#Shape of Dataset is
print(house_price_dataframe.shape)

#Missing Values of the Dataset are
print(house_price_dataframe.isnull().sum())

# Statistical Values of Dataset are
print(house_price_dataframe.describe())


# correlation = house_price_dataframe.corr()
# plt.figure(figsize=(10,10))
#
# #Constructing the HeatMap for the DataFrame
# sns.heatmap(correlation,cbar=True,square=True,fmt='.1f',annot=True,annot_kws={'size':8},cmap='Blues')
# plt.show()

#Splitting the Dataset for Training and Testing Data
X = house_price_dataframe.drop(['price'],axis=1)
Y = house_price_dataframe['price']
print(X)

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=42)
print(X.shape,X_train.shape,X_test.shape)

#Model Training
model = XGBRegressor()

#Training the Model with X_train
print(model.fit(X_train,Y_train))

#Accuracy for Prediction on Training Data
training_data_prediction = model.predict(X_train)
print(training_data_prediction)

#R Squared Error
score = metrics.r2_score(Y_train,training_data_prediction)
print("R Squared Error for training data is : ",score)

#Mean Absolute Error
score_2=metrics.mean_absolute_error(Y_train,training_data_prediction)
print("Mean Absolute Error for training data is : ",score_2)

plt.scatter(Y_train,training_data_prediction)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual Prices vs Predicted Prices")
plt.show()
#Prediction on Test Data
testing_data_prediction = model.predict(X_test)

score_3 = metrics.r2_score(Y_test,testing_data_prediction)
print("R Squared Error for testing data is : ",score_3)

score_4=metrics.mean_absolute_error(Y_test,testing_data_prediction)
print("Mean Absolute Error for testing data is : ",score_4)