import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score,r2_score,mean_squared_error,mean_absolute_error
from sklearn.model_selection import train_test_split
import pickle

data = pd.read_csv("D:\\Internship project\\Forest_prediciton\\forest.csv")

print(data.head())

check_isnull = data.isnull()
print(check_isnull)

description = data.describe()
print(description)

X = data.drop(columns=["Id","Cover_Type"],axis=1)
Y = data["Cover_Type"]

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,random_state=42,train_size=0.8)

dr = DecisionTreeRegressor()
dr.fit(X_train,Y_train)
y_pred_dr = dr.predict(X_test)
print("Decision Tree R2 ",r2_score(Y_test,y_pred_dr))
print("Decision Tree MSE ",mean_squared_error(Y_test,y_pred_dr))
print("Decision Tree MAE ",mean_absolute_error(Y_test,y_pred_dr))

rm = RandomForestRegressor()
rm.fit(X_train,Y_train)
y_pred_rm = rm.predict(X_test)
print("\n")
print("Ramdom Forest R2 ",r2_score(Y_test,y_pred_rm))
print("Random Forest MSE ",mean_squared_error(Y_test,y_pred_rm))
print("Random Forest MAE ",mean_absolute_error(Y_test,y_pred_rm))

with open("random_forest_model.pkl","wb") as file:
    pickle.dump(rm,file)
