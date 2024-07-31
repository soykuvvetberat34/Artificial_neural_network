from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import mean_squared_error,r2_score
import pandas as pd
import numpy as np

datas=pd.read_csv("C:\\Users\\berat\\pythonEğitimleri\\python\\Turkcell Makine Öğrenmesi\\Hitters.csv")
datas=datas.dropna()
dms=pd.get_dummies(datas[["League","Division","NewLeague"]])
y=datas["Salary"]
x_=datas.drop(["Salary","League","Division","NewLeague"],axis=1).astype("float64")
x=pd.concat([x_,dms],axis=1)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=42)

#sadece bağımsız değişkenleri standartize ediyoruz
sc=StandardScaler()
x_train_scaled=sc.fit_transform(x_train)
x_test_scaled=sc.fit_transform(x_test)
mlpregressor_model=MLPRegressor()
mlpregressor_model.fit(x_train_scaled,y_train)
predict_mlpr=mlpregressor_model.predict(x_test_scaled)
rmse=np.sqrt(mean_squared_error(y_test,predict_mlpr))

#Model tuning
mlp_params={
    "alpha":[0.1,0.01,0.02,0.001,0.0001],#ceza parametreleri
    "hidden_layer_sizes":[(10,20),(5,5),(100,100)]#gizli katman sayısı
    #hidden_layer_sizes 2 tane gizli katman koy ve bu her katman içerisine koyulacak nöron sayısı
    #örneğin 10,20 de iki katman ve ilk katmanda 10 nöron ikincide 20 nöron var
}
mlp_cv_=GridSearchCV(mlpregressor_model,mlp_params,cv=5,verbose=2,n_jobs=-1)
mlp_cv_.fit(x_train_scaled,y_train)
alpha=mlp_cv_.best_params_["alpha"]
hls=mlp_cv_.best_params_["hidden_layer_sizes"]
mlp_reg_tuned=MLPRegressor(alpha=alpha,hidden_layer_sizes=hls)
mlp_reg_tuned.fit(x_train_scaled,y_train)
predict_tuned_mlpr=mlp_reg_tuned.predict(x_test_scaled)
rmse2=np.sqrt(mean_squared_error(y_test,predict_tuned_mlpr))
print(rmse2)
print(rmse)






