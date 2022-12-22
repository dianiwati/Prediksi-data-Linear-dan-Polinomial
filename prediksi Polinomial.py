from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
import numpy as np                                      


#database 
#x = data, y = Target 
x = [[0],[1],[2],[3],[4],[5],[6],[7],[8],[9]] 
y = [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]          

#data uji 
predict = np.array([[40]])                         
poly = PolynomialFeatures(degree=2)               
x_=poly.fit_transform(x)                          
predict_ = poly.fit_transform(predict)            
regr = linear_model.LinearRegression()            
regr.fit(x_,y)                                    

#menampilkan data prediksi 
print ("Prediksi")
print ("input = ", predict)
print ("Output =", regr.predict(predict_))
