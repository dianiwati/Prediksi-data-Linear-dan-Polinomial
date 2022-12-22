import numpy as np
from sklearn.linear_model import LinearRegression

# x = Data, y = Target
x = [[1], [3], [5], [7], [9]]
y = [2, 6, 10, 14, 18]

regr = LinearRegression().fit(x,y)
regr.score(x, y)

#Data Uji
predict = np.array([[10]])

#Menampilkan Data Prediksi
print ("Prediksi")
print ("Input = ", predict)
print ("Output = ", regr.predict (predict))