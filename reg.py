#linear Regression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#Generate Synthetic data
np.random.seed(42)
X = 2 * np.random.rand(100, 1) #Random Feature
y = 4 + 3 * X + np.random.randn(100, 1) #Linear Relationship with Noise

#Create Linear Regression Model
model = LinearRegression()

#Fit the model to the data
model.fit(X, y)

#Get the parameters (slope and intercept)
slope = model.coef_[0][0]
intercept = model.intercept_[0]

#Make predictions using the model
X_new = np.array([[0] , [2]]) #two new data points for predictions
y_pred = model.predict(X_new)

#plot the data and the linear Regression line
plt.scatter(X, y, label = 'Data points')
plt.plot(X_new,y_pred,'r-',label = f'linear regression line(y={intercept:.2f}+{slope:.2f}x)')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.title('simple linear regression example')
plt.show()