##BEST FIT LINE - Cost function using un-constrained method - Gradient descent
##Ex 1. Use the downloaded data
##Ex 2. Convert this data to array
##Ex 3. Define the learning rate and no. of iterations as 0.0001 and 1000 respectively along with y-intercept and slope
##Ex 4. Create the functions to get the BEST FIT line 
##    1. Compute error for the line given the points
##    2. Step gradient function
##    3. Gradient descent

##Ex 5. Display using scatter plot the data points and the best fit line
##Ex 6. Display the Gradient and y-intercept value in the form y = mx+c
##Ex 7. Find the BEST FIT line i.e., m and c of y=mx+c with least error using trial and error method i.e., modify learning rate or iterations or both 


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy import *


def comp_err(b, m, pts):
	totErr = 0
	for i in range(0, len(pts)):
		x = pts[i, 0]
		y = pts[i, 1]
		totErr += (y - (m * x + b)) ** 2
		return totErr / float(len(pts))

def step_gradient(b_current, m_current, points, learningRate):
	b_gradient = 0
	m_gradient = 0
	N = float(len(points))
	for i in range(0, len(points)):
		x = points[i, 0]
		y = points[i, 1]
		b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
		m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current))
	new_b = b_current - (learningRate * b_gradient)
	new_m = m_current - (learningRate * m_gradient)
	return [new_b, new_m]

def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
	b = starting_b
	m = starting_m
	for i in range(num_iterations):
		b, m = step_gradient(b, m, array(points), learning_rate)
		return [b, m]


def plot_dp():

	Input_file = np.genfromtxt('auto-mpg.csv', delimiter=',', skip_header=1)
	Num = np.shape(Input_file)[0]
	X = np.hstack((np.ones(Num).reshape(Num, 1), Input_file[:, 4].reshape(Num, 1)))
	Y = Input_file[:, 0]
 
	X[:, 1] = (X[:, 1]-np.mean(X[:, 1]))/np.std(X[:, 1])
 
	wght = np.array([0, 0])
 
	max_iter = 1000
	eta = 1E-4
	for t in range(0, max_iter):
		grad_t = np.array([0., 0.])
		for i in range(0, Num):
			x_i = X[i, :]
			y_i = Y[i]
			h = np.dot(wght, x_i)-y_i
			grad_t += 2*x_i*h
			wght = wght - eta*grad_t
	tt = np.linspace(np.min(X[:, 1]), np.max(X[:, 1]), 10)
	bf_line = wght[0]+wght[1]*tt
	plt.plot(X[:, 1], Y, 'kx', tt, bf_line, 'r-')
	plt.xlabel('Weight (Normalized)')
	plt.ylabel('MPG')
	plt.title('Linear Regression')
	plt.show()


def execute():
	data = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data-original",
                       
                       delim_whitespace = True, header=None,
                       names = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration',
                               'model', 'origin', 'car_name'])

	data.to_csv('auto-mpg.csv', index=False)

	data.head()
	data_pts = genfromtxt("auto-mpg.csv", delimiter=",")
	plt_pts = data_pts
	learn_rate = 0.0001
	init_b = 0
	init_m = 0
	itr = 1000
	print ('Compute Error = ', comp_err(init_b,init_m,data_pts))
#   getting b and m in array through gradient descent
	[b, m] = gradient_descent_runner(data_pts, init_b, init_m, learn_rate, itr)
	print ('After Gradient Descent Function - Compute Error =', comp_err(b,m,data_pts))
	print('plottting..')
	plot_dp()
if __name__ == '__main__':
	execute()
