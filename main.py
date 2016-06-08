#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model

path = "./result/"
pass_count = 20000
dataset_size = 100

class Config:
	nn_input_dim = 2
	nn_output_dim = 2
	epsilon = 0.01
	reg_lambda = 0.01

def generate_data():
	np.random.seed(0)
	X, y = datasets.make_moons(dataset_size, noise=0.20)
	return X, y

def visualize_linear(X, y):
	# Визуализация предсказания с помощью библиотечной функции
	clf = linear_model.LogisticRegressionCV()
	clf.fit(X, y)
	plot_decision_boundary(lambda x: clf.predict(x), X, y, 0)
	plt.title("Logistic regression")

def visualize(X, y, model, i):
	plot_decision_boundary(lambda x: predict(x, model), X, y, i)
	plt.title("Logistic regression")

def plot_decision_boundary(pred_func, X, y, i):
	# Ищем макс и мин значения х и у исходного набора точек
	x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
	y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
	h = 0.01
	# создаем сетку точек с расстоянием h между ними
	# D - массив всех возможных точек от левого
	# нижнего угла до правого верхнего сетки координат, между точками h
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
	D = np.c_[xx.ravel(), yy.ravel()]
	# Z - массив точек [0, 1, 1, 0, 0, ...]
	# которые вернула pred_func
	# Zi - предсказание принадлежности Di к одному из классов
	Z = pred_func(D)
	Z = Z.reshape(xx.shape)

	plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
	plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)
	plt.savefig(path + str(i), format='png')
#	plt.show()

def calculate_loss(X, y, model):
	num_examples = len(X)
	W1, W2 = model["W1"], model["W2"]

	y_hat, _, _, _ = forward_propogation(X, model)

	correct_logprobs = -np.log(y_hat[range(num_examples), y])
	data_loss = np.sum(correct_logprobs)
	data_loss += Config.reg_lambda / 2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
	return 1. / num_examples * data_loss

def predict(X, model):
	y_hat, _, _, _ = forward_propogation(X, model)
	# возвращает массив с результатами предсказаний для всех точек на
	# координатной плоскости [0, 1, 1, 0, 0, ..]
	return np.argmax(y_hat, axis=1)

def forward_propogation(X, model):
	# Проход по нейронной сети вперед
	W1, b1, W2, b2 = model["W1"], model["b1"], model["W2"], model["b2"]
	# z1 - вектор-результат взвешенной суммы входного вектора
	# a1 - вектор-результат активационной функции tanh
	# между слоем input и hidden
	z1 = X.dot(W1) + b1
	a1 = np.tanh(z1)
	# то же самое между hidden и output слоем
	# но активацивационная функция - softmax
	z2 = a1.dot(W2) + b2
	exp_scores = np.exp(z2)
	y_hat = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
	return y_hat, z1, a1, z2

def backpropogation(X, y, y_hat, a1, model):
	# Обратное распространение ошибки
	num_examples = len(X)
	delta3 = y_hat
	delta3[range(num_examples), y] -= 1
	dW2 = (a1.T).dot(delta3)
	db2 = np.sum(delta3, axis=0, keepdims=True)
	delta2 = delta3.dot(model["W2"].T) * (1 - np.power(a1, 2))
	dW1 = np.dot(X.T, delta2)
	db1 = np.sum(delta2, axis=0)

	dW2 += Config.reg_lambda * model["W2"]
	dW1 += Config.reg_lambda * model["W1"]

	model["W1"] += -Config.epsilon * dW1
	model["b1"] += -Config.epsilon * db1
	model["W2"] += -Config.epsilon * dW2
	model["b2"] += -Config.epsilon * db2

	return model

def multilayer_perceptron(X, y, nn_hdim, num_passes=pass_count, print_loss=False):
	#	W1 - матрица весов между input и hidden слоем
	#		при создании заполненяется рандомными значениями
	#		рaзмерность (2х3) тк в input слое 2 нейрона, в hidden - 3
	#	b1 - вектор смещения для hidden слоя, на первой итерации заполнен
	#		нулями
	num_examples = len(X)
	np.random.seed(0)
	W1 = np.random.randn(Config.nn_input_dim, nn_hdim) / np.sqrt(Config.nn_input_dim)
	b1 = np.zeros((1, nn_hdim))
	# То же самое для связей между hidden и output слоем
	W2 = np.random.randn(nn_hdim, Config.nn_output_dim) / np.sqrt(nn_hdim)
	b2 = np.zeros((1, Config.nn_output_dim))

	model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

	# График предсказания встроенной функции
	visualize_linear(X, y)
	# График предсказания перцептрона
	visualize(X, y, model, 1)

	# Процесс обучения
	for i in range(0, num_passes):

		y_hat, _, a1, _ = forward_propogation(X, model)

		model = backpropogation(X, y, y_hat, a1, model)

		if i <= 750 and i % 50 == 0:
			print("Loss after iteration %i: %f" % (i, calculate_loss(X, y, model)))
			visualize(X, y, model, i)
		if print_loss and i % 1000 == 0:
			print("Loss after iteration %i: %f" % (i, calculate_loss(X, y, model)))

	return model

def main():
	X, y = generate_data()
	#X - массив точек типа np.array - array([[x1,y1],[x2,y2],...])
	#y - массив значений, принадлежность к классу - [1, 0, 0, 1, 1, ..]
	hidden_neuron_quantity = 3
	model = multilayer_perceptron(X, y, hidden_neuron_quantity, print_loss=True)
	visualize(X, y, model, pass_count)

if __name__ == "__main__":
	main()
