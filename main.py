#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import time
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model

path = "./result/"
dataset_size = 200
loss_delta = 0.0000001

class Profiler(object):
	def __enter__(self):
		self._startTime = time.time()

	def __exit__(self, type, value, traceback):
		print "\tElapsed time: {:.3f} sec".format(time.time() - self._startTime)

class Config:
	nn_input_dim = 2
	nn_output_dim = 2
	epsilon = 0.01
	reg_lambda = 0.01


def generate_data():
#	np.random.seed(0)
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
#	plt.scatter(X[:,0], X[:,1], s=40, c=y)
	plt.savefig(path + str(i) + ".png", format='png')


def calculate_loss(X, y, model):
	num_examples = len(X)
	W1, W2 = model["W1"], model["W2"]

	y_hat, _, _, _ = forward_propagation(X, model)

	correct_logprobs = -np.log(y_hat[range(num_examples), y])
	data_loss = np.sum(correct_logprobs)
	#регулязация
	data_loss += Config.reg_lambda / 2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))

	return 1. / num_examples * data_loss


def predict(X, model):
	output, _, _, _ = forward_propagation(X, model)
	# возвращает массив с результатами предсказаний для всех точек на
	# координатной плоскости [0, 1, 1, 0, 0, ..]
	return np.argmax(output, axis=1)


def forward_propagation(X, model):
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

	return dW1, db1, dW2, db2


def train_network(X, y, model, print_loss=False):
	# Процесс обучения
	current_loss = 0
	last_loss = 0
	i = 0

	while True:
		y_hat, _, a1, _ = forward_propagation(X, model)

		dW1, db1, dW2, db2 = backpropogation(X, y, y_hat, a1, model)

		dW2 += Config.reg_lambda * model["W2"]
		dW1 += Config.reg_lambda * model["W1"]

		model["W1"] += -Config.epsilon * dW1
		model["b1"] += -Config.epsilon * db1
		model["W2"] += -Config.epsilon * dW2
		model["b2"] += -Config.epsilon * db2

		current_loss = calculate_loss(X, y, model)
		if i % 50 == 0:
#			visualize(X, y, model, i)
			print("\tLoss after iteration %i: %f" % (i, current_loss))

		if abs(current_loss - last_loss) > loss_delta:
			last_loss = current_loss
		else:
			break

		i = i + 1

	print("\tMinimal loss calculated on %i iteration: loss = %f" % (i, current_loss))
	return model


def multilayer_perceptron(X, y, nn_hdim):
	#	W1 - матрица весов между input и hidden слоем
	#		при создании заполненяется рандомными значениями
	#		рaзмерность (2х3) тк в input слое 2 нейрона, в hidden - 3
	#	b1 - вектор смещения для hidden слоя, на первой итерации заполнен
	#		нулями
	np.random.seed(0)
	W1 = np.random.randn(Config.nn_input_dim, nn_hdim) / np.sqrt(Config.nn_input_dim)
	b1 = np.zeros((1, nn_hdim))
	# То же самое для связей между hidden и output слоем
	W2 = np.random.randn(nn_hdim, Config.nn_output_dim) / np.sqrt(nn_hdim)
	b2 = np.zeros((1, Config.nn_output_dim))


	model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

	return model


def main():
	X, y = generate_data()
	with Profiler() as p:
		#X - массив точек типа np.array - array([[x1,y1],[x2,y2],...])
		#y - массив значений, принадлежность к классу - [1, 0, 0, 1, 1, ..]
	#	visualize_linear(X, y)
		hidden_neuron_quantity = 3
		model = multilayer_perceptron(X, y, hidden_neuron_quantity)
		print "\n\tPerceptron before training:\n\t"
		pprint(model)

		visualize(X, y, model, "begin_py")
		model = train_network(X, y, model, print_loss=True)
		print "\n\tPerceptron after training:\n\t"
		pprint(model)

#======================================================================
	model2 = {
		"W1": [[-3.363171846284268,3.9442164237878488,2.9922150117631654],[0.7930534650839552,-0.7758100494565097,2.5574413557182742]],
		"b1": [-1.2451114742261244,-4.670694767699797,-2.244981504083405],
		"W2": [[3.5987838698415566,-2.0161703693843216],[-4.030738233948683,5.10436937137815],[4.598538159596516,-4.246981385708192]],
		"b2":[-0.7348899787424444,0.734889978742444]}
#======================================================================
	visualize(X, y, model, "python")
#	visualize(X, y, model2, "erlang")


if __name__ == "__main__":
	main()
