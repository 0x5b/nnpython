import matplotlib as plt
import numpy as np
from sklearn import datasets, linear_model

def generate_data():
	np.random.seed(0)
	X, y = datasets.make_moons(200, noise=0.20)
	return X, y

class Config:
	nn_input_dim = 2
	nn_output_dim = 2
	epsilon = 0.01
	reg_lambda = 0.01

def visualize(X, y, model):
	plot_decision_boundary(lambda x: predict(model, x), X, y)
	plt.title("Logistic regression")

def plot_decision_boundary(pred_func, X, y):
	pass

if __name__ == "__mail__":
	X, y = generate_data()
	plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)
	plt.show()
