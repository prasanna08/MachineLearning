import numpy as np
import cvxopt

class SVM(object):
	def __init__(self, data, targets, kernel='poly', threshold=1e-5, degree=1, sigma=1.0):
		self.X = data
		self.Y = targets
		self.kernel_type = kernel
		self.degree = degree
		self.sigma = sigma
		self.threshold = threshold

	def build_kernel(self):
		self.K = np.dot(self.X, self.X.T)

		if self.kernel_type == 'poly':
			self.K = (1 + self.K / self.sigma) ** self.degree

		elif self.kernel_type == 'rbf':
			b = np.ones((1, self.X.shape[0]))
			p = (np.diag(self.K) * np.ones((1, self.X.shape[0]))).T
			self.K = self.K - 0.5*(np.dot(p, b) + np.dot(b.T, p.T))
			self.K = np.exp(self.K / self.sigma**2)

	def train_svm(self):
		# build kernel.
		self.build_kernel()

		# Assemble the matrices for the constraints
		P = self.Y*self.Y.T*self.K
		q = -np.ones((self.X.shape[0],1))
		G = -np.eye(self.X.shape[0])
		h = np.zeros((self.X.shape[0],1))
		A = self.Y.reshape(1,self.X.shape[0])
		b = 0.0

		# Call the quadratic solver
		sol = cvxopt.solvers.qp(cvxopt.matrix(P),cvxopt.matrix(q),cvxopt.matrix(G),cvxopt.matrix(h), cvxopt.matrix(A), cvxopt.matrix(b))

		lambdas = np.array(sol['x'])
		self.sv = np.where(lambdas > self.threshold)[0]
		self.lambdas = lambdas[self.sv]
		self.X = self.X[self.sv]
		self.Y = self.Y[self.sv]

		self.bias = np.sum(self.Y)
		for n in range(len(self.sv)):
			self.bias -= np.sum(self.lambdas * self.Y * self.K[self.sv[n], self.sv].reshape(len(self.sv), 1))
			self.bias /= len(self.sv)

		self.nsupport = len(self.sv)
		print self.nsupport, "support vectors found" 

	def classifier(self, datapoints, soft=False):
		k = np.dot(self.X, datapoints.T)
		if self.kernel_type == 'poly':
			k = (1 + (k / self.sigma)) ** self.degree

		elif self.kernel_type == 'rbf':
			p = (np.sum(self.X**2, axis=1) * np.ones((1, self.nsupport))).T
			p = np.dot(p, np.ones((1, datapoints.shape[0])))
			b = (np.sum(datapoints**2, axis=1) * self.ones((1, datapoints.shape[0])))
			b = np.dot(np.ones((self.nsupport, 1)), b)
			k = k - 0.5*p - 0.5*b
			k = k / self.sigma**2

		l = self.lambdas.reshape(self.nsupport, 1)
		o = self.Y.reshape(self.nsupport, 1 )
		k = k * l * o
		predictions = (k.sum(axis=0) + self.bias)

		if not soft:
			predictions = np.sign(predictions)

		return predictions
