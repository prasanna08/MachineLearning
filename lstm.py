"""Simple LSTM layer implementation.

Source: Andrej Karpathy (https://gist.github.com/karpathy/587454dc0146a6ae21fc)
"""

import numpy as np

class LSTM(object):
	def init(self, n_input, n_hidden):
		WLSTM = np.random.rand(n_input + n_hidden + 1, 4 * n_hidden) / np.sqrt(n_input + n_hidden)
		WLSTM[0, :] = 0
		return WLSTM


	def Sigmoid(self, z):
		return 1.0 / (1.0 + np.exp(-z))

	def Forward(self, X, WLSTM, c0=None, h0=None):
		n_hidden = WLSTM.shape[1] / 4
		n_input = X.shape[2]
		batch_size = X.shape[1]
		num_steps = X.shape[0]

		Hin = np.zeros((num_steps, batch_size, n_input + n_hidden + 1))
		IFOG = np.zeros((num_steps, batch_size, 4 * n_hidden))
		IFOGf = np.zeros((num_steps, batch_size, 4 * n_hidden))
		C = np.zeros((num_steps, batch_size, n_hidden))
		Ct = np.zeros((num_steps, batch_size, n_hidden))
		Hout = np.zeros((num_steps, batch_size, n_hidden))

		h0 = h0 if h0 is not None else np.zeros((batch_size, n_hidden))
		c0 = c0 if c0 is not None else np.zeros((batch_size, n_hidden))

		for t in range(num_steps):
			Hin[t, :, 0] = 1
			Hin[t, :, 1:n_input+1] = X[t]
			Hin[t, :, n_input+1:] = h0 if t == 0 else Hout[t-1]

			IFOG[t] = np.dot(Hin[t], WLSTM) # input, input gate, forget, output
			IFOGf[t, :, :n_hidden] = np.tanh(IFOG[t, :, :n_hidden])
			IFOGf[t, :, n_hidden:] = self.Sigmoid(IFOG[t, :, n_hidden:])

			C[t] = IFOGf[t, :, :n_hidden] * IFOGf[t, :, n_hidden:2*n_hidden]
			prevc = c0 if t == 0 else C[t-1]
			C[t] += IFOGf[t, :, 2*n_hidden:3*n_hidden] * prevc

			Ct[t] = np.tanh(C[t])
			Hout[t] = Ct[t] * IFOGf[t, :, 3*n_hidden:]

		cache = {
			'Hin': Hin,
			'WLSTM': WLSTM,
			'Hout': Hout,
			'IFOG': IFOG,
			'IFOGf': IFOGf,
			'C': C,
			'Ct': Ct,
			'c0': c0
		}
		return Hout, C[t], Hout[t], cache

	def Backward(self, dHout_in, cache, dcn=None, dhn=None):
		WLSTM = cache['WLSTM']
		Hout = cache['Hout']
		Hin = cache['Hin']
		IFOG = cache['IFOG']
		IFOGf = cache['IFOGf']
		C = cache['C']
		Ct = cache['Ct']
		c0 = cache['c0']

		num_steps = Hin.shape[0]
		batch_size = Hin.shape[1]
		n_hidden = Hout.shape[2]
		n_input = Hin.shape[2] - n_hidden - 1

		dIFOGf = np.zeros(IFOGf.shape)
		dIFOG = np.zeros(IFOG.shape)
		dWLSTM = np.zeros(WLSTM.shape)
		dC = np.zeros(C.shape)
		dX = np.zeros((num_steps, batch_size, n_input))
		dHin = np.zeros(Hin.shape)

		dHout = dHout_in.copy()
		dh0 = np.zeros((batch_size, n_hidden))
		dc0 = np.zeros((batch_size, n_hidden))


		if dcn is not None:
			dC[num_steps-1] += dcn.copy()
		if dhn is not None:
			dHout[num_steps-1] += dhn.copy()

		for t in reversed(range(num_steps)):
			dIFOGf[t, :, 3*n_hidden:] = Ct[t] * dHout[t] # output gate.
			dC[t] += (1 - Ct[t]**2) * (IFOGf[t, :, 3*n_hidden:] * dHout[t])

			if t > 0:
				dIFOGf[t, :, 2*n_hidden:3*n_hidden] = dC[t] * C[t-1] # forget gate.	
				dC[t-1] = dC[t] * IFOGf[t, :, 2*n_hidden:3*n_hidden]
			else:
				dIFOGf[t, :, 2*n_hidden:3*n_hidden] = dC[t] * c0 # forget gate.
				dc0 = dC[t] * IFOGf[t, :, 2*n_hidden:3*n_hidden]

			
			dIFOGf[t, :, :n_hidden] = dC[t] * IFOGf[t, :, n_hidden:2*n_hidden] # input.
			dIFOGf[t, :, n_hidden:2*n_hidden] = dC[t] * IFOGf[t, :, :n_hidden] # input gate.

			dIFOG[t, :, :n_hidden] = (1 - IFOGf[t, :, :n_hidden] ** 2) * dIFOGf[t, :, :n_hidden]
			y = IFOGf[t, :, n_hidden:]
			dIFOG[t, :, n_hidden:] = y * (1 - y) * dIFOGf[t, :, n_hidden:]

			dWLSTM += np.dot(Hin[t].transpose(), dIFOG[t])

			dHin[t] = np.dot(dIFOG[t], WLSTM.transpose())
			dX[t] = dHin[t, :, 1:n_input+1]

			if t > 0:
				dHout[t-1] += dHin[t, :, n_input+1:]
			else:
				dh0 += dHin[t, :, n_input+1:]

		return dX, dWLSTM, dc0, dh0
