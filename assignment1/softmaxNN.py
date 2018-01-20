import numpy as np

class softmaxNN:

	def __init__(self,train_data,learning_rate=0.01):

		W1 = np.random.rand(128,784)
		W2 = np.random.rand(64,128)
		W3 = np.random.rand(10,64)
		b1 = np.random.rand(128)
		b2 = np.random.rand(64)
		b3 = np.random.rand(10)
		self.W = [W1,W2,W3]
		self.b = [b1,b2,b3]
		self.grad_W = [None,None,None]
		self.grad_b = [None,None,None]
		self.y = [None,None,None,None]
		self.z = [None,None,None,None]
		self.grad_y = [None,None,None,None]
		self.grad_z = [None,None,None,None]
		self.number_of_layers = len(self.W)

	def forward_pass(self,x,d):

		self.y[0] = x

		for i in range(1,self.number_of_layers+1):

			self.z[i] = np.dot(self.W[i-1],self.y[i-1]) + self.b[i-1]
			self.y[i] = self.relu(self.z[i])

	def back_prop(self,d):

		self.grad_z[3] = self.y-d

		for i in range(self.number_of_layers):

			self.grad_y[-i-1] =  np.dot(self.grad_z[-i],self.W[-i])
			self.grad_w[-i]   =  np.dot(self.y[-i-1],self.grad_z[-i])
			self.grad_b[-i]   =  self.grad_z[-i]
			tmp = self.y[-i-1]
			tmp[self.y[-i-1]<=0] = 0
			tmp[self.y[-i-1]] = 1
			self.grad_z[-i-1] = np.eye(self.y[-i].shape[0])*self.y[self]

	def relu(self,z):
		
		return np.maximum(0,z)

if __name__=="__main__":
	sn = softmaxNN(10)
	x  = np.ones(784) 
	sn.forward_pass(x,1)
	print(sn.y[2])
