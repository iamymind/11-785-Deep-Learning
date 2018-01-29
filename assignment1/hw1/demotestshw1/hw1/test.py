from hw1 import *
import numpy as np

[X,Y] = load_data('digitstrain.txt')


weights = [
           0.01 * np.random.randn(784,128),
	   0.01 * np.random.randn(128,64),
	   0.01 * np.random.randn(64,10)
	  # 0.01 * np.random.randn(100,10)
	  ]

bias    = [
	   np.zeros((1,128)),
	   np.zeros((1,64)),   
	   np.zeros((1,10))   
	   #np.zeros((1,10))
	  ]

lr      = 0.001
#[Ws,bs] = update_weights_double_layer(X, Y, weights, bias, lr)
batch_size = 100
#update_weights_double_layer_batch(X, Y, weights, bias, lr, batch_size)


activation = 'sigmoid'
momentum   = 0.9
#update_weights_double_layer_batch_act(X, Y, weights, bias, lr, batch_size, activation)
u_weights,u_bias = update_weights_double_layer_batch_act_mom(X, Y, weights, bias, lr, batch_size, activation, momentum)

print(u_weights[0].shape)
