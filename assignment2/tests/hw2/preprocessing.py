import numpy as np
def sample_zero_mean(x):
    """
    Make each sample have a mean of zero by subtracting mean along the feature axis.
    :param x: float32(shape=(samples, features))
    :return: array same shape as x
    """
    return (x.T-np.mean(x,axis=1)).T
    

def gcn(x, scale=55., bias=0.01):
    """
    GCN each sample (assume sample mean=0)
    :param x: float32(shape=(samples, features))
    :param scale: factor to scale output
    :param bias: bias for sqrt
    :return: scale * x / sqrt(bias + sample variance)
    """
    sample_variance = np.var(x,axis=1)
    return scale * (x.T / np.sqrt(bias + sample_variance)).T
    


def feature_zero_mean(x, xtest):
    """
    Make each feature have a mean of zero by subtracting mean along sample axis.
    Use train statistics to normalize test data.
    :param x: float32(shape=(samples, features))
    :param xtest: float32(shape=(samples, features))
    :return: tuple (x, xtest)
    """
    train_feature_mean = np.mean(x,axis=0)
    return (x-train_feature_mean,xtest-train_feature_mean)
    

def zca(x, xtest, bias=0.1):
	"""
	ZCA training data. Use train statistics to normalize test data.
	:param x: float32(shape=(samples, features)) (assume mean=0)
	:param xtest: float32(shape=(samples, features))
	:param bias: bias to add to covariance matrix
	:return: tuple (x, xtest)
	"""
	
	n = x.shape[0]
	temp = np.dot(x.T, x)/n + np.eye(x.shape[1])*bias
	u, s, v = np.linalg.svd(temp,full_matrices=True)
	pca = np.dot(np.dot(u, np.diag(1. / np.sqrt(s))),u.T) 
	zca_x = np.dot(x,pca)
	zca_xtest = np.dot(xtest,pca)

	return (zca_x,zca_xtest) 


def cifar_10_preprocess(x, xtest, image_size=32):
    """
    1) sample_zero_mean and gcn xtrain and xtest.
    2) feature_zero_mean xtrain and xtest.
    3) zca xtrain and xtest.
    4) reshape xtrain and xtest into NCHW
    :param x: float32 flat images (n, 3*image_size^2)
    :param xtest float32 flat images (n, 3*image_size^2)
    :param image_size: height and width of image
    :return: tuple (new x, new xtest), each shaped (n, 3, image_size, image_size)
    """
    n_train = x.shape[0]
    n_test  = xtest.shape[0]
    x, xtest = sample_zero_mean(x), sample_zero_mean(xtest)
    x, xtest = gcn(x), gcn(xtest)
    x, xtest = feature_zero_mean(x, xtest)
    x, xtest = zca(x, xtest)
    x = x.reshape(n_train, 3, image_size, image_size) 
    xtest = xtest.reshape(n_test, 3, image_size, image_size) 
    print(x.shape)
    return (x, xtest)
