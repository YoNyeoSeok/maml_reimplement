import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

class Gaussian_data():
    def __init__(self, l=1, e=.1):
    
        self.length_scale = l
        self.noise_level = e
        self.kernel =  RBF(length_scale=l) + WhilteKernel(noise_level=e)
        self.gp = GaussianProcessRegressor(kernel=self.kernel)

    def gen(self, n=5, x_min=-2, x_max=2):
        x = np.random.rand(x_min, x_max, n).sort()
        y = self.gp.sample_y(x)
        return x, y

class Sinewave_data():
    def __init__(self, A=1, w=1, b=0):
        self.A = A
        self.w = w
        self.b = b

    def gen(self, n=5, x_min=-4, x_max=4):
        x = (x_max - x_min) * (np.sort(np.random.rand(n)) - .5)
        y = self.A*np.sin(self.w*x + self.b)
        return x, y
