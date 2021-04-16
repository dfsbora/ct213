import numpy as np


class SimpleEvolutionStrategy:
    """
    Represents a simple evolution strategy optimization algorithm.
    The mean and covariance of a gaussian distribution are evolved at each generation.
    """
    def __init__(self, m0, C0, mu, population_size):
        """
        Constructs the simple evolution strategy algorithm.

        :param m0: initial mean of the gaussian distribution.
        :type m0: numpy array of floats.
        :param C0: initial covariance of the gaussian distribution.
        :type C0: numpy matrix of floats.
        :param mu: number of parents used to evolve the distribution.
        :type mu: int.
        :param population_size: number of samples at each generation.
        :type population_size: int.
        """
        self.m = m0
        self.C = C0
        self.mu = mu
        self.population_size = population_size
        self.samples = np.random.multivariate_normal(self.m, self.C, self.population_size)

    def ask(self):
        """
        Obtains the samples of this generation to be evaluated.
        The returned matrix has dimension (population_size, n), where n is the problem dimension.

        :return: samples to be evaluated.
        :rtype: numpy array of floats.
        """
        return self.samples

    def tell(self, fitnesses):
        """
        Tells the algorithm the evaluated fitnesses. The order of the fitnesses in this array
        must respect the order of the samples.

        :param fitnesses: array containing the value of fitness of each sample.
        :type fitnesses: numpy array of floats.
        """
        # Todo: implement this method
        #ordena fitness e aplica mesma ordem a samples, escolhe mi elementos
        indices = np.argsort(fitnesses)
        best_samples = self.samples[indices[0:self.mu],:]

        #calcula nova media e variancia
        array2d = (best_samples-self.m)#[np.newaxis]
        aux = array2d.transpose().dot(array2d)
        self.C = (1/self.mu) * aux
        #print("C(2,2): ", np.shape(self.C))

        self.m =  np.sum(best_samples, axis = 0)/self.mu
        #print("m(2,): ", np.shape(self.m))

        #amostra novos samples
        self.samples = np.random.multivariate_normal(self.m, self.C, self.population_size)


