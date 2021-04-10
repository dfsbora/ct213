import numpy as np
import random
from math import inf


class Particle:
    """
    Represents a particle of the Particle Swarm Optimization algorithm.
    """
    def __init__(self, lower_bound, upper_bound):
        """
        Creates a particle of the Particle Swarm Optimization algorithm.

        :param lower_bound: lower bound of the particle position.
        :type lower_bound: numpy array.
        :param upper_bound: upper bound of the particle position.
        :type upper_bound: numpy array.
        """
        # Todo: implement
        self.x = np.random.uniform(lower_bound, upper_bound)
        self.best = self.x
        self.best_cost = -float("inf")
        delta = upper_bound - lower_bound
        self.v = np.random.uniform(-delta,delta)
        self.cost = None


class ParticleSwarmOptimization:
    """
    Represents the Particle Swarm Optimization algorithm.
    Hyperparameters:
        inertia_weight: inertia weight.
        cognitive_parameter: cognitive parameter.
        social_parameter: social parameter.

    :param hyperparams: hyperparameters used by Particle Swarm Optimization.
    :type hyperparams: Params.
    :param lower_bound: lower bound of particle position.
    :type lower_bound: numpy array.
    :param upper_bound: upper bound of particle position.
    :type upper_bound: numpy array.
    """
    def __init__(self, hyperparams, lower_bound, upper_bound):
        # Todo: implement
        self.hyperparams = hyperparams

        self.index = 0  # particle index
        self.particles = []
        for i in range(hyperparams.num_particles):
            self.particles.append(Particle(lower_bound, upper_bound))

        self.best_global = Particle(lower_bound, upper_bound)
        self.best_global.x = np.zeros(np.size(lower_bound))
        self.best_global.cost = float("-inf")

        self.best_iteration = Particle(lower_bound, upper_bound)
        self.best_iteration.x = np.zeros(np.size(lower_bound))
        self.best_iteration.cost = float("-inf")

    def get_best_position(self):
        """
        Obtains the best position so far found by the algorithm.

        :return: the best position.
        :rtype: numpy array.
        """
        # Todo: implement
        return self.best_global.x

    def get_best_value(self):
        """
        Obtains the value of the best position so far found by the algorithm.

        :return: value of the best position.
        :rtype: float.
        """
        # Todo: implement
        return self.best_global.cost

    def get_position_to_evaluate(self):
        """
        Obtains a new position to evaluate.

        :return: position to evaluate.
        :rtype: numpy array.
        """
        # Todo: implement
        return self.particles[self.index].x

    def advance_generation(self):
        """
        Advances the generation of particles. Auxiliary method to be used by notify_evaluation().
        """
        # Todo: implement
        if self.particles[self.index].cost > self.particles[self.index].best_cost:
            self.particles[self.index].best_cost = self.particles[self.index].cost
            self.particles[self.index].best = self.particles[self.index].x

            if self.particles[self.index].cost > self.best_iteration.cost:
                self.best_iteration.cost = self.particles[self.index].cost
                self.best_iteration.x = self.particles[self.index].x

                if self.particles[self.index].cost > self.best_global.cost:
                    self.best_global.cost = self.particles[self.index].cost
                    self.best_global.x = self.particles[self.index].x


        w = self.hyperparams.inertia_weight
        phip = self.hyperparams.cognitive_parameter
        phig = self.hyperparams.social_parameter

        rp = random.uniform(0.0, 1.0)
        rg = random.uniform(0.0, 1.0)

        self.particles[self.index].v = w*self.particles[self.index].v + phip*rp*(self.particles[self.index].best-self.particles[self.index].x) + phig*rg*(self.best_global.x-self.particles[self.index].x)
        self.particles[self.index].x = self.particles[self.index].x + self.particles[self.index].v



    def notify_evaluation(self, value):
        """
        Notifies the algorithm that a particle position evaluation was completed.

        :param value: quality of the particle position.
        :type value: float.
        """
        # Todo: implement

        self.particles[self.index].cost = value

        self.advance_generation()

        self.index = self.index + 1
        if self.index >= self.hyperparams.num_particles:
            self.index = 0




