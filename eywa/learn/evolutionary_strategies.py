from .learner import Learner
import numpy as np

loss_to_reward_map = {'mse': lambda x, y: -np.sum((x - y) ** 2)}


class EvolutionaryStrategies(Learner):

	def __init__(self, npop=100, lr=0.001, std=0.1):
		self.npop = npop
		self.lr = lr
		self.std = std

	def __call__(self, f, W, epochs=100):

		for e in range(epochs):
			curr_reward = f()
			mutations = np.random.randn(*(self.npop,) + W.shape)
			jittered = self.std * mutations
			rewards = np.zeros(self.npop)
			for m in range(self.npop):
				delta = jittered[m]
				W += delta
				rewards[m] = f()
				W -= delta
			rewards -= curr_reward
			norm_rewards = (rewards - rewards.mean())
			std_rewards = rewards.std()
			if std_rewards:
				norm_rewards /= std_rewards
			W += self.lr * np.dot(mutations.T, norm_rewards) / (self.npop * self.std)


ES = EvolutionaryStrategies
