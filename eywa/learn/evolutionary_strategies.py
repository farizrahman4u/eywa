from ..utils import ProgressBar
from .learner import Learner
import numpy as np


class EvolutionaryStrategies(Learner):

	def __init__(self, npop=100, lr=0.001, std=0.1):
		super().__init__()
		self.npop = npop
		self.lr = lr
		self.std = std

	def __call__(self, f, W, epochs=100):
		pbar = ProgressBar(epochs)
		rewards = np.zeros(self.npop)
		const = self.lr / (self.npop * self.std)
		for e in range(epochs):
			curr_reward = f()
			mutations = np.random.randn(*(self.npop,) + W.shape)
			jittered = self.std * mutations
			rewards *= 0
			for m, delta in enumerate(jittered):
				W += delta
				rewards[m] = f()
				W -= delta
			rewards -= curr_reward
			std_rewards = rewards.std()
			rewards -= rewards.mean()
			try:
				rewards /= std_rewards
			except Exception:
				pass
			W += const * np.dot(mutations.T, rewards)
			pbar.update()


ES = EvolutionaryStrategies
