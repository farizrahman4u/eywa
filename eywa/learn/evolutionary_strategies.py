from .learner import Learner
import numpy as np

loss_to_reward_map = {'mse': lambda x, y: -np.sum((x - y) ** 2)}


class EvolutionaryStrategies(Learner):

	def __init__(self, npop=100, lr=0.001, std=0.1):
		self.npop = npop
		self.lr = lr
		self.std = std

	def __call__(self, f, W, X, Y, loss='mse', epochs=100):
		_f_r = loss_to_reward_map[loss]
		f_reward = lambda : np.sum([_f_r(f(W, x), y) for x, y in zip(X, Y)])
		for e in range(epochs):
			curr_reward = f_reward()
			mutations = np.random.randn(*(self.npop,) + W.shape)
			jittered = self.std * mutations
			rewards = np.zeros(self.npop)
			for m in range(self.npop):
				delta = jittered[m]
				W += delta
				rewards[m] = f_reward()
				W -= delta
			rewards -= curr_reward
			norm_rewards = (rewards - rewards.mean())
			std_rewards = rewards.std()
			if std_rewards:
				norm_rewards /= std_rewards
			W += self.lr * np.dot(mutations.T, norm_rewards) / (self.npop * self.std)


ES = EvolutionaryStrategies
