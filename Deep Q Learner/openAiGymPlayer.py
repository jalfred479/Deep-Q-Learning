import gym
import numpy as np
import tensorflow as tf
import math
import random

EPSILON = 1
EPSILON_DECAY = .9
MIN_EPSILON =  .1
MAX_STEPS = 400
MAX_MEMORY = 10000
BATCH_SIZE = 150
GAMMA = .99
MAX_EPISODES = 1000
UPDATE = 100


class NueralNetwork():
	def __init__(self, environment_shape, environment_output):
		self.sess = tf.Session()
		self.memory = []
		self.shape = environment_shape
		self.outputSize = environment_output
		self.inputStates = tf.placeholder(tf.float32,[None, self.shape])
		self.outputStates = tf.placeholder(tf.float32,[None, self.outputSize])
		self.rewards = tf.placeholder(tf.float32, [None, ])
		self.nextStates = tf.placeholder(tf.float32,[None, self.shape])
		self.create_weights()
		self.create_updates()
		self.linkNetwork()
		self.linkQPrime()
		self.trainInit()
	
	def create_weights(self):
		self.w1 = tf.Variable(tf.random_normal([self.shape, 200]))
		self.b1 = tf.Variable(tf.zeros([200]))
		self.w2 = tf.Variable(tf.random_normal([200, 200]))
		self.b2 = tf.Variable(tf.zeros([200]))
		self.w3 = tf.Variable(tf.random_normal([200, self.outputSize]))
		self.b3 = tf.Variable(tf.zeros([self.outputSize]))
		self.w1_= tf.Variable(self.w1.initialized_value())
		self.b1_= tf.Variable(self.b1.initialized_value())
		self.w2_= tf.Variable(self.w2.initialized_value())
		self.b2_= tf.Variable(self.b2.initialized_value())
		self.w3_= tf.Variable(self.w3.initialized_value())
		self.b3_= tf.Variable(self.b3.initialized_value())
		
	def create_updates(self):
		self.update_w1_ = self.w1_.assign(self.w1)
		self.update_b1_ = self.b1_.assign(self.b1)
		self.update_w2_ = self.w2_.assign(self.w2)
		self.update_b2_ = self.b2_.assign(self.b2)
		self.update_w3_ = self.w3_.assign(self.w3)
		self.update_b3_ = self.b3_.assign(self.b3)
	
	def linkNetwork(self):
		self.hidden_1 = tf.nn.relu(tf.matmul(self.inputStates, self.w1) + self.b1)
		self.hidden_2 = tf.nn.relu(tf.matmul(self.hidden_1, self.w2) + self.b2)
		self.actions = tf.matmul(self.hidden_2, self.w3) + self.b3
		self.Q = tf.reduce_sum(tf.mul(self.actions, self.outputStates), reduction_indices=1)
		
	def linkQPrime(self):
		self.hidden_1_ = tf.nn.relu(tf.matmul(self.nextStates, self.w1_) + self.b1_)
		self.hidden_2_ = tf.nn.relu(tf.matmul(self.hidden_1_, self.w2_) + self.b2_)
		self.actions_ = tf.matmul(self.hidden_2_, self.w3_) + self.b3_
		self.Q_ = self.rewards + GAMMA*tf.reduce_max(self.actions_, reduction_indices=1)
	
	def trainInit(self):
		self.loss = tf.reduce_mean(tf.square(self.Q_-self.Q))
		self.rewardLoss = tf.reduce_mean(tf.square(self.rewards-self.Q))
		self.training = tf.train.AdamOptimizer(0.0001).minimize(self.loss)
		self.rewardTraining = tf.train.AdamOptimizer(0.0001).minimize(self.rewardLoss)
		
	def makeDecision(self, state):
		decision = self.sess.run(self.actions, feed_dict={self.inputStates :np.array([state])})
		decision = np.argmax(action)
		return decision
		
	def run(self):
		self.sess.run(tf.initialize_all_variables())
	
	def end(self):
		self.see.close()
	
	def update(self):
		self.sess.run(update_w1_)
		self.sess.run(update_b1_)
		self.sess.run(update_w2_)
		self.sess.run(update_b2_)
		self.sess.run(update_w3_)
		self.sess.run(update_b3_)
		
	def rewardTrain(self, memState, memState_, action, rewards):
		feed = {
				self.inputStates : memState,
				self.nextStates : memState_,
				self.rewards : rewards,
				self.outputStates : action
				}
		self.sess.run([self.rewardLoss, self.rewardTraining], feed_dict = feed)
	
	def train(self, memState, memState_, action, rewards):
		feed = {
				self.inputStates : memState,
				self.nextStates : memState_,
				self.rewards : rewards,
				self.outputStates : action
				}
		self.sess.run([self.loss, self.training], feed_dict = feed)
	
class Learner():
	def __init__(self, environment, actions):
		self.environment = environment
		self.totalActions = actions
		self.nueral = NueralNetwork(environment.shape[0], actions.n)
		self.memState = np.zeros((MAX_MEMORY, environment.shape[0]))
		self.memState_ = np.zeros((MAX_MEMORY, environment.shape[0]))
		self.actions = np.zeros((MAX_MEMORY, actions.n))
		self.rewards = np.zeros((MAX_MEMORY))
		self.epsilon = 1
		self.epsilon_decay = .9
		self.epsilon_min = .1
	
	def makeAction(self, state):
		if self.epsilon > random.random():
			action = self.totalActions.sample()
		else:
			action = self.nueral.makeDecision(state)
		
		self.epsilon = self.epsilon * self.epsilon_decay
		if self.epsilon < self.epsilon_min:
			self.epsilon = self.epsilon_min
		
		return action
	
	def makeMemory(self, memNum, state, next, reward, action):
		self.memState[memNum] = np.array(state)
		self.memState_[memNum] = np.array(next)
		self.rewards[memNum] = reward
		self.actions[memNum] = np.zeros(self.totalActions.n)
		self.actions[memNum][action] = 1.0
	
	def rewardTrain(self, totalMem):
		batch = min(totalMem, BATCH_SIZE)
		size = min(totalMem, MAX_MEMORY)
		i = np.random.choice(size, batch, replace=True)
		self.nueral.rewardTrain(self.memState[i], self.memState_[i], self.actions[i], self.rewards[i])
		
	def train(self, totalMem):
		batch = min(totalMem, BATCH_SIZE)
		size = min(totalMem, MAX_MEMORY)
		i = np.random.choice(size, batch, replace=True)
		self.nueral.train(self.memState[i], self.memState_[i], self.actions[i], self.rewards[i])
	
	def update(self):
		self.nueral.update()
	
	def end(self):
		self.nueral.close()
		
	def run(self):
		self.nueral.run()
		
if __name__ == '__main__':
	env = gym.make('CartPole-v0')
	learner = Learner(env.observation_space, env.action_space)
	learner.run()
	memNum = 0
	totalMem = 0
	
	for episode in range(MAX_EPISODES):
		done = False
		reward = 0.0
		trainNum = 0
		state = env.reset()
		next_state = state
		
		for step in range(MAX_STEPS):
			action = learner.makeAction(state)
			#env.render()
			next_state, reward, done, _ = env.step(action)
		
			learner.makeMemory(memNum, state, next_state, reward, action)
			memNum += 1
		
			if memNum + 1 < MAX_MEMORY:
				memNum += 1
				if totalMem < MAX_MEMORY:
					totalMem += 1
			else:
				memNum = 0
		
			if totalMem > 0:
				if done:
					learner.rewardTrain(totalMem)
				else:
					learner.train(totalMem)
		
				if trainNum > UPDATE:
					learner.update()
				else:
					trainNum += 1
		
			state = next_state
		
			if done:
				print("Episode - {}, Steps - {}".format(episode, step))
				break
		
		
	learner.end();		
