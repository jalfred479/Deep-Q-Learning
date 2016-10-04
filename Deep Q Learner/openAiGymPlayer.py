import gym
import numpy as np
import tensorflow as tf
import math
import random

EPSILON = 1
EPSILON_DECAY = .9
MIN_EPSILON =  .1
MAX_STEPS = 400
MAX_MEMORY = 100000
BATCH_SIZE = 500
GAMMA = .99
MAX_EPISODES = 1500
UPDATE = 150


class NueralNetwork():
	def __init__(self, environment_shape, environment_output):
		tf.reset_default_graph()
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
		self.w1 = tf.Variable(tf.random_uniform([self.shape, 200], -.1, .1))
		self.b1 = tf.Variable(tf.random_uniform([200], -.1, .1))
		self.w2 = tf.Variable(tf.random_uniform([200, 200], -.1, .1))
		self.b2 = tf.Variable(tf.random_uniform([200], -.1, .1))
		self.w3 = tf.Variable(tf.random_uniform([200, self.outputSize], -.1, .1))
		self.b3 = tf.Variable(tf.random_uniform([self.outputSize], -.1, .1))
		self.w1_= tf.Variable(tf.random_uniform([self.shape, 200], -1, 1))
		self.b1_= tf.Variable(tf.random_uniform([200], -1, 1))
		self.w2_= tf.Variable(tf.random_uniform([200, 200], -1, 1))
		self.b2_= tf.Variable(tf.random_uniform([200], -1, 1))
		self.w3_= tf.Variable(tf.random_uniform([200, self.outputSize], -.01, .01))
		self.b3_= tf.Variable(tf.random_uniform([self.outputSize], -.01, .01))
		
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
		self.Q = tf.matmul(self.hidden_2, self.w3) + self.b3
		#self.Q = tf.reduce_sum(tf.mul(self.actions, self.outputStates), reduction_indices=1)
		
	def linkQPrime(self):
		self.hidden_1_ = tf.nn.relu(tf.matmul(self.nextStates, self.w1_) + self.b1_)
		self.hidden_2_ = tf.nn.relu(tf.matmul(self.hidden_1_, self.w2_) + self.b2_)
		self.Q_ = tf.matmul(self.hidden_2_, self.w3_) + self.b3_
		#self.Q_ = tf.matmul(self.actions
	
	def trainInit(self):
		self.actionTakenPlaceholder = tf.placeholder(tf.int32, [None], name="actionMasks")
		self.actionMasks = tf.one_hot(self.actionTakenPlaceholder, self.outputSize)
		self.filtered_Q = tf.reduce_sum(tf.mul(self.Q, self.actionMasks), reduction_indices=1)
		self.target_q_placeholder = tf.placeholder(tf.float32, [None,])
		self.loss = tf.reduce_mean(tf.square(self.filtered_Q - self.target_q_placeholder))
		#self.rewardLoss = tf.reduce_mean(tf.square(self.rewards-self.Q))
		self.training = tf.train.AdamOptimizer(0.0001).minimize(self.loss)
		#self.rewardTraining = tf.train.AdamOptimizer(0.0001).minimize(self.rewardLoss)
		
	def makeDecision(self, state):
		decision = self.sess.run(self.Q, feed_dict={self.inputStates :np.array([state])})
		decision = np.argmax(action)
		return decision
		
	def run(self):
		self.sess.run(tf.initialize_all_variables())
	
	def end(self):
		self.sess.close()
	
	def update(self):
		self.sess.run(self.update_w1_)
		self.sess.run(self.update_b1_)
		self.sess.run(self.update_w2_)
		self.sess.run(self.update_b2_)
		self.sess.run(self.update_w3_)
		self.sess.run(self.update_b3_)
		
	def train(self, memState, memState_, action, rewards, done):
		all_q_prime = self.sess.run(self.Q_, feed_dict = {self.nextStates : memState_})
		y_ = []
		state_samples = []
		actions = []
		for i in range(len(memState)):
			if done[i]:
				y_.append(rewards[i])
			else:
				this_q_prime = all_q_prime[i]
				maxQ = max(this_q_prime)
				y_.append(rewards[i] + GAMMA*maxQ)
			
			state_samples.append(memState[i])
			actions.append(action[i])
			
		feed = {
				self.inputStates : state_samples,
				self.target_q_placeholder : y_,
				self.actionTakenPlaceholder : actions
				}
		self.sess.run([self.training], feed_dict = feed)
	
	#def train(self, memState, memState_, action, rewards):
	#	feed = {
	#			self.inputStates : memState,
	#			self.nextStates : memState_,
	#			self.rewards : rewards,
	#			self.outputStates : action
	#			}
	#	self.sess.run([self.loss, self.training], feed_dict = feed)
	
class Learner():
	def __init__(self, environment, actions):
		self.environment = environment
		self.totalActions = actions
		self.nueral = NueralNetwork(environment.shape[0], actions.n)
		self.memState = np.zeros((MAX_MEMORY, environment.shape[0]))
		self.memState_ = np.zeros((MAX_MEMORY, environment.shape[0]))
		self.actions = np.zeros((MAX_MEMORY))
		self.rewards = np.zeros((MAX_MEMORY))
		self.done = np.zeros((MAX_MEMORY))
		self.epsilon = 1
		self.epsilon_decay = .99995
		self.epsilon_min = .1
		self.random = 0
		self.decision = 0
	
	def makeAction(self, state):
		if self.epsilon > random.random():
			action = self.totalActions.sample()
			self.random += 1
		else:
			action = self.nueral.makeDecision(state)
			self.decision += 1
		
		self.epsilon = self.epsilon * self.epsilon_decay
		if self.epsilon < self.epsilon_min:
			self.epsilon = self.epsilon_min
		
		return action
	
	def makeMemory(self, memNum, state, next, reward, action, done):
		self.memState[memNum] = np.array(state)
		self.memState_[memNum] = np.array(next)
		self.rewards[memNum] = reward
		self.actions[memNum] = action
		self.done[memNum] = done
	
	def train(self, totalMem):
		batch = min(totalMem, BATCH_SIZE)
		size = min(totalMem, MAX_MEMORY)
		i = np.random.choice(size, batch, replace=True)
		self.nueral.train(self.memState[i], self.memState_[i], self.actions[i], self.rewards[i], self.done[i])
		
	#def train(self, totalMem):
	#	batch = min(totalMem, BATCH_SIZE)
	#	size = min(totalMem, MAX_MEMORY)
	#	i = np.random.choice(size, batch, replace=True)
	#	self.nueral.train(self.memState[i], self.memState_[i], self.actions[i], self.rewards[i])
		
	def percentRand(self):
		print("Total Random - {} Total Decided = {}".format(self.random/(self.random + self.decision), self.decision/(self.decision + self.random)))
	
	def update(self):
		self.nueral.update()
	
	def end(self):
		self.nueral.end()
		
	def run(self):
		self.nueral.run()
		
if __name__ == '__main__':
	env = gym.make('CartPole-v0')
	learner = Learner(env.observation_space, env.action_space)
	learner.run()
	memNum = 0
	totalMem = 0
	trainNum = 0
	
	for episode in range(MAX_EPISODES):
		done = False
		reward = 0.0
		
		state = env.reset()
		next_state = state
		
		for step in range(MAX_STEPS):
			action = learner.makeAction(state)
			#env.render()
			next_state, reward, done, _ = env.step(action)
			
			learner.makeMemory(memNum, state, next_state, reward, action, done)
			memNum += 1
		
			if memNum + 1 < MAX_MEMORY:
				memNum += 1
				if totalMem < MAX_MEMORY:
					totalMem += 1
			else:
				memNum = 0
		
			if totalMem > 0:
				learner.train(totalMem)
		
				if trainNum > UPDATE:
					learner.update()
					trainNum = 0
				else:
					trainNum += 1
		
			state = next_state
		
			if done:
				if episode % 10 == 0:
					print("Episode - {}, Steps - {}".format(episode, step))
				learner.percentRand()
				
				break
		
		
	learner.end();		
