import gym
import numpy as np
import tensorflow as tf
import math
import random

EPSILON = 1
EPSILON_DECAY = .9995
MIN_EPSILON =  .1
MAX_STEPS = 199
MAX_MEMORY = 100000
BATCH_SIZE = 500
GAMMA = .99
MAX_EPISODES = 1500
UPDATE = 150


def main():
	sess = tf.Session()
	env = gym.make('CartPole-v0')
	env.monitor.start("/tmp/cartpole-experiment", force=True)
	
	#Input and output shapes created.
	inputStates = tf.placeholder(tf.float32,[None, env.observation_space.shape[0]])
	outputStates = tf.placeholder(tf.float32,[None, env.action_space.n])
	#nextStates = tf.placeholder(tf.float32,[None, env.observation_space.shape[0]])
	
	#Creating weights and biases for Q
	w1 = tf.Variable(tf.random_uniform([env.observation_space.shape[0], 200], -.1, .1))
	b1 = tf.Variable(tf.random_uniform([200], -.1, .1))
	w2 = tf.Variable(tf.random_uniform([200, 200], -.1, .1))
	b2 = tf.Variable(tf.random_uniform([200], -.1, .1))
	w3 = tf.Variable(tf.random_uniform([200, env.action_space.n], -.1, .1))
	b3 = tf.Variable(tf.random_uniform([env.action_space.n], -.1, .1))
	
	#Creating weights and biases for Q prime
	w1_= tf.Variable(tf.random_uniform([env.observation_space.shape[0], 200], -1, 1))
	b1_= tf.Variable(tf.random_uniform([200], -1, 1))
	w2_= tf.Variable(tf.random_uniform([200, 200], -1, 1))
	b2_= tf.Variable(tf.random_uniform([200], -1, 1))
	w3_= tf.Variable(tf.random_uniform([200, env.action_space.n], -.01, .01))
	b3_= tf.Variable(tf.random_uniform([env.action_space.n], -.01, .01))
	
	#Creating updates for weights and biases on Q prime
	update_w1_ = w1_.assign(w1)
	update_b1_ = b1_.assign(b1)
	update_w2_ = w2_.assign(w2)
	update_b2_ = b2_.assign(b2)
	update_w3_ = w3_.assign(w3)
	update_b3_ = b3_.assign(b3)

	#Linking main network
	hidden_1 = tf.nn.relu(tf.matmul(inputStates, w1) + b1)
	hidden_2 = tf.nn.relu(tf.matmul(hidden_1, w2) + b2)
	Q = tf.matmul(hidden_2, w3) + b3

	#Linking Q_ network
	hidden_1_ = tf.nn.relu(tf.matmul(inputStates, w1_) + b1_)
	hidden_2_ = tf.nn.relu(tf.matmul(hidden_1_, w2_) + b2_)
	Q_ = tf.matmul(hidden_2_, w3_) + b3_

	#Creating trainer
	actionTakenPlaceholder = tf.placeholder(tf.int32, [None], name="actionMasks")
	actionMasks = tf.one_hot(actionTakenPlaceholder, env.action_space.n)
	filtered_Q = tf.reduce_sum(tf.mul(Q, actionMasks), reduction_indices=1)
	target_q_placeholder = tf.placeholder(tf.float32, [None,])
	loss = tf.reduce_mean(tf.square(filtered_Q - target_q_placeholder))
	training = tf.train.AdamOptimizer(0.0001).minimize(loss)

	#Creating memory and associated attribute arrays.
	rewards = np.zeros((MAX_MEMORY))
	currentState = np.zeros((MAX_MEMORY, env.observation_space.shape[0]))
	nextState = np.zeros((MAX_MEMORY, env.observation_space.shape[0]))
	didFinish = np.zeros((MAX_MEMORY))
	actions = np.zeros((MAX_MEMORY))
	sess.run(tf.initialize_all_variables())

	memNum = 0
	totalMem = 0
	trainNum = 0
	state = env.reset()
	exploration = 0
	decision = 0
	epsilon = 1.0
	
	for episode in range(MAX_EPISODES):
		done = False
		reward = 0.0
		
		state = env.reset()
		
		
		for step in range(MAX_STEPS):
			
			#Make an action
			if epsilon > random.random():
				action = env.action_space.sample()
				exploration += 1
			else:
				action = sess.run(Q, feed_dict={inputStates:np.array([state])})
				action = np.argmax(action)
				decision += 1
		
			epsilon = epsilon * EPSILON_DECAY
			if epsilon < MIN_EPSILON:
				epsilon = MIN_EPSILON
			
			if episode % 10 == 0:	
				env.render()
			
			#Create the memories
			
			next_state, reward, done, _ = env.step(action)
			currentState[memNum] = state
			#print(action)
			#next_state, reward, done, _ = env.step(action)
			actions[memNum] = action
			didFinish[memNum] = done
			nextState[memNum] = next_state
			rewards[memNum] = reward
			
			if memNum + 1 < MAX_MEMORY:
				memNum += 1
				if totalMem < MAX_MEMORY:
					totalMem += 1
			else:
				memNum = 0
		
			if totalMem > 0:
				batch = min(totalMem, BATCH_SIZE)
				size = min(totalMem, MAX_MEMORY)
				i = np.random.choice(size, batch, replace=True)
				sampleNextState = nextState[i]
				sampleCurrentState = currentState[i]
				sampleRewards = rewards[i]
				sampleDidFinish = didFinish[i]
				sampleActions = actions[i]
				allQPrime = sess.run(Q_, feed_dict={inputStates:sampleNextState})
				y_ = []
				state_samples = []
				actionTaken = []
				for mem in range(len(sampleCurrentState)):
					if sampleDidFinish[mem]:
						y_.append(sampleRewards[mem])
					else:
						this_q_prime = allQPrime[mem]
						maxQ = max(this_q_prime)
						y_.append(sampleRewards[mem] + GAMMA*maxQ)
					state_samples.append(sampleCurrentState[mem])
					actionTaken.append(sampleActions[mem])
				feed = {
							inputStates: state_samples,
							target_q_placeholder: y_,
							actionTakenPlaceholder: actionTaken
						}
				sess.run([training], feed_dict=feed)
		
				if trainNum > UPDATE:
					sess.run(update_w1_)
					sess.run(update_b1_)
					sess.run(update_w2_)
					sess.run(update_b2_)
					sess.run(update_w3_)
					sess.run(update_b3_)
					trainNum = 0
				else:
					trainNum += 1
		
			state = next_state
		
			if done:
				if episode % 10 == 0:
					print("Episode - {}, Steps - {}".format(episode, step))
				
				break
				
	env.monitor.close()

if __name__ == "__main__":
	main()

