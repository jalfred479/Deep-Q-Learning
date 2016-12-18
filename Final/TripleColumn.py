import gym
import sys
import numpy as np
import tensorflow as tf
import math
import random
import argparse
import os
import csv

EPSILON = 1
EPSILON_DECAY = .9995
MIN_EPSILON =  .1
MAX_STEPS = 200
MAX_MEMORY = 100000
BATCH_SIZE = 500
GAMMA = .99
MAX_EPISODES = 500
UPDATE = 150

TITLE = "/TripleColumn/"
COLUMN1 = "/SingleColumnRight/"
COLUMN2 = "/DoubleColumnLeft/"

def main(load='False'):
	csvfile = open(os.getcwd() + TITLE + "tripleColumn.csv", 'w')
	data_writer = csv.writer(csvfile)
	sess = tf.Session()
	env = gym.make('CartPole-v0')
	env.monitor.start("/tmp/cartpole-experiment", force=True)
	
	#Input and output shapes created.
	with tf.name_scope('Input'):
		inputStates = tf.placeholder(tf.float32,[None, env.observation_space.shape[0]])
	outputStates = tf.placeholder(tf.float32,[None, env.action_space.n])
	
	#Need to import other columns, so we need to create the variables first.
	Column1_w1 = tf.Variable(tf.random_uniform([env.observation_space.shape[0], 200], -.1, .1), name="Column1_w1")
	Column1_b1 = tf.Variable(tf.random_uniform([200], -.1, .1), name="Column1_b1")
	Column1_w2 = tf.Variable(tf.random_uniform([200, 200], -.1, .1), name="Column1_w2")
	Column1_b2 = tf.Variable(tf.random_uniform([200], -.1, .1), name="Column1_b2")
	Column1_w3 = tf.Variable(tf.random_uniform([200, env.action_space.n], -.1, .1), name="Column1_w3")
	Column1_b3 = tf.Variable(tf.random_uniform([env.action_space.n], -.1, .1), name="Column1_b3")
	Column1_Saver = tf.train.Saver()
	Column1_Saver.restore(sess, os.getcwd() + COLUMN1 + "Weights")
	
	column1_hidden_1_input = tf.matmul(inputStates, Column1_w1) + Column1_b1
	column1_hidden_1 = tf.nn.relu(column1_hidden_1_input)
	column1_hidden_2_input = tf.matmul(column1_hidden_1, Column1_w2) + Column1_b2
	column1_hidden_2 = tf.nn.relu(column1_hidden_2_input)
	column1_Q = tf.matmul(column1_hidden_2, Column1_w3) + Column1_b3
	
	Column2_w1 = tf.Variable(tf.random_uniform([env.observation_space.shape[0], 200], -.1, .1), name="Column2_w1")
	Column2_b1 = tf.Variable(tf.random_uniform([200], -.1, .1), name="Column2_b1")
	Column2_w2 = tf.Variable(tf.random_uniform([200, 200], -.1, .1), name="Column2_w2")
	Column2_b2 = tf.Variable(tf.random_uniform([200], -.1, .1), name="Column2_b2")
	Column2_w3 = tf.Variable(tf.random_uniform([200, env.action_space.n], -.1, .1), name="Column2_w3")
	Column2_b3 = tf.Variable(tf.random_uniform([env.action_space.n], -.1, .1), name="Column2_b3")
	Column2_Saver = tf.train.Saver([Column2_w1, Column2_b1, Column2_w2, Column2_b2, Column2_w3, Column2_b3])
	Column2_Saver.restore(sess, os.getcwd() + COLUMN2 + "Weights")
	
	column2_hidden_1_input = tf.matmul(inputStates, Column1_w1) + Column1_b1
	column2_hidden_1_input += column1_hidden_1_input
	column2_hidden_1 = tf.nn.relu(column2_hidden_1_input)
	column2_hidden_2_input = tf.matmul(column1_hidden_1, Column1_w2) + Column1_b2
	column2_hidden_2_input += column1_hidden_2_input
	column2_hidden_2 = tf.nn.relu(column2_hidden_2_input)
	column2_Q = tf.matmul(column2_hidden_2, Column1_w3) + Column1_b3
	column2_Q += column1_Q
	
	#Created function for summary attachments
	def variable_summaries(var):
		with tf.name_scope('summaries'):
			mean = tf.reduce_mean(var)
			tf.summary.scalar('mean', mean)
			with tf.name_scope('stdddev'):
				stddev = tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
			tf.summary.scalar('stddev', stddev)
			tf.summary.scalar('max', tf.reduce_max(var))
			tf.summary.scalar('min', tf.reduce_min(var))
			tf.summary.histogram('histogram', var)
	
	#Creating weights and biases for Q
	with tf.name_scope("First-Layer"):
		with tf.name_scope("Weights"):
			w1 = tf.Variable(tf.random_uniform([env.observation_space.shape[0], 200], -.1, .1))
			variable_summaries(w1)
		with tf.name_scope("Bias"):
			b1 = tf.Variable(tf.random_uniform([200], -.1, .1))
			variable_summaries(b1)
		with tf.name_scope("Weight-plus-Bias"):
		#first layer doesn't link up with other columns.
			hidden_1_input = tf.matmul(inputStates, w1) + b1
			hidden_1 = tf.nn.relu(hidden_1_input)
			tf.summary.histogram('preactivation', hidden_1_input)
			tf.summary.histogram('activation', hidden_1)
			
	with tf.name_scope("Second-Layer"):
		with tf.name_scope("Weights"):
			w2 = tf.Variable(tf.random_uniform([200, 200], -.1, .1))
			variable_summaries(w2)
		with tf.name_scope("Bias"):
			b2 = tf.Variable(tf.random_uniform([200], -.1, .1))
			variable_summaries(b2)
		with tf.name_scope("Weight-plus-Bias"):
		#second layer does link up with other columns.
			#creating the inputs (practivations)
			hidden_2_input = tf.matmul(hidden_1, w2) + b2
			hidden_2_input += column1_hidden_2_input
			hidden_2_input += column2_hidden_2_input
			hidden_2 = tf.nn.relu(hidden_2_input)
			tf.summary.histogram('preactivation', hidden_2_input)
			tf.summary.histogram('activation', hidden_2)
		
	with tf.name_scope("Third-Layer"):
		with tf.name_scope("Weights"):
			w3 = tf.Variable(tf.random_uniform([200, env.action_space.n], -.1, .1))
			variable_summaries(w3)
		with tf.name_scope("Bias"):
			b3 = tf.Variable(tf.random_uniform([env.action_space.n], -.1, .1))
			variable_summaries(b2)
		with tf.name_scope("Weight-plus-Bias"):
		#last layer also links with old columns.
			Q = tf.matmul(hidden_2, w3) + b3
			Q += column1_Q
			Q += column2_Q
			tf.summary.histogram('preactivation', Q)


	
	#Creating weights and biases for Q prime
	w1_= tf.Variable(tf.random_uniform([env.observation_space.shape[0], 200], -1, 1))
	b1_= tf.Variable(tf.random_uniform([200], -1, 1))
	w2_= tf.Variable(tf.random_uniform([200, 200], -1, 1))
	b2_= tf.Variable(tf.random_uniform([200], -1, 1))
	w3_= tf.Variable(tf.random_uniform([200, env.action_space.n], -.01, .01))
	b3_= tf.Variable(tf.random_uniform([env.action_space.n], -.01, .01))
	
	#for saving variables.
	saver = tf.train.Saver({"Column3_w1":w1,
	 "Column3_b1":b1,
	 "Column3_w2":w2,
	 "Column3_b2":b2,
	 "Column3_w3":w3,
	 "Column3_b3":b3,
	 "Column3_w1_":w1_,
	 "Column3_b1_":b1_,
	 "Column3_w2_":w2_, 
	 "Column3_b2_": b2_,
	 "Column3_w3_": w3_,
	 "Column3_b3_": b3_})
	
	#Creating updates for weights and biases on Q prime
	update_w1_ = w1_.assign(w1)
	update_b1_ = b1_.assign(b1)
	update_w2_ = w2_.assign(w2)
	update_b2_ = b2_.assign(b2)
	update_w3_ = w3_.assign(w3)
	update_b3_ = b3_.assign(b3)

	
	#Linking Q_ network
	hidden_1_input_ = tf.matmul(inputStates, w1_) + b1_
	hidden_1_ = tf.nn.relu(hidden_1_input_)
	hidden_2_input_ = tf.matmul(hidden_1_, w2_) + b2_
	hidden_2_input_ += column1_hidden_2_input
	hidden_2_input_ += column2_hidden_2_input
	hidden_2_ = tf.nn.relu(hidden_2_input_)
	Q_ = tf.matmul(hidden_2_, w3_) + b3_
	Q_ += column1_Q
	Q_ += column2_Q

	#Creating trainer
	actionTakenPlaceholder = tf.placeholder(tf.int32, [None], name="actionMasks")
	actionMasks = tf.one_hot(actionTakenPlaceholder, env.action_space.n)
	filtered_Q = tf.reduce_sum(tf.mul(Q, actionMasks), reduction_indices=1)
	target_q_placeholder = tf.placeholder(tf.float32, [None,])
	with tf.name_scope('Loss'):
		loss = tf.reduce_mean(tf.square(filtered_Q - target_q_placeholder))
		tf.summary.scalar("loss", loss)
	
	with tf.name_scope('Train'):
		training = tf.train.AdamOptimizer(0.0001).minimize(loss)
	
	merged = tf.summary.merge_all()
	train_writer = tf.summary.FileWriter(os.getcwd() + TITLE +"train", sess.graph)
	
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
		totalReward = 0.0
		
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
			if action == 0:
				reward = 1.5
			elif action == 1:
				reward = 1.5
			totalReward += reward;
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
				summary, _ = sess.run([merged, training], feed_dict=feed)
				
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
		
			if done or step == MAX_STEPS - 1:
				train_writer.add_summary(summary, episode)
				data_writer.writerow([episode,totalReward,step])
				if episode % 10 == 0:
					print("Episode - {}, Steps - {}".format(episode, step))
				
				break
	
	train_writer.close()
	env.monitor.close()
	saver.save(sess, os.getcwd() + TITLE + "Weights");

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--load', choices=['True', 'False'], default='False')
	args = parser.parse_args()
	main(load=args.load)

