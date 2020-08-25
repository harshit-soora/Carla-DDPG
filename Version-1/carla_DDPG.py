import glob
import os
import sys
import random
import time
import numpy as np
import math
import environment as env
import utility as ut
import model as md
from keras.models import load_model
import matplotlib.pyplot as plt 

np.random.seed(32)
random.seed(32)
AGGREGATE_STATS_EVERY = 20

try:
	sys.path.append(glob.glob('../../../carla/dist/carla-*%d.%d-%s.egg' % (
		sys.version_info.major,
		sys.version_info.minor,
		'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
	pass
import carla


# Continuous action space as of now for straight maneuver 
def choose_action_straight(choice):
	# Choice will be a continuous value between 0-1
	action = []
	action = [choice, 0, 0, False]
	return action

#Train the straight learning agent
def train_straight_DDPG(episodes,agent):
	# Currently we need one action output that will be 
	# amount of acceleration of straight vehicle
	action_space = 1
	state_space = 2
	# Get the first state (speed, distance from junction)
	# Create model
	straight_model = md.DDPG(action_space,state_space,'Straight_Model')

	# Update rate of target
	tau = 0.01

	# To store reward history of each episode
	ep_reward_list = []
	# To store average reward history of last few episodes
	avg_reward_list = []
	# To store actor and critic loss
	actor_loss = []
	critic_loss = []

	for epi in range(episodes):
		try:
			agent.reset(False)
			time.sleep(1)

			start_state = [0,round(agent.get_location().x-19, 4)]
			state = np.reshape(start_state, (1,2))
			score = 0
			max_step = 1_000
			for i in range(max_step):
				choice = straight_model.policy(state)
				action = choose_action_straight(choice)

				p = 0
				if i%10==0:
					print(f"action----{action}-----epsilon----{straight_model.epsilon}")
					p = 1

				next_state, reward, done, _ = agent.step_straight(action,p)
				time.sleep(1)

				score += reward
				next_state = np.reshape(next_state, (1, 2))
				straight_model.remember(state, choice, reward, next_state, done)
				state = next_state

				# This is back-prop, updating weights
				lossActor, lossCritic = straight_model.replay()
				actor_loss.append(lossActor)
				critic_loss.append(lossCritic)

				# Update the target model, we do it slowly as it keep things stable
				straight_model.update_target(tau)
				if done:
					break

			# Append episode reward to a list
			ep_reward_list.append(score)
			print("\nepisode: {}/{}, score: {}".format(epi, episodes, score))

			avg_reward = np.mean(ep_reward_list[-AGGREGATE_STATS_EVERY:])
			print("\nEpisode * {} * Avg Reward is ==> {}\n".format(epi, avg_reward))
			avg_reward_list.append(avg_reward)

			# Update log stats (every given number of episodes)
			if not epi % AGGREGATE_STATS_EVERY or epi == 1:
				min_reward = min(ep_reward_list[-AGGREGATE_STATS_EVERY:])
				max_reward = max(ep_reward_list[-AGGREGATE_STATS_EVERY:])
				straight_model.tensorboard.update_stats(reward_avg=avg_reward, reward_min=min_reward, reward_max=max_reward, epsilon=straight_model.epsilon)
		
		finally:
			print(f"Task Completed! Episode {epi}")

			straight_model.save_model()
			if agent != None:
				agent.destroy()
				time.sleep(3)

	return actor_loss, critic_loss

if __name__ == '__main__':
	"""
	Main function
	"""
	agent = env.CarlaVehicle()

	#Code to train the models
	episodes = 25
	actor_Loss, critic_Loss = train_straight_DDPG(episodes,agent)

	print("\n\n--We need to Maxmise Actor Loss--Minimise Critic Loss--\n\n")
	x_label = 'Episodes'
	y_label = 'Actor Loss'
	ut.plot(actor_Loss, x_label, y_label)
	y_label = 'Critic Loss'
	ut.plot(critic_Loss,  x_label, y_label)


