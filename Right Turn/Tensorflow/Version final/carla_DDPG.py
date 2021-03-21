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
AGGREGATE_STATS_EVERY = 50
ROUNDING_FACTOR = 3

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


#action space for right turn
def choose_action_rightturn(choice):
	action = []
	action = [0.5, choice, 0, False]
	return action

#action space for left lane change
def choose_action_leftlanechange(choice):
	action = []
	if choice < 0:
		action = [0.6, choice, 0, False]
	if choice > 0:
		action = [0.5, choice, 0, False]
	return action



"""#overall action space
def choose_action_overall(choice):
	if choice==0:
		action = [0.5, 0, 0, False]
	elif choice==1:
		action = [0.6, 0, 0, False]
	elif choice==2:
		action = [0.5, 0.2, 0.0, False]
	elif choice==3:
		action = [0.5, 0.5, 0.0, False]
	else:
		action = [0.5, 1.0, 0.0, False]
	return action
"""

"""
# ==============================================================================
# -- Function trains Straight Driving DDPG Network ------------------
# -- Scenario Right turn on Traffic Jnction ------------------------------------
# ==============================================================================
def train_straight_DDPG(episodes,agent):
	# Currently we need one action output that will be 
	# amount of acceleration of straight vehicle
	# Shape is the number of neural inputs or output
	action_space = 1
	state_space = 2

	# Currently we didn't resize the Radar data to (4, len(radar))
	# as we have to flatten it anyway
	radar_space = 500
	
	# Get the first state (speed, distance from junction)
	# Create model
	straight_model = md.DDPG(action_space, state_space, radar_space, 'Straight_Model')

	# Update rate of target
	tau = 0.005

	# To store reward history of each episode
	ep_reward_list = []
	# To store average reward history of last few episodes
	avg_reward_list = []
	# To store actor and critic loss
	actor_loss = []
	critic_loss = []
	'''epi_count  = 0
	epirange = 10'''
	
	for epi in range(episodes):
		try:
			loc = random.randint(30, 130)
			print("--------Spawn Location-------")
			print(loc)
			print("\n")
			radar_state_prev = agent.reset(True, loc)
			time.sleep(1)
			radar_state_prev = np.reshape(radar_state_prev, (1,radar_space))

			start_state = [0,100]
			state = np.reshape(start_state, (1,state_space))
			score = 0
			max_step = 5_0

			actor_loss_epi = []
			critic_loss_epi = []
			for i in range(max_step):
				choice = straight_model.policy(radar_state_prev, state)
				action = choose_action_straight(choice)
				#For reward function debugging
				'''k = random.randint(0,80)
				if(epi>=epi_count and epi_count<epirange and k<50):
					action =  choose_action_straight(0.6)
					choice = 0.6'''
				
				print(f"action----{action}-----epsilon----{straight_model.epsilon}")

				radar_state_next, next_state, reward, done, _ = agent.step_straight(action, 1)
				
				score += reward

				next_state = np.reshape(next_state, (1, state_space))
				straight_model.remember(radar_state_prev, radar_state_next, state, choice, reward, next_state, done)
				state = next_state
				radar_state_prev = np.reshape(radar_state_next, (1, radar_space))

				# This is back-prop, updating weights
				lossActor, lossCritic = straight_model.replay()
				actor_loss_epi.append(lossActor)
				critic_loss_epi.append(lossCritic)

				# Update the target model, we do it slowly as it keep things stable, SOFT VERSION
				straight_model.update_target(tau)
				if done:
					break

			actor_loss.append(np.mean(actor_loss_epi))
			critic_loss.append(np.mean(critic_loss_epi))

			# Will do a HARD update now, setting it to critic and actor, set tau=1
			# straight_model.update_target(0.01)

			ep_reward_list.append(score)
			print("\nepisode: {}/{}, score: {}".format(epi, episodes, score))
			'''if(epi>=epi_count and epi_count<epirange):
				epi_count = epi_count+1'''

			avg_reward = np.mean(ep_reward_list[-AGGREGATE_STATS_EVERY:])
			print("\nEpisode * {} * Avg Reward is ==> {}\n".format(epi, avg_reward))
			avg_reward_list.append(avg_reward)

			# Update log stats (every given number of episodes)
			min_reward = min(ep_reward_list[-AGGREGATE_STATS_EVERY:])
			max_reward = max(ep_reward_list[-AGGREGATE_STATS_EVERY:])
			# straight_model.tensorboard.update_stats(reward_avg=avg_reward, reward_min=min_reward, reward_max=max_reward, epsilon=straight_model.epsilon)
			straight_model.tensorboard.update_stats(reward_avg=avg_reward, critic_loss=np.mean(critic_loss_epi), actor_loss=np.mean(actor_loss_epi))
			

			if(epi%100==0 and epi>1):
				x_label = 'Episodes'
				y_label = 'Actor Loss'
				ut.plot(actor_loss, x_label, y_label, epi)
				y_label = 'Critic Loss'
				ut.plot(critic_loss,  x_label, y_label, epi)
			if len(ep_reward_list)>100 and avg_reward > 1000:
				print('\n Task Completed! \n')
				break

		finally:
			print(f"Task Completed! Episode {epi}")

			straight_model.save_model()
			if agent != None:
				agent.destroy()
				time.sleep(1)

	return actor_loss, critic_loss
"""

# ==============================================================================
# -- Function Right Turn DDPG Network ------------------
# -- Scenario Right turn on Traffic Jnction ------------------------------------
# ==============================================================================
def train_rightturn_DDPG(episodes,agent):
	# Two action choice for output
	# amount of acceleration of straight vehicle
	# Shape is the number of neural inputs or output
	action_space = 1
	state_space = 2

	radar_space = 400
	
	# Get the first state (speed, distance from junction)
	# Create model
	rightturn_model = md.DDPG(action_space, state_space, radar_space, 'Right_Turn_Model')

	# Update rate of target
	tau = 0.005

	# To store reward history of each episode
	ep_reward_list = []
	# To store average reward history of last few episodes
	avg_reward_list = []
	# To store actor and critic loss
	actor_loss = []
	critic_loss = []

	#For debugging the reward function
	epi_count  = 150
	epirange = 200
	for epi in range(episodes):
		try:
			loc = random.randint(30, 130)
			print(f'--------Spawn Succeded RightTurn-----------')
			radar_state_prev = agent.reset(False, loc)
			radar_state_prev = np.reshape(radar_state_prev, (1,radar_space))
			start_state = [50,90]
			state = np.reshape(start_state, (1,state_space))
			score = 0
			max_step = 5_00

			actor_loss_epi = []
			critic_loss_epi = []
			for i in range(max_step):
				choice = rightturn_model.policy(radar_state_prev, state)
				action = choose_action_rightturn(choice)
				# print(f'action1------------{action}')
				# if(epi>=epi_count and epi_count<epirange):
				# 	action =  choose_action_rightturn(0.2)
				# 	choice = 0.2
				print(f'action----{action}-------epsilon----{rightturn_model.epsilon}')
				radar_state_next, next_state, reward, done, _ = agent.step_rightturn(action, 1)
				# print(f'next_state-----{next_state}-----reward---{next_state}----{done}')
				time.sleep(0.2)

				score += reward
				next_state = np.reshape(next_state, (1, state_space))
				rightturn_model.remember(radar_state_prev, radar_state_next, state, choice, reward, next_state, done)
				state = next_state
				radar_state_prev = np.reshape(radar_state_next, (1, radar_space))

				# This is back-prop, updating weights
				lossActor, lossCritic = rightturn_model.replay()
				actor_loss_epi.append(lossActor)
				critic_loss_epi.append(lossCritic)

				# Update the target model, we do it slowly as it keep things stable, SOFT VERSION
				rightturn_model.update_target(tau)

				if done:
					break
			actor_loss.append(np.mean(actor_loss_epi))
			critic_loss.append(np.mean(critic_loss_epi))

			# Will do a HARD update now, setting it to critic and actor, set tau=1
			rightturn_model.update_target(0.01)

			ep_reward_list.append(score)
			print("\nepisode: {}/{}, score: {}".format(epi, episodes, score))

			avg_reward = np.mean(ep_reward_list[-AGGREGATE_STATS_EVERY:])
			print("\nEpisode * {} * Avg Reward is ==> {}\n".format(epi, avg_reward))
			avg_reward_list.append(avg_reward)

			# Update log stats (every given number of episodes)
			min_reward = min(ep_reward_list[-AGGREGATE_STATS_EVERY:])
			max_reward = max(ep_reward_list[-AGGREGATE_STATS_EVERY:])
			# straight_model.tensorboard.update_stats(reward_avg=avg_reward, reward_min=min_reward, reward_max=max_reward, epsilon=straight_model.epsilon)
			rightturn_model.tensorboard.update_stats(reward_avg=avg_reward, critic_loss=np.mean(critic_loss_epi), actor_loss=np.mean(actor_loss_epi))
			
			if(epi%100==0 and epi>1):
				x_label = 'Episodes'
				y_label = 'Actor Loss'
				ut.plot(actor_loss, x_label, y_label, epi)
				y_label = 'Critic Loss'
				ut.plot(critic_loss,  x_label, y_label, epi)

			# # Average score of last 100 episode
			# if avg_reward > 500:
			# 	print('\n Task Completed! \n')
			# 	break

		finally:
			print(f"Task Completed! Episode {epi}")

			rightturn_model.save_model()
			if agent != None:
				agent.destroy()
				time.sleep(1)

	return actor_loss, critic_loss

"""
# ==============================================================================
# -- Function Right Turn DDPG Network ------------------
# -- Scenario Right turn on Traffic Jnction ------------------------------------
# ==============================================================================
def train_leftlanechange_DDPG(episodes,agent):
	# Two action choice for output
	# amount of acceleration of straight vehicle
	# Shape is the number of neural inputs or output
	action_space = 1
	state_space = 4

	radar_space = 500
	
	# Get the first state (speed, distance from junction)
	# Create model
	left_lane_model = md.DDPG(action_space, state_space, radar_space, 'LeftLane_Change_Model')

	# Update rate of target
	tau = 0.005

	# To store reward history of each episode
	ep_reward_list = []
	# To store average reward history of last few episodes
	avg_reward_list = []
	# To store actor and critic loss
	actor_loss = []
	critic_loss = []

	#For debugging the reward function
	'''epi_count  = 0
	epirange = 10'''
	for epi in range(episodes):
		try:
			loc = random.randint(30, 130)
			print(f'--------Spawn Succeded LeftLanChange-----------')
			radar_state_prev = agent.reset(True, loc)
			radar_state_prev = np.reshape(radar_state_prev, (1,radar_space))
			start_state = [5,5,5,5]
			state = np.reshape(start_state, (1,state_space))
			score = 0
			max_step = 1_000

			actor_loss_epi = []
			critic_loss_epi = []
			for i in range(max_step):
				choice = left_lane_model.policy(radar_state_prev, state)
				action = choose_action_leftlanechange(choice)
				print(f'action1------------{action}')
				#For Debugging the reward function
				'''if(epi>=epi_count and epi_count<epirange):
					if(agent.lane_id_ego!=agent.lane_id_target and action[1]>0):
						action =  choose_action_leftlanechange(-0.15)
						choice = -0.15
					elif agent.lane_id_ego==agent.lane_id_target:
						if(abs(abs(int(180+agent.yaw_vehicle))-abs(int(agent.yaw_target_road)))>1 and action[1]<0):
							action = choose_action_leftlanechange(0.2)
							choice = 0.2
					print(f'CHOICE FORCED------------{choice}')'''
				print(f'action2------------{action}')
				radar_state_next, next_state, reward, done, _ = agent.step_leftlanechange(action, 1)
				print(f'next_state------------{next_state}-----reward---{next_state}----done--{done}')
				#time.sleep(0.1)

				score += reward
				next_state = np.reshape(next_state, (1, state_space))
				left_lane_model.remember(radar_state_prev, radar_state_next, state, choice, reward, next_state, done)
				state = next_state
				radar_state_prev = np.reshape(radar_state_next, (1, radar_space))

				# This is back-prop, updating weights
				lossActor, lossCritic = left_lane_model.replay()
				actor_loss_epi.append(lossActor)
				critic_loss_epi.append(lossCritic)

				# Update the target model, we do it slowly as it keep things stable, SOFT VERSION
				#rightturn_model.update_target(tau)

				if done:
					break
			actor_loss.append(np.mean(actor_loss_epi))
			critic_loss.append(np.mean(critic_loss_epi))

			# Will do a HARD update now, setting it to critic and actor, set tau=1
			left_lane_model.update_target(0.01)

			ep_reward_list.append(score)
			print("\nepisode: {}/{}, score: {}".format(epi, episodes, score))

			avg_reward = np.mean(ep_reward_list[-AGGREGATE_STATS_EVERY:])
			print("\nEpisode * {} * Avg Reward is ==> {}\n".format(epi, avg_reward))
			avg_reward_list.append(avg_reward)

			# Update log stats (every given number of episodes)
			min_reward = min(ep_reward_list[-AGGREGATE_STATS_EVERY:])
			max_reward = max(ep_reward_list[-AGGREGATE_STATS_EVERY:])
			# straight_model.tensorboard.update_stats(reward_avg=avg_reward, reward_min=min_reward, reward_max=max_reward, epsilon=straight_model.epsilon)
			left_lane_model.tensorboard.update_stats(reward_avg=avg_reward, critic_loss=np.mean(critic_loss_epi), actor_loss=np.mean(actor_loss_epi))
			if(epi>=epi_count and epi_count<epirange):
					epi_count = epi_count+1
			
			if(epi%100==0 and epi>1):
				x_label = 'Episodes'
				y_label = 'Actor Loss'
				ut.plot(actor_loss, x_label, y_label, epi)
				y_label = 'Critic Loss'
				ut.plot(critic_loss,  x_label, y_label, epi)

			# Average score of last 100 episode
			if avg_reward > 500:
				print('\n Task Completed! \n')
				break

		finally:
			print(f"Task Completed! Episode {epi}")

			left_lane_model.save_model()
			if agent != None:
				agent.destroy()
				time.sleep(1)

	return actor_loss, critic_loss



def train_overall_dqn(episodes, agent):
	'''For comparison with RL agent 
	that learns the overall task'''
	#Create model
	loss = []
	overall_model = md.DQN(4,3,'Overall')
	for epi in range(episodes):
		try:
				#agent.reset(False)
				agent.reset(True)
				print(f'spawn suceeded----------')
				state = [0,50,0]
				state = np.reshape(state, (1,3))
				score = 0
				max_step = 1000
				for i in range(max_step):
					choice = overall_model.act(state)
					action = choose_action_overall(choice)
					print(f"action ------------------> {action}")
					next_state, reward, done, _ = agent.step_overall(action)
					print(f'obs----------->{next_state}-----reward--- {reward} -----done--{done}')
					time.sleep(0.5)
					score += reward
					next_state = np.reshape(next_state, (1, 3))
					overall_model.remember(state, choice, reward, next_state, done)
					state = next_state
					overall_model.replay(done,epi,loss)
					if done:
						print("episode: {}/{}, score: {}".format(epi, episodes, score))
						break
				loss.append(score)
				# Average score of last 100 episode
				is_solved = 0
				if len(loss)>=100:
					is_solved = np.mean(loss[-100:])
				if is_solved > 2000:
					print('n Task Completed! n')
					break
				print("Average over last 100 episode: {0:.2f} n".format(is_solved))
		finally:
			overall_model.save_model()
			if agent != None:
				agent.destroy()
				time.sleep(5)
	return loss

#Code for neural network right turn control
def nn_control():
	try:
		agent.reset(False)
		traffic_light = None
		straight_model = load_model('Straight_Model.h5')
		right_turn_model = load_model('Right_Turn_converged.h5')
		print(f'spawn suceeded----------')
		obs = [0,agent.get_location().x-19]
		while agent.get_location().x > 19:
			obs = np.reshape(obs, (1,2))
			choice = straight_model.predict(obs)
			action = choose_action_straight(np.argmax(choice[0]))
			obs, reward, done, _ = agent.step_straight(action)
			time.sleep(0.5)
			#Break if in traffic light trigger box
			if agent.get_vehicle().is_at_traffic_light():
				agent.step([0.0, 0.0, 1.0, False])
				#Stop at intersection
				traffic_light = agent.get_vehicle().get_traffic_light()
				break
		#Staye Stopped at traffic light
		agent.step([0.0, 0.0, 1.0, False])
		time.sleep(1)
		obs = [50,90]
		'''while(traffic_light.get_state() != carla.TrafficLightState.Green):
				print(traffic_light.get_state())'''
		done = False
		while(not done):
			obs = np.reshape(obs, (1,2))
			choice = right_turn_model.predict(obs)
			action = choose_action_rightturn(np.argmax(choice[0]))
			print(f'action-------------{action}')
			obs, reward, done, _ = agent.step_rightturn(action)
			print(f'obs----------->{obs}-----reward--- {reward} -----done--{done}')
			time.sleep(0.5)
		count = 0
		while count<10:
			print('Here')
			agent.step([0.5, 0.0, 0.0, False])
			time.sleep(0.5)
			count=count+1
		agent.step([0.0, 0.0, 1.0, False])
		time.sleep(5)
	finally:
		agent.destroy()
		time.sleep(5)


#Code for manual right turn control
def manual_control():
	try:
		agent.reset(False)
		traffic_light = None
		print(f'spawn suceeded----------')
		while agent.get_location().x > 19:
			obs, reward, done, _  = agent.step_overall([0.6, 0.0, 0.0, False])
			print(f'obs----------->{obs}-----reward--- {reward} -----done--{done}')
			time.sleep(0.5)
			#Break if in traffic light trigger box
			if agent.get_vehicle().is_at_traffic_light():
				agent.step([0.0, 0.0, 1.0, False])
				#Stop at intersection
				traffic_light = agent.get_vehicle().get_traffic_light()
				break
		#Staye Stopped at traffic light
		agent.step([0.0, 0.0, 1.0, False])
		time.sleep(1)
		'''while(traffic_light.get_state() != carla.TrafficLightState.Green):
				print(traffic_light.get_state())'''
		while(agent.get_location().y > 115):
			obs, reward, done, _  = agent.step_overall([0.5, 0.2, 0.0, False])
			print(f'obs----------->{obs}-----reward--- {reward} -----done--{done}')
			time.sleep(0.5)
		count = 0
		while count<10:
			print('Here')
			agent.step([0.5, 0.0, 0.0, False])
			time.sleep(0.5)
			count=count+1
		agent.step([0.0, 0.0, 1.0, False])
		time.sleep(5)
	finally:
		agent.destroy()
		time.sleep(5)

"""

if __name__ == '__main__':
	"""
	Main function
	"""
	agent = env.CarlaVehicle()

	#Code to train the models
	episodes = 5_00
	# actor_Loss, critic_Loss = train_straight_DDPG(episodes,agent)
	actor_Loss, critic_Loss = train_rightturn_DDPG(episodes,agent)

	print("\n\n--We need to Maxmise Actor Loss--Minimise Critic Loss--\n\n")
	x_label = 'Episodes'
	y_label = 'Actor Loss'
	ut.plot(actor_Loss, x_label, y_label)
	y_label = 'Critic Loss'
	ut.plot(critic_Loss,  x_label, y_label)
	#Code for Manual Control
	#manual_control()

