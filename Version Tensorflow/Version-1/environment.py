import glob
import os
import sys
import math
import time
import numpy as np
import cv2
import random

try:
	sys.path.append(glob.glob('../../../carla/dist/carla-*%d.%d-%s.egg' % (
		sys.version_info.major,
		sys.version_info.minor,
		'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
	pass

import carla


#Global Setting
SECONDS_PER_EPISODE = 80
np.random.seed(32)
random.seed(32)

class CarlaVehicle(object):
	"""
	class responsable of:
		-spawning the ego vehicle
		-destroy the created objects
		-providing environment for RL training
	"""
	def __init__(self):
		self.client = carla.Client('localhost',2000)
		self.client.set_timeout(5.0)
		self.radar_data = None

	def reset(self,Norender):
		'''reset function to reset the environment before 
		the begining of each episode
		:params Norender: to be set true during training
		'''
		self.collision_hist = []
		self.world = self.client.get_world()
		self.map = self.world.get_map()
		'''target_location: Orientation details of the target loacation
		(To be obtained from route planner)'''
		self.target_waypoint = carla.Transform(carla.Location(x = 1.89, y = 117.06, z=0), carla.Rotation(yaw=269.63))

		#Code for setting no rendering mode
		if Norender:
			settings = self.world.get_settings()
			settings.no_rendering_mode = True
			self.world.apply_settings(settings)

		self.actor_list = []
		self.blueprint_library = self.world.get_blueprint_library()
		self.bp = self.blueprint_library.filter("model3")[0]

		#create ego vehicle the reason for adding offset to z is to avoid collision
		init_pos = carla.Transform(carla.Location(x = 100, y = 130, z=2), carla.Rotation(yaw=180))
		self.vehicle = self.world.spawn_actor(self.bp, init_pos)
		self.actor_list.append(self.vehicle)

		#Create location to spawn sensors
		transform = carla.Transform(carla.Location(x=2.5, z=0.7))
		#Create Collision Sensors
		colsensor = self.blueprint_library.find("sensor.other.collision")
		self.colsensor = self.world.spawn_actor(colsensor, transform, attach_to=self.vehicle)
		self.actor_list.append(self.colsensor)
		self.colsensor.listen(lambda event: self.collision_data(event))
		self.episode_start = time.time()

	#record a collision event
	def collision_data(self, event):
		self.collision_hist.append(event)

	
	def step(self, action):
		#Apply Vehicle Action
		self.vehicle.apply_control(carla.VehicleControl(throttle=action[0], steer=action[1], brake=action[2], reverse=action[3]))


	#Method to take action by the DDPG Agent for straight drive
	def step_straight(self, action, p):
		done = False
		self.step(action)

		#Calculate vehicle speed
		kmh = self.get_speed()

		if p:
			print(f'collision_hist----{self.collision_hist}------kmh----{kmh}------light----{self.vehicle.is_at_traffic_light()}')
		
		reward = 0
		if len(self.collision_hist) != 0:
			done = True
			reward = reward - 100
		elif kmh<2:
			done = False
			reward = reward - 5
			reward = reward + float(kmh)
		elif kmh<25:
			done = False
			reward = reward + 2
			reward = reward + float(kmh/25)
		else:
			done = False
			reward = reward + 1
			reward = reward + float(kmh/50)

		# Build in function of Carla
		if self.vehicle.is_at_traffic_light() and kmh<10:
			done = True
			reward = reward+100
		elif self.vehicle.is_at_traffic_light() and kmh>10:
			done = True
			reward = reward-100

		if self.episode_start + SECONDS_PER_EPISODE < time.time():
			done = True

		location = self.get_location()
		return [round(kmh, 4), round(location.x-19, 4)] , reward, done, None


	def get_location(self):
		"""
			Get the position of a vehicle
			:param vehicle: the vehicle whose position is to get
			:return: speed as a float in Kmh
  		"""
		location = self.vehicle.get_location()
		return location


	def destroy(self):
		"""
			destroy all the actors
			:param self
			:return None
		"""
		print('destroying actors')
		for actor in self.actor_list:
			actor.destroy()
		print('done.')

	def get_speed(self):
		"""
			Compute speed of a vehicle in Kmh
			:param vehicle: the vehicle for which speed is calculated
			:return: speed as a float in Kmh
		"""
		vel = self.vehicle.get_velocity()
		return 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)
