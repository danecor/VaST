from vizdoom import *
import cv2
import skimage.transform
import numpy as np
import logging
import os
import position_tests
import matplotlib.pyplot as plt
import io

class Environment(DoomGame):

	def __init__(self, **kwargs):

		map_path = kwargs['map_path']
		high_res = kwargs.get('high_res', False)
		seed = kwargs.get('seed', 0)
		rng = kwargs.get('rng', None)
		length = kwargs.get('length', 0)
		play_doom = kwargs.get('play_doom', False)
		show_screen = kwargs.get('show_screen', False)
		include_use = kwargs.get('include_use', False)

		self.train_epoch_length = kwargs.get('train_epoch_length', np.inf)
		self.test_epoch_length = kwargs.get('test_epoch_length', np.inf)
		self.epoch_length = self.train_epoch_length
		self.epoch_steps = kwargs.get('epoch_steps', 0)
		self.hist_len = kwargs['hist_len']
		self.test = False
	    
		# Create DoomGame instance. It will run the game and communicate with you.
		super(Environment, self).__init__()
		self.act_skip = kwargs['act_skip']
		self.num_reset_steps = kwargs['num_reset_steps']
		self.max_turn_steps = kwargs['max_turn_steps']
		if include_use:
			self.actions = [[True, False, False, False], 
			           		[False, True, False, False], 
			           		[False, False, True, False],
			           		[False, False, False, True],
			           		]
		else:
			self.actions = [[True, False, False], 
			           		[False, True, False], 
			           		[False, False, True],
			           		]
		self.screen_size = tuple(kwargs['screen_size'])
		self.terminal_distance = kwargs['terminal_distance']
		self.position_test = position_tests.__dict__[kwargs['position_test']]
		self.terminal_reward = kwargs['terminal_reward']
		self.goal_reward = kwargs['goal_reward']
		self.living_reward = kwargs['living_reward']
		self.hist_len = kwargs['hist_len']
		self.xlim = kwargs['xlim']
		self.ylim = kwargs['ylim']
		if self.hist_len > 1:
			self.observations = np.zeros(kwargs['screen_size'] + [3*self.hist_len], 
			                             dtype=np.uint8)
		self.episode_positions = []
		self.record_positions = None
		if kwargs.get('record_positions', False):
			self.record_positions = []
		self.params = kwargs
		if rng is None:
			self.rng = np.random.RandomState(seed)
		else:
			self.rng = rng
		self.set_seed()

		assert os.path.isfile('_vizdoom.ini'), 'The file _vizdoom.ini must be in the current diectory.'
		
		# Now it's time for configuration!
		# load_config could be used to load configuration instead of doing it here with code.
		# If load_config is used in-code configuration will also work - most recent changes will add to previous ones.
		# game.load_config("../../scenarios/basic.cfg")
		
		# Sets path to additional resources wad file which is basically your scenario wad.
		# If not specified default maps will be used and it's pretty much useless... unless you want to play good old Doom.
		if os.path.exists("doom/maps/"):
			self.set_doom_scenario_path("doom/maps/%s.wad" % map_path)
		else:
			self.set_doom_scenario_path("maps/%s.wad" % map_path)
		
		# Sets map to start (scenario .wad files can contain many maps).
		self.set_doom_map(kwargs['map'])
		
		# Sets resolution. Default is 320X240
		if high_res:
			res = ScreenResolution.RES_640X480
		else:
			res = ScreenResolution.RES_160X120
		self.set_screen_resolution(res)
		
		# Sets the screen buffer format. Not used here but now you can change it. Defalut is CRCGCB.
		self.set_screen_format(ScreenFormat.RGB24)
		
		# Enables depth buffer.
		#self.set_depth_buffer_enabled(True)
		
		# Enables labeling of in self objects labeling.
		self.set_labels_buffer_enabled(True)
		
		# Enables buffer with top down map of the current episode/level.
		self.set_automap_buffer_enabled(False)
		
		# Sets other rendering options
		self.set_render_hud(False)
		self.set_render_minimal_hud(False)  # If hud is enabled
		self.set_render_crosshair(False)
		self.set_render_weapon(False)
		self.set_render_decals(False)
		self.set_render_particles(False)
		self.set_render_effects_sprites(False)
		self.set_render_messages(False)
		self.set_render_corpses(False)
		
		# Adds buttons that will be allowed. 
		self.add_available_button(Button.MOVE_FORWARD)
		self.add_available_button(Button.TURN_LEFT)
		self.add_available_button(Button.TURN_RIGHT)
		if include_use:
			self.add_available_button(Button.USE)
		#self.add_available_button(Button.MOVE_LEFT)
		#self.add_available_button(Button.MOVE_RIGHT)
		#self.add_available_button(Button.MOVE_BACKWARD)
		
		# Adds self variables that will be included in state.
		#self.add_available_self_variable(selfVariable.ARMOR)
		#self.add_available_self_variable(selfVariable.HEALTH)

		if kwargs.get('dm', True):
			self.add_game_args("-deathmatch")
		self.add_game_args("-use_mouse=false")
		
		# Sets the living reward (for each move) to -1
		self.set_living_reward(0)
		
		# Sets ViZDoom mode (PLAYER, ASYNC_PLAYER, SPECTATOR, ASYNC_SPECTATOR, PLAYER mode is default)
		if play_doom:
			self.set_mode(Mode.SPECTATOR)
			self.set_window_visible(True)
		else:
			self.set_mode(Mode.PLAYER)
			self.set_window_visible(show_screen)

		self.set_episode_timeout(length)

		# Enables engine output to console.
		#self.set_console_enabled(True)
		
		# Initialize the game. Further configuration won't take any effect from now on.
		self.init()

	def start(self):
		self.new_episode()

	def end(self, save_path=None):
		if (self.record_positions is not None) and (save_path is not None):
			record_positions = np.array(self.record_positions)
			np.save("%s/record_positions.npy" % save_path,
			        record_positions)
		self.close()

	def set_seed(self):
		super(Environment, self).set_seed(self.rng.randint(100000000))

	def init_epoch(self, test=False):
		self.epoch_steps = 0
		if test:
			self.epoch_length = self.test_epoch_length
			self.test = True
		else:
			self.epoch_length = self.train_epoch_length
			self.test = False

	def epoch_finished(self):
		return (self.epoch_steps >= self.epoch_length)

	def init_episode(self):
		self.episode_positions = []
		self.respawn_player()
		if self.hist_len > 1:
			self.observations *= 0
		if self.max_turn_steps>0:
			for i in range(self.rng.randint(self.max_turn_steps)):
				self.step(1, False)
		for i in range(self.num_reset_steps):
			self.step(self.rng.choice(len(self.actions)), False)
		_, terminal = self.get_rt()
		if terminal:
			return self.init_episode()
		if (self.record_positions is not None) and (not self.test):
			self.record_positions.append(self.get_position())
		return self.get_observations()

	def step(self, action_index, epoch_step=True):
		action = self.actions[action_index]
		super(Environment, self).make_action(action, self.act_skip)
		if epoch_step:
			self.epoch_steps += 1
			return self.get_action_results(epoch_step)

	def get_action_results(self, epoch_step=True):
		obs = self.get_observations()
		reward, terminal = self.get_rt(epoch_step)
		return obs, reward, terminal

	def get_observations(self):
		screen = self.get_state().screen_buffer
		processed_screen = cv2.resize(screen, self.screen_size[::-1], 
		                           interpolation=cv2.INTER_LINEAR)
		if self.hist_len==1:
			return processed_screen[:, :, ::-1]
		self.observations[:, :, 3:] = self.observations[:, :, :-3]
		self.observations[:, :, :3] = processed_screen[:, :, ::-1]
		return self.observations

	def get_rt(self, epoch_step=False):
		logging.debug('Get position variables')
		pos_x, pos_y, angle = self.get_position()
		if epoch_step:
			self.episode_positions.append([pos_x, pos_y, angle])
			if (self.record_positions is not None) and (not self.test):
				self.record_positions.append([pos_x, pos_y, angle])
		logging.debug('Test for terminal')
		return self.position_test(pos_x, pos_y, 
		                          self.terminal_distance, self.goal_reward, 
		                          self.terminal_reward, self.living_reward,
		                          self.rng)

	def get_position(self):
		pos_x = self.get_game_variable(GameVariable.POSITION_X)
		pos_y = self.get_game_variable(GameVariable.POSITION_Y)
		angle = self.get_game_variable(GameVariable.ANGLE)
		return pos_x, pos_y, angle

	def get_full_obs(self):
		return self.get_state().screen_buffer

	def get_episode_plots(self):
		return self.get_path_plot()

	def get_path_plot(self):
		logging.debug("Generating path plot.")
		x, y = np.array(self.episode_positions).T[:2]
		fig = plt.figure(num=0, figsize=(6, 4), dpi=200)
		fig.clf()
		plt.quiver(x[:-1],y[:-1],np.diff(x), np.diff(y), 
		           angles='xy', scale_units='xy', scale=1)
		plt.title("Episode Trajectory")
		plt.xlim(self.xlim)
		plt.ylim(self.ylim)
		plt.tight_layout()
		logging.debug("Path plot generated.")
		return {"episode_trajectory": fig}