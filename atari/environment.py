"""
Based on https://github.com/vvanirudh/Atari-DeepRL (Author: Nathan Sprague)
"""

import cv2
import numpy as np
import atari_py
import atari_py.ale_python_interface as ale_python_interface
import logging
from skimage import img_as_ubyte

class Environment(object):

	def __init__(self, **kwargs):

		game = kwargs['game']
		frame_skip = kwargs['frame_skip']
		max_start_nullops = kwargs['max_start_nullops']
		death_ends_episode = kwargs['death_ends_episode']
		hist_len = kwargs['hist_len']
		crop_top = kwargs['crop_top']
		crop_bottom = kwargs['crop_bottom']
		seed = kwargs.get('seed', 0)
		rng = kwargs.get('rng', None)

		self.train_epoch_length = kwargs['train_epoch_length']
		self.test_epoch_length = kwargs['test_epoch_length']
		self.epoch_length = None
		self.epoch_steps = 0

		self.ale = ale_python_interface.ALEInterface()
		#Implement frame skip ourselves
		self.frame_skip = frame_skip
		self.ale.setInt("frame_skip",1)
		self.ale.setBool('display_screen', False)
		self.ale.setFloat('repeat_action_probability', 0.)

		self.ale.loadROM(atari_py.get_game_path(game))
		self.max_start_nullops = max_start_nullops
		self.actions = self.ale.getMinimalActionSet()
		#Fire button not required for deterministic pong
		if game=='pong':
			self.actions = [0, 3, 4]

		self.width, self.height = self.ale.getScreenDims()

		self.buffer_length = 2
		self.buffer_count = 0
		self.screen_buffer = np.empty((self.buffer_length,
									   self.height, self.width),
									   dtype=np.uint8)
		self.observations = np.zeros((84, 84, hist_len), dtype=np.float32)

		self.death_ends_episode = death_ends_episode
		self.terminal_lol = False # Most recent episode ended on a loss of life

		self.crop_top = crop_top
		self.crop_bottom = crop_bottom

		if rng is None:
			self.rng = np.random.RandomState(seed)
		else:
			self.rng = rng

	def init_epoch(self, test=False):
		self.epoch_steps = 0
		if test:
			self.epoch_length = self.test_epoch_length
		else:
			self.epoch_length = self.train_epoch_length

	def epoch_finished(self):
		return (self.epoch_steps >= self.epoch_length)

	def init_episode(self):
		self.observations *= 0
		if not self.terminal_lol or self.ale.game_over():
			self.ale.reset_game()
			if self.max_start_nullops > 0:
				random_actions = self.rng.randint(0, self.max_start_nullops)
				for _ in range(random_actions):
					self._act(0)  # Null action
		# Make sure the screen buffer is filled at the beginning of
		# each episode...
		for i in range(2):
			self._act(0)
		self._update_observations(self.get_screen())
		return self.observations

	def start(self):
		pass

	def end(self, save_path):
		pass

	def set_seed(self):
		pass

	def step(self, action):
		""" Repeat one action the appropriate number of times and return
		the summed reward. """
		start_lives = self.ale.lives()
		reward = 0
		for _ in range(self.frame_skip):
		    reward += self._act(self.actions[action])
		self._update_observations(self.get_screen())
		self.terminal_lol = (self.death_ends_episode and (self.ale.lives() < start_lives))
		terminal = self.ale.game_over() or self.terminal_lol
		self.epoch_steps += 1
		return self.observations, reward, terminal

	def _act(self, action):
		"""Perform the indicated action for a single frame, return the
		resulting reward and store the resulting screen image in the
		buffer
		"""
		reward = self.ale.act(action)
		index = self.buffer_count % self.buffer_length
		self.ale.getScreenGrayscale(self.screen_buffer[index, ...])
		self.buffer_count += 1
		return reward

	def _update_observations(self, screen):
		self.observations[:, :, 1:] = self.observations[:, :, :-1]
		self.observations[:, :, 0] = screen[:, :, 0]

	def get_screen(self):
		""" Resize and merge the previous two screen images """
		assert self.buffer_count >= 2
		index = self.buffer_count % self.buffer_length - 1
		max_image = np.maximum(self.screen_buffer[index, ...],
		                       self.screen_buffer[index - 1, ...])
		return self.preprocess(max_image)[:, :, None]

	def preprocess(self, observation):
		observation = observation[self.crop_top:-self.crop_bottom-1]
		observation = cv2.resize(observation, (84, 84), 
		                         interpolation = cv2.INTER_LINEAR).astype(np.float32)
		observation -= observation.min()
		observation /= observation.max()
		return img_as_ubyte(observation)

	def get_full_obs(self):
		return self.ale.getScreenRGB()

	def get_episode_plots(self):
		return {}