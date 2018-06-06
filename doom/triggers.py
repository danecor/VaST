import logging
from environment import Environment
import numpy as np
import time
from position_tests import *

class Trigger(object):

	def __init__(self, trigger_step):
		self.trigger_step = trigger_step

	def attempt(self, model, agent, env):
		if agent.step == self.trigger_step:
			logging.info('Activating trigger: %s at step %s' % (str(self), 
			                                                    self.trigger_step))
			return self._activate(model, agent, env)
		return model, agent, env, False

	def _activate(self, model, agent, env):
		return model, agent, env, False

class ReuseableTrigger(Trigger):

	def attempt(self, model, agent, env):
		if agent.step % self.trigger_step == 0:
			logging.info('Activating trigger: %s at step %s' % (str(self), 
			                                                    self.trigger_step))
			return self._activate(model, agent, env)
		return model, agent, env, False

class ReverseRewards(Trigger):

	def __str__(self):
		return "Reverse Rewards to Hall Y"

	def _activate(self, model, agent, env):
		env.position_test = hall_y
		return model, agent, env, False

class SwitchMap(Trigger):

	def _activate(self, model, agent, env):
		env.end()
		params = env.params
		epoch_steps = env.epoch_steps
		rng = env.rng
		params['map_path'] = self.new_map_path
		env = Environment(**params)
		env.rng = rng
		env.epoch_steps = epoch_steps
		env.start()
		return model, agent, env, True

class AddTeleporter(SwitchMap):

	def __str__(self):
		return "Add teleporter to H-maze"

	def __init__(self, trigger_step):
		super(AddTeleporter, self).__init__(trigger_step)
		self.new_map_path = 'hmazeport'

class AddTeleporterAndPrimeFRunP(AddTeleporter):

	def __init__(self, trigger_step):
		super(AddTeleporterAndPrimeFRunP, self).__init__(trigger_step)
		self.memory_path = 'doom/maps/hmaze_teleport_frames_frunP.npz'

	def __str__(self):
		return "Add teleporter to H-maze and prime agent with a forced run"

	def _activate(self, model, agent, env):
		model, agent, env, _ = super(AddTeleporterAndPrimeFRunP, self)._activate(model, agent, 
		                                                                   	  env)
		forced_run = np.load(self.memory_path)
		obs = forced_run['obs']
		acts = forced_run['acts']
		reward = forced_run['reward']
		terminal = forced_run['terminal']
		agent.init_episode(obs[0], model)
		for i in range(len(acts)):
			env.epoch_steps += 1
			agent.observe(acts[i], reward[i], terminal[i], obs[i+1], model)
		return model, agent, env, True

class AddTeleporterAndPrimeFRunN(AddTeleporterAndPrimeFRunP):

	def __init__(self, trigger_step):
		super(AddTeleporterAndPrimeFRunN, self).__init__(trigger_step)
		self.memory_path = 'doom/maps/hmaze_teleport_frames_frunN.npz'

class FRunEx(Trigger):

	def __init__(self, trigger_step):
		super(FRunEx, self).__init__(trigger_step)
		self.memory_path = 'doom/maps/exmaze_frames.npz'

	def __str__(self):
		return "Forced run of alternate arm of T-maze"

	def _activate(self, model, agent, env):
		forced_run = np.load(self.memory_path)
		obs = forced_run['obs']
		acts = forced_run['acts']
		reward = forced_run['reward']
		terminal = forced_run['terminal']
		positions = forced_run['positions']
		agent.init_episode(obs[0], model)
		env.record_positions.append(positions[0])
		for i in range(len(acts)):
			env.epoch_steps += 1
			env.record_positions.append(positions[i])
			agent.observe(acts[i], reward[i], terminal[i], obs[i+1], model)
		time.sleep(50)
		return model, agent, env, True