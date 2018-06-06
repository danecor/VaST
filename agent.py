import numpy as np
import os
import logging
from table.lookup import *
from io_utils import get_max_steps
import cPickle as pickle
from replay_memory import ReplayMemory

class Agent(object):

	def __init__(self, save_path, **kwargs):
		self.step = 0
		self.save_path = save_path
		self.n_act = kwargs['n_act']
		self.n_z = kwargs['n_z']
		self.epsilon_period = kwargs['epsilon_period']
		self.min_epsilon = kwargs['min_epsilon']
		self.exp_eps_decay = kwargs['exp_eps_decay']
		self.burnin = kwargs['burnin']
		self.test = False
		self.test_epsilon = kwargs['test_epsilon']
		self.state = None
		self.track_repeats = kwargs.get('track_repeats', False)
		self.freeze_weights = kwargs.get('freeze_weights', False)
		seed = kwargs.get('seed', None)
		if seed is not None:
			self.rng = np.random.RandomState(seed)
		else:
			self.rng = np.random.RandomState()
		self.summary_variables = {'reward': 0, 'nonzero_rewards': 0, 
								  'episode_reward': 0}
		if self.track_repeats:
			self.hashed_obs = set()
			self.episode_hashed_obs = set()
			self.hashed_obs_matches = 0.
			self.hashed_obs_checks = 0.
		self.final_init(kwargs)

	def final_init(self, params):
		batch_size = params['minibatch_size']*params['concurrent_batches']
		replay_length = params['hist_len']+1
		self.replay_memory = ReplayMemory(params['max_replay_size'], 
										  params['obs_size'],
										  replay_length,
										  batch_size,
										  self.rng)
		self.index_vec = 2**np.arange(self.n_z, dtype='uint64')
		self.delete_old_episodes = params['delete_old_episodes']
		self.summary_variables['num_sweeps'] = 0
		self.lookup = Lookup(params['n_z'], self.n_act, params['discount'], 
							 params['init_capacity'], params['pri_cutoff'], 
							 self.save_path, self.rng)

	def get_states(self, zs):
		if len(zs.shape) == 1:
			zs.shape = [self.n_z, 2]
		else:
			zs.shape = [zs.shape[0], self.n_z, 2]
		keys = np.argmax(zs, -1).astype(np.bool)
		return np.dot(keys, self.index_vec)

	def get_epsilon(self):
		if self.test:
			return self.test_epsilon
		step = self.step-self.burnin
		if step < 0:
			return 1.
		if self.epsilon_period == 0:
			return self.min_epsilon
		if self.exp_eps_decay:
			return (1-self.min_epsilon)*np.exp(-step/float(self.epsilon_period)) + self.min_epsilon
		return max((1-self.min_epsilon)*(1 - step/float(self.epsilon_period)), 0) + self.min_epsilon

	def get_action(self, model):
		if self.step <= self.burnin:
			return self.rng.choice(range(self.n_act))
		epsilon = self.get_epsilon()
		if self.rng.rand() < epsilon:
			action = self.rng.choice(range(self.n_act))
		else:
			logging.debug("Estimating action for state %s" % str(self.state))
			action = self.lookup.estimate_max_action(self.state)
		return action

	def init_episode(self, obs, model):
		self.reset_episode_summary()
		zs = model.encode(obs)
		self.state = self.get_states(zs)
		if (not self.test) and (self.replay_memory is not None):
			pop_eps = self.replay_memory.append(0, 0, False, obs, self.state, start=True)
			if self.delete_old_episodes and (pop_eps is not None):
				self.lookup.delete_transition(*pop_eps)
			if self.track_repeats:
				self.check_repeated(obs)

	def observe(self, action, reward, terminal, obs_next, model):
		zs = model.encode(obs_next)
		state_next = self.get_states(zs)
		table_state_next = None if terminal else state_next
		if not self.test:
			self.step += 1
			self.lookup.add_transition(self.state, action, reward, table_state_next)
			pop_eps = self.replay_memory.append(action, reward, terminal, obs_next, state_next)
			if self.delete_old_episodes and (pop_eps is not None):
				self.lookup.delete_transition(*pop_eps)
			if self.track_repeats:
				self.check_repeated(obs_next)
		self.state = table_state_next
		self.update_summary_variables(reward)

	def train_and_update(self, model, summary_writer):
		if self.test or (self.step < self.burnin) or self.freeze_weights:
			return None
		batch = self.replay_memory.get_minibatch()
		zs, wait_time = model.train(self.step, summary_writer, batch)
		self.summary_variables['train_wait_time'] = wait_time
		self.update_transitions(zs)
		self.replay_memory.minibatch_inds = batch['inds']

	def finish_update(self, model):
		if self.test or (self.step < self.burnin) or self.freeze_weights:
			return None
		zs, wait_time = model.finish_training()
		self.summary_variables['train_wait_time'] = wait_time
		self.update_transitions(zs)

	def update_transitions(self, zs):
		logging.debug("Updating transitions.")
		if zs is None:
			return None
		states = self.get_states(zs)
		updated_transitions, pc_changed = self.replay_memory.get_updated_transitions(states)
		for updated_transition in updated_transitions:
			self.lookup.update_transition(*updated_transition)
		self.summary_variables['reassigned_states'] = pc_changed

	def update_summary_variables(self, reward):
		prefix = ''
		if self.test:
			prefix = 'eval_'
		self.summary_variables[prefix+'reward'] += reward
		self.summary_variables[prefix+'nonzero_rewards'] += (reward != 0)
		self.summary_variables[prefix+'episode_reward'] += reward
		self.summary_variables[prefix+'episode_length'] += 1

	def get_summary_variables(self):
		self.summary_variables['epsilon'] = self.get_epsilon()
		max_q, avg_q, table_size, num_sweeps, p_size, lookup_dist, lookup_val = self.lookup.get_summary_variables()
		self.summary_variables['table_size'] = table_size
		self.summary_variables['max_q'] = max_q
		self.summary_variables['avg_q'] = avg_q
		self.summary_variables['lookup_distance'] = lookup_dist
		self.summary_variables['lookup_values'] = lookup_val
		self.summary_variables['num_sweeps'] = num_sweeps
		self.summary_variables['priority_queue_size'] = p_size
		if self.track_repeats:
			self.summary_variables['obs_repeats'] = float(self.hashed_obs_matches)/self.hashed_obs_checks
		return self.summary_variables

	def reset_summary_variables(self):
		if self.test:
			self.summary_variables['eval_reward'] = 0.
			self.summary_variables['eval_nonzero_rewards'] = 0.
		else:
			self.summary_variables['reward'] = 0.
			self.summary_variables['nonzero_rewards'] = 0.
			self.lookup.reset_summary_variables()

	def reset_episode_summary(self):
		if self.test:
			self.summary_variables['eval_episode_reward'] = 0.
			self.summary_variables['eval_episode_length'] = 0.	
		else:		
			self.summary_variables['episode_reward'] = 0.
			self.summary_variables['episode_length'] = 0.
			if self.track_repeats:
				self.hashed_obs.update(self.episode_hashed_obs)
				self.episode_hashed_obs = set()
				#self.hashed_obs_matches = 0.
				#self.hashed_obs_checks = 0.

	def check_repeated(self, obs):
		hash_obs = hash(obs.tostring())
		self.episode_hashed_obs.add(hash_obs)
		if hash_obs in self.hashed_obs:
			self.hashed_obs_matches += 1.
		self.hashed_obs_checks += 1.

	def save(self):
		old_steps = get_max_steps(self.save_path, 'agent')
		if old_steps is not None:
			os.remove("%s/agent.ckpt-%s" % (self.save_path, old_steps))
		data = self.replay_memory.save_and_export(self.save_path, self.step,
		                                          old_steps)
		lookup = self.lookup
		self.lookup = None
		agent_save_path = "%s/agent.ckpt-%s" % (self.save_path, self.step)
		with open(agent_save_path, "wb") as handle:
			pickle.dump(self, handle, pickle.HIGHEST_PROTOCOL)
		self.replay_memory.load_memory(data)
		lookup.save(self.step, old_steps)
		self.lookup = lookup