import numpy as np
import logging
import time
import os

class ReplayMemory(object):

	def __init__(self, max_size, obs_shape, hist_len, batch_size, rng, 
	             return_obs_next=False):
		self.max_size = max_size
		self.position = 0
		self.size = 0
		self.hist_len = hist_len
		self.channels = obs_shape[2]
		self.batch_size = batch_size
		self.rng = rng
		self.return_obs_next = return_obs_next
		self.screens = np.empty((self.max_size, obs_shape[0], obs_shape[1], self.channels), 
		                    	dtype = np.uint8)
		self.actions = np.empty(self.max_size, dtype=np.uint8)
		self.rewards = np.empty(self.max_size, dtype=np.float32)
		self.starts = np.zeros(self.max_size, dtype=np.bool)
		self.terminals = np.zeros(self.max_size, dtype=np.bool)
		self.state_assignments = np.zeros(self.max_size, dtype=np.uint64)
		self.new = np.zeros(self.max_size, dtype=np.bool)
		if self.hist_len > 1:
			self.minibatch_obs = np.empty((batch_size, self.hist_len, self.screens.shape[1], 
			                               self.screens.shape[2], self.screens.shape[3]), 
										  dtype=np.uint8)
		self.minibatch_inds = None

	def append(self, action, reward, terminal, obs_next, state_next=None, start=False):
		pop_episode = None
		if self.is_full():
			if not self.starts[(self.position + 1) % self.max_size]:
				pop_state = self.state_assignments[self.position]
				pop_action = self.actions[(self.position + 1) % self.max_size]
				pop_reward = self.rewards[(self.position + 1) % self.max_size]
				if self.terminals[(self.position + 1) % self.max_size]:
					pop_next_state = None
				else:
					pop_next_state = self.state_assignments[(self.position + 1) % self.max_size]
				pop_episode = (pop_state, pop_action, pop_reward, pop_next_state)
		self.actions[self.position] = action
		self.screens[self.position] = obs_next[:, :, :self.channels]
		self.rewards[self.position] = reward
		self.starts[self.position] = start
		self.terminals[self.position] = terminal
		self.new[self.position] = 1
		if state_next is not None:
			self.state_assignments[self.position] = state_next
		self.size = min(self.size + 1, self.max_size)
		self.position = (self.position + 1) % self.max_size
		return pop_episode

	def __len__(self):
		return self.size

	def is_full(self):
		return self.size == self.max_size

	def get_minibatch(self, step=None):
		if self.return_obs_next:
			last_pos = (self.position - 1) % self.max_size
			delete_inds = last_pos + np.arange(self.hist_len+1)
		else:
			delete_inds = self.position + np.arange(self.hist_len-1)
		batchable_inds = np.delete(np.arange(self.size), delete_inds)
		inds = self.rng.choice(batchable_inds, self.batch_size)
		batch = {'acts': self.actions[inds], 
				 'starts': self.starts[inds], 
				 'rewards': self.rewards[inds],
				 'terminals': self.terminals[inds],
				 'inds': inds}
		logging.debug("Getting frame histories")
		batch['obs'] = self.get_obs(inds)
		if self.return_obs_next:
			batch['obs_next'] = self.get_obs((inds + 1) % self.max_size)
		logging.debug("Finished creating minibatch.")
		return batch

	def get_obs(self, inds):
		if self.hist_len == 1:
			return self.screens[inds]
		else:
			return self._get_frame_histories(inds)	

	def check_is_reassigned(self, inds, new_states):
		if type(inds) is list:
			inds = np.array(inds)
		is_new = self.new[inds]
		is_terminal = self.terminals[inds]
		old_states = self.state_assignments[inds]
		return (1-is_new)*(1-is_terminal)*(inds < self.size)*(old_states != new_states)

	def get_window(self, ind):
		assert not self.terminals[ind]
		pre_ind = (ind - 1) % self.max_size
		post_ind = (ind + 1) % self.max_size
		state_window = self.state_assignments[[pre_ind, ind, post_ind]].tolist()
		acts = self.actions[[ind, post_ind]]
		rewards = self.rewards[[ind, post_ind]]
		if self.terminals[post_ind]:
			state_window[-1] = None
		elif self.starts[post_ind]:
			state_window[-1] = -1
		if self.starts[ind]:
			state_window[0] = -1
		return (state_window, acts, rewards)

	def get_updated_transitions(self, states):
		self.new[self.position] = 1
		batch_size = len(self.minibatch_inds)
		replay_inds = np.empty([2*batch_size], dtype=np.uint32)
		replay_inds[::2] = self.minibatch_inds
		replay_inds[1::2] = (self.minibatch_inds - 1) % self.max_size
		is_reassigned = self.check_is_reassigned(replay_inds, states)
		reassigned_states = np.argwhere(is_reassigned).flatten()
		updated_transitions = []
		for ind in reassigned_states:
			replay_ind = replay_inds[ind]
			new_state = states[ind]
			state_window, actions, rewards = self.get_window(replay_ind)
			updated_transitions.append((state_window, actions, rewards, new_state))
			self.state_assignments[replay_ind] = new_state
		self.new[:] = 0
		self.minibatch_inds = None
		num_reassigned =  len(reassigned_states)/float(2*batch_size)
		return updated_transitions, num_reassigned

	def _get_frame_histories(self, inds):
		self.minibatch_obs[:] = 0
		for mb_ind, ind in enumerate(inds):
			for frame, hist_ind in enumerate(xrange(ind, ind - self.hist_len, -1)):
				hist_ind %= self.max_size
				self.minibatch_obs[mb_ind, frame] = self.screens[hist_ind]
				if self.starts[hist_ind]:
					break
		return self.minibatch_obs.transpose((0, 2, 3, 1, 4)).reshape((self.batch_size,
		                                                              self.screens.shape[1],
		                                                              self.screens.shape[2],
		                                                              -1))

	def save_and_export(self, save_path, step, old_steps):
		if old_steps is not None:
			os.remove("%s/agent_replay.ckpt-%s.npz" % (save_path, old_steps))
		data = {'screens': self.screens, 'actions': self.actions,
				'rewards': self.rewards, 'starts': self.starts,
				'terminals': self.terminals, 
				'state_assignments': self.state_assignments}
		np.savez("%s/agent_replay.ckpt-%s.npz" % (save_path, step), **data)
		self.screens = None
		self.actions = None
		self.rewards = None
		self.starts = None
		self.terminals = None
		self.state_assignments = None
		return data

	def load_memory(self, data):
		self.screens = data['screens']
		self.actions = data['actions']
		self.rewards = data['rewards']
		self.starts = data['starts']
		self.terminals = data['terminals']
		self.state_assignments = data['state_assignments']		