import numpy as np
import logging
import os
from table_utils import increment_dict
import bottleneck as bn
from table import TableProcess
import pyximport; pyximport.install()
import cPickle as pickle
from hamming import hamming_dist, hamming_neighbours

class Lookup(object):

	def __init__(self, n_z, n_acts, discount, init_capacity, pri_cutoff, save_path, rng):
		self.n_acts = n_acts
		self.n_z = n_z
		self.discount = discount
		self.init_capacity = init_capacity
		self.capacity = init_capacity
		self.pri_cutoff = pri_cutoff
		self.q_searches = 0
		self.dists = 0.
		self.values = 0.
		self.rng = rng or np.random.RandomState()
		self._state_to_table = increment_dict()
		self.save_path = save_path
		self.make_table()

	def make_table(self, data=None):
		self.table = TableProcess(self.n_acts, self.discount, 
		                          self.capacity, self.pri_cutoff, 
		                          self.save_path)
		self.table.start()
		if data is not None:
			logging.info("Restoring table data.")
			self.table.restore(data)

	def expand_table_capacity(self):
		self.capacity += int(0.5*self.init_capacity)
		logging.info("Expanding table capacity to %i" % self.capacity)
		self.restart_table()

	def restart_table(self):
		logging.info("Restarting table")
		self.table.save('tmp')
		self.table.shutdown()
		if len(self._state_to_table.orphans) == 0:
			orphans = self.table.clean_orphaned_states('tmp', self._state_to_table.max_fill)
			logging.info("Updating orphans.")
			self._state_to_table.update_orphans(orphans)
			logging.info("Finished updating orphans.")
		self.make_table('tmp')
		os.remove("%s/table.ckpt-tmp" % self.save_path)
		os.remove("%s/transitions.ckpt-tmp.h5" % self.save_path)

	def add_transition(self, state, action, reward, next_state):
		pre = self.get_table_index(state)
		post = self.get_table_index(next_state)
		self.table.add(pre, action, reward, post)

	def delete_transition(self, state, action, reward, next_state):
		pre = self.get_table_index(state)
		post = self.get_table_index(next_state)
		self.table.delete(pre, action, reward, post)

	def update_transition(self, state_window, actions, rewards, new_state):
		if state_window[0] != -1:
			self.delete_transition(state_window[0], actions[0], rewards[0], state_window[1])
			self.add_transition(state_window[0], actions[0], rewards[0], new_state)
		if state_window[-1] != -1:
			self.delete_transition(state_window[1], actions[1], rewards[1], state_window[2])
			self.add_transition(new_state, actions[1], rewards[1], state_window[2])

	def get_table_index(self, state):
		if state is None:
			return -1
		table_ind = self._state_to_table[int(state)]
		if self._state_to_table.is_full(self.table.capacity):
			self.expand_table_capacity()
		return table_ind

	def estimate_max_action(self, state):
		qs = self.get_qs(state)
		max_action = np.argmax(qs)
		self.values += qs[max_action]
		return max_action

	def get_qs(self, state):
		logging.debug("Getting q-values for state %s" % str(state))
		self.q_searches += 1
		state = int(state)
		dist = 0		
		distances = None
		table_ind = self._state_to_table.get(state)
		nsa = np.zeros(self.n_acts)
		sum_q = np.zeros(self.n_acts)
		if table_ind is not None:
			ind_nsa, ind_qs = self.table[table_ind]
			nsa += ind_nsa
			sum_q += np.nan_to_num(ind_qs)*nsa
		missing_qs = nsa==0
		while np.any(missing_qs):
			dist += 1
			if dist == 5:
				distances = self.get_table_hamming_distances(state)
			if dist > self.n_z:
				self.dists += dist
				return np.zeros(self.n_acts)
			neighbours = self.get_table_hamming_neighbours(state, dist, distances)
			neighbours_nsa, neighbours_qs = self.table[neighbours]
			nsa[missing_qs] += np.sum(neighbours_nsa, 0)[missing_qs]
			sum_q[missing_qs] += (bn.nansum(neighbours_nsa*neighbours_qs, 0))[missing_qs]
			missing_qs = nsa==0
		self.dists += dist
		return sum_q/nsa

	def get_table_hamming_distances(self, state):
		logging.debug("Getting all distances for state %s" % str(state))
		return dict([(val, hamming_dist(state, key)) for (key, val) in self._state_to_table.iteritems()])

	def get_table_hamming_neighbours(self, state, dist, distances):
		logging.debug("Getting neighbours for state %s with dist %i" % (str(state), dist))
		if distances is None:
			all_neighbours = hamming_neighbours(state, dist, self.n_z)
			table_neighbours = [self._state_to_table.get(nind) for nind in all_neighbours]
			return [neighbour for neighbour in table_neighbours if neighbour is not None]
		return [key for (key, val) in distances.iteritems() if val==dist]

	def get_summary_variables(self):
		query_returns = self.table.get_variables('max_q', 'avg_q', 'table_size', 
		                                         'num_sweeps', 'priority_length')
		max_q, avg_q, table_size, num_sweeps, priority_size = query_returns
		lookup_distance = self.dists/float(max(self.q_searches, 1))
		lookup_values = self.values/float(max(self.q_searches, 1))
		return max_q, avg_q, table_size, num_sweeps, priority_size, lookup_distance, lookup_values

	def reset_summary_variables(self):
		self.q_searches = 0
		self.dists = 0.
		self.values = 0.
		self.table.reset_summary_variables()

	def save(self, step, old_steps):
		if old_steps is not None:
			os.remove("%s/state_dict-%s" % (self.save_path, old_steps))
			os.remove("%s/lookup-%s" % (self.save_path, old_steps))
		table = self.table
		inc_dict = self._state_to_table
		self.table = None
		self._state_to_table = None
		inc_dict.save("%s/state_dict-%s" % (self.save_path, step))
		with open("%s/lookup-%s" % (self.save_path, step), "wb") as handle:
			pickle.dump(self, handle, pickle.HIGHEST_PROTOCOL)
		self.table = table
		self._state_to_table = inc_dict
		self.table.save(step, old_steps)

	def restore_dict(self, dict_path):
		self._state_to_table = increment_dict(path=dict_path)