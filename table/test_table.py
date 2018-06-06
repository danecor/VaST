import pytest
import os
import numpy as np
from lookup import Lookup
from h5table import restore
import cPickle as pickle
import bottleneck as bn
import time
import pyximport; pyximport.install(reload_support=True)

class TestTable(object):

	@pytest.fixture
	def lookup(self, tmpdir):
		lookup = Lookup(2, 3, 0.99, 4, 0.000001, str(tmpdir), None)
		yield lookup
		lookup.table.shutdown()

	@pytest.fixture
	def bigger_lookup(self, tmpdir):
		lookup = Lookup(2, 3, 0.99, 50, 0.000001, str(tmpdir), None)
		yield lookup
		lookup.table.shutdown()

	@pytest.fixture
	def big_lookup(self, tmpdir):
		lookup = Lookup(10, 3, 0.99, 10, 0.000001, str(tmpdir), None)
		yield lookup
		lookup.table.shutdown()

	@pytest.fixture
	def loaded_lookup_table(self, lookup):
		capacity = 4
		n_acts = lookup.n_acts
		states = np.arange(4)
		table_inds = [lookup.get_table_index(state) for state in states]
		experiences = [[[1],[1,3],[2,2]],
					   [[],[],[]],
					   [[],[2,3,3,1],[3]],
					   [[0,0,0,1,0],[],[]]]
		tmp_experiences = [[[3],[],[1,2,0]],
			   			   [[0],[2],[]],
			   			   [[1],[],[3]],
			   			   [[3],[],[1,2]]]
		pos_nsa = np.array([[len(experiences[i][a])>0 for a in range(3)] for i in range(4)])
		reward = pos_nsa*np.random.randn(capacity, n_acts)
		tmp_reward = np.random.randn(capacity, n_acts)
		for a in xrange(n_acts):
			for i in xrange(capacity):
				for outcome in experiences[i][a]:
					lookup.table.add(i, a, reward[i][a], outcome)
				for outcome in tmp_experiences[i][a]:
					lookup.table.add(i, a, tmp_reward[i][a], outcome)
		for a in xrange(n_acts):
			for i in xrange(capacity):
				for outcome in tmp_experiences[i][a]:
					lookup.table.delete(i, a, tmp_reward[i][a], outcome)
		lookup.table.save(1)
		return experiences, reward, lookup

	@pytest.fixture
	def episode(self):
		states = np.array([0, 1, 3, 2, 0])
		acts = np.array([0, 1, 2, 0])
		rewards = np.array([0, 0, 0, 1])
		return states, acts, rewards

	def test_update(self, loaded_lookup_table):
		experiences, true_reward, lookup = loaded_lookup_table
		capacity = lookup.table.capacity
		n_acts = lookup.n_acts
		nsas, nsas_inv, rewards = lookup.table.get_variables('nsas', 'nsas_inv', 'rewards')
		assert np.allclose(rewards, true_reward)
		for a in xrange(n_acts):
			for i in xrange(capacity):
				for j in xrange(capacity):
					num_nsa = sum(np.array(experiences[i][a]) == j)
					assert nsas[a][i][j] == num_nsa
					assert nsas_inv[a][j][i] == num_nsa

	def test_qs(self, loaded_lookup_table, epsilon=0.001):
		experiences, true_reward, lookup = loaded_lookup_table
		table = lookup.table
		capacity = lookup.table.capacity
		n_acts = lookup.n_acts
		probs = np.zeros((n_acts, capacity, capacity))
		for a in xrange(n_acts):
			for i in xrange(capacity):
				exp = np.array(experiences[i][a])
				if len(exp) > 0:
					for j in xrange(capacity):
						probs[a][i][j] = sum(exp == j)/float(len(exp))
		vals = value_iteration(probs, true_reward, experiences, epsilon=epsilon)
		priority_length, num_sweeps = table.get_variables('priority_length', 'num_sweeps')
		while (priority_length > 0) and (num_sweeps < 1000):
			time.sleep(0.01)
			priority_length, num_sweeps = table.get_variables('priority_length', 'num_sweeps')
		nsa, qs = table[:]
		assert np.all(abs(np.nan_to_num(bn.nanmax(qs, 1))-vals)<1.5*epsilon)
		table.reverse_rewards()
		vals = value_iteration(probs, -true_reward, experiences, epsilon=epsilon)
		priority_length, num_sweeps = table.get_variables('priority_length', 'num_sweeps')
		while (priority_length > 0):
			time.sleep(0.01)
			priority_length, num_sweeps = table.get_variables('priority_length', 'num_sweeps')
		nsa, qs = table[:]
		assert np.all(abs(np.nan_to_num(bn.nanmax(qs, 1))-vals)<1.5*epsilon)		

	def test_q_lookup(self, loaded_lookup_table, epsilon=0.001):
		experiences, true_reward, lookup = loaded_lookup_table
		priority_length, num_sweeps = lookup.table.get_variables('priority_length', 'num_sweeps')
		while (priority_length > 0) and (num_sweeps < 1000):
			time.sleep(0.01)
			priority_length, num_sweeps = lookup.table.get_variables('priority_length', 'num_sweeps')
		table_nsa, table_qs = lookup.table[:]
		lookup_qs = np.zeros((lookup.table.capacity, lookup.n_acts))
		for state in range(4):	
			lookup_qs[state] = lookup.get_qs(state)
		for state in range(4):
			nbs = lookup.get_table_hamming_neighbours(state, 1, None)
			weighted_neighbours = bn.nansum(table_qs[nbs]*table_nsa[nbs],0)
			for a in xrange(lookup.n_acts):
				if not np.isnan(table_qs[state, a]):
					assert np.isclose(lookup_qs[state][a], table_qs[state,a])
				else:
					mean_neighbours = weighted_neighbours[a]/np.sum(table_nsa[nbs, a], 0)
					assert np.isclose(lookup_qs[state][a], mean_neighbours)

	def test_add_episode_terminal(self, lookup, episode):
		states, acts, true_rewards = episode
		for i in range(len(acts)-1):
			lookup.add_transition(states[i], acts[i], true_rewards[i], states[i+1])
		lookup.add_transition(states[-2], acts[-1], true_rewards[-1], None)
		time.sleep(0.05)
		nsas, nsas_inv, rewards = lookup.table.get_variables('nsas', 'nsas_inv', 'rewards')
		for i, act in enumerate(acts):
			pre = lookup.get_table_index(states[i])
			post = lookup.get_table_index(states[i+1]) if (i < (len(acts)-1)) else -1
			assert nsas[act][pre][post] == 1
			assert nsas_inv[act][post][pre] == 1
			assert rewards[pre][act] == true_rewards[i]

	def test_add_episode_nonterminal(self, lookup, episode):
		states, acts, true_rewards = episode
		for i in range(len(acts)):
			lookup.add_transition(states[i], acts[i], true_rewards[i], states[i+1])
		time.sleep(0.05)
		nsas, nsas_inv, rewards = lookup.table.get_variables('nsas', 'nsas_inv', 'rewards')
		for i, act in enumerate(acts):
			pre = lookup.get_table_index(states[i])
			post = lookup.get_table_index(states[i+1])
			assert nsas[act][pre][post] == 1
			assert nsas_inv[act][post][pre] == 1
			assert rewards[pre][act] == true_rewards[i]

	def test_update_episode_orphan(self, bigger_lookup, episode):
		lookup = bigger_lookup
		states, acts, true_rewards = episode
		for i in range(len(acts)):
			lookup.add_transition(states[i], acts[i], true_rewards[i], states[i+1])
		lookup.update_transition([3,2,0], acts[2:], true_rewards[2:], 1)
		states[3] = 1
		time.sleep(0.05)
		assert len(lookup._state_to_table) == 4
		lookup.restart_table()
		assert len(lookup._state_to_table) == 3
		nsas, nsas_inv, rewards = lookup.table.get_variables('nsas', 'nsas_inv', 'rewards')
		for i, act in enumerate(acts):
			pre = lookup.get_table_index(states[i])
			post = lookup.get_table_index(states[i+1])
			assert nsas[act][pre][post] == 1
			assert nsas_inv[act][post][pre] == 1
			assert rewards[pre][act] == true_rewards[i]
		prev_state = 1
		for i in range(1000, 1100):
			lookup.update_transition([3,prev_state,0], acts[2:], 
			                         true_rewards[2:], i)
			prev_state = i
		states[3] = i
		for key in lookup._state_to_table.iterkeys():
			assert key == lookup._state_to_table.inverse[lookup._state_to_table[key]]
		lookup.restart_table()
		assert len(lookup._state_to_table) == 4
		nsas, nsas_inv, rewards = lookup.table.get_variables('nsas', 'nsas_inv', 'rewards')
		for i, act in enumerate(acts):
			pre = lookup.get_table_index(states[i])
			post = lookup.get_table_index(states[i+1])
			assert nsas[act][pre][post] == 1
			assert nsas_inv[act][post][pre] == 1
			assert rewards[pre][act] == true_rewards[i]

	def test_update_episode_middle(self, lookup, episode):
		states, acts, true_rewards = episode
		for i in range(len(acts)):
			pre_state = states[i] if states[i] != 3 else 2
			post_state = states[i+1] if states[i+1] != 3 else 2
			lookup.add_transition(pre_state, acts[i], true_rewards[i], post_state)
		lookup.update_transition([1,2,2], acts[1:3], true_rewards[1:3], 3)
		time.sleep(0.05)
		nsas, nsas_inv, rewards = lookup.table.get_variables('nsas', 'nsas_inv', 'rewards')
		for i, act in enumerate(acts):
			pre = lookup.get_table_index(states[i])
			post = lookup.get_table_index(states[i+1])
			assert nsas[act][pre][post] == 1
			assert nsas_inv[act][post][pre] == 1
			assert rewards[pre][act] == true_rewards[i]

	def test_update_episode_start(self, lookup, episode):
		states, acts, true_rewards = episode
		for i in range(len(acts)):
			pre_state = states[i] if (i > 0) else 2
			lookup.add_transition(pre_state, acts[i], true_rewards[i], states[i+1])
		lookup.update_transition([-1, 2, 1], [-1, acts[0]], [np.nan, true_rewards[0]], 0)
		time.sleep(0.02)
		nsas, nsas_inv, rewards = lookup.table.get_variables('nsas', 'nsas_inv', 'rewards')
		for i, act in enumerate(acts):
			pre = lookup.get_table_index(states[i])
			post = lookup.get_table_index(states[i+1])
			assert nsas[act][pre][post] == 1
			assert nsas_inv[act][post][pre] == 1
			assert rewards[pre][act] == true_rewards[i]

	def test_update_episode_terminal(self, lookup, episode):
		states, acts, true_rewards = episode
		for i in range(len(acts)-1):
			post_state = states[i+1] if (i < (len(acts)-2)) else 0
			lookup.add_transition(states[i], acts[i], true_rewards[i], post_state)
		lookup.add_transition(0, acts[-1], true_rewards[-1], None)
		lookup.update_transition([3, 0, None], acts[-2:], true_rewards[-2:], 2)
		time.sleep(0.02)
		nsas, nsas_inv, rewards = lookup.table.get_variables('nsas', 'nsas_inv', 'rewards')
		for i, act in enumerate(acts):
			pre = lookup.get_table_index(states[i])
			post = lookup.get_table_index(states[i+1]) if (i < (len(acts)-1)) else -1
			assert nsas[act][pre][post] == 1
			assert nsas_inv[act][post][pre] == 1
			assert rewards[pre][act] == true_rewards[i]

	def test_update_episode_nonterminal(self, lookup, episode):
		states, acts, true_rewards = episode
		for i in range(len(acts)):
			post_state = states[i+1] if (i < (len(acts)-1)) else 1
			lookup.add_transition(states[i], acts[i], true_rewards[i], post_state)
		lookup.update_transition([2, 1, -1], [acts[-1], -1], [true_rewards[-1], np.nan], 0)
		time.sleep(0.02)
		nsas, nsas_inv, rewards = lookup.table.get_variables('nsas', 'nsas_inv', 'rewards')
		for i, act in enumerate(acts):
			pre = lookup.get_table_index(states[i])
			post = lookup.get_table_index(states[i+1])
			assert nsas[act][pre][post] == 1
			assert nsas_inv[act][post][pre] == 1
			assert rewards[pre][act] == true_rewards[i]

	def test_save_restore_experience(self, loaded_lookup_table):
		experiences, true_reward, lookup = loaded_lookup_table
		table = lookup.table
		nsas, nsas_inv, rewards = table.get_variables('nsas', 'nsas_inv', 'rewards')
		nsas_save, nsas_inv_save = restore(table.save_path+'/transitions.ckpt-1.h5', 3)
		with open(table.save_path+'/table.ckpt-1', 'r') as handle:
			data = pickle.load(handle)
		rewards_save = data['rewards']
		assert np.allclose(rewards, rewards_save)
		for act in range(table.n_acts):
			for ind in xrange(table.capacity):
				assert nsas[act][ind] == nsas_save[act][ind]
				assert nsas_inv[act][ind] == nsas_inv_save[act][ind]

	def test_neighbours(self, big_lookup):
		index_vec = 2**np.arange(big_lookup.n_z)
		distances = None
		for dist in range(1, 10):
			key = np.random.randint(2, size=10)
			flips = np.random.choice(10, replace=False, size=dist)
			neighbour = np.copy(key)
			neighbour[flips] = 1 - neighbour[flips]
			state = np.dot(key, index_vec)
			neighbour_state = np.dot(neighbour, index_vec)
			neighbour_ind = big_lookup.get_table_index(neighbour_state)
			if dist >= 4:
				distances = big_lookup.get_table_hamming_distances(state)
			assert neighbour_ind in big_lookup.get_table_hamming_neighbours(state, dist, distances)

	def test_table_indexing_and_expand(self, big_lookup):
		length = 21
		unique_obs = []
		states = np.random.choice(2**10, length, replace=False)
		acts = np.random.randint(0, 3, length-1)
		true_rewards = np.random.randint(0, 1, length-1)
		for i in range(len(acts)):
			big_lookup.add_transition(states[i], acts[i], true_rewards[i], states[i+1])
		table_inds = [big_lookup.get_table_index(state) for state in states]
		assert np.all(table_inds == np.arange(length))
		nsas, nsas_inv, rewards = big_lookup.table.get_variables('nsas', 'nsas_inv', 'rewards')
		for i, act in enumerate(acts):
			pre = i
			post = i+1
			assert nsas[act][pre][post] == 1
			assert nsas_inv[act][post][pre] == 1
			assert rewards[pre][act] == true_rewards[pre]
		_, table_qs = big_lookup.table[:]
		for i, act in enumerate(acts):
			lookup_qs = big_lookup.get_qs(states[i])
			assert np.isclose(lookup_qs[act], table_qs[table_inds[i]][act])		

def value_iteration(T, R, E, discount=0.99, epsilon=0.001):
	"Solve an MDP by value iteration."
	n_act = T.shape[0]
	n_states = T.shape[1]
	U1 = np.zeros(n_states)
	while True:
		U = np.copy(U1)
		delta = 0
		for s in xrange(n_states):
			try:
				U1[s] = max([R[s,a] + discount*sum([p*U[s1] for (s1, p) in enumerate(T[a, s])])
							for a in xrange(n_act) if E[s][a]])
			except ValueError:
				U1[s] = 0
			delta = max(delta, abs(U1[s] - U[s]))
		if delta < epsilon * (1 - discount) / discount:
			return U