import pytest
import numpy as np
from replay_memory import ReplayMemory

class TestReplayMemory(object):

	@pytest.fixture
	def loaded_replay(self):
		loaded_replay = ReplayMemory(30, [1, 1, 1], 5, 200, np.random.RandomState(123))
		state = 1
		loaded_replay.append(0, np.nan, False, np.array([[[state]]]), state, True)
		start = False
		terminal = False
		for i in range(2, 40):
			state += 1
			if i in (12, 20, 30, 35):
				terminal = True
			loaded_replay.append(state, state, terminal, np.array([[[state]]]), state, state==1)
			if terminal:
				start = True
				state = 0
			terminal = False
		return loaded_replay

	@pytest.fixture
	def empty_replay(self):
		return ReplayMemory(30, [1, 1, 1], 5, 200, np.random.RandomState(456))

	def test_append(self, loaded_replay, empty_replay):
		assert loaded_replay.is_full()
		assert np.all(np.diff(loaded_replay.screens.flatten())[1-loaded_replay.terminals] == 1)
		popped_episode = loaded_replay.append(100, 100, False, np.array([[[100]]]), 100, False)
		assert popped_episode[1] == popped_episode[2] == popped_episode[3]
		assert popped_episode[0] == popped_episode[1]-1
		popped_episode = loaded_replay.append(110, 110, False, np.array([[[110]]]), 110, False)
		assert popped_episode[3] is None
		popped_episode = loaded_replay.append(120, 120, False, np.array([[[120]]]), 120, False)
		assert popped_episode is None
		popped_episode = empty_replay.append(0, np.nan, False, np.array([[[1]]]), 1, True)
		assert not empty_replay.is_full()
		assert popped_episode is None

	def test_get_minibatch(self, loaded_replay, empty_replay):
		minibatch = loaded_replay.get_minibatch()
		for i in range(loaded_replay.position, loaded_replay.position+3):
			assert not i in minibatch['inds']
		for i in range(50):
			ind = minibatch['inds'][i]
			obs = minibatch['obs'][i]
			act = minibatch['acts'][i]
			assert obs[0][0][0]==loaded_replay.screens[ind][0][0][0]
			#Check that frames match up
			assert np.all(obs[0][0][0]==act)
			flat_obs = obs.flatten()[::-1]
			frame_diffs = np.diff(flat_obs)
			#Check that observations are ordered sequentially
			assert np.all((frame_diffs==1)[flat_obs[1:]>0])
			#Unless they wrap around an episode boundary (and therefore correspond to 0)
			assert np.all((frame_diffs==0)[flat_obs[1:]==0])
		empty_replay.append(0, np.nan, False, np.array([[[1]]]), 1, True)
		minibatch = empty_replay.get_minibatch()
		assert np.all(minibatch['inds'] == 0)

	def test_is_reassigned(self, loaded_replay):
		check_reassigned_new = loaded_replay.check_is_reassigned(range(len(loaded_replay)), 
		                                                         loaded_replay.state_assignments)
		#New transitions don't count as changed
		assert np.all(check_reassigned_new==0)
		loaded_replay.new[:] = 0
		check_reassigned_same = loaded_replay.check_is_reassigned(range(len(loaded_replay)), 
		                                                          loaded_replay.state_assignments)
		assert np.all(check_reassigned_same==0)
		check_reassigned_diff = loaded_replay.check_is_reassigned(range(len(loaded_replay)), 
		                                                          loaded_replay.state_assignments+1)
		#All transitions have either changed or are terminal (not counted)
		assert np.all((check_reassigned_diff + loaded_replay.terminals) == 1)

	def test_get_window(self, loaded_replay, empty_replay):
		for ind in range(len(loaded_replay)):
			if loaded_replay.terminals[ind]:
				with pytest.raises(Exception):
					state_window, acts, rewards = loaded_replay.get_window(ind)
				continue
			state_window, acts, rewards = loaded_replay.get_window(ind)
			if loaded_replay.starts[ind]:
				assert state_window[0] == -1
			else:
				assert state_window[0] == loaded_replay.state_assignments[(ind - 1) % 30]
			if loaded_replay.terminals[(ind + 1) % 30]:
				assert state_window[-1] == None
			else:
				assert state_window[-1] == loaded_replay.state_assignments[(ind + 1) % 30]
			assert state_window[1] == loaded_replay.state_assignments[ind]
			assert acts[0] == loaded_replay.actions[ind]
			assert rewards[0] == loaded_replay.rewards[ind]
			assert acts[1] == loaded_replay.actions[(ind + 1) % 30]
			assert rewards[1] == loaded_replay.rewards[(ind + 1) % 30]
		loaded_replay.append(0, np.nan, False, np.array([[[100]]]), 100, True)
		end_ind = (loaded_replay.position - 2) % loaded_replay.max_size
		state_window, acts, rewards = loaded_replay.get_window(end_ind)
		assert state_window[-1] == -1
		assert np.isnan(rewards[-1])
		start_ind = (loaded_replay.position - 1) % loaded_replay.max_size
		state_window, _, rewards = loaded_replay.get_window(start_ind)
		assert state_window[0] == -1
		assert np.isnan(rewards[0])
		empty_replay.append(0, np.nan, False, np.array([[[1]]]), 1, True)
		empty_replay.append(1, 1, False, np.array([[[2]]]), 2, False)
		state_window, _, _ = empty_replay.get_window(1)
		assert state_window[-1] == 0