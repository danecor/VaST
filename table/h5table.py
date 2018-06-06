from tables import *
from table_utils import dd2
from scipy.sparse import dok_matrix, csr_matrix
import numpy as np

class Transition(IsDescription):
	pre  = Int32Col()
	post  = Int32Col()
	count = UInt32Col()

def save(file_path, nsas):
	h5file = open_file(file_path, mode = "w", title = "Transition Table")
	group = h5file.create_group("/", 'transition_table', 'Transitions')
	for a in xrange(len(nsas)):
		table = h5file.create_table(group, 'action%i' % a, Transition, "action", expectedrows=len(nsas[a])*10)
		transition = table.row
		for pre in nsas[a].iterkeys():
			for post, count in nsas[a][pre].iteritems():
				transition['pre']  = pre
				transition['post']  = post
				transition['count'] = count
				transition.append()
		table.flush()
	h5file.close()

def restore(file_path, n_acts):
	h5file = open_file(file_path, "r")
	act_tables = h5file.root.transition_table
	nsas = []
	nsas_inv = []
	for a in range(n_acts):
		nsas.append(dd2())
		nsas_inv.append(dd2())
		act_table = act_tables._v_children['action%i' % a]
		for trans in act_table.iterrows():
			pre = trans['pre']
			post = trans['post']
			count = trans['count']
			nsas[a][pre][post] = count
			nsas_inv[a][post][pre] = count
	h5file.close()
	return nsas, nsas_inv

def restore_to_matrix(file_path, n_acts, count_lim=1):
	h5file = open_file(file_path, "r")
	act_tables = h5file.root.transition_table
	nsas = {}
	for a in range(n_acts):
		act_table = act_tables._v_children['action%i' % a]
		for trans in act_table.iterrows():
			pre = trans['pre']
			post = trans['post']
			count = trans['count']
			if post == -1:
				continue
			if nsas.get(pre) is None:
				nsas[pre] = {}
				nsas[pre][post] = count
			else:
				if nsas[pre].get(post) is None:
					nsas[pre][post] = count
				else:
					nsas[pre][post] += count
			if nsas.get(post) is None:
				nsas[post] = {}
				nsas[post][pre] = count
			else:
				if nsas[post].get(pre) is None:
					nsas[post][pre] = count
				else:
					nsas[post][pre] += count
	h5file.close()
	ordered_state_lookup = {}
	ordered_count = 0
	ordered_nsas = {}
	for pre in nsas.keys():
		vals = [val for val in nsas[pre].values() if val >= count_lim]
		denom = float(np.sum(vals))
		for post in nsas[pre].keys():
			assert nsas[pre][post] == nsas[post][pre]
			if nsas[pre][post] >= count_lim:
				ordered_pre = ordered_state_lookup.get(pre)
				if ordered_pre is None:
					ordered_state_lookup[pre] = ordered_count
					ordered_pre = ordered_state_lookup[pre]
					ordered_count += 1
				ordered_post = ordered_state_lookup.get(post)
				if ordered_post is None:
					ordered_state_lookup[post] = ordered_count
					ordered_post = ordered_state_lookup[post]
					ordered_count += 1
				if ordered_nsas.get(ordered_pre) is None:
					ordered_nsas[ordered_pre] = {}
				ordered_nsas[ordered_pre][ordered_post] = nsas[pre][post]/denom
	mat = dok_matrix((len(ordered_nsas),len(ordered_nsas)), dtype=np.float32)
	for pre in ordered_nsas.keys():
		for post in ordered_nsas[pre].keys():
			mat[pre, post] = ordered_nsas[pre][post]
	return mat.tocsr()