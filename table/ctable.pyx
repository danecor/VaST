#cython: boundscheck=False, wraparound=False, initializedcheck=False, nonecheck=False, cdivision=True
# Cython-based implementation of prioritized sweeping with small backups (http://proceedings.mlr.press/v28/vanseijen13.pdf)
# Author: Dane Corneil
import cPickle as pickle
import h5table
import os
import time
from table_utils import dd2
import numpy as np
cimport numpy as np
from libc.math cimport isnan, isinf, INFINITY, NAN
from libc.stdint cimport uint32_t  # import the integer type from C
from cypridict.pridict import priority_dict

cdef class SweeperTable:

	cdef public np.float64_t[:,::1] rewards, qs
	cdef public np.float64_t[::1] vs, us
	cdef public np.uint32_t[:,::1] nsa
	cdef public list nsas, nsas_inv
	cdef public priorities
	cdef public unsigned int num_sweeps
	cdef unsigned int n_acts
	cdef unsigned int capacity
	cdef double discount
	cdef double pri_cutoff
	cdef str save_path

	def __init__(self, double discount, int capacity, int n_acts, double pri_cutoff,
				 str save_path):
		cdef np.float64_t[:,::1] qs = np.full([capacity, n_acts], np.nan, dtype=np.float64)
		cdef np.float64_t[:,::1] rewards = np.zeros([capacity, n_acts], dtype=np.float64)
		cdef np.float64_t[::1] vs = np.zeros(capacity, dtype=np.float64)
		cdef np.float64_t[::1] us = np.zeros(capacity, dtype=np.float64)
		cdef np.uint32_t[:,::1] nsa = np.zeros([capacity, n_acts], dtype=np.uint32)
		self.rewards = rewards
		self.vs = vs
		self.us = us
		self.qs = qs
		self.nsa = nsa
		self.discount = discount
		self.pri_cutoff = pri_cutoff
		self.capacity = capacity
		self.n_acts = n_acts
		self.save_path = save_path
		self.priorities = priority_dict()
		self.nsas = []
		self.nsas_inv = []
		#check = cymaxheap.MaxHeap()
		for i in range(self.n_acts):
			self.nsas.append(dd2())
			self.nsas_inv.append(dd2())
		self.num_sweeps = 0

	def save(self, steps, old_steps):
		data = {'qs': np.asarray(self.qs), 
				'nsa': np.asarray(self.nsa), 
				'rewards': np.asarray(self.rewards), 
				'vs': np.asarray(self.vs), 
				'us': np.asarray(self.us), 
				'priorities': dict(self.priorities)}
		with open("%s/table.ckpt-%s" % (self.save_path, steps), 'w') as handle:
			pickle.dump(data, handle)
		del data
		transition_path = "%s/transitions.ckpt-%s.h5" % (self.save_path, steps)
		h5table.save(transition_path, self.nsas)
		if old_steps is not None:
			os.remove("%s/table.ckpt-%s" % (self.save_path, old_steps))
			os.remove("%s/transitions.ckpt-%s.h5" % (self.save_path, old_steps))

	def restore(self, steps):
		table_path = "%s/table.ckpt-%s" % (self.save_path, steps)
		with open(table_path, 'r') as handle:
			data = pickle.load(handle)
		transition_path = "%s/transitions.ckpt-%s.h5" % (self.save_path, steps)
		nsas, nsas_inv = h5table.restore(transition_path, self.n_acts)
		old_capacity = data['rewards'].shape[0]
		cdef np.float64_t[:,::1] qs = data['qs']
		cdef np.float64_t[:,::1] rewards = data['rewards']
		cdef np.float64_t[::1] vs = data['vs']
		cdef np.float64_t[::1] us = data['us']
		cdef np.uint32_t[:,::1] nsa = data['nsa']
		self.rewards[:old_capacity] = rewards
		self.vs[:old_capacity] = vs
		self.us[:old_capacity] = us
		self.nsa[:old_capacity] = nsa
		self.qs[:old_capacity] = qs
		self.priorities = priority_dict(data['priorities'])
		self.nsas = nsas
		self.nsas_inv = nsas_inv

	def add(self, long ind, unsigned int act, np.float64_t reward, long ind_next):
		cdef np.float64_t sum_future_q, sum_rewards, qs
		self.nsas[act][ind][ind_next] += 1
		self.nsas_inv[act][ind_next][ind] = self.nsas[act][ind][ind_next]
		self.nsa[ind, act] += 1
		qs = self.qs[ind, act]
		if isnan(qs):
			qs = 0
		sum_future_q = (qs - self.rewards[ind, act])*(self.nsa[ind, act] - 1)
		if ind_next != -1:
			sum_future_q += self.discount*self.us[ind_next]
		sum_rewards = self.rewards[ind, act]*(self.nsa[ind, act] - 1) + reward
		self.qs[ind, act] = (sum_future_q + sum_rewards)/self.nsa[ind, act]
		self.rewards[ind, act] = sum_rewards/self.nsa[ind, act]
		self.update_tails(ind)

	def delete(self, long ind, unsigned int act, np.float64_t reward, long ind_next):
		cdef np.float64_t sum_future_q, sum_rewards
		self.nsas[act][ind][ind_next] -= 1
		self.nsas_inv[act][ind_next][ind] = self.nsas[act][ind][ind_next]
		self.nsa[ind, act] -= 1
		if self.nsa[ind, act] == 0:
			self.qs[ind, act] = NAN
			self.rewards[ind, act] = 0
		else:
			sum_future_q = (self.qs[ind, act] - self.rewards[ind, act])*(self.nsa[ind, act] + 1)
			if ind_next != -1:
				sum_future_q -= self.discount*self.us[ind_next]
			sum_rewards = self.rewards[ind, act]*(self.nsa[ind, act] + 1) - reward
			self.qs[ind, act] = (sum_future_q + sum_rewards)/self.nsa[ind, act]
			self.rewards[ind, act] = sum_rewards/self.nsa[ind, act]
		if self.nsas[act][ind][ind_next] == 0:
			del self.nsas_inv[act][ind_next][ind]
			del self.nsas[act][ind][ind_next]
		self.update_tails(ind)

	def priority_sweep(self):
		cdef long ind, pred_ind
		cdef np.float64_t delta_u, trans_prob
		cdef size_t act, trans_count
		ind = self.priorities.pop()
		if ind > -1:
			delta_u = self.vs[ind] - self.us[ind]
			self.us[ind] = self.vs[ind]
			for act in range(self.n_acts):
				for (pred_ind, trans_count) in self.nsas_inv[act][ind].iteritems():
					trans_prob = trans_count/(<np.float64_t>self.nsa[pred_ind, act])
					self.qs[pred_ind, act] += self.discount*trans_prob*delta_u
					self.update_tails(pred_ind)
			self.num_sweeps += 1

	cdef void update_tails(self, unsigned int ind):
		cdef np.float64_t[::1] qs = self.qs[ind]
		cdef np.float64_t new_val = nanmax(qs)
		if isinf(new_val):
			new_val = 0
		self.vs[ind] = new_val
		cdef np.float64_t priority = abs(self.us[ind] - self.vs[ind])
		if priority > self.pri_cutoff:
			self.priorities[ind] = priority

	def lookup(self, key_slice):
		nsa = np.asarray(self.nsa)[key_slice]
		qs = np.asarray(self.qs)[key_slice]
		return (nsa, qs)

	def parse_msg(self, unsigned int code, msg):
		if code==0:
			return self.lookup(msg)
		elif code==1:
			self.add(*msg)
			return False
		elif code==2:
			self.delete(*msg)
			return False
		elif code==3:
			return self.get_variables(msg)
		elif code==4:
			self.reset_summary_variables()
			return False
		elif code==5:
			self.save(*msg)
			return True
		elif code==6:
			self.restore(msg)
			return True
		elif code==7:
			self.reverse_rewards()
			return True
		return None

	cdef void reset_summary_variables(self):
		self.num_sweeps = 0

	def get_variables(self, tuple var_names):
		cdef list get_vars = []
		for ind in range(len(var_names)):
			var_name = var_names[ind]
			if var_name=='max_q':
				get_vars.append(nanmax2d(self.qs))
			elif var_name=='avg_q':
				get_vars.append(nanmean2d(self.qs))
			elif var_name=='table_size':
				get_vars.append(np.sum(np.asarray(self.nsa)>0))
			elif var_name=='priority_length':
				get_vars.append(len(self.priorities))
			elif var_name=='nsas':
				get_vars.append(self.nsas)
			elif var_name=='nsas_inv':
				get_vars.append(self.nsas_inv)
			elif var_name=='qs':
				get_vars.append(np.asarray(self.qs))
			elif var_name=='rewards':
				get_vars.append(np.asarray(self.rewards))
			elif var_name=='nsa':
				get_vars.append(np.asarray(self.nsa))
			elif var_name=='num_sweeps':
				get_vars.append(self.num_sweeps)
		return get_vars

	def reverse_rewards(self):
		cdef int ind, act
		for ind in range(self.rewards.shape[0]):
			for act in range(self.rewards.shape[1]):
				if self.rewards[ind, act] != 0:
					self.qs[ind, act] -= self.rewards[ind, act]
					self.rewards[ind, act] *= -1
					self.qs[ind, act] += self.rewards[ind, act]
					self.update_tails(ind)

cdef np.float64_t nanmax(np.float64_t[::1] vals):
	cdef np.float64_t maxval, val
	maxval = -INFINITY
	for ind in range(vals.shape[0]):
		val = vals[ind]
		if not isnan(val) and val > maxval:
			maxval = val
	return maxval

cdef np.float64_t nanmax2d(np.float64_t[:,::1] vals):
	cdef np.float64_t maxval, val
	maxval = -INFINITY
	for ind in range(vals.shape[0]):
		for ind2 in range(vals.shape[1]):
			val = vals[ind,ind2]
			if not isnan(val) and val > maxval:
				maxval = val
	return maxval

cdef np.float64_t nanmean2d(np.float64_t[:,::1] vals):
	cdef np.float64_t meanval, val, count
	meanval = 0.
	count = 0.
	for ind in range(vals.shape[0]):
		for ind2 in range(vals.shape[1]):
			val = vals[ind,ind2]
			if not isnan(val):
				meanval += val
				count += 1
	return meanval/count