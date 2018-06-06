import logging
import os
from multiprocessing import Process, Pipe
import numpy as np
import cPickle as pickle
import h5table
from table_utils import ExceptionWrapper
import pyximport; pyximport.install(setup_args={"include_dirs":np.get_include()},
                                    reload_support=True)
from ctable import SweeperTable

class TableProcess(Process):

	def __init__(self, n_acts, discount, capacity, pri_cutoff, save_path):
		self.n_acts = n_acts
		self.discount = discount
		self.capacity = capacity
		self.pri_cutoff = pri_cutoff
		self.save_path = save_path
		self.killed = False
		self.lookup_conn, self.table_conn = Pipe()
		super(TableProcess, self).__init__(name='TableProcess')

	def start(self):
		self.daemon = True
		super(TableProcess, self).start()

	def run(self):
		try:
			logging.info("Starting sweeper table")
			self.ctable = SweeperTable(self.discount, self.capacity, self.n_acts, 
			         			 	   self.pri_cutoff, self.save_path)
			self._loop()
			logging.info("Exiting sweeper table.")
		except Exception as e:
			self.table_conn.send(ExceptionWrapper(e))
			raise e

	def _loop(self):
		while not self.killed:
			#Check for q requests, experience updates or save requests
			self._empty_pipe()
			#Do a priority sweep
			self.ctable.priority_sweep()

	def __getitem__(self, key_slice):
		self.lookup_conn.send((0, key_slice))
		return self.get_pipe_output()

	def add(self, ind, act, reward, ind_next):
		self.lookup_conn.send((1, (ind, act, reward, ind_next)))
		if self.lookup_conn.poll():
			return self.get_pipe_output()

	def delete(self, ind, act, reward, ind_next):
		self.lookup_conn.send((2, (ind, act, reward, ind_next)))
		if self.lookup_conn.poll():
			return self.get_pipe_output()

	def get_variables(self, *var_names):
		self.lookup_conn.send((3, var_names))
		return self.get_pipe_output()

	def reset_summary_variables(self):
		self.lookup_conn.send((4, None))
		if self.lookup_conn.poll():
			return self.get_pipe_output()

	def save(self, filename, old_filename=None):
		self.lookup_conn.send((5, (filename, old_filename)))
		return self.get_pipe_output()

	def restore(self, filename):
		self.lookup_conn.send((6, filename))
		return self.get_pipe_output()

	def clean_orphaned_states(self, filename, max_state):
		logging.info("Determining orphan states.")
		table_path = "%s/table.ckpt-%s" % (self.save_path, filename)
		with open(table_path, 'r') as handle:
			data = pickle.load(handle)
		transition_path = "%s/transitions.ckpt-%s.h5" % (self.save_path, filename)
		nsas, nsas_inv = h5table.restore(transition_path, self.n_acts)
		qs = data['qs']
		rewards = data['rewards']
		vs = data['vs']
		us = data['us']
		nsa = data['nsa']
		priorities = data['priorities']
		orphan_candidates = np.argwhere(nsa[:max_state].sum(axis=1)==0)[:, 0]
		orphans = []
		for orphan_candidate in orphan_candidates:
			orphaned = True
			for act in range(self.n_acts):
				if len(nsas_inv[act][orphan_candidate]) > 0:
					orphaned = False
					break
			if orphaned:
				orphans.append(orphan_candidate)
				qs[orphan_candidate] = np.nan
				rewards[orphan_candidate] = 0
				vs[orphan_candidate] = 0
				us[orphan_candidate] = 0
				priorities.pop(orphan_candidate, None)
		data = {'qs': qs, 'nsa': nsa, 'rewards': rewards, 
				'vs': vs, 'us': us, 'priorities': priorities}
		with open("%s/table.ckpt-%s" % (self.save_path, filename), 'w') as handle:
			pickle.dump(data, handle)
		return np.array(orphans)

	def reverse_rewards(self):
		self.lookup_conn.send((7, None))
		return self.get_pipe_output()

	def kill(self):
		self.lookup_conn.send((8, None))
		return self.get_pipe_output()

	def shutdown(self):
		if self.is_alive():
			self.kill()
			self.join()

	def get_pipe_output(self):
		msg = self.lookup_conn.recv()
		if isinstance(msg, ExceptionWrapper):
			logging.exception("Exception in TableProcess")
			msg.re_raise()
		return msg

	def _empty_pipe(self):
		while self.table_conn.poll():
			code, msg = self.table_conn.recv()
			result = self.ctable.parse_msg(code, msg)
			if result is None:
				self.killed = True
				result = True
			if result:
				self.table_conn.send(result)