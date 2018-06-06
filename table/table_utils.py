import sys
import numpy as np
import logging
from collections import defaultdict, deque
from operator import itemgetter
import cPickle as pickle
import tblib.pickling_support
tblib.pickling_support.install()

class increment_dict(dict):

	def __init__(self, *args, **kwargs):
		restore = kwargs.get('path', None)
		if restore is not None:
			data = pickle.load(open(restore, 'rb'))
			super(increment_dict, self).__init__(data['dict'])
			self.orphans = data['orphans']
			self.max_fill = data['max_fill']
			self.inverse = data['inverse']				
		else:
			self.orphans = kwargs.get('orphans', deque())
			self.max_fill = 0
			self.inverse = {}
			super(increment_dict, self).__init__(*args, **kwargs)
			for key, value in self.iteritems():
				self.inverse[value] = key

	def __missing__(self, key):
		if len(self.orphans) == 0:
			self[key] = len(self)
			self.max_fill = max(self.max_fill, len(self))
		else:
			self[key] = self.orphans.pop()
		return self[key]

	def __setitem__(self, key, value):
		if key in self:
			del self.inverse[self[key]]
		super(increment_dict, self).__setitem__(key, value)
		try:
			self.inverse[value] = key
		except AttributeError:
			self.inverse = {}
			self.inverse[value] = key

	def __delitem__(self, key):
		del self.inverse[self[key]]
		super(increment_dict, self).__delitem__(key)

	def reverse_delete(self, key):
		del self[self.inverse[key]]

	def update_orphans(self, new_orphans):
		if len(new_orphans) > 0:
			delete_states = itemgetter(*new_orphans)(self.inverse)
			try:
				map(self.__delitem__, delete_states)
			except TypeError:
				map(self.__delitem__, [delete_states])
			self.orphans = new_orphans.tolist()

	def is_full(self, capacity):
		return (len(self.orphans) == 0) and (len(self) > capacity)

	def save(self, path):
		data = {'dict': dict(self), 'inverse': self.inverse,
				'max_fill': self.max_fill, 'orphans': self.orphans}
		with open(path, "wb") as handle:
			pickle.dump(data, handle, pickle.HIGHEST_PROTOCOL)

class dd(dict):

	def __setitem__(self, key, val):
		super(dd, self).__setitem__(key, np.uint32(val))

	def __missing__(self, key):
		self[key] = 0
		return self[key]

class dd2(dict):

	def __missing__(self, key):
		self[key] = dd()
		return self[key]

class ExceptionWrapper(object):

	def __init__(self, ee):
		self.ee = ee
		__,  __, self.tb = sys.exc_info()

	def re_raise(self):
		raise self.ee, None, self.tb