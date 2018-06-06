import math
import numpy as np 
import tensorflow as tf
import threading
import logging

def lrelu(input_tensor, reuse):
	act_func = tf.contrib.keras.layers.LeakyReLU(alpha=0.001)
	return act_func(input_tensor)

def pelu(x, reuse):
	"""Parametric Exponential Linear Unit (https://arxiv.org/abs/1605.09332v1)."""
	with tf.variable_scope('activation', reuse=reuse):
		alpha = tf.get_variable('alpha', (), initializer=tf.constant_initializer(0.2))
		beta = tf.get_variable('beta', (), initializer=tf.constant_initializer(0.2))
		if not reuse:
			tf.summary.scalar('alpha', alpha)
			tf.summary.scalar('beta', beta)
		alpha_clip = tf.maximum(alpha,0)
		beta_clip = tf.maximum(beta,0)
		positive = tf.nn.relu(x) * alpha_clip / (beta_clip + 1e-9)
		negative = alpha_clip * (tf.exp((-tf.nn.relu(-x)) / (beta_clip + 1e-9)) - 1)
		return negative + positive

def relu(x, reuse):
	return tf.nn.relu(x)

act_funcs = {'lrelu':lrelu, 'pelu':pelu, 'relu':relu}

def sample_logistic(shape, eps=1e-20): 
  """Sample from Logistic Variable (Difference of Gumbels)"""
  U = tf.random_uniform(shape,minval=0,maxval=1,seed=1)
  return tf.log(U + eps) - tf.log(1 - U + eps)

def sample_gumbel(shape, eps=1e-20): 
  """Sample from Gumbel(0, 1)"""
  U = tf.random_uniform(shape,minval=0,maxval=1,seed=1)
  return -tf.log(-tf.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature): 
  """ Draw a sample from the Gumbel-Softmax distribution"""
  y = logits + sample_gumbel(tf.shape(logits))
  return tf.nn.softmax( y / temperature)

def gumbel_softmax(logits, temperature, hard=False):
  """Sample from the Gumbel-Softmax distribution and optionally discretize.
  Args:
	logits: [batch_size, n_class] unnormalized log-probs
	temperature: non-negative scalar
	hard: if True, take argmax, but differentiate w.r.t. soft sample y
  Returns:
	[batch_size, n_class] sample from the Gumbel-Softmax distribution.
	If hard=True, then the returned sample will be one-hot, otherwise it will
	be a probabilitiy distribution that sums to 1 across classes
  """
  y = gumbel_softmax_sample(logits, tf.maximum(temperature, 1e-20))
  if hard:
	k = tf.shape(logits)[-1]
	#y_hard = tf.cast(tf.one_hot(tf.argmax(y,1),k), y.dtype)
	y_hard = tf.cast(tf.equal(y,tf.reduce_max(y,1,keep_dims=True)),y.dtype)
	y = tf.stop_gradient(y_hard - y) + y
  return y

def variable_summaries(var):
	"""Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
	with tf.name_scope('summaries'):
	  mean = tf.reduce_mean(var)
	  tf.summary.scalar('mean', mean)
	  with tf.name_scope('stddev'):
		stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
	  tf.summary.scalar('stddev', stddev)
	  tf.summary.scalar('max', tf.reduce_max(var))
	  tf.summary.scalar('min', tf.reduce_min(var))
	  tf.summary.histogram('histogram', var)

def normalize(vector, epsilon=1e-8):
	vector_norm = tf.norm(vector,axis=-1)
	vector_norm += epsilon
	return tf.div(vector,vector_norm[:, None])

def linear(x, output_dim, reuse=False, do_batch_norm=False, 
		   next_linear=True, regularizer=None, init=None):
	input_dim = x.get_shape()[1]
	with tf.variable_scope('linear', reuse=reuse):
		w = tf.get_variable('w', [input_dim, output_dim], 
							initializer=init or tf.contrib.layers.xavier_initializer(seed=2),
							regularizer=regularizer)
		if do_batch_norm:
		  return batch_norm(tf.matmul(x, w), next_linear, reuse)
		else:
		  b = tf.get_variable('b', [output_dim], initializer=tf.constant_initializer(0.0))
		  return tf.matmul(x, w) + b

def conv_lin(x, output_dim, filt_side, stride=2, reuse=False, do_batch_norm=False, next_linear=True):
	input_dim = x.get_shape()[-1]
	with tf.variable_scope('conv_lin', reuse=reuse):
		w = tf.get_variable('w', [filt_side, filt_side, input_dim, output_dim], 
							initializer=tf.contrib.layers.xavier_initializer_conv2d(seed=3))
		if do_batch_norm:
		  return batch_norm(conv(x, w, stride), next_linear, reuse)
		else:
		  b = tf.get_variable('b', [output_dim], initializer=tf.constant_initializer(0.0))
		  return conv(x, w, stride, b)

def deconv_lin(x, output_shape, filt_side, stride=2, reuse=False, do_batch_norm=False, next_linear=True):
	input_dim = x.get_shape()[-1]
	output_dim = output_shape[-1]
	with tf.variable_scope('deconv_lin', reuse=reuse):
		w = tf.get_variable('w', [filt_side, filt_side, output_dim, input_dim], 
							initializer=tf.contrib.layers.xavier_initializer_conv2d(seed=4))
		b = tf.get_variable('b', output_shape[1:], initializer=tf.constant_initializer(0.0))
		if do_batch_norm:
		  deconv_out = tf.reshape(deconv(x, w, output_shape, stride), output_shape)
		  return batch_norm(deconv_out, next_linear, reuse)
		else:
		  return deconv(x, w, output_shape, stride, b)

def conv(x, weights, stride, biases=None):
	conv = tf.nn.conv2d(x, weights, strides=[1,stride,stride,1], padding='SAME')
	if biases is not None:
	  conv = conv + biases
	return conv

def deconv(x, weights, out_shape, stride, biases=None):
	conv = tf.nn.conv2d_transpose(x, weights, out_shape, strides=[1,stride,stride,1], padding='SAME')
	if biases is not None:
	  conv = conv + biases
	return conv

def batch_norm(x, scale=True, reuse=True):
	scope = tf.contrib.framework.get_name_scope()
	return tf.contrib.layers.batch_norm(x,
					  decay=0.9, 
					  updates_collections=None,
					  epsilon=1e-5,
					  scale=scale,
					  scope=scope,
					  reuse=reuse,
					  is_training=True)

def dropout(x):
	return tf.nn.dropout(x, keep_prob=0.8)

def chunks(l, n):
	"""Yield successive n-sized chunks from l."""
	for i in xrange(0, len(l), n):
		yield l[i:i + n]

def make_compare_plot(obs, gen_obs, channels=None):
	if channels == 2:
		#Filler to plot image in red and green
		filler = tf.zeros([tf.shape(obs)[0], tf.shape(obs)[1], tf.shape(obs)[2], 1])
		obs = tf.concat([obs, filler], 3)
		gen_obs = tf.concat([gen_obs, filler], 3)
	elif channels > 3:
		hist_obs = [obs[:, :, :, i] for i in range(channels)]
		hist_obs = tf.concat(hist_obs, axis=1)
		hist_gen_obs = [gen_obs[:, :, :, i] for i in range(channels)]
		hist_gen_obs = tf.concat(hist_gen_obs, axis=1)
		return tf.concat([hist_obs, hist_gen_obs], axis=2)[:, :, :, None]
	return tf.concat([obs, gen_obs], axis=2)

class SummaryWriter(tf.summary.FileWriter):

	def __init__(self, path, graph):
		super(SummaryWriter, self).__init__(path+"/", graph)
		self.sess = tf.Session()
		self.summaries = {}
		self.write_lock = threading.Lock()

	def add_summary(self, summary, steps, run_metadata=None):
		with self.write_lock:
			super(SummaryWriter, self).add_summary(summary, steps)
			if run_metadata is not None:
				super(SummaryWriter, self).add_run_metadata(run_metadata, 'step%d' % steps)

	def value_summary(self, value, name):
		return tf.Summary(value=[tf.Summary.Value(tag=name, simple_value=value)])
	
	def summarize_model(self, model_summary, steps, run_metadata=None):
		if model_summary is not None:
			self.add_summary(model_summary, steps, run_metadata)
	
	def summarize_agent(self, agent_summary, steps, episode_update=False, keys=None):
		if agent_summary is not None:
			for key, value in agent_summary.iteritems():
				if 'eval_' in key:
					continue
				if (keys is not None) and (key not in keys):
					continue
				if episode_update and 'episode' in key:
					self.add_summary(self.value_summary(value, key), steps)
				elif not episode_update and not 'episode' in key:
					self.add_summary(self.value_summary(value, key), steps)

	def summarize_images(self, figures, steps):
		for key, fig in figures.iteritems():
			logging.debug("Getting trajectory image.")
			image = self.fig2rgb_array(fig)
			logging.debug("Getting trajectory image summary.")
			try:
				(placeholder, summary) = self.summaries[key]
			except KeyError:
				placeholder = tf.placeholder(tf.uint8, image.shape)
				summary = tf.summary.image(key, placeholder, max_outputs=1)
				self.summaries[key] = (placeholder, summary)
			logging.debug("Writing to trajectory image summary.")
			try:
  				self.add_summary(summary.eval(feed_dict={placeholder: image},
  				                              session=self.sess), steps)
  			except Exception as e:
  				logging.info("Exception making summary: %s" % str(e))

	def fig2rgb_array(self, fig, expand=True):
		logging.debug("Drawing canvas.")
  		fig.canvas.draw()
  		logging.debug("Converting to RGB string.")
  		buf = fig.canvas.tostring_rgb()
  		logging.debug("Getting shape.")
  		ncols, nrows = fig.canvas.get_width_height()
  		shape = (nrows, ncols, 3) if not expand else (1, nrows, ncols, 3)
  		logging.debug("Reading string buffer into numpy.")
  		return np.fromstring(buf, dtype=np.uint8).reshape(shape)
