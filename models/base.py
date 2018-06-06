import tensorflow as tf
from tensorflow.python import debug as tf_debug
from ops import act_funcs, conv_lin, linear
import logging

class BaseModel(object):

	def __init__(self, params, save_path, restore_step=None, debug=False, restore_path=None):
		self.net_arch = params['net_arch']
		self.save_path = save_path
		self.restore_path = restore_path
		if restore_path is None:
			self.restore_path = save_path
		self.debug = debug
		self.transfer_fct = act_funcs[params['act_func']]
		self.lrate = params["learning_rate"]
		self.minibatch_size = params["minibatch_size"]
		self.obs_size_x = params["obs_size"][0]
		self.obs_size_y = params["obs_size"][1]
		self.obs_channels = params["obs_size"][2]
		self.n_act = params["n_act"]
		self.hist_len = params["hist_len"]
		self.burnin = params["burnin"]
		self.grad_norm_clip = params["grad_norm_clip"]
		self.summary_step = params["summary_step"]
		self._final_init(params)
		self._create_graph(params, restore_step)
		self._post_graph_init()

	def _create_graph(self, params, restore_step):
		self.graph = tf.Graph()
		with self.graph.as_default():
			if restore_step is None:
				tf.set_random_seed(params['seed'])
			self._create_network()
			self._create_losses()
			self._create_optimizer()
			init = tf.global_variables_initializer()
			self._create_session(init, params.get("gpu_frac", None), 
			                     restore_step)
			self._init_tensorboard()
			self.graph.finalize()

	def _create_session(self, init, gpu_frac, restore_step):
		# Build the session
		config = tf.ConfigProto(log_device_placement=False)
		config.gpu_options.allow_growth = True
		if gpu_frac is not None:
			config.gpu_options.per_process_gpu_memory_fraction = gpu_frac
		self.sess = tf.Session(config=config)
		if self.debug:
			logging.info('Debug Mode!')
			self.sess = tf_debug.LocalCLIDebugWrapperSession(self.sess)
			self.sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
		#Saver to save graph
		self.saver = tf.train.Saver(max_to_keep=1)
		#Initialize or restore the weights
		if restore_step is None:
			self.sess.run(init)
		else:
			logging.info("Restoring: %s at step %s" % (self.restore_path, restore_step))
			self.saver.restore(self.sess, "%s/weights.ckpt-%s" % (self.restore_path, restore_step))

	def encoder_network(self, input_obs, out_size, reuse=False, net_name=''):
		shapes = self.net_arch
		with tf.variable_scope('%sencoder' % net_name):
			last_layer = input_obs
			num_conv_layers = len(shapes['encoder'][:-1])
			for layer in range(num_conv_layers):
				layer_name = 'enc_h%s' % str(layer+1)
				with tf.variable_scope(layer_name):
					channels, filt_size, stride = shapes['encoder'][layer]
					last_layer = self.transfer_fct(conv_lin(last_layer, channels, filt_size, 
					                                     	stride, reuse, False), reuse)
					if layer==num_conv_layers-1:
						last_layer = tf.contrib.layers.flatten(last_layer)
			z_out = self._final_encoder_layers(last_layer, shapes['encoder'][-1],
			                                   reuse, out_size)
		return z_out

	def _final_encoder_layers(self, layer_input, layer_shape, reuse, out_size):
		with tf.variable_scope('enc_full'):
			hidden_layer = self.transfer_fct(linear(layer_input, layer_shape, 
											   		reuse, False), reuse)
		with tf.variable_scope('enc_out'):
			z_out = linear(hidden_layer, out_size, reuse)
		return z_out
		
	def _final_init(self, params):
		pass

	def _create_network(self):
		self.model_step = tf.get_variable("model_step", trainable=False, initializer=0)

	def _create_losses(self):
		self.cost = 0

	def _create_input(self):
		pass

	def _post_graph_init(self):
		pass

	def _create_optimizer(self):
		adam_optimizer = tf.train.AdamOptimizer(learning_rate=self.lrate)
		if self.grad_norm_clip is not None:
			gradients, variables = zip(*adam_optimizer.compute_gradients(self.cost))
			gradients, global_norm = tf.clip_by_global_norm(gradients, self.grad_norm_clip)
			tf.summary.scalar('grad_global_norm', global_norm)
			self.optimizer = adam_optimizer.apply_gradients(zip(gradients, variables),
															global_step=self.model_step)
		else:
			self.optimizer = adam_optimizer.minimize(self.cost,
													 global_step=self.model_step)

	def _init_tensorboard(self):
		tf.summary.scalar('model_steps', self.model_step)
		self.merged = tf.summary.merge_all()

	def save(self, step):
		self.saver.save(self.sess, "%s/weights/weights.ckpt" % self.save_path, 
		                global_step=step)