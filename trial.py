import os
import logging
import numpy as np
from models.ops import SummaryWriter
from io_utils import save_experiment, ensure_dir, update_log
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class Trial(object):

	def __init__(self, path, vae, agent, environment, trigger=None):
		self.path = path
		self.model = vae
		self.agent = agent
		self.environment = environment
		self.trigger = trigger
		self.record_obs = None
		self.start_step = 0

	def run(self, num_steps, summary_step, update_step, train=True, record=False):
		if record:
			self.init_recording()
		summary_writer = SummaryWriter(self.path, self.model.sess.graph)
		self.agent.test = not train
		self.agent.reset_summary_variables()
		self.environment.start()
		self.environment.init_epoch()
		obs = self.init_episode()
		while self.agent.step < num_steps:
			if record:
				self.update_recording()
			new_eps = self.apply_trigger()
			if new_eps:
				obs = self.init_episode()
			if self.agent.step % update_step == 0:
				self.agent.train_and_update(self.model, summary_writer)
			obs, terminal = self.act_and_observe(obs)
			if terminal:
				self.update_summary(summary_writer, True)
				obs = self.init_episode()
			if self.agent.step % summary_step == 0:
				self.update_summary(summary_writer)
			if self.environment.epoch_finished():
				self.agent.finish_update(self.model)
				self.run_test_epoch(summary_writer)
				if train:
					self.save()
				self.environment.init_epoch()
				obs = self.init_episode()
		self.shutdown(train, record)

	def init_episode(self):
		logging.debug('Environment reset: %s' % self.agent.step)
		obs = self.environment.init_episode()
		logging.debug('Agent reset: %s' % self.agent.step)
		self.agent.init_episode(obs, self.model)
		return obs

	def act_and_observe(self, obs):
		logging.debug('Get action: %s' % self.agent.step)
		action = self.agent.get_action(self.model)
		logging.debug('Take action: %s' % self.agent.step)
		obs, reward, terminal = self.environment.step(action)
		logging.debug('Step update: %s' % self.agent.step)
		self.agent.observe(action, reward, terminal, obs, self.model)
		return obs, terminal

	def run_test_epoch(self, summary_writer):
		logging.debug("Starting evalation.")
		self.environment.init_epoch(test=True)
		self.agent.test = True
		self.agent.reset_summary_variables()
		episode_reward = 0
		num_episodes = 0
		obs = self.init_episode()
		while not self.environment.epoch_finished():
			logging.debug("Evaluation step: %i" % self.environment.epoch_steps)
			obs, terminal = self.act_and_observe(obs)
			if terminal:
				episode_reward += self.agent.summary_variables['eval_episode_reward']
				num_episodes += 1
				env_plots = self.environment.get_episode_plots()
				summary_writer.summarize_images(env_plots, self.agent.step)
				self.init_episode()
		logging.debug("Finished evaluation.")
		eval_reward = self.agent.summary_variables['eval_reward']/float(self.environment.test_epoch_length)
		summary_writer.add_summary(summary_writer.value_summary(eval_reward, "eval_reward"), 
		                           self.agent.step)
		if num_episodes == 0:
			eval_eps_reward = np.nan
		else:
			eval_eps_reward = episode_reward/float(num_episodes)
		summary_writer.add_summary(summary_writer.value_summary(eval_eps_reward, 
		                                                        "eval_episode_reward"), 
		                           self.agent.step)
		self.agent.reset_summary_variables()
		self.agent.test = False

	def update_summary(self, summary_writer, episode_update=False):
		logging.debug('Summary update: %s' % self.agent.step)
		agent_summary = self.agent.get_summary_variables()
		summary_writer.summarize_agent(agent_summary, self.agent.step, episode_update)
		if not episode_update:
			update_log(agent_summary, self.agent.step)
			self.agent.reset_summary_variables()

	def save(self):
		logging.debug('Save: %s' % self.agent.step)
		save_experiment(self.path, self.model, self.agent, self.environment)

	def apply_trigger(self):
		new_eps = False
		if self.trigger is not None:
			logging.debug('Trigger: %s' % self.agent.step)
			vae, agent, env, new_eps = self.trigger.attempt(self.model, 
			                                                self.agent, 
			                                            	self.environment)
			self.model = vae
			self.agent = agent
			self.environment = env
		return new_eps

	def init_recording(self):
		hr_obs = self.environment.get_full_obs()
		self.record_obs = np.empty([num_steps, 
		                 			hr_obs.shape[0], 
		                 			hr_obs.shape[1], 
		                 			hr_obs.shape[2]], dtype = np.uint8)
		self.start_step = self.agent.step

	def update_recording(self):
		self.record_obs[self.agent.step-self.start_step] = self.environment.get_full_obs()

	def shutdown(self, train, record):
		try:
			logging.info('Shutting down sweeper table.')
			self.agent.lookup.table.shutdown()
			logging.info('Shutting down VAE training.')
		except AttributeError:
			pass
		if train:
			try:
				self.model.training_thread.join()
			except AttributeError:
				pass
		self.model.sess.close()
		logging.info('Shutting down environment.')
		self.environment.end(self.agent.save_path)
		if record:
			logging.info('Recording trial.')
			self.record_trial()

	def record_trial(self):
		logging.info('Plotting frames.')
		ims = []
		fig = plt.figure()
		for ind in range(len(self.record_obs)):
			im = plt.imshow(self.record_obs[ind][:,:,::-1], animated=True)
			plt.xticks([])
			plt.yticks([])
			plt.tight_layout()
			ims.append([im])
		Writer = animation.writers['ffmpeg']
		writer = Writer(fps=20, metadata=dict(artist='Me'), bitrate=-1)
		file_path = '%s/record.mp4' % self.path
		if os.path.exists(file_path):
			i = 1
			while os.path.exists('%s/record%i.mp4' % (self.path, i)):
			    i += 1
			file_path = '%s/record%i.mp4' % (self.path, i)
		logging.info('Saving frames to %s.' % file_path)
		ani = animation.ArtistAnimation(fig, ims, blit=True)
		full_path = ensure_dir(file_path)
		ani.save(full_path, writer=writer)