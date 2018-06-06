#Run Experiment
import argparse
import importlib
import logging
import matplotlib
matplotlib.use('Agg')
import numpy as np
from io_utils import *
from trial import Trial
import agent

parser = argparse.ArgumentParser(description='Run experiments.')

#Run arguments
parser.add_argument('module', metavar='module', type=str, nargs=1)
parser.add_argument('experiment', metavar='experiment', type=str, nargs=1)
parser.add_argument('--debug', dest='debug', action='store_true')
parser.add_argument('--test', dest='test', action='store_true')
parser.add_argument('--record', dest='record', action='store_true')
parser.add_argument('--path_ext', dest='path_ext', type=str)
parser.add_argument('--restore_path', dest='restore_path', type=str, default=None)
parser.add_argument('--show_screen', dest='show_screen', action='store_true')
parser.add_argument('--gpu_frac', dest='gpu_frac', type=float)
parser.add_argument('--num_steps', dest='num_steps', type=int)
parser.add_argument('--train_epoch_length', dest='train_epoch_length', type=int)
parser.add_argument('--test_epoch_length', dest='test_epoch_length', type=int)
parser.add_argument('--concurrent_batches', dest='concurrent_batches', type=int)
parser.add_argument('--train_step', dest='train_step', type=int)
parser.add_argument('--summary_step', dest='summary_step', type=int)
parser.add_argument('--hist_len', dest='hist_len', type=int)
parser.add_argument('--map_path', dest='map_path', type=str)
parser.add_argument('--living_reward', dest='living_reward', type=float)
parser.add_argument('--trigger', dest='trigger', type=str)
parser.add_argument('--trigger_step', dest='trigger_step', type=int)
parser.add_argument('--num_reset_steps', dest='num_reset_steps', type=int)
parser.add_argument('--restore_weights_path', dest='restore_weights_path', type=str, default=None)

#Seeds
parser.add_argument('--seed', dest='seed', type=int, default=0)
parser.add_argument('--NP_SEED', dest='NP_SEED', type=int)
parser.add_argument('--TF_SEED', dest='TF_SEED', type=int)
parser.add_argument('--AG_SEED', dest='AG_SEED', type=int)
parser.add_argument('--ENV_SEED', dest='ENV_SEED', type=int)

#Model arguments
parser.add_argument('--n_z', dest='n_z', type=int)
parser.add_argument('--act_func', dest='act_func', type=str)
parser.add_argument('--tau_min', dest='tau_min', type=float)
parser.add_argument('--tau_max', dest='tau_max', type=float)
parser.add_argument('--tau_period', dest='tau_period', type=int)
parser.add_argument('--minibatch_size', dest='minibatch_size', type=int)
parser.add_argument('--beta_prior', dest='beta_prior', type=float)
parser.add_argument('--grad_norm_clip', dest='grad_norm_clip', type=float)
parser.add_argument('--train_reward', dest='train_reward', action='store_true')
parser.add_argument('--straight_through', dest='straight_through', action='store_true')
parser.add_argument('--learning_rate', dest='learning_rate', type=float)
parser.add_argument('--prior_tau', dest='prior_tau', type=float)
parser.add_argument('--net_arch', dest='net_arch', type=int)
parser.add_argument('--burnin', dest='burnin', type=int)

#Agent/Table/Replay arguments
parser.add_argument('--discount', dest='discount', type=float)
parser.add_argument('--exp_eps_decay', dest='exp_eps_decay', action='store_true')
parser.add_argument('--epsilon_period', dest='epsilon_period', type=int)
parser.add_argument('--pri_cutoff', dest='pri_cutoff', type=float)
parser.add_argument('--min_epsilon', dest='min_epsilon', type=float)
parser.add_argument('--test_epsilon', dest='test_epsilon', type=float)
parser.add_argument('--max_replay_size', dest='max_replay_size', type=int)
parser.add_argument('--init_capacity', dest='init_capacity', type=int)
parser.add_argument('--delete_old_episodes', dest='delete_old_episodes', action='store_true')
parser.add_argument('--track_repeats', dest='track_repeats', action='store_true')
parser.add_argument('--freeze_weights', dest='freeze_weights', action='store_true')

np.seterr(all='raise')
args = parser.parse_args()
print(args)
subdir = args.module[0]
experiment = args.experiment[0]
restore_weights_path = args.restore_weights_path
env_class = importlib.import_module(subdir + ".environment").Environment

if args.restore_path is None:
	params = load_params(subdir, experiment)
	params = process_and_update_from_args(params, args)
	path = gen_path(args, subdir, experiment)
	full_path = gen_full_path(subdir, path, unique=True)
	if restore_weights_path is not None:
		restore_weights_path = gen_full_path(subdir, 
		                                     args.restore_weights_path, 
		                                     unique=False)
	init_logger(full_path, args.debug)
	vae, agent, environment = init_experiment(full_path, params, env_class, agent,
	                                          restore_weights_path)
else:
	path = args.restore_path
	full_path = gen_full_path(subdir, path, unique=False)
	init_logger(full_path, args.debug)
	vae, agent, environment, params = restore_experiment(full_path, env_class, args)

trigger = params['env_params'].get('trigger')
if (trigger == 'None') or (trigger == ''):
	trigger = None
if trigger is not None:
	trigger_class = importlib.import_module(subdir + ".triggers")
	trigger = trigger_class.__dict__[trigger](params['env_params']['trigger_step'])

train = not params['test']
trial = Trial(full_path, vae, agent, environment, trigger)
try:
	update_step = params['concurrent_batches']*params['train_step']
	trial.run(params['num_steps'], params['summary_step'], update_step, 
	          train, params['record'])
except Exception as e:
	logging.exception("Exception in trial.")
finally:
	try:
		trial.agent.lookup.table.shutdown()
		trial.model.exit_signal.set()
	except AttributeError:
		pass
	try:
		trial.vae.sess.close()
	except:
		pass