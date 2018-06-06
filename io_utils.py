import sys
import os
import importlib
import logging
import yaml
import re
import glob
import cPickle as pickle
import numpy as np
from models.vae import FilteringVAE, ConcurrentVAE

def init_logger(path, is_debug):
	logger = logging.getLogger()
	level = logging.DEBUG if is_debug else logging.INFO
	logger.setLevel(level)
	log_path = ensure_dir("%s/output.log" % path)
	ch = logging.StreamHandler(sys.stdout)
	ch.setLevel(level)
	fh = logging.FileHandler(log_path)
	fh.setLevel(level)
	formatter = logging.Formatter('%(asctime)s - %(threadName)s - %(levelname)s: %(message)s')
	ch.setFormatter(formatter)
	fh.setFormatter(formatter)
	logger.addHandler(ch)
	logger.addHandler(fh)
	logging.info('Starting Experiment')

def gen_full_path(subdir, path, unique=True):
	file_path = "%s/data/%s" % (subdir, path)
	if unique and os.path.exists(file_path):
		i = 1
		while os.path.exists(file_path + str(i)):
		    i += 1
		file_path += str(i)
	return file_path

def gen_path(args, module, experiment):
	path = "%s_%s" % (module, experiment)
	args = vars(args)
	for key, val in args.iteritems():
		substr = ""
		if key != 'path_ext' and key != 'gpu_frac' and val is not None:
			substr = "_" + str(key)
			if type(val) is str:
				substr += "_" + str(val)
			elif type(val) is int:
				substr += str(val)
			elif type(val) is float:
				if abs(val) >= 0.01 and abs(val) < 1000:
					substr += "{:.2f}".format(val)
				else:
					substr += "{:.1E}".format(val)
			elif type(val) is bool and not val:
				substr = ""
		path += substr
	if 'path_ext' in args:
		path += "_" + args['path_ext']
	return path

def load_params(subdir, experiment):
	with open("%s/params.yaml" % subdir, 'r') as stream:
		params = yaml.load(stream)
	experiment_params = params['env_params']['experiments'][experiment]
	for (key, val) in experiment_params.iteritems():
		params['env_params'][key] = val
	params['env_params'].pop('experiments')
	for (key, val) in params.iteritems():
		if (not "_params" in key) and (key != 'net_arches'):
			params['env_params'][key] = val
			params['model_params'][key] = val
			params['agent_params'][key] = val
	return params

def init_experiment(path, params, env_class, agents, restore_weights_path=None):
	environment = env_class(**params['env_params'])
	params['env_params']['rng'] = environment.rng
	params['model_params']['n_act'] = len(environment.actions)
	with open("%s/params" % path, "wb") as handle:
		pickle.dump(params, handle)
	agent_class = agents.Agent
	rstep = None
	if restore_weights_path is not None:
		rstep = get_weight_steps(restore_weights_path)
		restore_weights_path += '/weights'
	if params['concurrent_batches'] == 1:
		model = FilteringVAE(params['model_params'], path, 
		                     restore_step=rstep, restore_path=restore_weights_path)
	else:
		model = ConcurrentVAE(params['model_params'], path, 
		                      restore_step=rstep, restore_path=restore_weights_path)
	params['agent_params']['n_act'] = len(environment.actions)
	agent = agent_class(path, **params['agent_params'])
	return model, agent, environment

def restore_experiment(path, env_class, args=None, steps=None, include_replay=True):
	with open("%s/params" % path, 'rb') as handle:
		params = pickle.load(handle)
	if args is not None:
		params = update_from_args(params, args)
	if steps is None:
		steps = get_max_steps(path, 'agent')
	agent = pickle.load(open("%s/agent.ckpt-%s" % (path, steps), 'rb'))
	agent.save_path = path
	agent.lookup = pickle.load(open("%s/lookup-%s" % (path, steps), 'rb'))
	agent.lookup.save_path = path
	agent.lookup.restore_dict("%s/state_dict-%s" % (path, steps))
	agent.lookup.make_table(steps)
	if not params['test'] and include_replay:
		replay_path = "%s/agent_replay.ckpt-%s.npz" % (path, steps)
		agent.replay_memory.load_memory(np.load(replay_path))
	else:
		agent.replay_memory = None
	vae_path = "%s/weights" % path
	if params['concurrent_batches'] == 1:
		restore_vae = FilteringVAE(params['model_params'], vae_path, steps)
	else:
		restore_vae = ConcurrentVAE(params['model_params'], vae_path, steps)
	environment = env_class(**params['env_params'])
	return restore_vae, agent, environment, params

def save_experiment(path, vae, agent, env):
	logging.info("Saving... %s" % agent.step)
	vae.save(agent.step)
	agent.save()
	param_path = "%s/params" % path
	with open(param_path, 'rb') as handle:
		params = pickle.load(handle)
	params['env_params']['rng'] = env.rng
	params['env_params']['epoch_steps'] = env.epoch_steps
	env.set_seed()
	with open(param_path, 'wb') as handle:
		pickle.dump(params, handle)

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    return file_path

def get_max_steps(path, checkfile):
	path = "%s/%s.ckpt*" % (path, checkfile)
	filenames = glob.glob(path)
	substrs = [re.findall(r'ckpt-\d+', filename)[0] for filename in filenames]
	try:
		return max([int(substr[5:]) for substr in substrs])
	except ValueError:
		return None

def get_weight_steps(path):
	path = "%s/weights/weights.ckpt-*.meta" % path
	filenames = glob.glob(path)
	return int(re.findall(r'weights.ckpt-\d+', filenames[0])[0][13:])

def update_from_args(params, args):
	args = vars(args)
	for key in args.keys():
		if args[key] is None:
			args.pop(key)
		else:
			if key in params['env_params']:
				params['env_params'][key] = args[key]
			if key in params['model_params']:
				params['model_params'][key] = args[key]
			if key in params['agent_params']:
				params['agent_params'][key] = args[key]
	for key in ['module', 'experiment']:
		args.pop(key)
	params.update(args)
	return params

def process_and_update_from_args(params, args):
	params = update_from_args(params, args)
	rng = np.random.RandomState(args.seed)
	seeds = {'NP_SEED': rng.randint(1e5), 'TF_SEED': rng.randint(1e5), 
			 'AG_SEED': rng.randint(1e5), 'ENV_SEED': rng.randint(1e5)}
	np.random.seed(seeds['NP_SEED'])
	params['env_params']['seed'] = seeds['ENV_SEED']
	params['agent_params']['seed'] = seeds['AG_SEED']
	params['model_params']['seed'] = seeds['TF_SEED']
	net_arch_key = params['model_params']['net_arch']
	params['model_params']['net_arch'] = params['net_arches'][net_arch_key]
	params.pop('net_arches')
	return params

def update_log(agent_summary, steps):
	logging.info(" ".join(["Step: ", '%05d' % steps, 
	                	   "Reward: %d" % agent_summary['reward'],
						   "Epsilon: %.3f" % agent_summary['epsilon']]))


def shuffle_textures():
	from random import shuffle
	textures = ['BRNBIGC\x00', 'BRNPOIS\x00', 'STARTAN1', 'BIGDOOR2', 
				'ZDOORB1\x00', 'CEMENT1\x00', 'COMPBLUE', 'BLODRIP2', 
				'CRACKLE2', 'CEMENT4\x00', 'BLAKWAL2', 'LITE3\x00\x00\x00', 
				'DOORYEL\x00', 'CEMENT8\x00', 'WOODMET1', 'BRNSMAL1', 
				'LITEBLU4', 'DOORRED\x00', 'COMPSTA1', 'BROVINE2', 
				'PANRED\x00\x00', 'BRICK3\x00\x00', 'CEMENT9\x00', 
				'CRATWIDE', 'ASHWALL7', 'BRICK11\x00', 'ASHWALL2', 
				'CRACKLE4', 'PANBLUE\x00', 'PANBLACK', 'BRICKLIT', 
				'ZZWOLF1\x00', 'ZZWOLF9\x00', 'ZIMMER8\x00']
	sd_file = open('doom/maps/train_mazes/sd_data','rb')
	sd_data = sd_file.read()
	sd_file.close()
	for i in range(500):
		new_sd_data = sd_data
		new_textures = list(textures)
		shuffle(new_textures)
		for j, offset in enumerate(range(0,1020,30)):
			new_sd_data = new_sd_data[:offset+20] + new_textures[j] + new_sd_data[offset+28:]
			assert len(new_sd_data) == len(sd_data)
		new_file = open('doom/maps/train_mazes/sd_data'+str(i),'wb')
		new_file.write(new_sd_data)
		new_file.close()