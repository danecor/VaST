hallmaze_hazards = [[0, -320], [256, -64], [-128, 320], [-384, -64],
					[128, -192], [0, 192]]

def hall_1(pos_x, pos_y, terminal_distance, goal_reward, 
           terminal_reward, living_reward, rng):
	if max(abs(pos_x),abs(pos_y)) > terminal_distance:
		reward = terminal_reward
		terminal = True
		if pos_x > terminal_distance:
			reward = goal_reward
		return reward, terminal
	return living_reward, False

def hall_x(pos_x, pos_y, terminal_distance, goal_reward, 
           terminal_reward, living_reward, rng):
	if max(abs(pos_x),abs(pos_y)) > terminal_distance:
		reward = terminal_reward
		terminal = True
		if abs(pos_x) > terminal_distance:
			reward = goal_reward
		return reward, terminal
	return living_reward, False

def hall_y(pos_x, pos_y, terminal_distance, goal_reward, 
           terminal_reward, living_reward, rng):
	if max(abs(pos_x),abs(pos_y)) > terminal_distance:
		reward = terminal_reward
		terminal = True
		if abs(pos_y) > terminal_distance:
			reward = goal_reward
		return reward, terminal
	return living_reward, False

def exmaze(pos_x, pos_y, terminal_distance, goal_reward, 
           terminal_reward, living_reward, rng):
	if pos_y < -terminal_distance:
		return goal_reward, True
	return living_reward, False

def no_test(pos_x, pos_y, terminal_distance, goal_reward, 
            terminal_reward, living_reward, rng):
	return living_reward, False

def hmaze(pos_x, pos_y, terminal_distance, goal_reward, 
          terminal_reward, living_reward, rng):
	if (abs(pos_y) > terminal_distance) and (pos_x > 0):
		if pos_y > 0:
			reward = goal_reward
		else:
			reward = terminal_reward
		return reward, True
	return living_reward, False

def hallmaze(pos_x, pos_y, terminal_distance, goal_reward, 
          terminal_reward, living_reward, rng):
	if max(abs(pos_x),abs(pos_y)) < terminal_distance:
		return goal_reward, True
	return living_reward, False

def hallmaze_hazard(pos_x, pos_y, terminal_distance, goal_reward, 
          terminal_reward, living_reward, rng):
	if max(abs(pos_x),abs(pos_y)) < terminal_distance:
		return goal_reward, True
	else:
		if rng.random_sample() < 0.25:
			for hazard in hallmaze_hazards:
				if (pos_x > hazard[0]) and (pos_x < (hazard[0]+64)):
					if (pos_y > hazard[1]) and (pos_y < (hazard[1]+64)):
							return -1., False
	return living_reward, False