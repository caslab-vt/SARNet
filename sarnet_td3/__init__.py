class MAgentTrainer(object):
    def __init__(self, name, model, obs_shape, act_space, args):
        raise NotImplemented()

    def action(self, input, is_bench):
        raise NotImplemented()

    def process_experience(self, obs, act, rew, new_obs, done, terminal):
        raise NotImplemented()

    def preupdate(self):
        raise NotImplemented()

    def update(self, agents, buffer, t):
        raise NotImplemented()