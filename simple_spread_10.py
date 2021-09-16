import numpy as np
from multiagent.core import World, Agent, Landmark, Border
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = 10
        num_landmarks = 10
        world.collaborative = False

        # Control partial observability of the agents
        self.vision_range = 3  # multiplier of agent size
        self.land_vision_count = 4
        self.agent_vision_count = 3  # include the self agent, that is +1

        # add agents
        world.agents = [Agent() for _ in range(num_agents)]

        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.15 / (num_agents / 6)

        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.05 / (num_landmarks / 6)

        self.occ_land_dist = (world.agents[0].size - world.landmarks[0].size) / 2
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0

        dists = [np.sqrt(np.sum(np.square(agent.state.p_pos - l.state.p_pos))) for l in world.landmarks]
        rew -= min(dists)
        if min(dists) < self.occ_land_dist:
            occupied_landmarks += 1

        if agent.collide:
            for a in world.agents:
                if a is not agent:
                    if self.is_collision(a, agent):
                        rew -= 1
                        collisions += 1
        info = {'success': [], 'collisions': [], 'rew': [], 'min_dists': [], 'occ_land': []}
        info['collisions'].append(collisions)
        info['occ_land'].append(occupied_landmarks)
        info['rew'].append(rew)
        info['min_dists'].append(min(dists))
        return info

        # return (rew, collisions, min_dists, occupied_landmarks)

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = 0
        dists = [np.sqrt(np.sum(np.square(agent.state.p_pos - l.state.p_pos))) for l in world.landmarks]
        rew -= min(dists)
        if min(dists) < self.occ_land_dist:
            rew += 1

        if agent.collide:
            for a in world.agents:
                if a is not agent:
                    if self.is_collision(a, agent):
                        rew -= 1
        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = self.get_land_vision(agent, world)
        other_pos = self.get_agent_vision(agent, world)

        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos)  # + comm

    def get_agent_vision(self, agent, world):
        dist = []
        for other in world.agents:
            dist.append(np.sqrt(np.sum(np.square(other.state.p_pos - agent.state.p_pos))))

        sorted_dist = np.argsort(dist)
        agent_idx = sorted_dist[1:self.agent_vision_count+1]  # max. number of agents it can observe

        agent_vision = []
        for i, other in enumerate(world.agents):
            if i in agent_idx:
                agent_vision.append(other.state.p_pos - agent.state.p_pos)

        return agent_vision

    def get_land_vision(self, agent, world):
        dist = []
        for entity in world.landmarks:
            dist.append(np.sqrt(np.sum(np.square(entity.state.p_pos - agent.state.p_pos))))
        # Ascending sort, and retrieve the index of landmarks in that order
        sorted_dist = np.argsort(dist)
        land_idx = sorted_dist[0:self.land_vision_count]  # max. number of landmarks that it can observe

        # Check if these landmarks are in the vision range and populate observation
        land_vision = []
        for i, entity in enumerate(world.landmarks):
            if i in land_idx:
                land_vision.append(entity.state.p_pos - agent.state.p_pos)

        return land_vision