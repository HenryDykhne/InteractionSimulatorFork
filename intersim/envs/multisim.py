
from enum import Enum
from typing import List, Tuple
import functools
from ai_traineree.types import DataSpace

import torch
import torch.nn.functional as F
import numpy as np
import pickle
from intersim.envs.simulator_jax import generate_paths
from matplotlib import pyplot as plt

from intersim.utils import ssdot_to_simstates, to_circle, get_map_path, get_svt, powerseries, horner_scheme
from intersim import StackedVehicleTraj
from intersim.viz import animate, build_map
from intersim.collisions import check_collisions, check_collisions_trajectory, collision_matrix, minSheepDistances
from intersim.default_policies.rulepolicy import RulePolicy
from intersim.default_policies.idm import IDM, IDM2

import gymnasium
from gymnasium import spaces
from pettingzoo.utils.env import ParallelEnv
from pettingzoo.utils.to_parallel import from_parallel
from pettingzoo.utils import wrappers
#from pettingzoo.test import parallel_api_test

# import gym
# from gym import error, spaces, utils
# from gym.utils import seeding

import logging
logger = logging.getLogger(__name__)

from typing import Callable


dog_names = ['Meg', 'Nell', 'Ben', 'Fly', 'Glen', 'Moss', 'Roy', 'Jess', 'Spot', 'CapFlash', 
            'Jill', 'Bess', 'Sweep', 'Jim', 'Tess', 'Lassie', 'Sam', 'Shep', 'Bob', 'Queen', 
            'Floss', 'Mirk', 'Lad', 'Mist', 'Bill', 'Don', 'Dot', 'Craig', 'Tweed', 'Nan', 
            'Lass', 'Gyp', 'Jan', 'Nap', 'Mac', 'Bet', 'Kim', 'Jack', 'Joe', 'Jed', 'Maid']

def wrap_env(env):
    env = from_parallel(env)
    #env = wrappers.ClipOutOfBoundsWrapper(env)
    #env = wrappers.OrderEnforcingWrapper(env)
    #env.dones = {agent: True for agent in env.possible_agents}
    return env

def wrapped_env(render_mode=None):
    """
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    env = raw_env(render_mode=internal_render_mode)
    # This wrapper is only for environments which print results to the terminal
    if render_mode == "ansi":
        env = wrappers.CaptureStdoutWrapper(env)
    # Clips actions for continuous action space 
    env = wrappers.ClipOutOfBoundsWrapper(env)
    # Provides a wide vareity of helpful user errors
    # Strongly recommended
    env = wrappers.OrderEnforcingWrapper(env)
    return env

def raw_env(render_mode=None):
    """
    To support the AEC API, the raw_env() function just uses the from_parallel
    function to convert from a ParallelEnv to an AEC env
    """
    env = MultiSim(render_mode=render_mode)
    env = parallel_to_aec(env)
    return env

class MultiSim(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "multisim"}

    """
    scenario_name:
    track:
    controlled_agents: list of track_ids (starts at 1)
    blackbox_agents: list of track_ids (starts at 1)
    acc_limits: linear acceleration limits in m/s^2
    phi_limits: steering rate limits in rad/s
    delta_limits: steering angle limits in rad
    traj_length: int defining number of timesteps to simulate
    """
    def __init__(self,
            scenario_name: str = "DR_USA_Intersection_MA",
            track: int = 1,
            controlled_agents: List[int] = [1],
            blackbox_agents: List[int] = [2],
            blackbox_policy = IDM,
            acc_limits: Tuple[float, float] = (-4.0, 4.0),
            phi_limits: Tuple[float, float] = (-0.5, 0.5),
            delta_limits: Tuple[float, float] = (-1.0, 1.0),
            traj_length: int = 30,
            render_mode=None,
            reward_coeffs = [1]*4
        ):
        # Ensure no duplicates or overlapping between controlled and idm
        assert len(set(controlled_agents)) == len(controlled_agents)
        assert len(set(blackbox_agents)) == len(blackbox_agents)
        assert len(set(controlled_agents).intersection(blackbox_agents)) == 0

        # Load tracks and map
        svt, filename = get_svt(scenario_name, track, controlled_agents=controlled_agents, blackbox_agents=blackbox_agents)
        map_path = get_map_path(scenario_name)

        # Initialize the state [x,y,v,psi,psidot]
        #print(svt._xlist)

        self.reward_coeffs = reward_coeffs

        self.possible_agents = [dog_names[i-1] for i in controlled_agents]
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )

        self.render_mode = render_mode

        #print(map_path)
        # simulator fields
        # make a mask of size nv that indicates which are controlled (0 for idm, 1 for controlled)
        self._nv = svt.nv
        self._dt = svt.dt
        self._svt = svt
        self._traj_length = traj_length
        #self._state = torch.zeros(self._nv, 5) * np.nan
        #self._exceeded = torch.tensor([False] * self._nv)
        # self._stop_on_collision = stop_on_collision
        # self._check_collisions = check_collisions
        self._map_path = map_path
        self._map_info = self.extract_map_info()
        self._lengths = self._svt.lengths
        self._widths = self._svt.widths
        self._acc_limits = acc_limits
        self._phi_limits = phi_limits
        self._delta_limits = delta_limits

        self._prev_steps = self._svt._masks.shape[-1] - self._traj_length

        self._state = self.get_padded_state(self._svt)

        # Convert agent track IDs to indices and create masks
        # Note: Assumes the two lists have no overlapping items
        agent_track_ids = controlled_agents + blackbox_agents
        self._cagent_idx = torch.tensor([agent_track_ids.index(tid) for tid in controlled_agents])
        self._bagent_idx = torch.tensor([agent_track_ids.index(tid) for tid in blackbox_agents])


        self._cagent_mask = torch.zeros(self._nv, dtype=bool)
        self._cagent_mask[self._cagent_idx] = 1

        # Initialize black box policy
        self._blackbox_policy = blackbox_policy(self._lengths, v0=30.0, s0=3., dth=4., amax=3., b=4., half_angle=20)

        # gym fields
        # using intersim.Box over spaces.Box for pytorch
        # action is: acceleration, steering_rate
        #action_shape = (self._traj_length, 2)
        action_shape = (self._traj_length, 2)
        low_action = np.array([[acc_limits[0], phi_limits[0]]])
        low_action = np.repeat(low_action, self._traj_length, axis=0)
        high_action = np.array([[acc_limits[1], phi_limits[1]]])
        high_action = np.repeat(high_action, self._traj_length, axis=0)
        #low_action = .expand(action_shape)
        self.action_box = spaces.Box(low=low_action.flatten(), high=high_action.flatten())

        self.action_spaces = {agent: self.action_box for agent in self.possible_agents}
        self.observation_spaces = {agent: spaces.Discrete(1) for agent in self.possible_agents}

        #low_state = torch.tensor([[-np.inf, -np.inf, 0., -np.pi, -np.inf]]).expand(self._nv, 5)
        #high_state = torch.tensor([[np.inf,np.inf,np.inf,np.pi, np.inf]]).expand(self._nv, 5)
        #self.state_space = Box(low=low, high=high, shape=(self._nv, 5))
        #self.observation_space = spaces.Discrete(1)

        super(MultiSim, self).__init__() # is this necessary??

    # # this cache ensures that same space object is returned for the same agent
    # # allows action space seeding to work as expected
    # @functools.lru_cache(maxsize=None)
    # def observation_space(self, agent):
    #     # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/
    #     return spaces.Discrete(1)

    # @functools.lru_cache(maxsize=None)
    # def action_space(self, agent):
    #     return self.action_box

    def step(self, actions):
        """
        step(action) takes in an action for each agent and should return the
        - observations
        - rewards
        - terminations
        - truncations
        - infos
        dicts where each dict looks like {agent_1: item_1, agent_2: item_2}
        """
        # If a user passes in actions with no agents, then just return empty observations, etc.
        if not actions:
            self.agents = []
            return {}, {}, {}, {}, {}
        
        # Make sure IDs are in assumed order
        agent_ids = [self.agent_name_mapping[agent_name] for agent_name in self.agents]
        assert agent_ids == sorted(agent_ids)

        actions = [torch.tensor(a) for a in actions.values()]
        action_arr = torch.stack(actions).view(-1, self._traj_length, 2)

        # action: (num_controlled_agents, traj_length, 2)
        idx0 = self._prev_steps
        prev_state = self._state[:idx0].clone()
        self.prev_state = prev_state
        gt_state = self._state[idx0:].clone()
        self.gt_state = gt_state

        # print('---')
        # print(prev_state)
        # print(gt_state)

        # Assume bicycle model wheelbase (L) is 60% of bounding box length
        # Average from this dataset https://rpubs.com/Arnav_Jain/DSP_AutomobileDatasetAnalysis
        L = 0.6 * self._lengths[self._cagent_idx]

        # Can calculate steering angle (delta) from bicycle model equations
        dt = self._svt._dt
        v0 = prev_state[-1, self._cagent_idx, 2]
        psi0 = prev_state[-1, self._cagent_idx, 3]
        psi1 = gt_state[0, self._cagent_idx, 3]
        delta0 = torch.atan(L / (v0 * dt) * (psi1 - psi0))

        # Set final value in state for controlled agents to delta
        # Final value is still psidot for blackbox agents
        # Doing this just because it is convenient
        cur_state = prev_state[-1]
        cur_state[self._cagent_idx, 4] = delta0
        cur_path_state = self._svt._simstate[idx0-1, self._bagent_idx].clone()

        collision_matrices = []
        min_dists = []
        simulated_traj = []
        sim_states = []
        for t in range(self._traj_length):
            # Calculate acceleration actions for blackbox agents
            # Note: Does not use the psidot value, this is why we were able to set it to delta earlier
            blackbox_policy_acc = self._blackbox_policy.compute_action(cur_state.reshape(-1))[self._bagent_idx]

            # Update state of blackbox agents based on computed actions
            nextv = cur_path_state[:, 1:2] + blackbox_policy_acc * dt
            nextv = nextv.clamp(0., np.inf) # not differentiable!
            cur_path_state[:, 0:1] += 0.5 * dt * (nextv + cur_path_state[:, 1:2])
            cur_path_state[:, 1:2] = nextv

            # See which states exceeded their maxes
            exceeded = (cur_path_state[:, 0] > self._svt.smax[self._bagent_idx])
            cur_path_state[exceeded] = np.nan

            # Convert to [x, y, v, psi, psidot] state
            projected_state = ssdot_to_simstates(
                cur_path_state[:, 0].unsqueeze(0),
                cur_path_state[:, 1].unsqueeze(0),
                self._svt._xpoly[self._bagent_idx],
                self._svt._dxpoly[self._bagent_idx],
                self._svt._ddxpoly[self._bagent_idx],
                self._svt._ypoly[self._bagent_idx],
                self._svt._dypoly[self._bagent_idx],
                self._svt._ddypoly[self._bagent_idx]
            )[0]
            cur_state[self._bagent_idx] = projected_state.double()

            # Update state of controlled agents using bicycle model
            cur_state[self._cagent_idx, 0] += dt * cur_state[self._cagent_idx, 2] * torch.cos(cur_state[self._cagent_idx, 3])
            cur_state[self._cagent_idx, 1] += dt * cur_state[self._cagent_idx, 2] * torch.sin(cur_state[self._cagent_idx, 3])
            cur_state[self._cagent_idx, 3] += dt * cur_state[self._cagent_idx, 2] * torch.tan(cur_state[self._cagent_idx, 4]) / L
            cur_state[self._cagent_idx, 2] += dt * action_arr[:, t, 0]
            cur_state[self._cagent_idx, 4] += dt * action_arr[:, t, 1]

            simulated_traj.append(cur_state[:, 0:2].clone())

            # Calculate collision matrix and min blackbox distance
            collision_matrices.append(collision_matrix(cur_state, self._lengths, self._widths))
            min_dists.append(minSheepDistances(cur_state, self._lengths, self._widths, self._cagent_mask))

            sim_states.append(cur_state.detach().clone())

        # For rendering
        self.sim_states = torch.stack(sim_states)
        #print(self.sim_states)

        # Calculate reward
        simulated_traj = torch.stack(simulated_traj)
        collision_matrices = torch.stack(collision_matrices)
        min_dists = torch.tensor(min_dists)
        #print(min_dists)
        gt_traj = gt_state[:, :, 0:2].unsqueeze(1)
        rewards = {}
        for cagent_idx, agent in zip(self._cagent_idx, self.agents):
            reward = self.calculate_reward(simulated_traj, gt_traj, collision_matrices, min_dists, self._cagent_mask, cagent_idx)
            rewards[agent] = reward

        # return observations, rewards, terminations, truncations, infos
        obs = {agent: torch.tensor(0.0).unsqueeze(0) for agent in self.agents}
        # terminations = {agent: True for agent in self.agents}
        dones = {agent: True for agent in self.agents}
        # truncations = {agent: False for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        return obs, rewards, dones, infos
    
    def calculate_reward(self, simulatedTrajectories, realisticTrajectories, collisionMatricies, minIdmDistances, controlMask, agentIDX):
        #realisticTrajectories expected dim = [30 timesteps, 6 modes, N agents, 2 coordinates]
        #simulatedTrajectories expected dim = [30, N, 2]
        #collisionMatricies expected dim = [30, N, N]
        #minIdmDistances expected dim = [30]
        #controlMask expected dim = [N]
        a, b, c, d = self.reward_coeffs

        crashLearner = 0
        dogSelfCol = 0
        interSheepCol = 0
        idmEnd, learnerEnd = 30, 30
        for i in range(30):
            if interSheepCol == 0 and 1 in collisionMatricies[i][~controlMask, ~controlMask]: #idmCollided
                interSheepCol = 1
                idmEnd = i
            if 1 in collisionMatricies[i][controlMask]: #agentCollided
                crashLearner = 1 
                dogSelfCol = 1 if 1 in collisionMatricies[i][agentIDX] else 0 
                learnerEnd = i
                break
            
        end = idmEnd if crashLearner == 0 else learnerEnd
        # print('-----')
        # print('minIdmDistances1', minIdmDistances) 
        simulatedTrajectories = simulatedTrajectories[:end]
        realisticTrajectories = realisticTrajectories[:end]
        minIdmDistances = minIdmDistances[:end]

        ades = []
        # print('----')
        # print(controlMask)
        # print(simulatedTrajectories[:, controlMask].size())
        # print(idmEnd, learnerEnd, end)
        for j in range(realisticTrajectories.size()[1]):
            x = (realisticTrajectories[:, j, controlMask] - simulatedTrajectories[:,controlMask])
            y = torch.norm(x, dim=2)
            z = torch.mean(y, dim=0)
            ade = torch.mean(z)
            ades.append(ade)

        bestMode = torch.argmin(torch.tensor(ades)) 
        x = (realisticTrajectories[:, bestMode, agentIDX] - simulatedTrajectories[:, agentIDX])
        #print('x', x.size())
        y = torch.norm(x, dim=1)
        z = torch.mean(y, dim=0)
        minLearnerADE = z
        
        #print('minIdmDistances', minIdmDistances) 
        minIdmDistance = torch.min(minIdmDistances)

        reward = 0  
        reward += a*minIdmDistance #0 = idm, 1 = learningAgent
        reward += b*dogSelfCol
        reward += c*minLearnerADE
        reward += d*interSheepCol
        # print('---')
        # print('minIdmDistance', minIdmDistance)
        # print('2', interSheepCol)
        # print('minLearnerADE', minLearnerADE)
        # print('4', dogSelfCol)
        # print('5',reward)
        return reward.item()

    # def reset(self):
    #     return [0] * self._cagent_idx.shape[0]
    
    def reset(self, seed=None, options=None):
        """
        Reset needs to initialize the `agents` attribute and must set up the
        environment so that render(), and step() can be called without issues.
        Here it initializes the `num_moves` variable which counts the number of
        hands that are played.
        Returns the observations for each agent
        """
        self.agents = self.possible_agents[:]
        #self.num_moves = 0
        observations = {agent: torch.tensor(0.0).unsqueeze(0) for agent in self.agents}

        return observations

    @staticmethod
    def get_padded_state(svt):
        """
        Return state of size (timesteps, nv, 5)
        """
        full_size = svt._masks[0].shape[0]
        padded_state = []
        for i in range(svt._t0.shape[0]):
            num_points = svt._xlist[i].shape[0]
            pre_pad = round((svt._t0[i].item() - svt._dt) / svt._dt)
            post_pad = full_size - pre_pad - num_points
            state = torch.stack((
                svt._xlist[i],
                svt._ylist[i],
                svt._v[i],
                svt._psilist[i],
                svt._psidotlist[i]
            ))
            padded_state.append(F.pad(state, (pre_pad, post_pad), "constant", float("nan")).T)
        padded_state = torch.stack(padded_state, axis=1)

        return padded_state

    def render_anim(self, filestr='render'):
        print(f"Saving render to {filestr}")
        animate(
            self._map_path,
            self._cagent_idx,
            self._bagent_idx,
            self.prev_state,
            self.gt_state,
            self.sim_states,
            self._lengths,
            self._widths,
            filestr=filestr
        )
        
    def extract_map_info(self):
        """
        Generates information about environment map
        Returns:
            map_info: information about map
        """
        map_info, _ = build_map(self._map_path)
        return map_info

    def close(self):
        """
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        """
        pass
        

if __name__=="__main__":
    SCENARIO_NAME = "DR_USA_Intersection_MA"
    TRACK = 100
    CONTROLLED_AGENTS = [2, 4]
    BLACKBOX_AGENTS = [5, 6]

    env = MultiSim(
        scenario_name=SCENARIO_NAME,
        track=100,
        controlled_agents=CONTROLLED_AGENTS,
        blackbox_agents=BLACKBOX_AGENTS
    )
    #parallel_api_test(env, num_cycles=1000)
    # env.reset()
    # env.step(torch.zeros(2, 60))
