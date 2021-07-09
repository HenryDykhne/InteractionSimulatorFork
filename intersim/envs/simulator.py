# simulator.py

from enum import Enum

import torch
import numpy as np
import pickle
from intersim.utils import ssdot_to_simstates, to_circle, get_map_path, get_svt
from intersim import Box, StackedVehicleTraj, InteractionGraph
from intersim.viz.utils import build_map

import gym
from gym import error, spaces, utils
from gym.utils import seeding

import logging
logger = logging.getLogger(__name__)

from typing import Callable

class RewardMethod(Enum):
    """
    NONE: 0 reward throughout
    CUSTOM: user-specified reward function R(next_state, action)
    """
    NONE = 'none'
    CUSTOM = 'custom'

class ObserveMethod(Enum):
    """
    LIGHT: simple state and graph information
    FULL: simple and relative state information, graph information, and map information
    HIDDEN: FULL state information and hidden environment information

    """
    LIGHT = 'light'
    FULL = 'full'
    HIDDEN = 'hidden' 

class InteractionSimulator(gym.Env):
    
    metadata = {'render.modes': ['live','file','post']}

    def __init__(self, 
                 loc: int = 0,
                 track: int = 0,
                 svt: StackedVehicleTraj = None, 
                 map_path: str = None, 
                 graph: InteractionGraph = InteractionGraph(),
                 min_acc: float=-1.0, max_acc: float=1.0,
                 reward_method='none', observe_method='full',
                 custom_reward: Callable[[dict, torch.Tensor], float] = lambda x, y: 0.
                 ):
        """
        Roundabout simulator.
        Args:
            loc (int): location index of scene
            track (into): track number of scene
            svt (StackedVehicleTraj): vehicle trajectory to reference
            map_path (str): path to map .osm file
            graph (InteractionGraph): graph depicting vehicle interactions
            min_acc (float): minimum acceleration allowed
            max_acc (float): maximum acceleration allowed
            reward_method (str): reward type. see RewardMethod class
            observe_method (str): observation type. see ObserveMethod class
            custom_reward (Callable): custom reward function R(s',a) to be used w/ 'custom' reward method
        Notes:
            action_space is [min_acc,max_acc] ^ nvehicles
            state_space is [x,y,v,psi,psidot] ^ nvehicles
        """

        # Get stacked vehicle trajectory
        if not svt:
            svt, filename = get_svt(loc, track)
            print('Vehicle Trajectory Paths: {}'.format(filename))
        else:
            print('Custom Vehicle Trajectory Paths')
            
        # Get map path
        if not map_path:
            map_path = get_map_path(loc)
        print('Map Path: {}'.format(map_path))
        
        # simulator fields 
        self._nv = svt.nv
        self._dt = svt.dt
        self._svt = svt
        self._state = torch.zeros(self._nv, 2) * np.nan
        self._t = svt.minT
        self._exceeded = torch.tensor([False] * self._nv)
        self._min_acc = min_acc
        self._max_acc = max_acc
        self._map_path = map_path
        self._map_info = self.extract_map_info()
        self._xpoly = self._svt.xpoly
        self._ypoly = self._svt.ypoly
        self._lengths = self._svt.lengths
        self._widths = self._svt.widths
        self._graph = graph
        self._graph.update_graph(self.projected_state)
        self._reward_method = RewardMethod(reward_method)
        self._observe_method = ObserveMethod(observe_method)
        self._custom_reward = custom_reward

        # gym fields
        # using intersim.Box over spaces.Box for pytorch
        self.action_space = Box(low=min_acc, high=max_acc, shape=(self._nv,1))
        low = torch.tensor([[-np.inf,-np.inf,0.,-np.pi, -np.inf]]).expand(self._nv,5)
        high = torch.tensor([[np.inf,np.inf,np.inf,np.pi, np.inf]]).expand(self._nv,5)
        self.state_space = Box(low=low, high=high, shape=(self._nv,5))
        self.done= False
        self.info = {
            'raw_state': self._state.clone(),
            'xpoly': self._xpoly,
            'dxpoly': self._svt.dxpoly,
            'ddxpoly': self._svt.ddxpoly,
            'ypoly': self._ypoly,
            'dypoly': self._svt.dypoly,
            'ddypoly': self._svt.ddypoly,
            'smax': self._svt.smax,
            'lengths': self._lengths,
            'widths': self._widths,
        }

        # rendering fields
        self._state_list = []
        self._graph_list = []

        super(InteractionSimulator, self).__init__() # is this necessary??

    @property
    def state(self):
        """
        Return [s, v]
        Returns:
            state (torch.tensor): (nv, 2) state in terms of s, sdot
        """
        return self._state
    
    @property
    def path_coefs(self):
        """
        Return coefficients for computing x(s) and y(s)
        Returns:
            xpoly (torch.tensor): (nv, deg) x polynomial path coefficients
            ypoly (torch.tensor): (nv, deg) y polynomial path coefficients
        """
        svt = self._svt
        return svt.xpoly, svt.ypoly
    
    @property
    def smax(self):
        """
        Return maximum path lengths
        Returns:
            smax (torch.tensor): (nv,) maximum path lengths
        """
        return self._svt.smax
    
    @property
    def projected_state(self):
        """
        Projects [s, v] to [x,y,v,psi,psidot]
        Returns:
            projstate (torch.tensor): (nv, 5) projected state
        """
        svt = self._svt
        projstate = ssdot_to_simstates(self._state[:,0].unsqueeze(0),self._state[:,1].unsqueeze(0),
                svt.xpoly, svt.dxpoly, svt.ddxpoly,
                                svt.ypoly, svt.dypoly, svt.ddypoly)
        projstate = projstate[0].reshape(self._nv, 5)
        projstate[...,3:5] = to_circle(projstate[...,3:5])
        return projstate

    @property
    def relative_state(self):
        return self._relative_state(self.projected_state)

    def _relative_state(self, projstate):
        """
        Compute relative state information from [x,y,v,psi,psidot] projected state
        Args:
            projstate (torch.tensor): (nv,5) projected state
        Returns:
            relstate (torch.tensor): (nv, nv, 5) projected state, where relstate[i,j,k] = projstate[j,k] - projstate[i,k]
        """
        relstate = projstate.unsqueeze(0) - projstate.unsqueeze(1)
        relstate[...,3:5] = to_circle(relstate[...,3:5])
        return relstate
    
    @property
    def map_info(self):
        """
        Return information about environment map
        Returns:
            map_info: information about map
        """
        return self._map_info
    
    def extract_map_info(self):
        """
        Generates information about environment map
        Returns:
            map_info: information about map
        """
        map_info, _ = build_map(self._map_path)
        return map_info

    def step(self, action):
        """
        Step simulation forward 1 time-step.
        Args:
            action (torch.tensor): (nvehicles,1) accelerations
        Returns:
            observation (dict): observation dictionary
            reward (float): reward
            episode_over (bool): whether the episode has ended
            info (dict): diagnostic information useful for debugging
        """

        self._t += self._dt

        # check for proper action input
        # 1) make sure all non-nan states have actions
        # 2) set actions for all nan state to 0
        # 3) make sure final action is inside action space, if not clamp
        nns = ~torch.isnan(self._state[:,0]) 
        nna = ~torch.isnan(action[:,0]) 
        nns_na = nns & ~nna
        if torch.any(nns_na):
            raise Exception('Time: {}. Some cars receive nan actions'.format(self._t))
        action[~nna] = 0.
        if not self.action_space.contains(action):
            print('Time: {}. Warning: requested action outside of bounds, being clamped'.format(self._t))
            action = action.clamp(self._min_acc, self._max_acc)

        # euler step the state
        nextv = self._state[:,1:2] + action * self._dt
        nextv = nextv.clamp(0., np.inf) # not differentiable!
        self._state[:,0:1] = self._state[:,0:1] + 0.5*self._dt*(nextv + self._state[:,1:2])
        self._state[:,1:2] = nextv

        # see which states exceeded their maxes
        self._exceeded = (self._state[:,0] > self._svt.smax) | self._exceeded
        self._state[self._exceeded] = np.nan
        self.done = torch.all(self._exceeded)

        # see which new vehicles should spawn
        
        # TODO: add an isFree checker 
        free = torch.tensor([True] * self._nv)
        should_spawn = (self._svt.t0 < self._t) & ~self._exceeded & ~nns & free

        # for now, just spawn them
        spawned = should_spawn
        self._state[spawned,0] = 0.001 # done to avoid potential DivByZero
        self._state[spawned,1] = self._svt.v0[spawned]
        
        projstate = self.projected_state

        # update interaction graph
        self._graph.update_graph(projstate)

        # generate info
        self.info.update({
            'exceeded': np.argwhere(self._exceeded.detach().numpy()).flatten(),
            'should_spawn': np.argwhere(should_spawn.detach().numpy()).flatten(),
            'spawned': np.argwhere(spawned.detach().numpy()).flatten(),
            'raw_state': self._state.clone(),
            'action_taken': action.clone()
        })

        # generate observation
        ob = self._get_observation(projstate)
        
        # generate reward from new state and action
        reward = self._get_reward(projstate, action)
        return ob, reward, self.done, self.info   
    
    def _get_observation(self, next_state):
        """
        Generate observation
        Args:
            next_state (torch.tensor): (nv,5) projected next state
        Returns:
            observation (dict): observation with following potential keys
                state (torch.tensor): (nv*5,) projected next state
                neighbor_dict (dict): interaction graph neighbor dictionary
                relative_state (torch.tensor): (nv,nv,5) relative state information where [i,j,k] encodes state[j,k] - state[i,k]
                map_info: map information embedding
                hidden_info: external information that would usually stay hidden for training
        """
        observation = {}
        observation['state'] = next_state.reshape(-1)
        observation['neighbor_dict'] = self._graph.neighbor_dict
        if self._observe_method in [ObserveMethod.FULL, ObserveMethod.HIDDEN]:
            observation['relative_state'] = self._relative_state(next_state)
            observation['map_info'] = self.map_info
        if self._observe_method in [ObserveMethod.HIDDEN]:
            observation['hidden_info'] = self.info
        return observation
    
    def _get_reward(self, next_state, action) -> float:
        """
        Generate reward
        Args:
            next_state (torch.tensor): (nv,5) next projected state
            action (torch.tensor): (nv,) action taken
        Returns:
            reward (float): immediate reward
        """
        if self._reward_method == RewardMethod.NONE:
            return 0.
        elif self._reward_method == RewardMethod.CUSTOM:
            return self._custom_reward(next_state, action)
        else:
            raise ValueError('Invalid reward method')

    def reset(self):
        """
        Reset simulation.
        Returns:
        """

        self._state = self._svt.state0
        self._t = self._svt.minT
        self._exceeded = torch.tensor([False] * self._nv)
        self._graph.update_graph(self.projected_state)
        self.done = False

        # rendering fields
        self._state_list = []
        self._graph_list = []
        print('Environment Reset')
        return None
        #return self.projected_state.reshape(-1), {'raw_state': self._state.clone()}

    def render(self, mode='post'):
        """
        Render environment
        Args:
            mode (str): different rendering options
                live: render a live matplotlib
                file: save necessary information to file
                post: save necessary information to file, and generate video upon closing
            filestr (str): base file string to save states, graphs, and videos to
        """
        self._mode = mode # latest mode

        if mode == 'live':
            import matplotlib
            import matplotlib.pyplot as plt
            raise NotImplementedError
        if mode in ['file', 'post']:
            self._state_list.append(self.projected_state.reshape(-1))
            self._graph_list.append(self._graph.edges)

    def close(self, **kwargs):
        """
        Clean up the environment
        Args:
            filestr (str): base file string to save states, graphs, and videos to
        """ 
        if self._mode == 'live':
            import matplotlib
            import matplotlib.pyplot as plt
            plt.close()
        if self._mode in ['file', 'post']:
            filestr = kwargs.get('filestr', 'render')
            stacked_states = torch.stack(self._state_list)
            torch.save(stacked_states, filestr+'_states.pt')
            pickle.dump(self._graph_list,open(filestr+'_graphs.pkl', 'wb'))
            torch.save(self._lengths, filestr+'_lengths.pt')
            torch.save(self._widths, filestr+'_widths.pt')
            torch.save(self._xpoly, filestr+'_xpoly.pt')
            torch.save(self._ypoly, filestr+'_ypoly.pt')
            if self._mode == 'post':
                self._animate(self._map_path, stacked_states, self._lengths, self._widths, 
                                self._graph_list, filestr, **kwargs)

    def _animate(self, osm, states, lengths, widths, graphs, file, **kwargs):
        """
        Wrapper for animating simulation once finished
        Args:
            osm (str): path to .osm map file
            states (torch.tensor): (frames, nv*5) tensor of vehicle states
            lengths (torch.tensor): (nv,) array of vehicle lengths 
            widths (torch.tensor): (nv,) array of vehicle widths
            graphs (list[list[tuple]]): list of list of edges. Outer list indexes frame.
            file (str): base file string to save animation to
        """
        import matplotlib
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        from intersim.viz.animatedviz import AnimatedViz
        fps = kwargs.get('fps', 15)
        bitrate = kwargs.get('bitrate', 1800)
        enc = kwargs.get('encoder', 'ffmpeg')
        iv = kwargs.get('interval', 20)
        blit = kwargs.get('blit', True)

        Writer = animation.writers[enc]
        writer = Writer(fps=fps, bitrate=bitrate)
        fig = plt.figure()
        ax = plt.axes()
        av = AnimatedViz(ax, osm, states, lengths, widths, graphs=graphs)
        ani = animation.FuncAnimation(fig, av.animate, frames=len(states),
                        interval=iv, blit=blit, init_func=av.initfun,repeat=False)
        ani.save(file+'_ani.mp4', writer)
