# animatedviz.py
from numpy import pi
import numpy as np
import torch
import matplotlib
from matplotlib import cm
import matplotlib.patches
import matplotlib.transforms
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from intersim.viz.utils import batched_rotate_around_center, draw_map_without_lanelet, build_map

import os

opj = os.path.join

def animate(osm, cagent_idx, bagent_idx, prev_state, gt_state, sim_state, lengths, widths, graphs=None, filestr='render', **kwargs):
    """
    Wrapper for animating simulation once finished
    Args:
        osm (str): path to .osm map file
        states (torch.tensor): (frames, nv, 5) tensor of vehicle states
        lengths (torch.tensor): (nv,) array of vehicle lengths 
        widths (torch.tensor): (nv,) array of vehicle widths
        graphs (list[list[tuple]]): list of list of edges. Outer list indexes frame.
        filestr (str): base file string to save animation to
    """
    fps = kwargs.get('fps', 10)
    bitrate = kwargs.get('bitrate', 1800)
    enc = kwargs.get('encoder', 'ffmpeg')
    iv = kwargs.get('interval', 20)
    blit = kwargs.get('blit', True)
    Writer = animation.writers[enc]
    writer = Writer(fps=fps, bitrate=bitrate)
    fig = plt.figure(figsize=(10, 6))
    ax = plt.axes()
    av = AnimatedViz(ax, osm, cagent_idx, bagent_idx, prev_state, gt_state, sim_state, lengths, widths, graphs=graphs)
    ani = animation.FuncAnimation(fig, av.animate, frames=len(gt_state),
                    interval=iv, blit=blit, init_func=av.initfun, repeat=False)
    ani.save(filestr+'_ani.mp4', writer)

class AnimatedViz:
    '''
    Animated visualization of a state sequence
    '''

    def __init__(self, ax, osm, cagent_idx, bagent_idx, prev_state, gt_state, sim_state, lengths, widths, graphs=None):
        '''
        Args:
            ax (plt.Axis): matplotlib Axis
            osm (str): osm file name
            states (torch.tensor): (T, nv, 5) states
            lengths (torch.tensor): (nv,) car lengths
            widths (torch.tensor): (nv,) car widths
            graphs (list of list of tuples): graphs[i][j] is the jth edge tuple in the ith frame
        ''' 

        assert prev_state.ndim == 3, 'Invalid state dim {} in AnimatedViz. Should be (T, nv, 5)'.format(prev_state.shape)
        assert gt_state.ndim == 3, 'Invalid state dim {} in AnimatedViz. Should be (T, nv, 5)'.format(gt_state.shape)
        self._T = gt_state.shape[0]
        self._nv = gt_state.shape[1]

        self.cagent_idx = cagent_idx
        self.bagent_idx = bagent_idx

        self._ax = ax
        self._osm = osm
        self._map_info, self._point_dict = build_map(osm)
        self.prev_x = prev_state[...,0].detach().numpy()
        self.prev_y = prev_state[...,1].detach().numpy()
        self.prev_psi = prev_state[...,3].detach().numpy()
        self.gt_x = gt_state[...,0].detach().numpy()
        self.gt_y = gt_state[...,1].detach().numpy()
        self.gt_psi = gt_state[...,3].detach().numpy()
        self.sim_x = sim_state[...,0].detach().numpy()
        self.sim_y = sim_state[...,1].detach().numpy()
        self.sim_psi = sim_state[...,3].detach().numpy()

        self._lengths = lengths.detach().numpy()
        self._widths = widths.detach().numpy()
        
        self._graphs = graphs
        
    @property
    def lines(self):
        return self._carrects + self._edges + [self._text]

    def initfun(self):
        ax = self._ax 

        draw_map_without_lanelet(self._map_info, self._point_dict, ax)

        # init car patches
        carrects = []
        cmap = cm.get_cmap('jet')
        self.car_colors = cmap(np.linspace(0,1,num=self._nv))
        self.car_colors = ["green", "red", "blue", "magenta"]
        for i in range(self._nv):
            rectpts = np.array([(-1.,-1.), (1.,-1), (1.,1.), (-1.,1.)])
            rect = matplotlib.patches.Polygon(rectpts, closed=True, color=self.car_colors[i], zorder=2.5, ec='k')
            ax.add_patch(rect)
            carrects.append(rect)
        self._carrects = carrects
        self._edges = []
        self._text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

        return self.lines

    def animate(self, i):
        '''
        Args:
            i (int): animation timestep
        '''
        ax = self._ax
        alpha = 0.3
        s = 4
        if i == 0:
            # draw prev traj points
            for i in self.cagent_idx:
                i = int(i)
                ax.scatter(self.prev_x[:, i], self.prev_y[:, i], color=self.car_colors[i], s=s, label=f"dog_{i}", alpha=alpha)
                ax.scatter(self.gt_x[:, i], self.gt_y[:, i], color=self.car_colors[i], s=s, alpha=alpha)
                ax.scatter(self.sim_x[:, i], self.sim_y[:, i], color=self.car_colors[i], s=s+1, marker='+')
            for i in self.bagent_idx:
                i = int(i)
                ax.scatter(self.prev_x[:, i], self.prev_y[:, i], color=self.car_colors[i], s=s, label=f"sheep_{i}", alpha=alpha)
                ax.scatter(self.gt_x[:, i], self.gt_y[:, i], color=self.car_colors[i], s=s, alpha=alpha)
                ax.scatter(self.sim_x[:, i], self.sim_y[:, i], color=self.car_colors[i], s=s+1, marker='+')
            ax.legend()

        x, y, lengths, widths = self.sim_x, self.sim_y, self._lengths, self._widths
        #print(x.shape) # (40, 4)
        
        psi = self.sim_psi

        nni = ~np.isnan(x[i])
        x = x[i, nni]
        y = y[i, nni]
        lengths = lengths[nni]
        widths = widths[nni]
        psi = psi[i, nni]

        self._text.set_text('i=%d' % i)
            
        T = self._T

        i = min(T-1, i)

        lowleft = np.stack([x - lengths / 2., y - widths / 2.], axis=-1)
        lowright = np.stack([x + lengths / 2., y - widths / 2.], axis=-1)
        upright = np.stack([x + lengths / 2., y + widths / 2.], axis=-1)
        upleft = np.stack([x - lengths / 2., y + widths / 2.], axis=-1)

        rotcorners = batched_rotate_around_center(np.stack([lowleft, lowright, upright, upleft],axis=1), 
                np.stack([x, y], axis=-1), yaw=psi)

        all_corners = np.stack([np.array([(-1.,-1.), (1.,-1), (1.,1.), (-1.,1.)])]*self._nv)
        all_corners[nni] = rotcorners

        for corners, carrect in zip(all_corners, self._carrects):
            carrect.set_xy(corners)
        
        edges = []
        if self._graphs is not None:
            for edge in self._edges:
                edge.remove()
            
            graph = self._graphs[i]
            for e in graph:
                stidx, enidx = e

                if np.isnan([
                    self._x[i, stidx], self._y[i, stidx],
                    self._x[i, enidx], self._y[i, enidx],
                    ]).any():
                    continue

                #arrow_func = matplotlib.patches.ConnectionPatch if (enidx, stidx) in graph else matplotlib.patches.Arrow
                ars = '<|-|>' if (enidx, stidx) in graph else '-|>'
                arrow = matplotlib.patches.FancyArrowPatch(posA = (self._x[i,stidx], self._y[i,stidx]),
                    posB = (self._x[i,enidx], self._y[i,enidx]),
                    arrowstyle=ars, mutation_scale=15, color='w', zorder=2.9, ec='k',)

                # arrow = matplotlib.patches.Arrow(self._x[i,stidx], self._y[i,stidx], 
                #                                  self._x[i,enidx] - self._x[i,stidx], 
                #                                  self._y[i,enidx] - self._y[i,stidx],  
                #                                  width=3.0, color='c', zorder=0.1, ec='k')

                ax.add_patch(arrow)
                edges.append(arrow)
        self._edges = edges
        return self.lines
