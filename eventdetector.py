import numpy as np
import os as os
from scipy.signal import (find_peaks, decimate)
import scipy.spatial.distance as dist
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.patches import Ellipse
import time as time
from concurrent.futures import ProcessPoolExecutor as Executor
from concurrent.futures import as_completed
from threading import Lock
import h5py as h5py
import sys as sys
from PySide2.QtWidgets import QMessageBox
from PySide2.QtCore import Signal, Slot, QObject
import mmap
import queue
import pandas as pd

d_type = np.dtype([('time', 'f8'), ('scale', 'f8'), ('coeff', 'f8'), ('N', 'f8'), ('label', 'i8')])

def spectral_cluster_2(all_events, threshold, selectivity=1, plot=False):
    _points = np.array(list(zip(all_events['time'],all_events['scale'])))
    _dist = dist.cdist(_points, _points, 'euclidean')
    _amp = np.log(all_events['coeff']/threshold)
#     _amp = 1
    _rr = np.tile(0.5/selectivity*np.multiply(np.multiply(all_events['N'],_amp),all_events['scale']),len(all_events)).reshape(_dist.shape)
    _rr += _rr.transpose()
    
    _adjacency = _rr-_dist
    # _rr = np.tile((0.5/selectivity*np.multiply(all_events['N'],all_events['scale']))**2,len(all_events)).reshape(_dist.shape)
    # _rr += np.tile((0.5/selectivity*all_events['scale'])**2,len(all_events)).reshape(_dist.shape)
    # _adjacency = _rr - _dist
    _mask = np.greater_equal(_rr,_dist).astype(np.bool)
    if plot:
        f,a = plt.subplots(1,1,figsize=(5,5))
        a.imshow(_adjacency)
        f.savefig(f"adj0_{all_events['time'][0]:0.4f}.svg")
        a.cla()
        a.imshow(_mask)
        f.savefig(f"adj1_{all_events['time'][0]:0.4f}.svg")
    for f in np.argwhere(np.sum(_mask,axis=0) == 1).flatten():
        all_events[f]['label'] = -1
    _degree_sort = np.argsort(np.sum(_adjacency,axis=0))
    all_events = all_events[_degree_sort]
    _adjacency = _adjacency[_degree_sort,:]
    _adjacency = _adjacency[:,_degree_sort]
    _mask = _mask[_degree_sort,:]
    _mask = _mask[:,_degree_sort]
    _orders = np.arange(len(all_events))
    for n in range(len(all_events)):
        _mask[n,:n] = True
        _idx = list(np.where(_mask[n,:]==True)[0])+list(np.where(_mask[n,:]==False)[0])
        _orders = _orders[_idx]
        _mask = _mask[_idx,:]
        _mask = _mask[:,_idx]
    _mask = np.triu(_mask, k=1)
    if plot:
        a.cla()
        a.imshow(_mask)
        f.savefig(f"adj2_{all_events['time'][0]:0.4f}.svg")
    _slices = np.argwhere(np.sum(_mask,axis=0)==0).flatten()[1:]
    _islands = np.split(all_events[_orders], _slices)
    selected_events = np.array([_island[np.argmax(_island['coeff'])] for _island in _islands])
    return selected_events

def spectral_cluster(all_events, threshold, selectivity=1, plot=False):
    _points = np.array(list(zip(all_events['time'],all_events['scale'])))
    _dist = dist.cdist(_points, _points, 'euclidean')
    _amp = np.log(all_events['coeff']/threshold)
#     _amp = 1
    _rr = np.tile(0.5/selectivity*np.multiply(np.multiply(all_events['N'],_amp),all_events['scale']),len(all_events)).reshape(_dist.shape)
    _rr += _rr.transpose()
    _adjacency = _rr-_dist
    # _rr = np.tile((0.5/selectivity*np.multiply(all_events['N'],all_events['scale']))**2,len(all_events)).reshape(_dist.shape)
    # _rr += np.tile((0.5/selectivity*all_events['scale'])**2,len(all_events)).reshape(_dist.shape)
    # _adjacency = _rr - _dist
    _mask = np.greater_equal(_rr,_dist).astype(np.bool)
    if plot:
        f,a = plt.subplots(1,1,figsize=(5,5))
        a.imshow(_adjacency)
        f.savefig(f"adj0_{all_events['time'][0]:0.4f}.svg")
        a.cla()
        a.imshow(_mask)
        f.savefig(f"adj1_{all_events['time'][0]:0.4f}.svg")
    for f in np.argwhere(np.sum(_mask,axis=0) == 1).flatten():
        all_events[f]['label'] = -1
    _degree_sort = np.argsort(np.sum(_adjacency,axis=0))
    all_events = all_events[_degree_sort]
    _adjacency = _adjacency[_degree_sort,:]
    _adjacency = _adjacency[:,_degree_sort]
    _mask = _mask[_degree_sort,:]
    _mask = _mask[:,_degree_sort]
    _orders = np.arange(len(all_events))
    for n in range(len(all_events)):
        _mask[n,:n] = True
        _idx = list(np.where(_mask[n,:]==True)[0])+list(np.where(_mask[n,:]==False)[0])
        _orders = _orders[_idx]
        _mask = _mask[_idx,:]
        _mask = _mask[:,_idx]
    _mask = np.triu(_mask, k=1)
    if plot:
        a.cla()
        a.imshow(_mask)
        f.savefig(f"adj2_{all_events['time'][0]:0.4f}.svg")
    _slices = np.argwhere(np.sum(_mask,axis=0)==0).flatten()[1:]
    _islands = np.split(all_events[_orders], _slices)
    selected_events = np.array([_island[np.argmax(_island['coeff'])] for _island in _islands])
    return selected_events

def detect_islands(all_events, threshold, selectivity=1):
    all_events_t_l = all_events['time']-0.5/selectivity*np.multiply(all_events['N'],all_events['scale'])
    _index_l = np.argsort(all_events_t_l)
    all_events_t_r = all_events['time']+0.5/selectivity*np.multiply(all_events['N'],all_events['scale'])
    _index_r = np.argsort(all_events_t_r)
    all_events_overlap = all_events_t_r[_index_r[:-1]]-all_events_t_l[_index_l[1:]]
    # print(_index_l,_index_r)
    # all_events.sort(order='time')
    # all_events_t_diff = np.diff(all_events['time'])
    # all_events_dt_add = 0.5/selectivity*(np.multiply(all_events['N'][:-1],all_events['scale'][:-1])+\
                            #  np.multiply(all_events['N'][1:],all_events['scale'][1:]))
    # all_events_overlap = np.sign(all_events_dt_add - all_events_t_diff)
    _slices = np.argwhere(all_events_overlap <= 0).flatten()+1
    _islands = np.split(all_events[_index_l], _slices, axis=0)
    return _islands

def select_events(events, threshold, selectivity=1, extent=1, plot=False):
    selected_events = []
    events = np.sort(events,order='coeff')[::-1]
    while(len(events) > 0):
        _dt = np.abs(events['time'][0]-events['time'])
        _rr = 0.5*extent*events['N'][0]*events['scale'][0]+0.5*extent*events['N']*events['scale']
        _rm_i = np.argwhere(_rr-_dt > 0).flatten()
#         print(len(_rm_i))
        if len(_rm_i) > selectivity:
            selected_events.append(events[0])
#         plt.plot(_rr-_dt)
#         plt.show()
        events = np.delete(events,_rm_i)
    return np.array(selected_events, dtype=events.dtype)

def filter_events(all_events, selectivity, refine=True):
    all_events_t_l = all_events['time']-0.5/selectivity*np.multiply(all_events['N'],all_events['scale'])
    _index_l = np.argsort(all_events_t_l)
    all_events_t_r = all_events['time']+0.5/selectivity*np.multiply(all_events['N'],all_events['scale'])
    _index_r = np.argsort(all_events_t_r)
    all_events_overlap = all_events_t_r[_index_r[:-1]]-all_events_t_l[_index_l[1:]]
    # print(_index_l,_index_r)
    # all_events.sort(order='time')
    # all_events_t_diff = np.diff(all_events['time'])
    # all_events_dt_add = 0.5/selectivity*(np.multiply(all_events['N'][:-1],all_events['scale'][:-1])+\
                            #  np.multiply(all_events['N'][1:],all_events['scale'][1:]))
    # all_events_overlap = np.sign(all_events_dt_add - all_events_t_diff)
    _slices = np.argwhere(all_events_overlap <= 0).flatten()+1
    _islands = np.split(all_events[_index_l], _slices, axis=0)
    if refine:
        _islands = [_cluster for _island in _islands for _cluster in spectral_cluster(_island,selectivity,plot=False)]
    selected_events = np.array([_island[np.argmax(_island['coeff'])] for _island in _islands])
    return selected_events

def find_events(signal, wavelets, scales, pad, slice_l, thresh, selectivity, dt, log=False, plot=False):
    _events = np.empty((0,), dtype=d_type)
    # _mean, _std = [], []
    if plot:
        _cwt_list = {}
        for i,k in enumerate(wavelets.keys()):
            _cwt = np.empty((len(wavelets[k]['wavelets']), len(signal[pad:-pad])))
            if np.iscomplexobj(wavelets[k]['wavelets'][0]):
                for n, w in enumerate(wavelets[k]['wavelets']):
                    _cwt[n,:] = np.abs(np.correlate(signal, w, mode='same'))[pad:-pad]
                    _index, _ = find_peaks(_cwt[n,:], distance=wavelets[k]['N']*scales[n]/dt, height=thresh)
                    # _index, _ = find_peaks(_cwt[n,:], distance=wavelets[k]['N']*scales[n]/dt, prominence=thresh)
                    _events = np.append(_events, np.array(list(zip((slice_l+_index)*dt, \
                                                                    [scales[n]]*len(_index), \
                                                                    _cwt[n,_index], \
                                                                    [wavelets[k]['N']]*len(_index), \
                                                                    [i]*len(_index))), dtype=d_type), axis=0)
            else:
                for n, w in enumerate(wavelets[k]['wavelets']):
                    _cwt[n,:] = (0.5*np.correlate(signal, w, mode='same'))[pad:-pad]
                    _cwt[n,:] += np.abs(_cwt[n,:])
                    # print(f'mean:{np.mean(_cwt[n,:])}, std:{np.std(_cwt[n,:])}')
                    _index, _ = find_peaks(_cwt[n,:], distance=wavelets[k]['N']*scales[n]/dt, height=thresh)
                    # _index, _ = find_peaks(_cwt[n,:], distance=wavelets[k]['N']*scales[n]/dt, prominence=thresh)
                    _events = np.append(_events, np.array(list(zip((slice_l+_index)*dt, \
                                                                    [scales[n]]*len(_index), \
                                                                    _cwt[n,_index], \
                                                                    [wavelets[k]['N']]*len(_index), \
                                                                    [i]*len(_index))), dtype=d_type), axis=0)
            _cwt_list[k] = (_cwt)
            # print(_events)
        return _events, _cwt_list, slice_l*dt, _cwt.shape[1]*dt
            # fig, ax1 = plt.subplots(1,1,figsize=(14,4))
            # ax2 = plt.twinx(ax1)
            # ax1.yaxis.tick_right()
            # ax2.yaxis.tick_left()
            # ax2.yaxis.set_label_position('left')
            # ax1.imshow(_cwt, extent=[slice_l*dt, _cwt.shape[1]*dt, scales[0]*1e3, scales[-1]*1e3], origin='lower', cmap='inferno')
            # ax1.set_yticks([])
            # ax1.axis('auto')
            # ax2.set_ylabel(f'\u0394t [ms]')
            # [ax2.add_artist(Ellipse((e['time'], e['scale']*1e3), width=1/selectivity*e['scale'], height=1/selectivity*e['scale']*1e3, clip_on=True, zorder=10, linewidth=1,
            #         edgecolor=(0,1,1,0.2), facecolor=(1, 0, 0, .025))) for e in _events]
            # ax2.plot(_events['time'], _events['scale']*1e3, '.', color='green')
            # ax2.set_ylim(scales[0]*1e3,scales[-1]*1e3)
            # ax1.set_xlim(slice_l*dt, _cwt.shape[1]*dt)
            # if log:
            #     ax2.set_yscale('log')
            # plt.show()
    else:
        for i,k in enumerate(wavelets.keys()):
            if np.iscomplexobj(wavelets[k]['wavelets'][0]):
                for n, w in enumerate(wavelets[k]['wavelets']):
                    _cwt = np.abs(np.correlate(signal, w, mode='same'))[pad:-pad]
                    _index, _ = find_peaks(_cwt, distance=wavelets[k]['N']*scales[n]/dt, height=thresh)
                    # _index, _ = find_peaks(_cwt, distance=wavelets[k]['N']*scales[n]/dt, prominence=thresh)
                    _events = np.append(_events, np.array(list(zip((slice_l+_index)*dt, \
                                                                    [scales[n]]*len(_index), \
                                                                    _cwt[_index], \
                                                                    [wavelets[k]['N']]*len(_index), \
                                                                    [i]*len(_index))), dtype=d_type), axis=0)
            else:
                for n, w in enumerate(wavelets[k]['wavelets']):
                    _cwt = (0.5*np.correlate(signal, w, mode='same'))[pad:-pad]
                    # print(np.max(_cwt))
                    _cwt += np.abs(_cwt)
#                     _std = np.std(_cwt)
                    # _mean.append(np.mean(_cwt))
                    # _std.append(np.std(_cwt))
#                     with thread_lock:
#                         print(f"std at {scales[n]}: {_std}")
                    _index, _ = find_peaks(_cwt, distance=wavelets[k]['N']*scales[n]/dt, height=thresh)
                    # _index, _ = find_peaks(_cwt, distance=wavelets[k]['N']*scales[n]/dt, prominence=thresh)
                    _events = np.append(_events, np.array(list(zip((slice_l+_index)*dt, \
                                                                    [scales[n]]*len(_index), \
                                                                    _cwt[_index], \
                                                                    [wavelets[k]['N']]*len(_index), \
                                                                    [i]*len(_index))), dtype=d_type), axis=0)
        # print(_events)
        # print(f'mean:{np.mean(_mean)}, std:{np.mean(_std)}')
        return _events

class eventdetector(QObject):
    progress = Signal(int)
    started = Signal(bool)
    showevents = Signal(object)
    drawcwt = Signal(object)
    message = Signal(str)
    def __init__(self):
        QObject.__init__(self)
        self.ts = None
        self.scales = {}
        self.window = {}
        self.resolution = 10
        self.threshold = 1
        self.selectivity = 1
        self.extent = 1
        self.chunksize = 1000
        self.log = True
        self.refine = True
        self.save = False
        self.save_cwt = True
        self.cwt_plot = True
        self.selected_events = []

    def analyze_trace(self, wavelets):
        self.started.emit(True)
        if self.log:
            scales = np.logspace(np.log10(self.scales['min']), np.log10(self.scales['max']), self.scales['count'], dtype=np.float64)
        else:
            scales = np.linspace(self.scales['min'], self.scales['max'], self.scales['count'], dtype=np.float64)
        pad = max([max([len(w) for w in wavelets[k]['wavelets']]) for k in wavelets.keys()])
        self.ts.dt = min(scales)
        self.ts.active = True
        dt = min(scales)/self.resolution
        if self.ts.type == 'ptu':       # TimeHarp Traces
            self.ts.load(plot=True, relim=False)
            self.ts.updateplot.emit(False)
            self.message.emit('Detecting events...')
            with h5py.File(os.path.splitext(self.ts.filename)[0]+'.hdf5', 'a') as f:
                print(f.keys())
                if f'{dt:010.6f}' not in f.keys():
                    self.ts.dt = dt
                    self.ts.processHT2()
                _xlim = [0, -1]
                signal = f[f'{dt:010.6f}']
                print(signal['time'][0],signal['time'][-1])
                if (self.window['l'] == 0):
                    _xlim[0] = 0
                else:
                    _xlim[0] = np.where(signal['time'][:] >= self.window['l'])[0][0]
                if (self.window['r'] == -1):
                    _xlim[1] = -1
                else:
                    _xlim[1] = np.where(signal['time'][:] <= self.window['r'])[0][-1]
                
                slices = list(range(self.chunksize, _xlim[1]-_xlim[0]+1, self.chunksize))
                if _xlim[0] < pad:
                    if _xlim[1]+pad > len(signal['time'][:])-1:
                        slices_l = [0]+[s-pad for s in slices]
                        slices_r = [s+pad for s in slices]+[_xlim[1]-_xlim[0]]
                        slices  = [0]+slices
                        signals = [signal['count'][_xlim[0]+slices_l[n]:_xlim[0]+slices_r[n]] for n in range(len(slices_l))]
                        signals[0] = np.pad(signals[0], (pad,0), mode='constant', constant_values=(0,0))
                        signals[-1] = np.pad(signals[-1], (0,pad), mode='constant', constant_values=(0,0))
                    else:
                        slices_l = [0]+[s-pad for s in slices]
                        slices_r = [s+pad for s in slices]+[_xlim[1]-_xlim[0]+pad]
                        slices  = [0]+slices
                        signals = [signal['count'][_xlim[0]+slices_l[n]:_xlim[0]+slices_r[n]] for n in range(len(slices_l))]
                        signals[0] = np.pad(signals[0], (pad,0), mode='constant', constant_values=(0,0))
                else:
                    if _xlim[1]+pad > len(signal['time'][:])-1:
                        slices_l = [-pad]+[s-pad for s in slices]
                        slices_r = [s+pad for s in slices]+[_xlim[1]-_xlim[0]]
                        slices  = [0]+slices
                        signals = [signal['count'][_xlim[0]+slices_l[n]:_xlim[0]+slices_r[n]] for n in range(len(slices_l))]
                        signals[-1] = np.pad(signals[-1], (0,pad), mode='constant', constant_values=(0,0))
                    else:
                        slices_l = [-pad]+[s-pad for s in slices]
                        slices_r = [s+pad for s in slices]+[_xlim[1]-_xlim[0]+pad]
                        slices  = [0]+slices
                        signals = [signal['count'][_xlim[0]+slices_l[n]:_xlim[0]+slices_r[n]] for n in range(len(slices_l))]
                n, total = 0, len(signals)
                # print(total)
                _events = []
                self.selected_events = []
        #         find_events(signal['count'][_xlim[0]:_xlim[1]], wavelets, scales, pad, slices[0], thresh, selectivity, dt, log=log, plot=True)
        #         return
                with Executor() as e:
                    # _events = self.find_events(signals[0], wavelets, scales, pad, slices[0], threshold, selectivity, dt, log=log, plot=False)
                    # print('events',_events)
                    _futures = [e.submit(find_events, s, wavelets, scales, pad, slices[m], self.threshold, self.selectivity, dt, log=self.log, plot=self.cwt_plot) for m,s in enumerate(signals)]
                    for _f in as_completed(_futures):
                        if self.cwt_plot:
                            _result = _f.result()
                            _events.append(_result[0])
                            _cwt_list = [_result[1],_result[2]+signal['time'][_xlim[0]],_result[3]]
                            self.drawcwt.emit(_cwt_list)
                        else:
                            _events.append(_f.result())
                        n += 1
                        progress = 100*n/total
                        self.progress.emit(progress)
                    _events = np.concatenate(tuple(_events), axis=0)
        #             _events['time'] += signal['time'][_xlim[0]]*1e-12
        #             return _events
                    if len(_events) > 0:
                        for i,k in enumerate(wavelets.keys()):
                            if (np.count_nonzero(_events['label']==i) == 0):
                                continue
                            _islands = detect_islands(_events[_events['label'] == i],self.threshold)
                            n, total = 0, len(_islands)
                            if self.refine:
        #                         _futures = [e.submit(spectral_cluster,_island,thresh,selectivity,cwt_plot) for _island in _islands]
                                _futures = [e.submit(select_events,_island,self.threshold,self.selectivity,self.extent,self.cwt_plot) for _island in _islands]
                                for _f in as_completed(_futures):
                                    n += 1
                                    self.selected_events.append(_f.result())
                                    progress = 100*n/total
                                    self.progress.emit(progress)
                            else:
                                self.selected_events.append([np.array(_island[np.argmax(_island['coeff'])]) for _island in _islands])
                        self.selected_events = np.concatenate(tuple(self.selected_events), axis=0)
                        if len(wavelets.keys()) > 1:
        #                     self.selected_events = [spectral_cluster(_island,thresh,selectivity,cwt_plot) for _island in detect_islands(self.selected_events,thresh,selectivity)]
                            self.selected_events = [select_events(_island,self.threshold,plot=self.cwt_plot) for _island in detect_islands(self.selected_events,self.threshold)]
                            # self.selected_events = filter_events(self.selected_events, selectivity=selectivity, refine=refine)
                            self.selected_events = np.concatenate(tuple(self.selected_events), axis=0)
                        self.selected_events['time'] += signal['time'][_xlim[0]]
                        if self.save:
                            if f'events/{self.threshold:.2f}' in f[f'{dt:010.6f}'].keys():
                                del f[f'{dt:010.6f}'][f'events/{self.threshold:.2f}']
                            for s in self.selected_events.dtype.names:
                                f[f'{dt:010.6f}'].create_dataset(f'events/{self.threshold:.2f}/{s}', data=self.selected_events[s])
        elif (self.ts.type == 'nanopore'):  # Nanopore Traces
            self.ts.dt = dt
            self.ts.load(plot=True, relim=False)
            # self.ts.trace = self.ts.trace.rolling(window=int(dt/self.ts.dt), win_type='hamming').mean()
            # self.ts.trace = pd.DataFrame({'time':self.ts.trace['time'][::int(dt/self.ts.nanopore_globres)],
            #                               'current':decimate(self.ts.trace['current'],int(dt/self.ts.nanopore_globres)),
            #                               'voltage':decimate(self.ts.trace['voltage'],int(dt/self.ts.nanopore_globres))})
            _xlim = [0, -1]
            signal = self.ts.trace
            self.message.emit('Detecting events...')
            if (self.window['l'] == 0):
                _xlim[0] = 0
            else:
                _xlim[0] = np.where(signal['time'] >= self.window['l'])[0][0]
            if (self.window['r'] == -1):
                _xlim[1] = -1
            else:
                _xlim[1] = np.where(signal['time'] <= self.window['r'])[0][-1]
            
            slices = list(range(self.chunksize, _xlim[1]-_xlim[0]+1, self.chunksize))
            if _xlim[0] < pad:
                if _xlim[1]+pad > len(signal['time'])-1:
                    slices_l = [0]+[s-pad for s in slices]
                    slices_r = [s+pad for s in slices]+[_xlim[1]-_xlim[0]]
                    slices  = [0]+slices
                    signals = [signal['current'][_xlim[0]+slices_l[n]:_xlim[0]+slices_r[n]] for n in range(len(slices_l))]
                    signals[0] = np.pad(signals[0], (pad,0), mode='constant', constant_values=(0,0))
                    signals[-1] = np.pad(signals[-1], (0,pad), mode='constant', constant_values=(0,0))
                else:
                    slices_l = [0]+[s-pad for s in slices]
                    slices_r = [s+pad for s in slices]+[_xlim[1]-_xlim[0]+pad]
                    slices  = [0]+slices
                    signals = [signal['current'][_xlim[0]+slices_l[n]:_xlim[0]+slices_r[n]] for n in range(len(slices_l))]
                    signals[0] = np.pad(signals[0], (pad,0), mode='constant', constant_values=(0,0))
            else:
                if _xlim[1]+pad > len(signal['time'])-1:
                    slices_l = [-pad]+[s-pad for s in slices]
                    slices_r = [s+pad for s in slices]+[_xlim[1]-_xlim[0]]
                    slices  = [0]+slices
                    signals = [signal['current'][_xlim[0]+slices_l[n]:_xlim[0]+slices_r[n]] for n in range(len(slices_l))]
                    signals[-1] = np.pad(signals[-1], (0,pad), mode='constant', constant_values=(0,0))
                else:
                    slices_l = [-pad]+[s-pad for s in slices]
                    slices_r = [s+pad for s in slices]+[_xlim[1]-_xlim[0]+pad]
                    slices  = [0]+slices
                    signals = [signal['current'][_xlim[0]+slices_l[n]:_xlim[0]+slices_r[n]] for n in range(len(slices_l))]
            n, total = 0, len(signals)
            # print(total)
            _events = []
            self.selected_events = []
    #         find_events(signal['count'][_xlim[0]:_xlim[1]], wavelets, scales, pad, slices[0], thresh, selectivity, dt, log=log, plot=True)
    #         return
            with Executor() as e:
                # _events = self.find_events(signals[0], wavelets, scales, pad, slices[0], threshold, selectivity, dt, log=log, plot=False)
                # print('events',_events)
                _futures = [e.submit(find_events, s, wavelets, scales, pad, slices[m], self.threshold, self.selectivity, dt, log=self.log, plot=self.cwt_plot) for m,s in enumerate(signals)]
                for _f in as_completed(_futures):
                    if self.cwt_plot:
                        _result = _f.result()
                        _events.append(_result[0])
                        _cwt_list = [_result[1],_result[2]+signal['time'][_xlim[0]],_result[3]]
                        self.drawcwt.emit(_cwt_list)
                    else:
                        _events.append(_f.result())
                    n += 1
                    progress = 100*n/total
                    self.progress.emit(progress)
                _events = np.concatenate(tuple(_events), axis=0)
    #             _events['time'] += signal['time'][_xlim[0]]*1e-12
    #             return _events
                if len(_events) > 0:
                    for i,k in enumerate(wavelets.keys()):
                        if (np.count_nonzero(_events['label']==i) == 0):
                            continue
                        _islands = detect_islands(_events[_events['label'] == i],self.threshold)
                        n, total = 0, len(_islands)
                        if self.refine:
    #                         _futures = [e.submit(spectral_cluster,_island,thresh,selectivity,cwt_plot) for _island in _islands]
                            _futures = [e.submit(select_events,_island,self.threshold,self.selectivity,self.extent,self.cwt_plot) for _island in _islands]
                            for _f in as_completed(_futures):
                                n += 1
                                self.selected_events.append(_f.result())
                                progress = 100*n/total
                                self.progress.emit(progress)
                        else:
                            self.selected_events.append([np.array(_island[np.argmax(_island['coeff'])]) for _island in _islands])
                    self.selected_events = np.concatenate(tuple(self.selected_events), axis=0)
                    if len(wavelets.keys()) > 1:
    #                     self.selected_events = [spectral_cluster(_island,thresh,selectivity,cwt_plot) for _island in detect_islands(self.selected_events,thresh,selectivity)]
                        self.selected_events = [select_events(_island,self.threshold,plot=self.cwt_plot) for _island in detect_islands(self.selected_events,self.threshold)]
                        # self.selected_events = filter_events(self.selected_events, selectivity=selectivity, refine=refine)
                        self.selected_events = np.concatenate(tuple(self.selected_events), axis=0)
                    self.selected_events['time'] += signal['time'][_xlim[0]]
                    if self.save:
                        print(self.selected_events)
        if len(self.selected_events) == 0:
            self.selected_events = np.empty((0,), dtype=d_type)
        self.started.emit(False)
        self.message.emit('Displaying events...')
        self.showevents.emit(self.selected_events)
        self.ts.updateplot.emit(False)
        return self.selected_events

    def analyze_chunk(self, chunk, wavelets):
        _events = np.empty((0,), dtype=d_type)
        for i,k in enumerate(wavelets.keys()):
            if np.iscomplexobj(wavelets[k]['wavelets']):
                _cwt = np.abs(np.correlate(signal, wavelets[k]['wavelets'], mode='same'))[pad:-pad]
                _index, _ = find_peaks(_cwt, distance=wavelets[k]['N']*scales[n]/dt, height=thresh)
                _events = np.append(_events, np.array(list(zip((slice_l+_index)*dt, \
                                                                [scales[n]]*len(_index), \
                                                                _cwt[_index], \
                                                                [wavelets[k]['N']]*len(_index), \
                                                                [i]*len(_index))), dtype=d_type), axis=0)
            else:
                _cwt = (0.5*np.correlate(signal, wavelets[k]['wavelets'], mode='same'))[pad:-pad]
                _cwt += np.abs(_cwt)
                _index, _ = find_peaks(_cwt, distance=wavelets[k]['N']*scales[n]/dt, height=thresh)
                _events = np.append(_events, np.array(list(zip((slice_l+_index)*dt, \
                                                                [scales[n]]*len(_index), \
                                                                _cwt[_index], \
                                                                [wavelets[k]['N']]*len(_index), \
                                                                [i]*len(_index))), dtype=d_type), axis=0)
        # print(_events)
        return _events

    def analyze_trace_rt(self, wavelets):
        self.started.emit(True)
        if self.log:
            scales = np.logspace(np.log10(self.scales['min']), np.log10(self.scales['max']), self.scales['count'], dtype=np.float64)
        else:
            scales = np.linspace(self.scales['min'], self.scales['max'], self.scales['count'], dtype=np.float64)
        dt = min(scales)/self.resolution
        pad = max([max([len(w) for w in wavelets[k]['wavelets']]) for k in wavelets.keys()])
        with h5py.File(os.path.splitext(self.ptufile.filename)[0]+'.hdf5', 'a') as f:
            print(f.keys())
            if f'{dt:010.6f}' not in f.keys():
                self.ptufile.binsize = dt
                self.ptufile.processHT2()
            _xlim = [0, -1]
            signal = f[f'{dt:010.6f}']
            if (self.window['l'] == 0):
                _xlim[0] = 0
            else:
                _xlim[0] = np.where(signal['time'][:] >= self.window['l'])[0][0]
            if (self.window['r'] == -1):
                _xlim[1] = -1
            else:
                _xlim[1] = np.where(signal['time'][:] <= self.window['r'])[0][-1]
            
            slices = list(range(self.chunksize, len(signal['count'][_xlim[0]:_xlim[1]]), self.chunksize))
            slices_l = [0]+[s-pad for s in slices]
            slices_r = [s+pad for s in slices]+[len(signal['count'][_xlim[0]:_xlim[1]])-1]
            slices  = [0]+slices
            signals = [signal['count'][_xlim[0]+slices_l[n]:_xlim[0]+slices_r[n]] for n in range(len(slices_l))]
            signals[0] = np.pad(signals[0], (pad,0), mode='constant', constant_values=(0,0))
            signals[-1] = np.pad(signals[-1], (0,pad), mode='constant', constant_values=(0,0))
            n, total = 0, len(signals)
            # print(total)
            _events = []
            self.selected_events = []
    #         find_events(signal['count'][_xlim[0]:_xlim[1]], wavelets, scales, pad, slices[0], thresh, selectivity, dt, log=log, plot=True)
    #         return
            with Executor() as e:
                # _events = self.find_events(signals[0], wavelets, scales, pad, slices[0], threshold, selectivity, dt, log=log, plot=False)
                # print('events',_events)
                _futures = [e.submit(find_events, s, wavelets, scales, pad, slices[m], self.threshold, self.selectivity, dt, log=self.log, plot=self.cwt_plot) for m,s in enumerate(signals)]
                for _f in as_completed(_futures):
                    if self.cwt_plot:
                        _result = _f.result()
                        _events.append(_result[0])
                        _cwt_list = [_result[1],_result[2]+signal['time'][_xlim[0]]]
                        self.drawcwt.emit(_cwt_list)
                    else:
                        _events.append(_f.result())
                    n += 1
                    progress = 100*n/total
                    self.progress.emit(progress)
                _events = np.concatenate(tuple(_events), axis=0)
    #             _events['time'] += signal['time'][_xlim[0]]*1e-12
    #             return _events
                if len(_events) > 0:
                    for i,k in enumerate(wavelets.keys()):
                        if (np.count_nonzero(_events['label']==i) == 0):
                            break
                        _islands = detect_islands(_events[_events['label'] == i],self.threshold)
                        n, total = 0, len(_islands)
                        if self.refine:
    #                         _futures = [e.submit(spectral_cluster,_island,thresh,selectivity,cwt_plot) for _island in _islands]
                            _futures = [e.submit(select_events,_island,self.threshold,self.selectivity,self.extent,self.cwt_plot) for _island in _islands]
                            for _f in as_completed(_futures):
                                n += 1
                                self.selected_events.append(_f.result())
                                progress = 100*n/total
                                self.progress.emit(progress)
                        else:
                            self.selected_events.append([np.array(_island[np.argmax(_island['coeff'])]) for _island in _islands])
                    self.selected_events = np.concatenate(tuple(self.selected_events), axis=0)
                    if len(wavelets.keys()) > 1:
    #                     self.selected_events = [spectral_cluster(_island,thresh,selectivity,cwt_plot) for _island in detect_islands(self.selected_events,thresh,selectivity)]
                        self.selected_events = [select_events(_island,self.threshold,self.cwt_plot) for _island in detect_islands(self.selected_events,self.threshold)]
                        # self.selected_events = filter_events(self.selected_events, selectivity=selectivity, refine=refine)
                        self.selected_events = np.concatenate(tuple(self.selected_events), axis=0)
                    self.selected_events['time'] += signal['time'][_xlim[0]]
                    if self.save:
                        if f'events/{self.threshold:.2f}' in f[f'{dt:010.6f}'].keys():
                            del f[f'{dt:010.6f}'][f'events/{self.threshold:.2f}']
                        for s in self.selected_events.dtype.names:
                            f[f'{dt:010.6f}'].create_dataset(f'events/{self.threshold:.2f}/{s}', data=self.selected_events[s])
            
        if len(self.selected_events) == 0:
            self.selected_events = np.empty((0,), dtype=d_type)
        self.started.emit(False)
        self.showevents.emit(self.selected_events)
        return self.selected_events
