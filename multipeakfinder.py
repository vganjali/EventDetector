import numpy as np
import os as os
from scipy.signal import find_peaks
import scipy.spatial.distance as dist
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import time as time
from concurrent.futures import ProcessPoolExecutor as Executor
from concurrent.futures import as_completed
import h5py as h5py
import sys as sys

def progress(count, total, title='', status='', length=50):
    per = count/total
    sys.stdout.write(f" [{'#'*round(per*length)+'-'*(length-round(per*length))}] {per*100:3.1f}% | {status:30s} \r")
    sys.stdout.flush()

def spectral_cluster(all_events, selectivity=1, plot=False):
    _points = np.array(list(zip(all_events['time'],all_events['scale'])))
    _dist = dist.cdist(_points, _points, 'sqeuclidean')
    _rr = np.tile(0.5/selectivity*np.multiply(all_events['N'],all_events['scale']),len(all_events)).reshape(_dist.shape)
    _rr += _rr.transpose()
    _adjacency = _rr**2-_dist
    # _rr = np.tile((0.5/selectivity*np.multiply(all_events['N'],all_events['scale']))**2,len(all_events)).reshape(_dist.shape)
    # _rr += np.tile((0.5/selectivity*all_events['scale'])**2,len(all_events)).reshape(_dist.shape)
    # _adjacency = _rr - _dist
    _mask = np.greater_equal(_rr**2,_dist).astype(np.bool)
    for f in np.argwhere(np.sum(_mask,axis=0) == 1).flatten():
        all_events[f]['name'] = -1
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
    _slices = np.argwhere(np.sum(_mask,axis=0)==0).flatten()[1:]
    _islands = np.split(all_events[_orders], _slices)
    return _islands

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
    d_type = np.dtype([('time', 'f8'), ('scale', 'f8'), ('coeff', 'f8'), ('N', 'f8'), ('name', 'i8')])
    _events = np.empty((0,), dtype=d_type)
    if plot:
        _cwt_list = {}
        for k in wavelets.keys():
            _cwt = np.empty((len(wavelets[k]['wavelets']), len(signal[pad:-pad])))
            for n, w in enumerate(wavelets[k]['wavelets']):
                _cwt[n,:] = (0.5*np.correlate(signal, w, mode='same')*np.sqrt(dt))[pad:-pad]
                _cwt[n,:] += np.abs(_cwt[n,:])
                _index, _ = find_peaks(_cwt[n,:], distance=wavelets[k]['N']*scales[n]/dt, height=thresh)
                _events = np.append(_events, np.array(list(zip((slice_l+_index)*dt, \
                                                                [scales[n]]*len(_index), \
                                                                _cwt[n,_index], \
                                                                [wavelets[k]['N']]*len(_index), \
                                                                [k]*len(_index))), dtype=d_type), axis=0)
            _cwt_list[k] = (_cwt)
        return _events, _cwt_list
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
            for n, w in enumerate(wavelets[k]['wavelets']):
                _cwt = (0.5*np.correlate(signal, w, mode='same')*np.sqrt(dt))[pad:-pad]
                _cwt += np.abs(_cwt)
                _index, _ = find_peaks(_cwt, distance=wavelets[k]['N']*scales[n]/dt, height=thresh)
                _events = np.append(_events, np.array(list(zip((slice_l+_index)*dt, \
                                                                [scales[n]]*len(_index), \
                                                                _cwt[_index], \
                                                                [wavelets[k]['N']]*len(_index), \
                                                                [i]*len(_index))), dtype=d_type), axis=0)
    
        return _events

def analyze_trace(filename, wavelets, scales, xlim, resolution, thresh, selectivity, chunksize, log=True, save=False, plot=True, cwt_plot=False, image_fmt='tiff'):
    dt = min(scales)/resolution
    pad = max([max([len(w) for w in wavelets[k]['wavelets']]) for k in wavelets.keys()])
    with h5py.File(filename+'.hdf5', 'a') as f:
        _xlim = [0, -1]
        signal = f[f'{dt:010.6f}']
        if (xlim[0] == 0):
            _xlim[0] = 0
        else:
            _xlim[0] = np.where(signal['time'][:] >= xlim[0]*1e12)[0][0]
        if (xlim[1] == -1):
            _xlim[1] = -1
        else:
            _xlim[1] = np.where(signal['time'][:] <= xlim[1]*1e12)[0][-1]
        
        if plot:
            fig, ax = plt.subplots(2,1,figsize=(10,5), dpi=120, sharex=True)
            plt.subplots_adjust(hspace=0)
            ax[0].plot(signal['time'][_xlim[0]:_xlim[1]]*1e-12, signal['count'][_xlim[0]:_xlim[1]], color='black', linewidth=0.5)
            ax[0].set_title(filename)
            ax[0].set_ylabel(f'signal [cnts/{dt*1e3:.3f}ms]')
            ax[0].set_ylim(bottom=0)
            plt.tight_layout()
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.1)
        slices = list(range(chunksize, len(signal['count'][_xlim[0]:_xlim[1]]), chunksize))
        slices_l = [0]+[s-pad for s in slices]
        slices_r = [s+pad for s in slices]+[len(signal['count'][_xlim[0]:_xlim[1]])-1]
        slices  = [0]+slices
        signals = [signal['count'][_xlim[0]+slices_l[n]:_xlim[0]+slices_r[n]] for n in range(len(slices_l))]
        signals[0] = np.pad(signals[0], (pad,0), mode='constant', constant_values=(0,0))
        signals[-1] = np.pad(signals[-1], (0,pad), mode='constant', constant_values=(0,0))
        total = len(signals)
        n, t0 = 0, time.time()
        _events = []
        with Executor() as e:
            progress(n,total,status=f'analyzing {filename}, remaining time: ...',length=50)
            # find_events(signal['count'], wavelets, scales, pad, slices[0], thresh, selectivity, dt, log=log, plot=True)
            _futures = [e.submit(find_events, s, wavelets, scales, pad, slices[m], thresh, selectivity, dt, log=log, plot=cwt_plot) for m,s in enumerate(signals)]
            for _f in as_completed(_futures):
                if cwt_plot:
                    _result = _f.result()
                    _events.append(_result[0])
                    _cwt_list = _result[1]
                    for k in _cwt_list.keys():
                        _cwt = _cwt_list[k]
                        fig, ax1 = plt.subplots(1,1,figsize=(14,4))
                        ax2 = plt.twinx(ax1)
                        ax1.yaxis.tick_right()
                        ax2.yaxis.tick_left()
                        ax2.yaxis.set_label_position('left')
                        ax1.imshow(_cwt, extent=[0, _cwt.shape[1]*dt, scales[0]*1e3, scales[-1]*1e3], origin='lower', cmap='inferno')
                        ax1.set_yticks([])
                        ax1.axis('auto')
                        ax2.set_ylabel(f'\u0394t [ms]')
                        [ax2.add_artist(Ellipse((e['time'], e['scale']*1e3), width=1/selectivity*e['N']*e['scale'], height=1/selectivity*e['scale']*1e3, clip_on=True, zorder=10, linewidth=1,
                                edgecolor=(0,1,1,0.2), facecolor=(1, 0, 0, .025))) for e in _result[0]]
                        ax2.plot(_result[0]['time'], _result[0]['scale']*1e3, '.', color='green')
                        ax2.set_ylim(scales[0]*1e3,scales[-1]*1e3)
                        ax1.set_xlim(0, _cwt.shape[1]*dt)
                        if log:
                            ax2.set_yscale('log')
                        plt.show()
                else:
                    _events.append(_f.result())
                n += 1
                rem_time = int((total-n)*(time.time()-t0)/n)
                progress(n,total,status=f'analyzing {filename}, remaining time:{rem_time}[s]',length=50)
            _events = np.concatenate(tuple(_events), axis=0)
            if (len(_events) > 0):
                selected_events = []
                total = len(wavelets.keys())
                n, t0 = 0, time.time()
                progress(n,total,status=f'filtering events, remaining time: ...',length=50)
                for k,_ in enumerate(wavelets.keys()):
                    n += 1
                    selected_events.append(filter_events(_events[_events['name'] == k], selectivity=selectivity, refine=True))
                    rem_time = int((total-n)*(time.time()-t0)/n)
                    progress(n,total,status=f'filtering events, remaining time:{rem_time}[s]',length=50)
                selected_events = np.concatenate(tuple(selected_events), axis=0)
                selected_events = filter_events(selected_events, selectivity=selectivity, refine=True)
                # print(selected_events[:10])
                selected_events['time'] += signal['time'][_xlim[0]]*1e-12
                if save:
                    if f'events/{thresh:.2f}' in f[f'{dt:010.6f}'].keys():
                        del f[f'{dt:010.6f}'][f'events/{thresh:.2f}']
                    for s in selected_events.dtype.names:
                        f[f'{dt:010.6f}'].create_dataset(f'events/{thresh:.2f}/{s}', data=selected_events[s])
        
        if plot:
            if len(selected_events) > 0:
                for n,k in enumerate(wavelets.keys()):
                    _loc = np.argwhere(selected_events['name'] == n)
                    ax[1].plot(selected_events['time'][_loc], selected_events['scale'][_loc]*1e3, 'o', markersize=4, fillstyle='none', markeredgecolor=f'C{n}', alpha=1, label=f'{k} ({len(_loc)})')
                    fig.canvas.draw()
                    fig.canvas.flush_events()
                    plt.pause(0.1)
                ax[1].set_xlabel('time [s]')
                ax[1].set_ylabel(f'\u0394t [ms]')
                ax[1].set_ylim(float(scales[0]*1e3),float(scales[-1]*1e3))
                if log:
                    ax[1].set_yscale('log')
                plt.legend()
                plt.tight_layout()
                if save:
                    os.makedirs(f'{filename}_plots/', exist_ok=True)
                    fig.savefig(f'{filename}_plots/thresh_{thresh}_trace.{image_fmt}', format=image_fmt, bbox_inches='tight', transparent=True)
                plt.show()
                fig, ax = plt.subplots(1,3, figsize=(12,4), dpi=120)
                ax[0].hist([selected_events[selected_events['name']==n]['scale']*1e3 for n,k in enumerate(wavelets.keys())], label=list(wavelets.keys()), rwidth=0.9)
                ax[0].set_title(filename)
                ax[0].set_xlabel(f'\u0394t [ms]')
                ax[0].set_ylabel('# of events')
                ax[0].legend()
                ax[1].hist([selected_events[selected_events['name']==n]['coeff'] for n,k in enumerate(wavelets.keys())], label=list(wavelets.keys()), rwidth=0.9)
                ax[1].set_title(filename)
                ax[1].set_xlabel('CWT coeff. [a.u.]')
                ax[1].set_ylabel('# of events')
                ax[1].legend()
                ax[2].hist([selected_events[selected_events['name']==n]['time'] for n,k in enumerate(wavelets.keys())], stacked=True, label=list(wavelets.keys()), rwidth=0.9)
                ax[2].set_title(filename)
                ax[2].set_xlabel('Time [s]')
                ax[2].set_ylabel(f"events/{(selected_events['time'][-1]-selected_events['time'][0])/10:.3f} [s]")
                ax[2].legend()
                plt.tight_layout()
                if save:
                    fig.savefig(f'{filename}_plots/thresh_{thresh}_hists.{image_fmt}', format=image_fmt, bbox_inches='tight', transparent=True)
                plt.show()
                plt.pause(0.1)
                # fig, ax = plt.subplots(1,2,figsize=(12,3))
                # _loc = [np.where(signal['time'] >= e['time'])[0][0] for e in selected_events]
                # _loc = [[_loc-int(0.6*e['N']*e['scale'])]]
                # _events_avg = np.mean([signal['count'][_loc[n]-] for n,e in enumerate(selected_events)])
                # freq = np.fft.fftshift(np.fft.fftfreq(len(w)))[int(len(w)/2):]
                # FFT_w = np.abs(np.fft.fftshift(np.fft.fft(w))[int(len(w)/2):])
    return selected_events