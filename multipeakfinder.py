import numpy as np
import os as os
from scipy.signal import find_peaks
import scipy.spatial.distance as dist
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
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
    selected_events = np.array([_island[np.argmax(_island['coeff'])] for _island in _islands])
    return selected_events

def detect_islands(all_events, selectivity):
    all_events_t_l = all_events['time']-0.5/selectivity*np.multiply(all_events['N'],all_events['scale'])
    _index_l = np.argsort(all_events_t_l)
    all_events_t_r = all_events['time']+0.5/selectivity*np.multiply(all_events['N'],all_events['scale'])
    _index_r = np.argsort(all_events_t_r)
    all_events_overlap = all_events_t_r[_index_r[:-1]]-all_events_t_l[_index_l[1:]]
    _slices = np.argwhere(all_events_overlap <= 0).flatten()+1
    _islands = np.split(all_events[_index_l], _slices, axis=0)
    return _islands

def filter_events(all_events, selectivity, refine=True):
    all_events_t_l = all_events['time']-0.5/selectivity*np.multiply(all_events['N'],all_events['scale'])
    _index_l = np.argsort(all_events_t_l)
    all_events_t_r = all_events['time']+0.5/selectivity*np.multiply(all_events['N'],all_events['scale'])
    _index_r = np.argsort(all_events_t_r)
    all_events_overlap = all_events_t_r[_index_r[:-1]]-all_events_t_l[_index_l[1:]]
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
        for i,k in enumerate(wavelets.keys()):
            _cwt = np.empty((len(wavelets[k]['wavelets']), len(signal[pad:-pad])))
            for n, w in enumerate(wavelets[k]['wavelets']):
                _cwt[n,:] = (0.5*np.correlate(signal, w, mode='same')*np.sqrt(dt))[pad:-pad]
                _cwt[n,:] += np.abs(_cwt[n,:])
                _index, _ = find_peaks(_cwt[n,:], distance=wavelets[k]['N']*scales[n]/dt, height=thresh)
                _events = np.append(_events, np.array(list(zip((slice_l+_index)*dt, \
                                                                [scales[n]]*len(_index), \
                                                                _cwt[n,_index], \
                                                                [wavelets[k]['N']]*len(_index), \
                                                                [i]*len(_index))), dtype=d_type), axis=0)
            _cwt_list[k] = (_cwt)
        return _events, _cwt_list
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

def analyze_trace(filename, wavelets, scales, xlim, resolution, thresh, selectivity, chunksize, log=True, refine=True, save=False, plot=True, cwt_plot=False, image_fmt='tiff'):
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
            fig_trace, ax_trace = plt.subplots(2,1,figsize=(10,5), dpi=120, sharex=True, num=f'Trace [{filename}]')
            plt.subplots_adjust(hspace=0)
            ax_trace[0].plot(signal['time'][_xlim[0]:_xlim[1]]*1e-12, signal['count'][_xlim[0]:_xlim[1]], color='black', linewidth=0.5)
            ax_trace[0].set_title(filename.split('/')[-1])
            ax_trace[0].set_ylabel(f'signal [cnts/{dt*1e3:.3f}ms]')
            ax_trace[0].set_ylim(bottom=0)
            fig_trace.tight_layout()
            fig_trace.canvas.draw()
            fig_trace.canvas.flush_events()
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
        selected_events = []
        with Executor() as e:
            progress(n,total,status=f'detecting events, remaining time: ...',length=30)
            _futures = [e.submit(find_events, s, wavelets, scales, pad, slices[m], thresh, selectivity, dt, log=log, plot=cwt_plot) for m,s in enumerate(signals)]
            for _f in as_completed(_futures):
                if cwt_plot:
                    _result = _f.result()
                    _events.append(_result[0])
                    _cwt_list = _result[1]
                    for k in _cwt_list.keys():
                        _cwt = _cwt_list[k]
                        fig_cwt, ax_cwt1 = plt.subplots(1,1,figsize=(14,4), num=f'CWT plot for {k} [{filename}]')
                        ax_cwt2 = plt.twinx(ax_cwt1)
                        ax_cwt1.yaxis.tick_right()
                        ax_cwt2.yaxis.tick_left()
                        ax_cwt2.yaxis.set_label_position('left')
                        pos = ax_cwt1.imshow(_cwt, extent=[0, _cwt.shape[1]*dt, scales[0]*1e3, scales[-1]*1e3], origin='lower', cmap='inferno')
                        ax_cwt1.set_yticks([])
                        ax_cwt1.axis('auto')
                        ax_cwt1.set_xlabel('Time [s]')
                        ax_cwt2.set_ylabel(f'\u0394t [ms]')
                        [ax_cwt2.add_artist(Ellipse((e['time'], e['scale']*1e3), width=1/selectivity*e['N']*e['scale'], height=1/selectivity*e['scale']*1e3, clip_on=True, zorder=10, linewidth=1,
                                edgecolor=(0,1,1,0.2), facecolor=(1, 0, 0, .025))) for e in _result[0]]
                        ax_cwt2.plot(_result[0]['time'], _result[0]['scale']*1e3, '.', color='green')
                        ax_cwt2.set_ylim(scales[0]*1e3,scales[-1]*1e3)
                        ax_cwt1.set_xlim(0, _cwt.shape[1]*dt)
                        if log:
                            ax_cwt2.set_yscale('log')
                            ax_cwt2.yaxis.set_major_formatter(FormatStrFormatter('%g'))
                            ax_cwt2.yaxis.set_minor_formatter(FormatStrFormatter('%g'))
                        cbar = fig_cwt.colorbar(pos, ax=ax_cwt2)
                        cbar.minorticks_on()
                        fig_cwt.tight_layout()
                        plt.show()
                else:
                    _events.append(_f.result())
                n += 1
                rem_time = int((total-n)*(time.time()-t0)/n)
                progress(n,total,status=f'detecting events, remaining time:{rem_time}[s]',length=30)
            _events = np.concatenate(tuple(_events), axis=0)
            if (len(_events) > 0):
                for i,k in enumerate(wavelets.keys()):
                    progress(n,total,status=f'filtering {k} events, remaining time: ...',length=30)
                    _islands = detect_islands(_events[_events['name'] == i], selectivity)
                    total = len(_islands)
                    n, t0 = 0, time.time()
                    if refine:
                        _futures = [e.submit(spectral_cluster,_island,selectivity,plot=False) for _island in _islands]
                        for _f in as_completed(_futures):
                            n += 1
                            selected_events.append(_f.result())
                            rem_time = int((total-n)*(time.time()-t0)/n)
                            progress(n,total,status=f'filtering {k} events, remaining time:{rem_time}[s]',length=30)
                    else:
                        selected_events = np.array([_island[np.argmax(_island['coeff'])] for _island in _islands])
                selected_events = np.concatenate(tuple(selected_events), axis=0)
                selected_events = [spectral_cluster(_island) for _island in detect_islands(selected_events, selectivity)]
                selected_events = np.concatenate(tuple(selected_events), axis=0)
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
                    ax_trace[1].plot(selected_events['time'][_loc], selected_events['scale'][_loc]*1e3, 'o', markersize=4, fillstyle='none', markeredgecolor=f'C{n}', alpha=1, label=f'{k} ({len(_loc)})')
                    fig_trace.canvas.draw()
                    fig_trace.canvas.flush_events()
                    plt.pause(0.1)
                ax_trace[1].set_xlabel('time [s]')
                ax_trace[1].set_ylabel(f'\u0394t [ms]')
                ax_trace[1].set_ylim(float(scales[0]*1e3),float(scales[-1]*1e3),)
                if log:
                    ax_trace[1].set_yscale('log')
                    ax_trace[1].yaxis.set_major_formatter(FormatStrFormatter('%g'))
                    ax_trace[1].yaxis.set_minor_formatter(FormatStrFormatter(''))
                ax_trace[1].legend()
                fig_trace.tight_layout()
                if save:
                    os.makedirs(f'{filename}_plots/', exist_ok=True)
                    fig_trace.savefig(f'{filename}_plots/thresh_{thresh}_xlim_{xlim[0]}_{xlim[1]}_trace.{image_fmt}', format=image_fmt, bbox_inches='tight', transparent=True)
                plt.show()
                fig_hist, ax_hist = plt.subplots(1,3, figsize=(12,4), dpi=120, num=f'Events statistics [{filename}]')
                ax_hist[0].hist([selected_events[selected_events['name']==n]['scale']*1e3 for n,k in enumerate(wavelets.keys())], label=list(wavelets.keys()), rwidth=0.9)
                ax_hist[0].set_title(filename.split('/')[-1])
                ax_hist[0].set_xlabel(f'\u0394t [ms]')
                ax_hist[0].set_ylabel('# of events')
                ax_hist[0].legend()
                ax_hist[1].hist([selected_events[selected_events['name']==n]['coeff'] for n,k in enumerate(wavelets.keys())], label=list(wavelets.keys()), rwidth=0.9)
                ax_hist[1].set_title(filename.split('/')[-1])
                ax_hist[1].set_xlabel('CWT coeff. [a.u.]')
                ax_hist[1].set_ylabel('# of events')
                ax_hist[1].legend()
                ax_hist[2].hist([selected_events[selected_events['name']==n]['time'] for n,k in enumerate(wavelets.keys())], stacked=True, label=list(wavelets.keys()), rwidth=0.9)
                ax_hist[2].set_title(filename.split('/')[-1])
                ax_hist[2].set_xlabel('Time [s]')
                ax_hist[2].set_ylabel(f"events/{(selected_events['time'][-1]-selected_events['time'][0])/10:.3f} [s]")
                ax_hist[2].legend()
                fig_hist.tight_layout()
                if save:
                    fig_hist.savefig(f'{filename}_plots/thresh_{thresh}_xlim_{xlim[0]}_{xlim[1]}_hists.{image_fmt}', format=image_fmt, bbox_inches='tight', transparent=True)
                plt.show()
                plt.pause(0.1)
    return selected_events
