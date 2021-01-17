import numpy as np
from scipy.special import erf
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import wavelets

d_type = np.dtype([('time', 'f8'), ('scale', 'f8'), ('coeff', 'f8'), ('N', 'f8')])
def find_events(data, wavelet, N, scales, threshold, dt):
    _events = np.empty((0,), dtype=d_type) 
    _cwt = np.empty((len(data),len(wavelet)))
    for n, w in enumerate(wavelet):
        _cwt[:,n] = np.abs(np.correlate(data, w, mode='same')) 
        # _cwt[:,n] = (0.5*np.correlate(data, w, mode='same')) 
        # _cwt[:,n] += np.abs(_cwt[:,n])
        _index, _ = find_peaks(_cwt[:,n], distance=N*scales[n]/dt, height=threshold)
        _events = np.append(_events, np.array(list(zip((_index)*dt, [scales[n]]*len(_index), _cwt[_index,n], [N]*len(_index))), dtype=d_type), axis=0)
    return _events, _cwt 
def detect_islands(all_events, extent=1): 
    all_events_t_l = all_events['time']-0.5*extent*np.multiply(all_events['N'],all_events['scale']) 
    _index_l = np.argsort(all_events_t_l) 
    all_events_t_r = all_events['time']+0.5*extent*np.multiply(all_events['N'],all_events['scale']) 
    _index_r = np.argsort(all_events_t_r) 
    all_events_overlap = all_events_t_r[_index_r[:-1]]-all_events_t_l[_index_l[1:]] 
    _slices = np.argwhere(all_events_overlap <= 0).flatten()+1 
    _islands = np.split(all_events[_index_l], _slices, axis=0) 
    return _islands
def select_events(events, selectivity=1, w=1, h=1, threshold=0.2): 
    selected_events = [] 
    events = np.sort(events,order='coeff')[::-1] 
    while(len(events) > selectivity):
        _n_0, _n_i = events['N'][0], events['N'] 
        _t_0, _t_i = events['time'][0], events['time'] 
        _s_0, _s_i = events['scale'][0], events['scale'] 
        _c_0, _c_i = events['coeff'][0], events['coeff'] 
        _dt = _t_0 - _t_i 
        _ds = _s_0 - _s_i 
        _theta_i = np.arctan(_ds/_dt) 
        _dist_square = _dt**2 + _ds**2 
        _r_0 = (w*h*_n_0*_s_0)/np.sqrt((w*_n_0*np.sin(_theta_i))**2+(h*np.cos(_theta_i))**2) 
        _r_i = (w*h*_n_i*_s_i)/np.sqrt((w*_n_i*np.sin(_theta_i))**2+(h*np.cos(_theta_i))**2)*np.sqrt((_c_i-_c_i[-1])/(_c_0-_c_i[-1]))
        _dr_square = (_r_i+_r_0)**2
        _adjacency = np.argwhere(np.nan_to_num(_dr_square,np.inf)-_dist_square >= 0).flatten()
        if len(_adjacency) > selectivity: selected_events.append(events[0])
        events = np.delete(events,_adjacency)
    return np.array(selected_events, dtype=d_type) 
def ed_cwt(data, scales, wavelet, resolution, threshold=4, selectivity=3, extent=4, w=1, h=1, dt=1):
    selected_events = [] 
    dt = min(scales)/resolution 
    if wavelet == 'ricker': 
        wvlts = [wavelets.ricker(s/dt) for s in scales]
        N = 1 
    elif wavelet[:4] == 'msg-':
        N = int(wavelet[4:]) 
        wvlts = [wavelets.msg(s/dt, N=N, mod=0.8, shift=1, skewness=0.5) for s in scales] 
    elif wavelet[:5] == 'msge-':
        N = len(wavelet[5:]) 
        wvlts = [wavelets.msg_encoded(s/dt, pattern=wavelet[5:], mod=1.5, shift=-2.9, skewness=0.04) for s in scales]
    elif wavelet[:7] == 'morlet-':
        N = len(wavelet[7:]) 
        wvlts = [wavelets.morlet(s/dt, N=N, is_complex=False) for s in scales]
    elif wavelet[:8] == 'cmorlet-':
        N = len(wavelet[8:]) 
        wvlts = [wavelets.morlet(s/dt, N=N, is_complex=True) for s in scales]
    all_events, cwt = find_events(data, wvlts, N, scales, threshold, dt)
    islands = detect_islands(all_events,extent) 
    for n,island in enumerate(islands): 
        selected_events.append(select_events(island,selectivity,w,h))
    return np.concatenate(tuple(selected_events),axis=0), all_events, cwt
