import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import time as time
from concurrent.futures import ThreadPoolExecutor as Executor
from concurrent.futures import as_completed
import multipeakfinder as mpf
import ptu as ptu
import wavelets as wlts
import h5py as h5py
import sys as sys

def progress(count, total, title='', status='', length=50):
    per = count/total
    sys.stdout.write(f" [{'#'*round(per*length)+'-'*(length-round(per*length))}] {per*100:3.1f}% | {status:30s} \r")
    sys.stdout.flush()

def main():
    scale = [0.1e-3, 1.0e-3, 30]
    log = True
    xlim = [0, 10]
    resolution = 10
    plot = True
    save = False
    dt = scale[0]/resolution
    thresh = 5
    dirname = ''
    filenames = ['default_007']
    globRes = 250e-12
    selectivity = 1
    if log:
        scales = np.logspace(np.log10(scale[0]), np.log10(scale[1]), scale[2], dtype=np.float64)
    else:
        scales = np.linspace(scale[0], scale[1], scale[2], dtype=np.float64)
    wavelets = {
                '7p':{'N': 7, \
                      'wavelets': [wlts.mmi_gaussian(s, 7, window=0.7, weight=1, mod=0.5, shift=1, dt=dt) for s in scales]}, \
                '8p':{'N': 8, \
                      'wavelets': [wlts.mmi_gaussian(s, 8, window=0.6, weight=1, mod=0.5, shift=1, dt=dt) for s in scales]}, \
                '9p':{'N': 9, \
                      'wavelets': [wlts.mmi_gaussian(s, 9, window=0.5, weight=1, mod=0.5, shift=1, dt=dt) for s in scales]}
                }
    if plot:
        for k in wavelets.keys():
            w = wavelets[k]['wavelets'][0]
            plt.plot(np.arange(len(w))-len(w)/2, w, label=k)
        plt.legend()
        plt.show()
    chunksize = 100000
    pad = max([max([len(w) for w in wavelets[k]['wavelets']]) for k in wavelets.keys()])
    total = len(filenames)
    n, t0 = 0, time.time()
    with Executor() as e:
        progress(n,total,status=f'binning files, remaining time: ...',length=50)
        _futures = [e.submit(ptu.processHT2, filename=dirname+f, globRes=globRes, binsize=dt, chunksize=1000000) for f in filenames]
        for f in as_completed(_futures):
            n += 1
            rem_time = int((total-n)*(time.time()-t0)/n)
            progress(n,total,status=f'binning files, remaining time:{rem_time} [s]',length=50)
    print('')
    total = len(filenames)
    n, t0 = 0, time.time()
    for f in filenames:
        progress(n,total,status=f'analyzing {f}, remaining time: ...',length=50)
        events_len = len(mpf.analyze_trace(dirname+f, wavelets, scales, xlim, resolution, thresh, selectivity, chunksize, log=log, save=save, plot=plot, cwt_plot=False))
        n += 1
        rem_time = int((total-n)*(time.time()-t0)/n)
        progress(n,total,status=f'{f}: {events_len} events detected, remaining time:{rem_time:.0f} [s]',length=50)
    print('')
if __name__ == '__main__':
    main()