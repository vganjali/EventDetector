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
    scale = [0.1e-3, 1.0e-3, 30]    # dt ranges used for scaling wavelets [min, max, count]
    log = True                      # linear/logarthmic range for dt
    xlim = [26, 34]                  # time window to do analysis in [s]. [0,-1] means the entire trace
    resolution = 10                 # number of points to consider for minimum dt as the binning size
    refine = False                   # refine initial local maximas using Euclidean Distance
    plot = True                     # show the output plot, including trace and detected events in time-scale space
    cwt_plot = True                # show CWT colorplot (for long traces will slow down significantly!)
    save = False                     # save the detected events (and plots) into the .hdf5 file
    image_format = 'tiff'
    thresh = 4                      # threshold value used to detect events
    # dirname = 'G:/My Drive/PhD/experiments/APD_Traces/Gopi/Gopi_plasmids_Dec_2019/2019-11-22 [TC7 D9 GB 200nm bead testing]/'                    # directory of the .ptu files
    # dirname = 'G:/My Drive/PhD/experiments/APD_Traces/Gopi/Gopi_plasmids_Dec_2019/2019-10-30 [KPC VIM NDM BYU prep plasmid TC7 G]/'
    # dirname = 'C:/Users/vg88/Dropbox/Codes/Python/'
    dirname = '/home/vahid/Downloads/'
    filenames = ['default_000']
    #filenames = ['default_004','default_005','default_006']     # list of .ptu files to analyze
    globRes = 250e-12               # globRes of T2 traces
    selectivity = 3                 # how selectively it should detect events
    chunksize = 100000              # chunksize used to split the binned trace to process in parallel
    
    dt = scale[0]/resolution
    if log:
        scales = np.logspace(np.log10(scale[0]), np.log10(scale[1]), scale[2], dtype=np.float64)
    else:
        scales = np.linspace(scale[0], scale[1], scale[2], dtype=np.float64)
    wavelets = {
                # '2p':{'N': 2, \
                #       'wavelets': [wlts.mmi_gaussian_skewed(s, 2, window=1, weight=1, mod=0.6, shift=1, dt=dt) for s in scales]},
                # '3p':{'N': 3, \
                #       'wavelets': [wlts.mmi_gaussian_skewed(s, 3, window=1, weight=1, mod=0.6, shift=1, dt=dt) for s in scales]},
                #'CH1':{'N': 11, \
                      #'wavelets': [wlts.mmi_gaussian_skewed(s, 11, window=1, weight=1, mod=0.6, shift=1, skewness=0.4, dt=dt) for s in scales]},
                # 'CH1_1':{'N': 12, \
                #       'wavelets': [wlts.mmi_gaussian_skewed(s, 12, window=1, weight=1, mod=0.6, shift=1, skewness=0.4, dt=dt) for s in scales]},
                #'CH2':{'N': 8, \
                      #'wavelets': [wlts.mmi_gaussian_skewed(s, 8, window=1, weight=1, mod=0.6, shift=1, skewness=0.4, dt=dt) for s in scales]},
                # 'CH2_1':{'N': 9, \
                #       'wavelets': [wlts.mmi_gaussian_skewed(s, 9, window=1, weight=1, mod=0.6, shift=1, skewness=0.4, dt=dt) for s in scales]},
                'CH3':{'N': 5, \
                      'wavelets': [wlts.mmi_gaussian_skewed(s, 5, window=1, weight=1, mod=0.6, shift=1, skewness=0.4, dt=dt) for s in scales]},
                # 'CH3_1':{'N': 6, \
                #       'wavelets': [wlts.mmi_gaussian_skewed(s, 6, window=1, weight=1, mod=0.6, shift=1, skewness=0.4, dt=dt) for s in scales]},
                # '9p':{'N': 9, \
                #       'wavelets': [wlts.mmi_gaussian_skewed(s, 9, window=1, weight=1, mod=0.6, shift=1, skewness=0.4, dt=dt) for s in scales]},
                # '9p':{'N': 9, \
                #       'wavelets': [wlts.mmi_gaussian_skewed(s, 9, window=1, weight=1, mod=0.6, shift=1, dt=dt) for s in scales]},
                # '7pp':{'N': 7, \
                #       'wavelets': [wlts.mmi_gaussian(s, 7, window=1, weight=1, mod=0.6, shift=1, dt=dt) for s in scales]}, \
                # '8p':{'N': 8, \
                #       'wavelets': [wlts.mmi_gaussian(s, 8, window=0.6, weight=1, mod=0.5, shift=1, dt=dt) for s in scales]}, \
                # '9p':{'N': 9, \
                #       'wavelets': [wlts.mmi_gaussian(s, 9, window=0.5, weight=1, mod=0.5, shift=1, dt=dt) for s in scales]},
                }
    if plot:
        plt.ion()
        fig_wlt, ax_wlt = plt.subplots(1,2,figsize=(12,3), num='Wavelets used for analysis')
        for n,k in enumerate(wavelets.keys()):
            w = wavelets[k]['wavelets'][0]
            freq = np.fft.fftshift(np.fft.fftfreq(len(w)))[int(len(w)/2):]
            FFT_w = np.abs(np.fft.fftshift(np.fft.fft(w))[int(len(w)/2):])
            ax_wlt[0].plot((np.arange(len(w))-len(w)/2)*dt*1e3, w, color=f'C{n}', label=k)
            ax_wlt[0].set_title('time domain')
            _end = int(len(freq)/2)
            ax_wlt[1].plot(freq[:-_end], FFT_w.real[:-_end], color=f'C{n}', linestyle='-', label=f'{k} (real)')
            ax_wlt[1].plot(freq[:-_end], FFT_w.imag[:-_end], color=f'C{n}', linestyle=':', label=f'{k} (imag)')
            ax_wlt[1].set_title('frequency domain (FFT)')
        for a in ax_wlt:
            a.grid()
            a.legend()
        fig_wlt.tight_layout()
        plt.show()
        fig_wlt.canvas.draw()
        fig_wlt.canvas.flush_events()
        plt.pause(0.1)
    # return

    total = len(filenames)
    n, t0 = 0, time.time()
    with Executor() as e:
        progress(n,total,status=f'binning files, remaining time: ...',length=30)
        _futures = [e.submit(ptu.processHT2, filename=dirname+f, globRes=globRes, binsize=dt, chunksize=1000000) for f in filenames]
        for f in as_completed(_futures):
            n += 1
            rem_time = int((total-n)*(time.time()-t0)/n)
            progress(n,total,status=f'binning files, remaining time:{rem_time} [s]',length=30)
    print(f"\nbinning finished in {time.time()-t0:.2f} [s]")
    total = len(filenames)
    n, t0 = 0, time.time()
    for f in filenames:
        print(f'\n>> analyzing {f}')
        events = mpf.analyze_trace(dirname+f, wavelets, scales, xlim, resolution, thresh, selectivity, chunksize, log=log, refine=refine, save=save, plot=plot, cwt_plot=cwt_plot, image_fmt=image_format)
        n += 1
        rem_time = int((total-n)*(time.time()-t0)/n)
        progress(n,total,status=f"{f}: {np.count_nonzero(events['name']!=-1)} event(s) detected, remaining time:{rem_time:.0f} [s]",length=30)
    print(f"\n>> finished in {time.time()-t0:.2f} [s]")
    if plot:
        plt.ioff()
        plt.show()
if __name__ == '__main__':
    main()
