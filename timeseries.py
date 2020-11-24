import os
import io
from math import (floor, ceil)
import numpy as np
import pandas as pd
import h5py as h5py
import mmap as mmap
from scipy.sparse import csr_matrix
from scipy.signal import decimate
from PySide2.QtCore import (Signal, Slot, QObject, QTimer)
import queue
import time

class timeseries(QObject):
    progress = Signal(int)
    started = Signal(bool)
    updateplot = Signal(bool)
    message = Signal(str)
    updateplot_timer = QTimer()
    def __init__(self, filename=''):
        QObject.__init__(self)
        self.filename = filename
        self.trace = []
        self.type = ''
        self.dt = 1e-3
        self.unit = ''
        self.chunksize = 100000
        self.ptu_globres = 250e-12
        self.nanopore_globres = 4e-6
        self.pm_globres = 2e-3
        self.queue = queue.Queue(maxsize=20)
        self.active = False
        self.update_interval = 50 #ms
        self.updateplot_timer.setInterval(self.update_interval)

    def load(self, plot=True, relim=True):
        self.started.emit(True)
        if self.active:
            self.message.emit('Reading file...')
            if os.path.splitext(self.filename)[1] == '.ptu':       # TimeHarp Traces
                self.type = 'ptu'
                self.unit = f"Intensity [cnts/{self.dt*1e3:g}ms]"
                self.processHT2()
            elif (os.path.splitext(self.filename)[1] == '.txt'):  # Nanopore Traces
                try:
                    tmp = pd.read_csv(self.filename, sep='\t', header=None, names=['time','current','voltage'], engine='c')
                    _columns = tmp.columns
                    print(_columns)
                    self.unit = u"Current [\u00B5A]"
                    self.trace = {'time':tmp[_columns[0]].to_numpy(), 'current':tmp[_columns[1]].to_numpy()}
                    self.nanopore_globres = self.trace[_columns[0]][1]-self.trace[_columns[0]][0]
                    # self.trace = self.trace.rolling(window=int(self.dt/self.nanopore_globres), win_type='hamming').mean().dropna()
                    self.type = 'nanopore'
                except Exception as excpt:
                    # print(excpt)
                    self.started.emit(False)
                    self.message.emit('Unsupported file')
                    return
                # bins = np.arange(len(self.trace), int(self.dt/self.nanopore_globres))
                _dt = floor(self.dt/self.nanopore_globres)
                if _dt == 0: _dt = 1
                _idx = np.arange(0,len(self.trace['time']),_dt)
                _decimated = {'time': self.trace['time'][_idx[:-1]]}
                _tmp = np.split(self.trace['current'], _idx[1:])
                # [print(_t) for _t in _tmp]
                _decimated['current'] = np.stack([np.mean(_t) for _t in _tmp[:-1]])
                self.trace = _decimated
                # self.trace = {'current':decimate(self.trace['current'], floor(self.dt/self.nanopore_globres), ftype='iir')}
                # self.trace = pd.DataFrame({k:decimate(self.trace[k],int(self.dt/self.nanopore_globres)) for k in self.trace.columns[1:]})
                # self.trace['time'] = np.arange(len(self.trace['current']), dtype=np.uint64)*floor(self.dt/self.nanopore_globres)*self.nanopore_globres
                self.dt = _dt*self.nanopore_globres
                # self.trace = self.trace.groupby(lambda x: x/(self.dt/dt)).mean().groupby(lambda y: y/(self.dt/dt), axis=1).mean()
                # print(self.trace)
            elif (os.path.splitext(self.filename)[1] == '.andor'):  # Andor binned Traces
                try:
                    tmp = pd.read_csv(self.filename, sep='\t', header=None, names=['time','intensity'], engine='c')
                    _columns = tmp.columns
                    print(_columns)
                    self.unit = u"Intensity [a.u.]"
                    self.trace = {'time':tmp[_columns[0]].to_numpy(), 'intensity':tmp[_columns[1]].to_numpy()}
                    self.nanopore_globres = self.trace[_columns[0]][1]-self.trace[_columns[0]][0]
                    # self.trace = self.trace.rolling(window=int(self.dt/self.nanopore_globres), win_type='hamming').mean().dropna()
                    self.type = 'andor'
                except Exception as excpt:
                    # print(excpt)
                    self.started.emit(False)
                    self.message.emit('Unsupported file')
                    return
                # bins = np.arange(len(self.trace), int(self.dt/self.nanopore_globres))
                _dt = floor(self.dt/self.nanopore_globres)
                if _dt == 0: _dt = 1
                _idx = np.arange(0,len(self.trace['time']),_dt)
                _decimated = {'time': self.trace['time'][_idx[:-1]]}
                _tmp = np.split(self.trace['intensity'], _idx[1:])
                # [print(_t) for _t in _tmp]
                _decimated['intensity'] = np.stack([np.mean(_t) for _t in _tmp[:-1]])
                self.trace = _decimated
                # self.trace = {'current':decimate(self.trace['current'], floor(self.dt/self.nanopore_globres), ftype='iir')}
                # self.trace = pd.DataFrame({k:decimate(self.trace[k],int(self.dt/self.nanopore_globres)) for k in self.trace.columns[1:]})
                # self.trace['time'] = np.arange(len(self.trace['current']), dtype=np.uint64)*floor(self.dt/self.nanopore_globres)*self.nanopore_globres
                self.dt = _dt*self.nanopore_globres
                # self.trace = self.trace.groupby(lambda x: x/(self.dt/dt)).mean().groupby(lambda y: y/(self.dt/dt), axis=1).mean()
                # print(self.trace)
            elif (os.path.splitext(self.filename)[1] == '.csv'):    # Power Meter
                try:
                    tmp = pd.read_csv(self.filename,sep=';',skiprows=5,skipfooter=8,date_parser=True,skip_blank_lines=True,skipinitialspace=True,engine='python')
                    _columns = tmp.columns
                    print(_columns)
                    self.unit = F"Power [{_columns[2].split('(')[1].split(')')[0]}]"
                    _time = pd.to_datetime(tmp[_columns[0]]+tmp[_columns[1]],format = '%m/%d/%Y%H:%M:%S.%f')
                    self.pm_globres = _time.diff().min().total_seconds()
                    self.type = 'powermeter'
                except Exception as excpt:
                    # print(excpt)
                    self.started.emit(False)
                    self.message.emit('Unsupported file')
                    return
                _dt = floor(self.dt/self.pm_globres)
                if _dt == 0: _dt = 1
                _power = pd.Series(data=tmp[_columns[2]].values, index=_time).fillna(method='bfill').resample(f'{_dt}ms',origin='epoch').mean().fillna(method='bfill')
                _idx = np.arange(0,len(_time),_dt)
                _decimated = {'time': np.arange(len(_idx[1:]))*self.dt}
                _tmp = np.split(_power, _idx[1:])
                # [print(_t) for _t in _tmp]
                _decimated['power'] = np.stack([np.mean(_t) for _t in _tmp[:-1]])
                self.trace = _decimated
                self.dt = _dt*self.pm_globres
            if plot:
                self.updateplot.emit(relim)
        self.started.emit(False)
        self.active = False
        return
            

    def processHT2(self):
        T2WRAPAROUND_V2 = 33554432
        ofcorr = 0
        _dt = floor(self.dt/self.ptu_globres)
        bin_edge = {'time':0, 'count':0}
        self.timeout = 5
        _chunksize = int(self.chunksize)
        with h5py.File(os.path.splitext(self.filename)[0]+'.hdf5', 'a') as fout:
            if (f'{self.dt:010.6f}' not in fout.keys()):
                grp = fout.create_group(f'{self.dt:010.6f}')
                ds_time = grp.create_dataset('time', (0,), chunks=(10000000,), maxshape=(None,), dtype=np.float64, compression='lzf')
                ds_count = grp.create_dataset('count', (0,), chunks=(10000000,), maxshape=(None,), dtype=np.uint64, compression='lzf')
            else:
                ds_time = fout[f'{self.dt:010.6f}']['time']
                ds_count = fout[f'{self.dt:010.6f}']['count']
            with open(self.filename, mode='rb') as fin:
                f_mmap = mmap.mmap(fin.fileno(),0,access=mmap.ACCESS_READ)
                offset = int(f_mmap.find(str.encode('Header_End'))+48)
                # print(os.path.getsize(self.filename))
                progress_total = os.path.getsize(self.filename)/(_chunksize*4)+1
                progress_current = 0
                while self.timeout > 0:
                    # print(_chunksize)
                    try:
                        records = np.frombuffer(f_mmap,dtype=np.uint32,count=_chunksize,offset=offset)
                        special = np.bitwise_and(np.right_shift(records,31), 0x01).astype(np.byte)
                        channel = np.bitwise_and(np.right_shift(records,25), 0x3F).astype(np.byte)
                        timetag = np.bitwise_and(records, 0x1FFFFFF).astype(np.uint64)
                        overflowCorrection = np.zeros(_chunksize, dtype=np.uint64)
                        overflowCorrection[np.where((channel == 0x3F) & (timetag == 0))[0]] = T2WRAPAROUND_V2
                        _loc = np.where((channel == 0x3F) & (timetag != 0))[0]
                        overflowCorrection[_loc] = np.multiply(timetag[_loc], T2WRAPAROUND_V2)
                        overflowCorrection = np.cumsum(overflowCorrection)+ofcorr
                        ofcorr = overflowCorrection[-1]
                        timetag += overflowCorrection
                        _tmp = timetag[np.where((special != 1) | (channel == 0x00))[0]].astype(np.uint64)
                        binned_sparse_time, binned_sparse_count = np.unique(np.insert(\
                            np.floor_divide(_tmp,_dt),0,[bin_edge['time']]*bin_edge['count'])\
                            .astype(np.uint64),return_counts=True)
                        bin_edge['time'], bin_edge['count'] = binned_sparse_time[-1], binned_sparse_count[-1]
                        ds_count.resize((int(binned_sparse_time[-1]+1),))
                        dense_data = csr_matrix((binned_sparse_count,([0]*len(binned_sparse_time),binned_sparse_time-binned_sparse_time[0])), shape=(1,int(binned_sparse_time[-1]-binned_sparse_time[0]+1)), dtype=np.int32).toarray()[0]
                        ds_count[binned_sparse_time[0]:int(binned_sparse_time[-1]+1)] = dense_data
                        offset += _chunksize*4
                        progress_current += 100/progress_total
                        self.progress.emit(int(progress_current))
                    except Exception as excpt:
                        # print('warning:',excpt)
                        if (len(f_mmap[offset:])/4 < _chunksize):
                            _chunksize = int(len(f_mmap[offset:])/4)
                        self.timeout -= 1
            ds_count.resize((int(bin_edge['time']+1),))
            ds_count[bin_edge['time']] = bin_edge['count']
            ds_time.resize((ds_count.size,))
            ds_time[:] = np.arange(ds_time.size, dtype=np.uint64)*_dt*self.ptu_globres
            self.dt = _dt*self.ptu_globres
            # print('trace length:',ds_count.size*self.dt,'[s]')
        return
    
    def resample(self, dt):
        ts_sig = np.empty((0,0), dtype=np.uint64)
        # ts_time = np.empty((0,0), dtype=np.uint64)
        _chunksize = int(self.chunksize)
        if self.type == 'ptu':
            T2WRAPAROUND_V2 = 33554432
            ofcorr = 0
            _dt = floor(dt/self.ptu_globres)
            bin_edge = {'time':0, 'count':0}
            self.timeout = 5
            with open(self.filename, mode='rb') as fin:
                f_mmap = mmap.mmap(fin.fileno(),0,access=mmap.ACCESS_READ)
                offset = int(f_mmap.find(str.encode('Header_End'))+48)
                # print(os.path.getsize(self.filename))
                # progress_total = os.path.getsize(self.filename)/(self.chunksize*4)+1
                # progress_current = 0
                while self.timeout > 0:
                    try:
                        records = np.frombuffer(f_mmap,dtype=np.uint32,count=_chunksize,offset=offset)
                        special = np.bitwise_and(np.right_shift(records,31), 0x01).astype(np.byte)
                        channel = np.bitwise_and(np.right_shift(records,25), 0x3F).astype(np.byte)
                        timetag = np.bitwise_and(records, 0x1FFFFFF).astype(np.uint64)
                        overflowCorrection = np.zeros(_chunksize, dtype=np.uint64)
                        overflowCorrection[np.where((channel == 0x3F) & (timetag == 0))[0]] = T2WRAPAROUND_V2
                        _loc = np.where((channel == 0x3F) & (timetag != 0))[0]
                        overflowCorrection[_loc] = np.multiply(timetag[_loc], T2WRAPAROUND_V2)
                        overflowCorrection = np.cumsum(overflowCorrection)+ofcorr
                        ofcorr = overflowCorrection[-1]
                        timetag += overflowCorrection
                        _tmp = timetag[np.where((special != 1) | (channel == 0x00))[0]].astype(np.uint64)
                        binned_sparse_time, binned_sparse_count = np.unique(np.insert(\
                            np.floor_divide(_tmp,_dt),0,[bin_edge['time']]*bin_edge['count'])\
                            .astype(np.uint64),return_counts=True)
                        bin_edge['time'], bin_edge['count'] = binned_sparse_time[-1], binned_sparse_count[-1]
                        ts_sig = np.resize(ts_sig, (int(binned_sparse_time[-1]+1),))
                        dense_data = csr_matrix((binned_sparse_count,([0]*len(binned_sparse_time),binned_sparse_time-binned_sparse_time[0])), shape=(1,int(binned_sparse_time[-1]-binned_sparse_time[0]+1)), dtype=np.int32).toarray()[0]
                        ts_sig[binned_sparse_time[0]:int(binned_sparse_time[-1]+1)] = dense_data
                        offset += _chunksize*4
                        # progress_current += 100/progress_total
                        # self.progress.emit(int(progress_current))
                    except Exception as excpt:
                        # print('warning:',excpt)
                        if (len(f_mmap[offset:])/4 < _chunksize):
                            _chunksize = int(len(f_mmap[offset:])/4)
                        self.timeout -= 1
            ts_sig = np.resize(ts_sig, (int(bin_edge['time']+1),))
            ts_sig[bin_edge['time']] = bin_edge['count']
            ts_time = np.arange(ts_sig.size, dtype=np.uint64)*_dt*self.ptu_globres
            # self.dt = _dt*self.ptu_globres
        elif self.type == 'nanopore':
            _dt = floor(dt/self.nanopore_globres)
            if _dt == 0: _dt = 1
            _idx = np.arange(0,len(self.trace['time']),_dt)
            _decimated = {'time': self.trace['time'][_idx[:-1]]}
            _tmp = np.split(self.trace['current'], _idx[1:])
            # [print(_t) for _t in _tmp]
            _decimated['current'] = np.stack([np.mean(_t) for _t in _tmp[:-1]])
            self.trace = _decimated
            # _dt = floor(dt/self.nanopore_globres)
            # _decimated = {'time': self.trace['time'][::_dt]}
            # _decimated['current'] = np.mean(np.split(self.trace['current'], np.arange(len(_decimated['time'])*_dt)), axis=1)
            # _decimated = {'current':decimate(self.trace['current'], floor(dt/self.nanopore_globres), ftype='iir')}
            # _decimated['time'] = np.arange(len(_decimated['current']), dtype=np.uint64)*floor(dt/self.nanopore_globres)*self.nanopore_globres
            # self.dt = floor(dt/self.nanopore_globres)*self.nanopore_globres
            ts_time, ts_sig = _decimated['time'], _decimated['current']
        elif self.type == 'andor':
            _dt = floor(dt/self.nanopore_globres)
            if _dt == 0: _dt = 1
            _idx = np.arange(0,len(self.trace['time']),_dt)
            _decimated = {'time': self.trace['time'][_idx[:-1]]}
            _tmp = np.split(self.trace['intensity'], _idx[1:])
            # [print(_t) for _t in _tmp]
            _decimated['intensity'] = np.stack([np.mean(_t) for _t in _tmp[:-1]])
            self.trace = _decimated
            # _dt = floor(dt/self.nanopore_globres)
            # _decimated = {'time': self.trace['time'][::_dt]}
            # _decimated['current'] = np.mean(np.split(self.trace['current'], np.arange(len(_decimated['time'])*_dt)), axis=1)
            # _decimated = {'current':decimate(self.trace['current'], floor(dt/self.nanopore_globres), ftype='iir')}
            # _decimated['time'] = np.arange(len(_decimated['current']), dtype=np.uint64)*floor(dt/self.nanopore_globres)*self.nanopore_globres
            # self.dt = floor(dt/self.nanopore_globres)*self.nanopore_globres
            ts_time, ts_sig = _decimated['time'], _decimated['intensity']
        return ts_time, ts_sig
        # print('trace length:',ds_count.size*self.dt,'[s]')
    
    def processHT2_rt(self):
        T2WRAPAROUND_V2 = 33554432
        ofcorr = 0
        _dt = int(self.dt/self.ptu_globres)
        bin_edge = {'time':0, 'count':0}
        buffer = {'time':[], 'count':[]}
        # self.chunksize = max(self.chunksize, 10000000)
        # self.started.emit(True)
        while not os.path.exists(self.filename):
            time.sleep(1)
        with io.FileIO(self.filename, mode='r') as fin:
            while True:
                try:
                    f_mmap = mmap.mmap(fin.fileno(),0,access=mmap.ACCESS_READ)
                    offset = int(f_mmap.find(str.encode('Header_End'))+48)
                    fin.seek(offset)
                    break
                except Exception as excpt:
                    print(excpt)
                    time.sleep(0.5)
            # print(os.path.getsize(self.filename))
            while self.active:
                try:
                    records = np.frombuffer(f_mmap,dtype=np.uint32,count=self.chunksize,offset=offset)
                except Exception as e:
                    # print(e)
                    time.sleep(self.update_interval/1000)
                    # records = np.fromfile(fin,dtype=np.uint32,count=-1,offset=offset)
                    records = []
                if len(records) > 0:
                    special = np.bitwise_and(np.right_shift(records,31), 0x01).astype(np.byte)
                    channel = np.bitwise_and(np.right_shift(records,25), 0x3F).astype(np.byte)
                    timetag = np.bitwise_and(records, 0x1FFFFFF).astype(np.uint64)
                    overflowCorrection = np.zeros(len(records), dtype=np.uint64)
                    overflowCorrection[np.where((channel == 0x3F) & (timetag == 0))[0]] = T2WRAPAROUND_V2
                    _loc = np.where((channel == 0x3F) & (timetag != 0))[0]
                    overflowCorrection[_loc] = np.multiply(timetag[_loc], T2WRAPAROUND_V2)
                    overflowCorrection = np.cumsum(overflowCorrection)+ofcorr
                    ofcorr = overflowCorrection[-1]
                    timetag += overflowCorrection
                    _tmp = timetag[np.where((special != 1) | (channel == 0x00))[0]].astype(np.uint64)
                    binned_sparse_time, binned_sparse_count = np.unique(np.insert(\
                        np.floor_divide(_tmp,_dt),0,[bin_edge['time']]*bin_edge['count'])\
                        .astype(np.uint64),return_counts=True)
                    buffer['count'] = binned_sparse_count[:-1]
                    buffer['time'] = binned_sparse_time[:-1]*self.dt*1e12
                    bin_edge['time'], bin_edge['count'] = binned_sparse_time[-1], binned_sparse_count[-1]
                    offset += len(records)*4
                    # while self.queue.full():
                    # 	print('full')
                    self.queue.put(buffer.copy())
                    # print('put:',buffer['time'][0]*1e-12,buffer['time'][-1]*1e-12)