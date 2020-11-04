"""
version:0.1.1
"""
import sys
import os
import time
import importlib
import multiprocessing
from concurrent.futures import ProcessPoolExecutor as Executor
from concurrent.futures import as_completed

from PySide2.QtGui import QPixmap
# from PySide2.QtUiTools import QUiLoader
from PySide2.QtWidgets import (QSplashScreen, QMainWindow, QApplication, QMessageBox, QWidget)
from PySide2.QtCore import (Qt, QObject, Signal)
import requests
import re
# import pandas as pd
# import pyqtgraph as pg
# import vaex
# import numpy as np
# import h5py
# import pyqtgraph as pg
# import mainwindow as mw
# import ptu
# import wavelet
# import eventdetector as ed
# import threading
modules = {'mainwindow':'mw', 'timeseries':'ts', 'wavelet':'', 'eventdetector':'ed', 'threading':'', 'pyqtgraph':'pg', 'numpy':'np', 'h5py':'', 'pandas':'pd', 'eventsinfo':'ei'}

# class InfoWindow(QWidget):
#     def __init__(self, parent=None):
#         super().__init__(parent)
#         self.parent = parent
#         self.hide()

#     def show(self):
#         self.show()

#     def hide(self, QCloseEvent):
#         self.hide()


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = mw.Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.target_update_list()
        self.ei_window = ei.Ui_Info()
        self.ei_window.setupUi(self.ei_window, parent=self)
        self.ts = ts.timeseries()
        self.eventdetector = ed.eventdetector()
        self.ts.progress.connect(self.ui.progressBar.setValue)
        self.ts.started.connect(self.ui.progressBar.setVisible)
        self.ts.updateplot.connect(self.update_time_plot)
        self.ts.updateplot_timer.timeout.connect(self.update_time_plot_rt)
        self.ts.message.connect(self.statusBar().showMessage)
        self.eventdetector.progress.connect(self.ui.progressBar.setValue)
        self.eventdetector.started.connect(self.ui.progressBar.setVisible)
        self.eventdetector.message.connect(self.statusBar().showMessage)
        # self.eventdetector.started.connect(self.update_statusbar)
        # self.eventdetector.showevents.connect(print)
        self.eventdetector.showevents.connect(self.update_events)
        self.eventdetector.drawcwt.connect(self.update_cwt_plot)
        self.ui.listWidget_files.itemClicked.connect(lambda sig: self.read_file(filename=sig,plot=True,relim=True))
        self.read_file_thread = threading.Thread(target=self.ts.processHT2_rt, args=(),daemon=False)
        self.ui.pushButton_realtime.clicked.connect(self.detect_events_rt)
        self.ui.doubleSpinBox_binsize.editingFinished.connect(lambda: self.read_file(filename=self.ui.listWidget_files.currentItem(),plot=True,relim=False))
        self.ui.pushButton_cwt.clicked.connect(self.detect_events)
        self.ui.pushButton_info_events.clicked.connect(self.ei_window.show_window)
        self.ei_window.progress.connect(self.ui.progressBar.setValue)
        self.ei_window.started.connect(self.ui.progressBar.setVisible)
        # self.ei_window.started.connect(self.update_statusbar)
        self.ei_window.message.connect(self.statusBar().showMessage)
        # with h5py.File("C:/Users/vahid/.vaex/data/helmi-dezeeuw-2000-FeH-v2-10percent.hdf5",'r') as f:
        #     print(f['table/columns'].keys())

    def read_file(self, filename, plot=True, relim=False):
        try:
            # self.ui.params['binsize'] = self.ui.doubleSpinBox_binsize.value()*1e-3
            # self.ui.params['buffer'] = self.ui.spinBox_buffersize.value()*1024
            if (not self.read_file_thread.is_alive()):
                if ((self.ts.dt != self.ui.params['binsize']) or (self.ts.filename != self.ui.params['currentdir']+filename.text())):
                    self.ts.filename = self.ui.params['currentdir']+filename.text()
                    self.ts.dt = self.ui.params['binsize']
                    self.ts.chunksize = int(self.ui.params['buffer']*1024*1024/4)
                    self.ts.active = True
                    self.read_file_thread = threading.Thread(target=self.ts.load, args=(plot,relim,),daemon=False)
                    # print(f"binning {ptufile.filename}")
                    self.ui.progressBar.reset()
                    self.read_file_thread.start()
            elif self.read_file_thread.is_alive():
                self.ts.active = False
                # self.statusBar().showMessage('Idle')
        except Exception as excpt:
            print(excpt)
            pass
    
    def time_plot_rt(self):
        try:
            if not self.read_file_thread.is_alive():
                # print('start')
                # self.ui.params['binsize'] = self.ui.doubleSpinBox_binsize.value()*1e-3
                # self.ui.params['window'] = self.ui.doubleSpinBox_windowsize.value()
                self.ts.filename = self.ui.params['currentdir']+self.ui.lineEdit_filename.text()+'.ptu'
                self.ts.binsize = self.ui.params['binsize']
                self.ts.buffersize = int(self.ui.params['buffer']*1024*1024/20/4)
                self.ts.active = True
                while not self.ts.queue.empty():
                    self.ts.queue.get()
                self.ui.plot_line.setLabel('left',f"Intensity [cnts/{self.ts.binsize*1e3:g}ms]")
                self.ui.plot_line.disableAutoRange()
                self.rt_plot_line = self.ui.plot_line.plot(
                    [],
                    [],
                    pen=pg.mkPen(color='b'),
                    clear=True,
                    name='CH 1'
                    # fillLevel=0,
                    # fillBrush=pg.mkBrush(color='b'),
                    )
                self.read_file_thread = threading.Thread(target=self.ts.processHT2_rt, args=(),daemon=True)
                self.read_file_thread.start()
                self.ts.updateplot_timer.start()
                self.ui.pushButton_realtime.setText('Stop')
                self.ui.listWidget_files.setEnabled(False)
                self.statusBar().showMessage('Realtime Plotting...')
            else:
                # print('stopped')
                self.ts.active = False
                self.read_file_thread = threading.Thread(target=self.ts.processHT2_rt, args=(),daemon=False)
                self.ts.updateplot_timer.stop()
                while not self.ts.queue.empty():
                    self.ts.queue.get()
                self.ui.pushButton_realtime.setText('Start')
                self.ui.listWidget_files.setEnabled(True)
                self.statusBar().showMessage('Idle')
            # self.plot_thread = threading.Thread(target=self.update_time_plot_worker, args=(),daemon=False).start()
        except Exception as e:
            print(e)
            while not self.ts.queue.empty():
                self.ts.queue.get()
            self.read_file_thread = threading.Thread(target=self.ts.processHT2_rt, args=(),daemon=False)
            self.ts.active = False



    def detect_events(self):
        import h5py, eventdetector
        # print('started')
        self.statusBar().showMessage('Analyzing')
        _wavelets = self.generate_wavelets()
        # [print(k,len(_wavelets[k]['wavelets'])) for k in _wavelets.keys()]
        self.ui.params['cwt']['window'] = {'l':list(self.ui.plot_line.getAxis('bottom').range)[0],'r':list(self.ui.plot_line.getAxis('bottom').range)[1]}
        # if self.ts.type == 'nanopore':
        #     self.ui.params['cwt']['resolution'] = self.ui.params['cwt']['scales']['min']/self.ts.nanopore_globres
        self.ts.dt = self.ui.params['cwt']['scales']['min']/self.ui.params['cwt']['resolution']
        self.ts.chunksize = int(self.ui.params['buffer']*1024*1024/4)
        for k in self.ui.params['cwt']:
            try:
                setattr(self.eventdetector, k, self.ui.params['cwt'][k])
            except Exception as e:
                print(e)
        self.eventdetector.ts = self.ts
        # self.eventdetector.chunksize = int(self.ui.params['buffer']*8)
        # self.eventdetector.chunksize = 100
        self.eventdetector.window = self.ui.params['cwt']['window']
        if self.ui.checkBox_show_cwt.checkState() == Qt.CheckState.Checked:
            self.eventdetector.cwt_plot = True
        else:
            self.eventdetector.cwt_plot = False
        self.detect_events_thread = threading.Thread(target=self.eventdetector.analyze_trace, args=(_wavelets,),daemon=False)
        # print(f"binning {ptufile.filename}")
        self.ui.progressBar.reset()
        self.ui.image_cwt = {name:[] for name in self.ui.params['targets']['name']}
        self.ui.scatter_events = {name:[] for name in self.ui.params['targets']['name']}
        self.ui.scatter_events_time = {name:[] for name in self.ui.params['targets']['name']}
        # print(self.ui.plot_line.removeItem(0))
        self.ui.image_cwt_vb.clear()
        self.ui.scatter_events_vb.clear()
        self.ui.scatter_events_time_vb.clear()
        for n,c in enumerate(self.ui.params['targets']['color']):
            self.ui.scatter_events[self.ui.params['targets']['name'][n]] = \
                pg.ScatterPlotItem([],[],symbol='s',pen=pg.mkPen(color=eval(c.split('rgb')[-1]),width=2),brush=None,pxMode=True, size=10, 
                name=self.ui.params['targets']['name'][n])
            self.ui.scatter_events_time[self.ui.params['targets']['name'][n]] = \
                pg.ScatterPlotItem([],[],symbol='+',pen=pg.mkPen(color=eval(c.split('rgb')[-1]),width=2),brush=None,pxMode=True, size=10, 
                name=self.ui.params['targets']['name'][n])
            self.ui.scatter_events[self.ui.params['targets']['name'][n]].setZValue(99)
            self.ui.scatter_events_time[self.ui.params['targets']['name'][n]].setZValue(99)
            self.ui.scatter_events_vb.addItem(self.ui.scatter_events[self.ui.params['targets']['name'][n]])
            self.ui.scatter_events_time_vb.addItem(self.ui.scatter_events_time[self.ui.params['targets']['name'][n]])
        # print(self.ui.plot_line.listDataItems())
        self.detect_events_thread.start()
        # print(f"binning {ptufile.filename}")
        # events = self.eventdetector.analyze_trace(os.path.splitext(self.ts.filename)[0], _wavelets, **self.ui.params['cwt'])
        # print(events)
        # self.statusBar().showMessage('Idle')

    def detect_events_rt(self):
        try:
            if not self.read_file_thread.is_alive():
                # print('start')
                # self.ui.params['binsize'] = self.ui.doubleSpinBox_binsize.value()*1e-3
                # self.ui.params['window'] = self.ui.doubleSpinBox_windowsize.value()
                if self.ui.params['cwt']['log']:
                    scales = np.logspace(np.log10(self.ui.params['cwt']['scales']['min']), np.log10(self.ui.params['cwt']['scales']['max']), self.ui.params['cwt']['scales']['count'], dtype=np.float64)
                else:
                    scales = np.linspace(self.ui.params['cwt']['scales']['min'], self.ui.params['cwt']['scales']['max'], self.ui.params['cwt']['scales']['count'], dtype=np.float64)
                self.ts.binsizes = [int(scale/self.ui.params['cwt']['resolution']/self.ts.globres) for scale in scales]
                self.ts.filename = self.ui.params['currentdir']+self.ui.lineEdit_filename.text()+'.ptu'
                self.ts.binsize = self.ui.params['binsize']
                self.ts.buffersize = int(self.ui.params['buffer']*1024*1024/20/4/self.ui.params['cwt']['scales']['count'])
                self.ts.active = True
                while not self.ts.queue.empty():
                    self.ts.queue.get()
                self.ui.plot_line.setLabel('left',f"Intensity [cnts/{self.ts.binsize*1e3:g}ms]")
                self.ui.plot_line.disableAutoRange()
                self.rt_plot_line = self.ui.plot_line.plot(
                    [],
                    [],
                    pen=pg.mkPen(color='b'),
                    clear=True,
                    name='CH 1'
                    # fillLevel=0,
                    # fillBrush=pg.mkBrush(color='b'),
                    )
                self.read_file_thread = threading.Thread(target=self.ts.processHT2_rt, args=(),daemon=True)
                self.read_file_thread.start()
                self.ts.updateplot_timer.start()
                self.ui.pushButton_realtime.setText('Stop')
                self.ui.listWidget_files.setEnabled(False)
                self.statusBar().showMessage('Realtime Plotting...')
            else:
                # print('stopped')
                self.ts.active = False
                self.read_file_thread = threading.Thread(target=self.ts.processHT2_rt, args=(),daemon=False)
                self.ts.updateplot_timer.stop()
                while not self.ts.queue.empty():
                    self.ts.queue.get()
                self.ui.pushButton_realtime.setText('Start')
                self.ui.listWidget_files.setEnabled(True)
                self.statusBar().showMessage('Idle')
            # self.plot_thread = threading.Thread(target=self.update_time_plot_worker, args=(),daemon=False).start()
        except Exception as e:
            print(e)
            while not self.ts.queue.empty():
                self.ts.queue.get()
            self.read_file_thread = threading.Thread(target=self.ts.processHT2_rt, args=(),daemon=False)
            self.ts.active = False

    def generate_wavelets(self):
        _wavelets = {}
        _dt = self.ui.params['cwt']['scales']['min']/self.ui.params['cwt']['resolution']
        if self.ui.params['cwt']['log']:
            scales = np.logspace(np.log10(self.ui.params['cwt']['scales']['min']), np.log10(self.ui.params['cwt']['scales']['max']), self.ui.params['cwt']['scales']['count'], dtype=np.float64)
        else:
            scales = np.linspace(self.ui.params['cwt']['scales']['min'], self.ui.params['cwt']['scales']['max'], self.ui.params['cwt']['scales']['count'], dtype=np.float64)
        # print(scales)
        for n,name in enumerate(self.ui.params['targets']['name']):
            wname = self.ui.params['targets']['wavelet']['name'][n]
            args = self.ui.params['targets']['wavelet']['parameters'][n]
            if self.ui.params['targets']['active'][n]:
                _wavelets[name] = {'N': args['N'],
                'wavelets': [getattr(wavelet, wname.lower().replace('-','_').replace(' ','_').replace('(','').replace(')',''))(s, dt=_dt, **args) for s in scales]}
            # else:
            #     _wavelets[name] = {'N': args['N'],
            #     'wavelets': [[0]]}
        return _wavelets

    def update_statusbar(self, signal):
        if (signal):
            self.statusBar().showMessage('Busy')
        else:
            self.statusBar().showMessage('Idle')

    def update_time_plot(self,relim=False):
        self.statusBar().showMessage('Updating plot...')
        if self.ts.type == 'ptu':       # TimeHarp Traces
            with h5py.File(os.path.splitext(self.ts.filename)[0]+'.hdf5', 'r') as f:
                self.ui.plot_line.disableAutoRange()
                self.ui.plot_line.setLabel('left',f"Intensity [cnts/{self.ts.dt*1e3:g}ms]")
                self.ui.plot_line.plot(
                    f[f"{self.ts.dt:010.6f}"]['time'][:],
                    f[f"{self.ts.dt:010.6f}"]['count'][:],
                    pen=pg.mkPen(color='b'),
                    clear=True,
                    name='CH 1',
                    zvalue=-1
                    # fillLevel=0,
                    # fillBrush=pg.mkBrush(color='b'),
                    )
                # self.ui.plot_line.setDownsampling(ds=True, auto=False, mode='mean')
                # self.ui.plot_line.setClipToView(clip=True)
                if relim:
                    self.ui.plot_line.setXRange(f[f"{self.ts.dt:010.6f}"]['time'][0],f[f"{self.ts.dt:010.6f}"]['time'][-1])
                self.ui.setYRange()
        elif (self.ts.type == 'nanopore'):  # Nanopore Traces
            self.ui.plot_line.disableAutoRange()
            self.ui.plot_line.setLabel('left',"Current [pA]")
            self.ui.plot_line.plot(
                self.ts.trace['time'],
                self.ts.trace['current'],
                pen=pg.mkPen(color='b'),
                clear=True,
                name='Current',
                zvalue=-1
                # fillLevel=0,
                # fillBrush=pg.mkBrush(color='b'),
                )
            # self.ui.plot_line.setDownsampling(ds=True, auto=False, mode='mean')
            # self.ui.plot_line.setClipToView(clip=True)
            if relim:
                self.ui.plot_line.setXRange(self.ts.trace['time'][0],self.ts.trace['time'][-1])
            self.ui.setYRange()
        
        self.statusBar().showMessage('Idle')

    def update_time_plot_rt(self):
        if not self.ts.queue.empty():
            buffer = self.ts.queue.get(block=False)
            # print('get:',buffer['time'][0]*1e-12,buffer['time'][-1]*1e-12)
            # self.ui.plot_line.plot(
            #     buffer['time']*1e-12,
            #     buffer['count'],
            #     pen=pg.mkPen(color='b'),
            #     clear=False,
            #     name='CH 1'
            #     # fillLevel=0,
            #     # fillBrush=pg.mkBrush(color='b'),
            #     )
            print(len(buffer['count'][0]))
            self.rt_plot_line.setData(
                np.append(self.rt_plot_line.xData,buffer['time'][0]),
                np.append(self.rt_plot_line.yData,buffer['count'][0]))

            # self.ui.plot_line.setDownsampling(ds=True, auto=False, mode='mean')
            # self.ui.plot_line.setClipToView(clip=True)
            self.ts.queue.task_done()
            self.ui.plot_line.setXRange(max(0,buffer['time'][0][-1]-self.ui.params['window']),buffer['time'][0][-1])
            self.ui.setYRange()
        # elif not self.read_file_thread.is_alive():
        #     self.ts.updateplot_timer.stop()
        #     self.ui.pushButton_realtime.setText('Start')
        #     self.statusBar().showMessage('Idle')

    def update_cwt_plot(self,cwt):
        if self.eventdetector.cwt_plot == True:
            self.ui.image_cwt_ax.disableAutoRange()
            # print(cwt)
            _view_rect = mw.QRectF(
                cwt[1],
                # 0,
                np.log10(self.ui.params['cwt']['scales']['min']*1e3),
                # cwt[0][list(cwt[0].keys())[0]].shape[1]*(self.ui.params['cwt']['scales']['min']/self.ui.params['cwt']['resolution']),
                cwt[2],
                np.log10(self.ui.params['cwt']['scales']['max']*1e3)-np.log10(self.ui.params['cwt']['scales']['min']*1e3))
                # self.ui.params['cwt']['scales']['count'])
            for key, value in cwt[0].items():
                self.ui.image_cwt[key].append(pg.ImageItem(image=value))
                self.ui.image_cwt[key][-1].setOpts(autoDownsample=True, 
                                                    axisOrder='row-major', 
                                                    # lut=self.ui.lut, 
                                                    update=True)
                self.ui.image_cwt[key][-1].setRect(_view_rect)
            # self.ui.image_cwt_ax.setLabel('left',f"Intensity [cnts/{self.ts.binsize*1e3:g}ms]")
            # self.ui.image_cwt.setImage(cwt[0]['6p'],
            #     # pen=pg.mkPen(color='b'),
            #     # clear=True,
            #     name='CH 1',
            #     autoRange=False
            #     # fillLevel=0,
            #     # fillBrush=pg.mkBrush(color='b'),
            #     )
                # self.ui.plot_line.setDownsampling(ds=True, auto=False, mode='mean')
                # self.ui.plot_line.setClipToView(clip=True)
                # if relim:
                #     xrange = [max(f[f"{ptufile.binsize:010.6f}"]['time'][0]*1e-12,self.ui.doubleSpinBox_window_l.value()),
                #             min(f[f"{ptufile.binsize:010.6f}"]['time'][-1]*1e-12,self.ui.doubleSpinBox_window_r.value())]
                #     if xrange[1] == 0: xrange[1] = f[f"{ptufile.binsize:010.6f}"]['time'][-1]*1e-12
                #     self.ui.plot_line.setXRange(xrange[0],xrange[1])
                # self.ui.setYRange()
            # self.ui.image_cwt.scale(_view_rect.width/cwt['6p'].shape[1],_view_rect.heigth/cwt['6p'].shape[0])
            # self.ui.image_cwt.translate(self.ui.params['cwt']['window'])
            self.ui.image_cwt_vb.addItem(self.ui.image_cwt[self.ui.comboBox_showcwt.currentText()][-1])
            self.ui.image_cwt_vb.setLimits(yMin=np.log10(0.9*self.ui.params['cwt']['scales']['min']*1e3), 
                                            yMax=np.log10(1.1*self.ui.params['cwt']['scales']['max']*1e3),
                                            minYRange=np.log10(1.1*self.ui.params['cwt']['scales']['max']*1e3)-np.log10(0.9*self.ui.params['cwt']['scales']['min']*1e3),
                                            maxYRange=np.log10(1.1*self.ui.params['cwt']['scales']['max']*1e3)-np.log10(0.9*self.ui.params['cwt']['scales']['min']*1e3))
            self.ui.image_cwt_vb.setYRange(np.log10(0.9*self.ui.params['cwt']['scales']['min']*1e3),np.log10(1.1*self.ui.params['cwt']['scales']['max']*1e3))
            # self.ui.image_cwt_vb.setLimits(yMin=-0.1*self.ui.params['cwt']['scales']['count'],
            #                                 yMax=1.1*self.ui.params['cwt']['scales']['count'],
            #                                 minYRange=1.2*self.ui.params['cwt']['scales']['count'],
            #                                 maxYRange=1.2*self.ui.params['cwt']['scales']['count'])
            # self.ui.image_cwt_vb.update()
        
    def update_events(self, events):
        # print(events)
        # colors = ['r','g','b']
        events.sort(order='time')
        for n,c in enumerate(self.ui.params['targets']['color']):
            self.ui.scatter_events[self.ui.params['targets']['name'][n]].setData(events[events['label']==n]['time'],np.log10(events[events['label']==n]['scale']*1e3))
            self.ui.scatter_events_time[self.ui.params['targets']['name'][n]].setData(events[events['label']==n]['time'],events[events['label']==n]['coeff'])
            # self.ui.scatter_events[self.ui.params['targets']['name'][n]].setZValue(99)
            # self.ui.scatter_events_time[self.ui.params['targets']['name'][n]].setZValue(99)
            # self.ui.scatter_events_vb.addItem(self.ui.scatter_events[self.ui.params['targets']['name'][n]])
            # self.ui.plot_line.getViewBox().addItem(self.ui.scatter_events_time[self.ui.params['targets']['name'][n]])
        self.ui.image_cwt_ax.addLegend()
        # self.ui.image_cwt_ax.getAxis('left').setLogMode(False)
        self.ui.scatter_events_vb.setLimits(yMin=np.log10(0.9*self.ui.params['cwt']['scales']['min']*1e3), 
                                            yMax=np.log10(1.1*self.ui.params['cwt']['scales']['max']*1e3),
                                            minYRange=np.log10(1.1*self.ui.params['cwt']['scales']['max']*1e3)-np.log10(0.9*self.ui.params['cwt']['scales']['min']*1e3),
                                            maxYRange=np.log10(1.1*self.ui.params['cwt']['scales']['max']*1e3)-np.log10(0.9*self.ui.params['cwt']['scales']['min']*1e3))
        self.ui.scatter_events_vb.setYRange(np.log10(0.9*self.ui.params['cwt']['scales']['min']*1e3),np.log10(1.1*self.ui.params['cwt']['scales']['max']*1e3))
        # self.ui.image_cwt_ax.getAxis('left').setLogMode(True)
        self.statusBar().showMessage(f"{len(events)} events detected")
        print(f"Total detected events: {len(events)}")
        self.ui.eventsmodel = mw.Events_Model(events)
        self.ui.tableView_events.setModel(self.ui.eventsmodel)
        self.ui.gen_summary(events)
        self.ui.gen_dist(events)
        # pxMode=True,)
        # [self.ui.image_cwt_ax.addItem(self.ui.scatter_events[k]) for k in self.ui.scatter_events.keys()]

    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'Quit',
                                     "Are you sure to quit?", QMessageBox.Yes |
                                     QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.ei_window.hide()
            event.accept()
        else:
            event.ignore()

if __name__ == '__main__':
    if sys.platform.startswith('win'):
        multiprocessing.freeze_support()
    app = QApplication(sys.argv)
    pixmap = QPixmap("splash.png")
    splash = QSplashScreen(pixmap)
    splash.show()

    # Loading some items
    QApplication.processEvents()
    # Checking for updates
    splash.showMessage(f'Checking for updates...',
                            Qt.AlignBottom | Qt.AlignLeft,
                            Qt.white)
    with open(__file__) as main_file:
        current_version = int(re.findall(r'version:.*\d*.\d*.\d*',main_file.read())[0].split(':')[1].replace(' ','').replace('.',''))
    try:
        r = requests.get('https://raw.github.com/vganjali/EventDetector/realtime/main.py',timeout=1)
        new_version = int(re.findall(r'version:.*\d*.\d*.\d*',r.text)[0].split(':')[1].replace(' ','').replace('.',''))
    except:
        new_version = 0
    if new_version > current_version:
        print('Updates are available')
        ret = QMessageBox.information(
            None,"Update",
            "Updates are available.\nDo you want to update?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes
        )
        if ret == QMessageBox.Yes:
            from pathlib import Path
            ul = requests.get(f'https://raw.github.com/vganjali/EventDetector/master/updatelist',timeout=1)
            update_list = ul.json()
            for f in update_list['files']:
                print(f)
                r = requests.get(f'https://raw.github.com/vganjali/EventDetector/master/{f}',timeout=3)
                with open(f'C:/Users/Public/appdata/local/SMD Analysis/{f}', 'wb') as f:
                    f.write(r.content)
            ret = QMessageBox.information(
                None,"Update",
                "SMD Analysis updated.\nPlease restart the program."
            )
            splash.finish(main_window)
            sys.exit(app.exec_())
            # Save was clicked
        elif ret == QMessageBox.No:
            print('')
        else:
            print('')
            # should never be reached

    for module,name in modules.items():
        splash.showMessage(f'Loading {module}...',
                            Qt.AlignBottom | Qt.AlignLeft,
                            Qt.white)
        if name == '':
            globals()[module] = importlib.import_module(module)
        else:
            globals()[name] = importlib.import_module(module)
        # QtCore.QThread.msleep(1000)
    # from mainwindow import *
    # import ptu
    # import wavelet
    # import eventdetector
    # import threading

    main_window = MainWindow()
    # threads_watch_thread = threading.Thread(target=main_window.threads_watch,args=(),daemon=True)
    # threads_watch_thread.start()

    main_window.show()
    splash.finish(main_window)
    main_window.ei_window.hide()
    sys.exit(app.exec_())
