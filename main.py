import sys
import os
import time
from mainwindow import *
import ptu
import wavelet
import eventdetector
import threading
from concurrent.futures import ProcessPoolExecutor as Executor
from concurrent.futures import as_completed

from PySide2.QtUiTools import QUiLoader
from PySide2.QtWidgets import *
from PySide2.QtCore import *
import pyqtgraph as pg
# import vaex
# import h5py

class MainWindow(QMainWindow):
	def __init__(self):
		super(MainWindow, self).__init__()
		self.ui = Ui_MainWindow()
		self.ui.setupUi(self)
		self.ui.target_update_list()
		self.ptufile = ptu.ptu()
		self.eventdetector = eventdetector.eventdetector()
		self.ptufile.progress.connect(self.ui.progressBar.setValue)
		self.ptufile.started.connect(self.ui.progressBar.setVisible)
		self.ptufile.started.connect(self.update_statusbar)
		self.ptufile.updateplot.connect(self.update_time_plot)
		self.eventdetector.progress.connect(self.ui.progressBar.setValue)
		self.eventdetector.started.connect(self.ui.progressBar.setVisible)
		self.eventdetector.started.connect(self.update_statusbar)
		# self.eventdetector.showevents.connect(print)
		self.eventdetector.showevents.connect(self.update_events)
		self.eventdetector.drawcwt.connect(self.update_cwt_plot)
		self.ui.listWidget_files.clicked.connect(lambda sig: self.read_file(sig,True))
		self.ui.doubleSpinBox_binsize.editingFinished.connect(lambda: self.read_file(self.ui.listWidget_files.currentIndex(),False))
		self.ui.pushButton_cwt.clicked.connect(self.detect_events)
		# with h5py.File("C:/Users/vahid/.vaex/data/helmi-dezeeuw-2000-FeH-v2-10percent.hdf5",'r') as f:
		# 	print(f['table/columns'].keys())

	def read_file(self, filename, relim=False):
		try:
			self.ui.params['binsize'] = self.ui.doubleSpinBox_binsize.value()*1e-3
			self.ui.params['window'] = [self.ui.doubleSpinBox_window_l.value(), self.ui.doubleSpinBox_window_r.value()]
			if ((self.ui.params['currentdir']+filename.data() != self.ptufile.filename) or 
				(self.ui.params['binsize'] != self.ptufile.binsize)):
				self.ptufile.filename = self.ui.params['currentdir']+filename.data()
				self.ptufile.binsize = self.ui.params['binsize']
				self.read_file_thread = threading.Thread(target=self.ptufile.processHT2, args=(relim,),daemon=False)
				# print(f"binning {ptufile.filename}")
				self.ui.progressBar.reset()
				self.read_file_thread.start()
		except Exception as e:
			print(e)
			pass

	def detect_events(self):
		import h5py, eventdetector
		print('started')
		self.statusBar().showMessage('Analyzing')
		_wavelets = self.generate_wavelets()
		# [print(k,len(_wavelets[k]['wavelets'])) for k in _wavelets.keys()]
		self.ui.params['cwt']['window'] = {'l':list(self.ui.plot_line.getAxis('bottom').range)[0],'r':list(self.ui.plot_line.getAxis('bottom').range)[1]}
		self.ptufile.binsize = self.ui.params['cwt']['scales']['min']/self.ui.params['cwt']['resolution']
		for k in self.ui.params['cwt']:
			try:
				setattr(self.eventdetector, k, self.ui.params['cwt'][k])
			except Exception as e:
				print(e)
		self.eventdetector.ptufile = self.ptufile
		self.detect_events_thread = threading.Thread(target=self.eventdetector.analyze_trace, args=(_wavelets,),daemon=False)
		# print(f"binning {ptufile.filename}")
		self.ui.progressBar.reset()
		self.ui.image_cwt = {name:[] for name in self.ui.params['targets']['name']}
		self.ui.scatter_events = {name:[] for name in self.ui.params['targets']['name']}
		self.ui.image_cwt_vb.clear()
		self.ui.scatter_events_vb.clear()
		self.detect_events_thread.start()
		# print(f"binning {ptufile.filename}")
		# events = self.eventdetector.analyze_trace(os.path.splitext(self.ptufile.filename)[0], _wavelets, **self.ui.params['cwt'])
		# print(events)
		self.statusBar().showMessage('Idle')

	def generate_wavelets(self):
		import numpy as np
		_wavelets = {}
		_dt = self.ui.params['cwt']['scales']['min']/self.ui.params['cwt']['resolution']
		if self.ui.params['cwt']['log']:
			scales = np.logspace(np.log10(self.ui.params['cwt']['scales']['min']), np.log10(self.ui.params['cwt']['scales']['max']), self.ui.params['cwt']['scales']['count'], dtype=np.float64)
		else:
			scales = np.linspace(self.ui.params['cwt']['scales']['min'], self.ui.params['cwt']['scales']['max'], self.ui.params['cwt']['scales']['count'], dtype=np.float64)
		print(scales)
		for n,name in enumerate(self.ui.params['targets']['name']):
			if self.ui.params['targets']['active'][n]:
				wname = self.ui.params['targets']['wavelet']['name'][n]
				args = self.ui.params['targets']['wavelet']['parameters'][n]
				_wavelets[name] = {'N': args['N'],
				'wavelets': [getattr(wavelet, wname.lower().replace('-','_').replace(' ','_'))(s, dt=_dt, **args) for s in scales]}
		return _wavelets

	def update_statusbar(self, signal):
		if (signal):
			self.statusBar().showMessage('Busy')
		else:
			self.statusBar().showMessage('Idle')

	def update_time_plot(self,relim=False):
		import h5py
		self.statusBar().showMessage('Updating Plot')
		with h5py.File(os.path.splitext(self.ptufile.filename)[0]+'.hdf5', 'r') as f:
			self.ui.plot_line.disableAutoRange()
			self.ui.plot_line.setLabel('left',f"Intenisty [cnts/{self.ptufile.binsize*1e3:g}ms]")
			self.ui.plot_line.plot(
				f[f"{self.ptufile.binsize:010.6f}"]['time'][:]*1e-12,
				f[f"{self.ptufile.binsize:010.6f}"]['count'][:],
				pen=pg.mkPen(color='b'),
				clear=True,
				name='CH 1'
				# fillLevel=0,
				# fillBrush=pg.mkBrush(color='b'),
				)
			# self.ui.plot_line.setDownsampling(ds=True, auto=False, mode='mean')
			# self.ui.plot_line.setClipToView(clip=True)
			if relim:
				xrange = [max(f[f"{self.ptufile.binsize:010.6f}"]['time'][0]*1e-12,self.ui.doubleSpinBox_window_l.value()),
						min(f[f"{self.ptufile.binsize:010.6f}"]['time'][-1]*1e-12,self.ui.doubleSpinBox_window_r.value())]
				if xrange[1] == 0: xrange[1] = f[f"{self.ptufile.binsize:010.6f}"]['time'][-1]*1e-12
				self.ui.plot_line.setXRange(xrange[0],xrange[1])
			self.ui.setYRange()
		self.statusBar().showMessage('Idle')


	def update_cwt_plot(self,cwt):
		import numpy as np
		self.statusBar().showMessage('Updating CWT Plot')
		self.ui.image_cwt_ax.disableAutoRange()
		# print(cwt)
		_view_rect = QRectF(
			cwt[1],
			0,
			cwt[0][list(cwt[0].keys())[0]].shape[1]*(self.ui.params['cwt']['scales']['min']/self.ui.params['cwt']['resolution']),
			self.ui.params['cwt']['scales']['count'])
		for key, value in cwt[0].items():
			self.ui.image_cwt[key].append(pg.ImageItem(image=value))
			self.ui.image_cwt[key][-1].setOpts(autoDownsample=True, 
												axisOrder='row-major', 
												# lut=self.ui.lut, 
												update=True)
			self.ui.image_cwt[key][-1].setRect(_view_rect)
		# self.ui.image_cwt_ax.setLabel('left',f"Intenisty [cnts/{self.ptufile.binsize*1e3:g}ms]")
		# self.ui.image_cwt.setImage(cwt[0]['6p'],
		# 	# pen=pg.mkPen(color='b'),
		# 	# clear=True,
		# 	name='CH 1',
		# 	autoRange=False
		# 	# fillLevel=0,
		# 	# fillBrush=pg.mkBrush(color='b'),
		# 	)
			# self.ui.plot_line.setDownsampling(ds=True, auto=False, mode='mean')
			# self.ui.plot_line.setClipToView(clip=True)
			# if relim:
			# 	xrange = [max(f[f"{ptufile.binsize:010.6f}"]['time'][0]*1e-12,self.ui.doubleSpinBox_window_l.value()),
			# 			min(f[f"{ptufile.binsize:010.6f}"]['time'][-1]*1e-12,self.ui.doubleSpinBox_window_r.value())]
			# 	if xrange[1] == 0: xrange[1] = f[f"{ptufile.binsize:010.6f}"]['time'][-1]*1e-12
			# 	self.ui.plot_line.setXRange(xrange[0],xrange[1])
			# self.ui.setYRange()
		# self.ui.image_cwt.scale(_view_rect.width/cwt['6p'].shape[1],_view_rect.heigth/cwt['6p'].shape[0])
		# self.ui.image_cwt.translate(self.ui.params['cwt']['window'])
		self.ui.image_cwt_vb.addItem(self.ui.image_cwt[self.ui.comboBox_showcwt.currentText()][-1])
		self.ui.image_cwt_vb.setLimits(yMin=0,
										yMax=self.ui.params['cwt']['scales']['count'],
										minYRange=self.ui.params['cwt']['scales']['count'],
										maxYRange=self.ui.params['cwt']['scales']['count'])
		# self.ui.image_cwt_vb.update()
		self.statusBar().showMessage('Idle')
		
	def update_events(self, events):
		import numpy as np
		# print(events)
		# colors = ['r','g','b']
		events.sort(order='time')
		for n,c in enumerate(self.ui.params['targets']['color']):
			self.ui.scatter_events[self.ui.params['targets']['name'][n]] = \
				pg.ScatterPlotItem(events[events['name']==n]['time'],np.log10(events[events['name']==n]['scale']*1e3),
				symbol='s',pen=pg.mkPen(color=eval(c.split('rgb')[-1]),width=2),brush=None,pxMode=True, size=10, 
				name=self.ui.params['targets']['name'][n])
			self.ui.scatter_events[self.ui.params['targets']['name'][n]].setZValue(99)
			self.ui.scatter_events_vb.addItem(self.ui.scatter_events[self.ui.params['targets']['name'][n]])
		self.ui.image_cwt_ax.addLegend()
		# self.ui.image_cwt_ax.getAxis('left').setLogMode(False)
		self.ui.scatter_events_vb.setLimits(yMin=np.log10(self.ui.params['cwt']['scales']['min']*1e3), 
											yMax=np.log10(self.ui.params['cwt']['scales']['max']*1e3),
											minYRange=np.log10(self.ui.params['cwt']['scales']['max']*1e3)-np.log10(self.ui.params['cwt']['scales']['min']*1e3),
											maxYRange=np.log10(self.ui.params['cwt']['scales']['max']*1e3)-np.log10(self.ui.params['cwt']['scales']['min']*1e3))
		self.ui.scatter_events_vb.setYRange(np.log10(self.ui.params['cwt']['scales']['min']*1e3),np.log10(self.ui.params['cwt']['scales']['max']*1e3))
		# self.ui.image_cwt_ax.getAxis('left').setLogMode(True)
		print(f"Total detected events: {len(events)}")
		self.ui.eventsmodel = Events_Model(events)
		self.ui.tableView_events.setModel(self.ui.eventsmodel)
		self.ui.gen_summary(events)
		self.ui.gen_dist(events)
		# pxMode=True,)
		# [self.ui.image_cwt_ax.addItem(self.ui.scatter_events[k]) for k in self.ui.scatter_events.keys()]

if __name__ == '__main__':
	app = QApplication(sys.argv)

	main_window = MainWindow()
	# threads_watch_thread = threading.Thread(target=main_window.threads_watch,args=(),daemon=True)
	# threads_watch_thread.start()

	main_window.show()
	sys.exit(app.exec_())
