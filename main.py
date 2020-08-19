import sys
import os
import time
import importlib
import multiprocessing
from concurrent.futures import ProcessPoolExecutor as Executor
from concurrent.futures import as_completed

from PySide2.QtGui import QPixmap
# from PySide2.QtUiTools import QUiLoader
from PySide2.QtWidgets import (QSplashScreen, QMainWindow, QApplication)
from PySide2.QtCore import Qt
# import pyqtgraph as pg
# import vaex
# import h5py
modules = {'mainwindow':'mw', 'ptu':'', 'wavelet':'', 'eventdetector':'ed', 'threading':'', 'pyqtgraph':'pg', 'numpy':'np'}

class MainWindow(QMainWindow):
	def __init__(self):
		super(MainWindow, self).__init__()
		self.ui = mw.Ui_MainWindow()
		self.ui.setupUi(self)
		self.ui.target_update_list()
		self.ptufile = ptu.ptu()
		self.eventdetector = ed.eventdetector()
		self.ptufile.progress.connect(self.ui.progressBar.setValue)
		self.ptufile.started.connect(self.ui.progressBar.setVisible)
		self.ptufile.started.connect(self.update_statusbar)
		self.ptufile.updateplot.connect(self.update_time_plot)
		self.ptufile.updateplot_timer.timeout.connect(self.update_time_plot_rt)
		self.eventdetector.progress.connect(self.ui.progressBar.setValue)
		self.eventdetector.started.connect(self.ui.progressBar.setVisible)
		self.eventdetector.started.connect(self.update_statusbar)
		# self.eventdetector.showevents.connect(print)
		self.eventdetector.showevents.connect(self.update_events)
		self.eventdetector.drawcwt.connect(self.update_cwt_plot)
		self.ui.listWidget_files.clicked.connect(lambda sig: self.read_file(filename=sig,update=True,relim=True))
		self.read_file_thread = threading.Thread(target=self.ptufile.processHT2_rt, args=(),daemon=False)
		self.ui.pushButton_realtime.clicked.connect(self.time_plot_rt)
		self.ui.doubleSpinBox_binsize.editingFinished.connect(lambda: self.read_file(filename=self.ui.listWidget_files.currentIndex(),update=True,relim=False))
		self.ui.pushButton_cwt.clicked.connect(self.detect_events)
		# with h5py.File("C:/Users/vahid/.vaex/data/helmi-dezeeuw-2000-FeH-v2-10percent.hdf5",'r') as f:
		# 	print(f['table/columns'].keys())

	def read_file(self, filename, update=True, relim=False):
		try:
			# self.ui.params['binsize'] = self.ui.doubleSpinBox_binsize.value()*1e-3
			# self.ui.params['buffer'] = self.ui.spinBox_buffersize.value()*1024
			if ((self.ui.params['currentdir']+filename.data() != self.ptufile.filename) or 
				(self.ui.params['binsize'] != self.ptufile.binsize)):
				self.ptufile.filename = self.ui.params['currentdir']+filename.data()
				self.ptufile.binsize = self.ui.params['binsize']
				self.ptufile.buffer = int(self.ui.params['buffer']*1024*1024/4)
				self.read_file_thread = threading.Thread(target=self.ptufile.processHT2, args=(update,relim,),daemon=False)
				# print(f"binning {ptufile.filename}")
				self.ui.progressBar.reset()
				self.read_file_thread.start()
		except Exception as e:
			print(e)
			pass
	
	def time_plot_rt(self):
		try:
			if not self.read_file_thread.is_alive():
				print('start')
				# self.ui.params['binsize'] = self.ui.doubleSpinBox_binsize.value()*1e-3
				# self.ui.params['window'] = self.ui.doubleSpinBox_windowsize.value()
				self.ptufile.filename = self.ui.params['currentdir']+self.ui.lineEdit_filename.text()+'.ptu'
				self.ptufile.binsize = self.ui.params['binsize']
				self.ptufile.buffer = int(self.ui.params['buffer']*1024*1024/20/4)
				self.ptufile.rt_active = True
				while not self.ptufile.queue.empty():
					self.ptufile.queue.get()
				self.ui.plot_line.setLabel('left',f"Intensity [cnts/{self.ptufile.binsize*1e3:g}ms]")
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
				self.read_file_thread = threading.Thread(target=self.ptufile.processHT2_rt, args=(),daemon=True)
				self.read_file_thread.start()
				self.ptufile.updateplot_timer.start()
				self.ui.pushButton_realtime.setText('Stop')
				self.statusBar().showMessage('Realtime Plotting...')
			else:
				print('stopped')
				self.ptufile.rt_active = False
				self.read_file_thread = threading.Thread(target=self.ptufile.processHT2_rt, args=(),daemon=False)
				self.ptufile.updateplot_timer.stop()
				while not self.ptufile.queue.empty():
					self.ptufile.queue.get()
				self.ui.pushButton_realtime.setText('Start')
				self.statusBar().showMessage('Idle')
			# self.plot_thread = threading.Thread(target=self.update_time_plot_worker, args=(),daemon=False).start()
		except Exception as e:
			print(e)
			while not self.ptufile.queue.empty():
				self.ptufile.queue.get()
			self.read_file_thread = threading.Thread(target=self.ptufile.processHT2_rt, args=(),daemon=False)
			self.ptufile.rt_active = False



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
		if self.ui.checkBox_store_cwt.checkState() == Qt.CheckState.Checked:
			self.eventdetector.cwt_plot = True
		else:
			self.eventdetector.cwt_plot = False
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
		# self.statusBar().showMessage('Idle')

	def generate_wavelets(self):
		import numpy as np
		_wavelets = {}
		_dt = self.ui.params['cwt']['scales']['min']/self.ui.params['cwt']['resolution']
		if self.ui.params['cwt']['log']:
			scales = np.logspace(np.log10(self.ui.params['cwt']['scales']['min']), np.log10(self.ui.params['cwt']['scales']['max']), self.ui.params['cwt']['scales']['count'], dtype=np.float64)
		else:
			scales = np.linspace(self.ui.params['cwt']['scales']['min'], self.ui.params['cwt']['scales']['max'], self.ui.params['cwt']['scales']['count'], dtype=np.float64)
		# print(scales)
		for n,name in enumerate(self.ui.params['targets']['name']):
			if self.ui.params['targets']['active'][n]:
				wname = self.ui.params['targets']['wavelet']['name'][n]
				args = self.ui.params['targets']['wavelet']['parameters'][n]
				_wavelets[name] = {'N': args['N'],
				'wavelets': [getattr(wavelet, wname.lower().replace('-','_').replace(' ','_').replace('(','').replace(')',''))(s, dt=_dt, **args) for s in scales]}
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
			self.ui.plot_line.setLabel('left',f"Intensity [cnts/{self.ptufile.binsize*1e3:g}ms]")
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
				self.ui.plot_line.setXRange(f[f"{self.ptufile.binsize:010.6f}"]['time'][0]*1e-12,f[f"{self.ptufile.binsize:010.6f}"]['time'][-1]*1e-12)
			self.ui.setYRange()
		self.statusBar().showMessage('Idle')

	def update_time_plot_rt(self):
		if not self.ptufile.queue.empty():
			buffer = self.ptufile.queue.get(block=False)
			# print('get:',buffer['time'][0]*1e-12,buffer['time'][-1]*1e-12)
			# self.ui.plot_line.plot(
			# 	buffer['time']*1e-12,
			# 	buffer['count'],
			# 	pen=pg.mkPen(color='b'),
			# 	clear=False,
			# 	name='CH 1'
			# 	# fillLevel=0,
			# 	# fillBrush=pg.mkBrush(color='b'),
			# 	)
			self.rt_plot_line.setData(
				np.append(self.rt_plot_line.xData,buffer['time']*1e-12),
				np.append(self.rt_plot_line.yData,buffer['count']))
			# self.ui.plot_line.setDownsampling(ds=True, auto=False, mode='mean')
			# self.ui.plot_line.setClipToView(clip=True)
			self.ptufile.queue.task_done()
			self.ui.plot_line.setXRange(max(0,buffer['time'][-1]*1e-12-self.ui.params['window']),buffer['time'][-1]*1e-12)
			self.ui.setYRange()
		# elif not self.read_file_thread.is_alive():
		# 	self.ptufile.updateplot_timer.stop()
		# 	self.ui.pushButton_realtime.setText('Start')
		# 	self.statusBar().showMessage('Idle')

	def update_cwt_plot(self,cwt):
		import numpy as np
		if self.eventdetector.cwt_plot == True:
			self.statusBar().showMessage('Updating CWT Plot')
			self.ui.image_cwt_ax.disableAutoRange()
			# print(cwt)
			_view_rect = mw.QRectF(
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
			# self.ui.image_cwt_ax.setLabel('left',f"Intensity [cnts/{self.ptufile.binsize*1e3:g}ms]")
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
				pg.ScatterPlotItem(events[events['label']==n]['time'],np.log10(events[events['label']==n]['scale']*1e3),
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
		self.ui.eventsmodel = mw.Events_Model(events)
		self.ui.tableView_events.setModel(self.ui.eventsmodel)
		self.ui.gen_summary(events)
		self.ui.gen_dist(events)
		# pxMode=True,)
		# [self.ui.image_cwt_ax.addItem(self.ui.scatter_events[k]) for k in self.ui.scatter_events.keys()]

if __name__ == '__main__':
	from PySide2.QtCore import Qt
	if sys.platform.startswith('win'):
		multiprocessing.freeze_support()
	app = QApplication(sys.argv)
	pixmap = QPixmap("splash.png")
	splash = QSplashScreen(pixmap)
	splash.show()

	# Loading some items
	QApplication.processEvents()
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
	sys.exit(app.exec_())
