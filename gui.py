import sys
import os
import json
from PySide2.QtWidgets import *
from PySide2.QtCore import *
from PySide2.QtGui import *

class MainWindow(QMainWindow):
	def __init__(self):
		super().__init__()
		self.resize(1000, 600)
		self.setWindowTitle("Event Detector")
		self.setWindowIcon(QIcon("icon.ico"))

		self.init_config()
		self.createMenus()
		self.createtoolbox()
		self.fileexplorer()
		self.filesetting()
		# self.menubar = QMenuBar(self)
		# self.menubar.addMenu("File")
		# self.menubar.addMenu("Edit")
		# self.menubar.addMenu("Help")
		self.statusBar().showMessage('Ready')
		self.progressBar = QProgressBar()
		self.statusBar().addPermanentWidget(self.progressBar)
		self.progressBar.setMaximum(100)
		self.progressBar.setVisible(False)

		self.dockwidget_l.setWidget(self.vsplitter)
		self.dockwidget_l.setFloating(False)
		# dockWidget_b.setWidget(self.folderlayout)
		# self.dockWidget_b.setFloating(False)

		# self.setCentralWidget(QTextBlock())
		self.addDockWidget(Qt.LeftDockWidgetArea, self.dockwidget_l)
		self.addDockWidget(Qt.BottomDockWidgetArea, self.dockwidget_b)
		# self.setStatusBar(self.statusbar)
	
	def init_config(self):
		if os.path.exists("config.json"):
			with open("config.json") as cf:
				self.params = json.load(cf)
		else:
			self.params = {
				'currentdir': '',
				'binsize': 1e-3,
				'xlim': [0,-1],
				'rebin': False
			}
			self.save_config()

	def load_config(self):
		filename = QFileDialog.getOpenFileName(self,'Load Configurations', self.params['currentdir'], 'JSON Files (*.json)')
		if filename[0] is not '':
			with open(filename[0], 'r') as cf:
				self.params = json.load(cf)

	def save_config(self):
		with open('config.json', 'w') as cf:
			json.dump(self.params, cf, sort_keys=True, indent=4)
			
	def save_config_as(self):
		filename = QFileDialog.getSaveFileName(self,'Save Configurations as', self.params['currentdir'], 'JSON Files (*.json)')
		if filename[0] is not '':
			with open(filename[0], 'w') as cf:
				json.dump(self.params, cf, sort_keys=True, indent=4)

	def createMenus(self):
		fileMenu = self.menuBar().addMenu("File")
		loadAction = QAction('Load', self)
		loadAction.setShortcut('Ctrl+o')
		loadAction.setStatusTip('Load configurations')
		loadAction.triggered.connect(self.load_config)
		saveAction = QAction('Save', self)
		saveAction.setShortcut('Ctrl+s')
		saveAction.setStatusTip('Save default configurations')
		saveAction.triggered.connect(self.save_config)
		saveasAction = QAction('Save as', self)
		saveasAction.setShortcut('Ctrl+Shift+s')
		saveasAction.setStatusTip('Save configurations as')
		saveasAction.triggered.connect(self.save_config_as)
		exitAction = QAction('Exit', self)
		exitAction.setShortcut('Ctrl+Q')
		exitAction.setStatusTip('Exit application')
		exitAction.triggered.connect(self.close)
		# fileMenu.addAction(Act)
		fileMenu.addAction(loadAction)
		fileMenu.addAction(saveAction)
		fileMenu.addAction(saveasAction)
		fileMenu.addSeparator()
		fileMenu.addAction(exitAction)
		fileMenu
	
	def createtoolbox(self):
		self.toolbox = QToolBox()
		

	def fileexplorer(self):
		self.dockwidget_l = QDockWidget('File Explorer', self)
		self.dockwidget_l.setFeatures(QDockWidget.DockWidgetFloatable)
		self.filelist = QListWidget()
		self.dirmodel = QFileSystemModel()
		self.dirmodel.setRootPath(self.params['currentdir'])
		self.dirmodel.setFilter(QDir.AllDirs | QDir.NoDotAndDotDot)
		self.foldertree =  QTreeView()
		self.foldertree.setModel(self.dirmodel)
		self.foldertree.setCurrentIndex(self.dirmodel.index(self.params['currentdir']))
		self.foldertree.setSortingEnabled(False)
		self.foldertree.hideColumn(1)
		self.foldertree.hideColumn(2)
		self.foldertree.hideColumn(3)
		self.foldertree.setHeaderHidden(True)
		self.get_file_list(self.dirmodel.index(self.params['currentdir']))
		self.foldertree.clicked.connect(self.get_file_list)
		self.vsplitter = QSplitter(orientation=Qt.Vertical)
		self.vsplitter.addWidget(self.foldertree)
		self.vsplitter.addWidget(self.filelist)
		self.dockwidget_l.setWidget(self.vsplitter)
		self.dockwidget_l.setFloating(False)

	def filesetting(self):
		self.dockwidget_b = QDockWidget('Tools', self)
		self.dockwidget_b.setFeatures(QDockWidget.DockWidgetFloatable)
		self.dockwidget_b_layout = QHBoxLayout()
		self.tabtools = QTabWidget()
		self.tabtools.setTabPosition(QTabWidget.TabPosition.West)
		# self.tabtools.setLayout(self.dockwidget_b_layout)
		# self.dockwidget_b_layout.addWidget(self.tabtools)
		self.dockwidget_b_layout.addWidget(self.tabtools)
		self.ptutab = QWidget()
		self.ptu = {
			'binsize':{'label':QLabel('Bin size [ms]:'), 'input':[QDoubleSpinBox()]},
			'window':{'label':QLabel('Window [s]:'), 'input':[QDoubleSpinBox(),QDoubleSpinBox()]},
			'sparse':QCheckBox('Sparse'),
			}
		self.ptu['binsize']['input'][0].setDecimals(3)
		self.ptu['window']['input'][0].setDecimals(3)
		self.ptu['window']['input'][1].setDecimals(3)
		self.ptutablayout = QGridLayout()
		# [self.ptutablayout.addWidget(self.ptu[k]['label'],n,0) for n,k in enumerate(['binsize','window'])]
		# [self.ptutablayout.addWidget(w,n,m+1) for n,k in enumerate(['binsize','window']) for m,w in enumerate(self.ptu[k]['input'])]
		# self.ptutablayout.addWidget(self.ptu['sparse'],2,0)
		# self.ptutab.setLayout(self.ptutablayout)
		# self.wavelettab = QWidget(self.tabtools)
		# self.wavelettab.setLayout(self.ptutablayout)
		self.tabtools.addTab(self.ptutab,'PTU')
		# self.tabtools.addTab(self.wavelettab,'Wavelet')
		self.dockwidget_b.setLayout(self.dockwidget_b_layout)
		self.dockwidget_b.setFloating(False)

	def get_file_list(self, signal):
		self.params['currentdir'] = self.dirmodel.filePath(signal)+'/'
		files = QDir(self.dirmodel.filePath(signal),"*.ptu *.PTU *.csv *.CSV *.txt *.TXT").entryList()
		self.filelist.clear()
		[self.filelist.addItem(f) for f in files]
