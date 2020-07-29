import os, json
from PySide2.QtCore import (QCoreApplication, QDate, QDateTime, QMetaObject,
    QObject, QPoint, QRect, QSize, QTime, QUrl, QDir, Qt, QModelIndex, QAbstractTableModel)
from PySide2.QtGui import (QBrush, QColor, QConicalGradient, QCursor, QFont,
    QFontDatabase, QIcon, QKeySequence, QLinearGradient, QPalette, QPainter,
    QPixmap, QRadialGradient, QStandardItem, QStandardItemModel)
from PySide2.QtWidgets import *

from matplotlib.backends.qt_compat import QtCore, QtWidgets, is_pyqt5
if is_pyqt5():
    from matplotlib.backends.backend_qt5agg import (
        FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
else:
    from matplotlib.backends.backend_qt4agg import (
        FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure
from matplotlib import cm
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 6, 'figure.facecolor': 'none'})
import numpy as np
# from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
# from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
# from matplotlib import pyplot as plt
# from PySide2.QtWebEngine import QtWebEngine
# from PySide2.QtWidgets import QApplication
# from PySide2.QtWebEngineWidgets import *
import pyqtgraph as pg

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        self.config_init()
        MainWindow.resize(1200, 800)
        icon = QIcon(u"icon.ico")
        MainWindow.setWindowIcon(icon)
        self.actionLoad = QAction(MainWindow)
        self.actionLoad.setObjectName(u"actionLoad")
        self.actionLoad.setShortcut('Ctrl+O')
        self.actionSave = QAction(MainWindow)
        self.actionSave.setObjectName(u"actionSave")
        self.actionSave.setShortcut('Ctrl+S')
        self.actionSave_As = QAction(MainWindow)
        self.actionSave_As.setObjectName(u"actionSave_As")
        self.actionSave_As.setShortcut('Ctrl+Shift+S')
        self.actionExit = QAction(MainWindow)
        self.actionExit.setObjectName(u"actionExit")
        self.actionExit.setShortcut('Ctrl+Q')
        self.actionHelp = QAction(MainWindow)
        self.actionHelp.setObjectName(u"actionHelp")
        self.actionAbout = QAction(MainWindow)
        self.actionAbout.setObjectName(u"actionAbout")
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.verticalLayout_centralwidget = QVBoxLayout(self.centralwidget)
        self.verticalLayout_centralwidget.setObjectName(u"verticalLayout_centralwidget")
        self.verticalLayout_centralwidget.setContentsMargins(0, 0, 0, 0)
        self.splitter_plot = QSplitter(self.centralwidget)
        self.splitter_plot.setObjectName(u"splitter_plot")
        self.splitter_plot.setOrientation(Qt.Vertical)
        self.splitter_plot.setChildrenCollapsible(False)
        # self.widget_plot = QWidget(self.centralwidget)
        # self.widget_plot.setObjectName(u"widget_plot")
        # QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
        # QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_ShareOpenGLContexts)
        pg.setConfigOptions(background='w', foreground=(30,30,30), leftButtonPan=False, antialias=True)
        self.widget_plot_timetrace = pg.GraphicsLayoutWidget(parent=self.splitter_plot, show=True)
        self.widget_plot_timetrace.setObjectName(u"widget_plot_timetrace")
        self.widget_plot_timetrace.setAntialiasing(False)
        self.widget_plot_timetrace.setCursor(QCursor(Qt.CrossCursor))
        self.plot_line = pg.PlotItem(parrent=self.widget_plot_timetrace)
        self.widget_plot_timetrace.addItem(self.plot_line)
        self.plot_line.setDownsampling(auto=True)
        self.plot_line.setClipToView(clip=True)
        self.plot_line.setLabels(left='Intenisty [Counts/s]', bottom='Time [s]')
        # self.plot_line.enableAutoRange(self.plot_line.vb.YAxis,True)
        self.plot_line.disableAutoRange()

        self.widget_plot_cwt = pg.GraphicsLayoutWidget(parent=self.splitter_plot, show=True)
        self.widget_plot_cwt.setObjectName(u"widget_plot_cwt")
        self.widget_plot_cwt.setAntialiasing(False)
        # self.widget_plot_cwt.setBackground('k')
        self.widget_plot_cwt.setCursor(QCursor(Qt.CrossCursor))
        self.image_cwt_vb = pg.ViewBox()
        self.scatter_events_vb = pg.ViewBox()
        self.image_cwt = {}
        self.scatter_events = {}
        # import numpy as np
        # self.image_cwt = pg.ImageItem(np.random.random((100,20)))
        # from matplotlib import cm
        # pos, rgba_colors = zip(*cmapToColormap(cm.cubehelix))
        # self.cmap = pg.ColorMap(pos, rgba_colors)
        # self.image_cwt.setLookupTable(cmap.getLookupTable())
        # self.image_cwt.setAutoDownsample(True)
        # self.scatter_events = pg.ScatterPlotItem([1,4,55,12],[4,1,6,3],
        #                         symbol='s',pen='g',brush=None,pxMode=True, size=6, name='ss')
        self.image_cwt_ax = self.widget_plot_cwt.addPlot()
        # self.image_cwt.setParentItem(self.image_cwt_ax)
        # self.scatter_events.setParentItem(self.image_cwt_ax)
        # self.image_cwt_ax.showAxis('right')
        self.image_cwt_ax.scene().addItem(self.image_cwt_vb)
        self.image_cwt_ax.scene().addItem(self.scatter_events_vb)
        self.image_cwt_vb.setGeometry(self.image_cwt_ax.vb.sceneBoundingRect())
        self.scatter_events_vb.setGeometry(self.image_cwt_ax.vb.sceneBoundingRect())
        self.image_cwt_ax.getAxis('right').linkToView(self.image_cwt_vb)
        self.image_cwt_ax.getAxis('left').linkToView(self.scatter_events_vb)
        self.image_cwt_vb.setXLink(self.image_cwt_ax)
        self.scatter_events_vb.setXLink(self.image_cwt_ax)
        # self.image_cwt_ax.enab
        self.image_cwt_ax.getAxis('left').setLogMode(True)
        # self.image_cwt_ax.getAxis('right').setLogMode(False)
        # self.image_cwt_vb.addItem(self.image_cwt)
        # self.scatter_events_vb.addItem(self.scatter_events)
        # self.legend = self.image_cwt_ax.addLegend()
        # Get the colormap
        # colormap = cm.get_cmap("RdBu_r")  # cm.get_cmap("CMRmap")
        # colormap._init()
        # colormap = pg.GradientEditorItem()   # for example
        # colormap.loadPreset('thermal')
        # # img.setLookupTable(map.getLookupTable())
        # self.lut = colormap.getLookupTable(256)
        # # self.lut = (colormap._lut[:-1] * 255).view(np.ndarray)  # Convert matplotlib colormap from 0-1 to 0 -255 for Qt

        # self.scatter_events.setZValue(99)
        self.image_cwt_ax.setXLink(self.plot_line)
        self.image_cwt_ax.setLabels(left='Scale [ms]', bottom='Time [s]')
        # self.image_cwt.setLabels(left='Intenisty [Counts/s]', bottom='Time [s]')
        # self.image_cwt.disableAutoRange()
        # self.time_ax = MyStaticMplCanvas(MainWindow, width=5, height=4, dpi=100)
        self.splitter_plot.addWidget(self.widget_plot_timetrace)
        self.splitter_plot.addWidget(self.widget_plot_cwt)
        self.verticalLayout_centralwidget.addWidget(self.splitter_plot)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 1200, 21))
        self.menuFile = QMenu(self.menubar)
        self.menuFile.setObjectName(u"menuFile")
        self.menuHelp = QMenu(self.menubar)
        self.menuHelp.setObjectName(u"menuHelp")
        MainWindow.setMenuBar(self.menubar)
        self.dockWidget_controlpanel = QDockWidget(MainWindow)
        self.dockWidget_controlpanel.setObjectName(u"dockWidget_controlpanel")
        self.dockWidget_controlpanel.setFeatures(QDockWidget.DockWidgetFloatable|QDockWidget.DockWidgetMovable|QDockWidget.DockWidgetVerticalTitleBar)
        self.dockWidget_controlpanel.setAllowedAreas(Qt.LeftDockWidgetArea|Qt.RightDockWidgetArea)
        self.dockWidget_controlpanel_Contents = QWidget()
        self.dockWidget_controlpanel_Contents.setObjectName(u"dockWidget_controlpanel_Contents")
        self.verticalLayout_binningandwindow_2 = QVBoxLayout(self.dockWidget_controlpanel_Contents)
        self.verticalLayout_binningandwindow_2.setObjectName(u"verticalLayout_binningandwindow_2")
        self.verticalLayout_binningandwindow_2.setContentsMargins(4, 4, 4, 4)
        self.toolBox = QToolBox(self.dockWidget_controlpanel_Contents)
        self.toolBox.setObjectName(u"toolBox")
        self.explorer_page = QWidget()
        self.explorer_page.setObjectName(u"explorer_page")
        self.explorer_page.setGeometry(QRect(0, 0, 360, 501))
        self.verticalLayout_explorer = QVBoxLayout(self.explorer_page)
        self.verticalLayout_explorer.setObjectName(u"verticalLayout_explorer")
        self.verticalLayout_explorer.setContentsMargins(0, 0, 0, 0)
        self.splitter_explorer = QSplitter(self.explorer_page)
        self.splitter_explorer.setObjectName(u"splitter_explorer")
        self.splitter_explorer.setOrientation(Qt.Vertical)
        self.splitter_explorer.setChildrenCollapsible(False)
        self.listWidget_files = QListWidget(self.splitter_explorer)
        self.listWidget_files.setObjectName(u"listWidget_files")
        self.listWidget_files.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.dirmodel = QFileSystemModel()
        self.dirmodel.setRootPath(self.params['currentdir'])
        self.dirmodel.setFilter(QDir.AllDirs | QDir.NoDotAndDotDot)
        self.treeView_explorer = QTreeView(self.splitter_explorer)
        self.treeView_explorer.setObjectName(u"treeView_explorer")
        self.treeView_explorer.setIndentation(10)
        self.treeView_explorer.setWordWrap(True)
        self.treeView_explorer.setModel(self.dirmodel)
        self.treeView_explorer.setCurrentIndex(self.dirmodel.index(self.params['currentdir']))
        self.treeView_explorer.setSortingEnabled(False)
        self.treeView_explorer.hideColumn(1)
        self.treeView_explorer.hideColumn(2)
        self.treeView_explorer.hideColumn(3)
        self.treeView_explorer.setHeaderHidden(True)
        self.get_file_list(self.dirmodel.index(self.params['currentdir']))
        self.splitter_explorer.addWidget(self.treeView_explorer)
        self.treeView_explorer.header().setVisible(False)
        self.splitter_explorer.addWidget(self.listWidget_files)

        self.verticalLayout_explorer.addWidget(self.splitter_explorer)

        self.toolBox.addItem(self.explorer_page, u"File Explorer")

        self.device_page = QWidget()
        self.device_page.setObjectName(u"device_page")
        self.device_page.setGeometry(QRect(0, 0, 292, 518))
        self.verticalLayout_fileinfo = QVBoxLayout(self.device_page)
        self.verticalLayout_fileinfo.setObjectName(u"verticalLayout_fileinfo")
        self.verticalLayout_fileinfo.setContentsMargins(0, 0, 0, 10)
        self.groupBox = QGroupBox(self.device_page)
        self.groupBox.setObjectName(u"groupBox")
        self.groupBox.setAlignment(Qt.AlignCenter)
        self.gridLayout_3 = QGridLayout(self.groupBox)
        self.gridLayout_3.setObjectName(u"gridLayout_3")
        self.label_3 = QLabel(self.groupBox)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout_3.addWidget(self.label_3, 1, 0, 1, 1)

        self.label_4 = QLabel(self.groupBox)
        self.label_4.setObjectName(u"label_4")
        self.label_4.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout_3.addWidget(self.label_4, 1, 2, 1, 1)

        
        self.canvas_exc = FigureCanvas(Figure(figsize=(1, 1),constrained_layout=True))
        self.canvas_exc.setStyleSheet("background-color:transparent;")
        self.ax_exc = self.canvas_exc.figure.subplots()
        # self.ax_exc.plot([1,2],[4,2])

        self.gridLayout_3.addWidget(self.canvas_exc, 0, 0, 1, 4)

        self.spinBox_n_exc = QSpinBox(self.groupBox)
        self.spinBox_n_exc.setObjectName(u"spinBox_n_exc")

        self.gridLayout_3.addWidget(self.spinBox_n_exc, 1, 1, 1, 1)

        self.doubleSpinBox_v_exc = QDoubleSpinBox(self.groupBox)
        self.doubleSpinBox_v_exc.setObjectName(u"doubleSpinBox_v_exc")

        self.gridLayout_3.addWidget(self.doubleSpinBox_v_exc, 1, 3, 1, 1)

        self.label = QLabel(self.groupBox)
        self.label.setObjectName(u"label")
        self.label.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout_3.addWidget(self.label, 2, 0, 1, 1)

        self.doubleSpinBox_h_exc = QDoubleSpinBox(self.groupBox)
        self.doubleSpinBox_h_exc.setObjectName(u"doubleSpinBox_h_exc")

        self.gridLayout_3.addWidget(self.doubleSpinBox_h_exc, 2, 3, 1, 1)

        self.doubleSpinBox_w_exc = QDoubleSpinBox(self.groupBox)
        self.doubleSpinBox_w_exc.setObjectName(u"doubleSpinBox_w_exc")

        self.gridLayout_3.addWidget(self.doubleSpinBox_w_exc, 2, 1, 1, 1)

        self.label_2 = QLabel(self.groupBox)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout_3.addWidget(self.label_2, 2, 2, 1, 1)


        self.verticalLayout_fileinfo.addWidget(self.groupBox)

        self.groupBox_2 = QGroupBox(self.device_page)
        self.groupBox_2.setObjectName(u"groupBox_2")
        self.groupBox_2.setAlignment(Qt.AlignCenter)
        self.gridLayout_4 = QGridLayout(self.groupBox_2)
        self.gridLayout_4.setObjectName(u"gridLayout_4")
        self.label_6 = QLabel(self.groupBox_2)
        self.label_6.setObjectName(u"label_6")
        self.label_6.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout_4.addWidget(self.label_6, 1, 2, 1, 1)

        self.doubleSpinBox_w_ch = QDoubleSpinBox(self.groupBox_2)
        self.doubleSpinBox_w_ch.setObjectName(u"doubleSpinBox_w_ch")

        self.gridLayout_4.addWidget(self.doubleSpinBox_w_ch, 1, 1, 1, 1)

        self.label_5 = QLabel(self.groupBox_2)
        self.label_5.setObjectName(u"label_5")
        self.label_5.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout_4.addWidget(self.label_5, 1, 0, 1, 1)

        self.doubleSpinBox_h_ch = QDoubleSpinBox(self.groupBox_2)
        self.doubleSpinBox_h_ch.setObjectName(u"doubleSpinBox_h_ch")

        self.gridLayout_4.addWidget(self.doubleSpinBox_h_ch, 1, 3, 1, 1)

        self.widget_3 = QWidget(self.groupBox_2)
        self.widget_3.setObjectName(u"widget_3")
        self.widget_3.setMinimumSize(QSize(0, 40))

        self.gridLayout_4.addWidget(self.widget_3, 0, 0, 1, 4)


        self.verticalLayout_fileinfo.addWidget(self.groupBox_2)

        self.groupBox_3 = QGroupBox(self.device_page)
        self.groupBox_3.setObjectName(u"groupBox_3")
        self.groupBox_3.setAlignment(Qt.AlignCenter)
        self.gridLayout_5 = QGridLayout(self.groupBox_3)
        self.gridLayout_5.setObjectName(u"gridLayout_5")
        self.label_8 = QLabel(self.groupBox_3)
        self.label_8.setObjectName(u"label_8")
        self.label_8.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout_5.addWidget(self.label_8, 1, 2, 1, 1)

        self.doubleSpinBox_h_c = QDoubleSpinBox(self.groupBox_3)
        self.doubleSpinBox_h_c.setObjectName(u"doubleSpinBox_h_c")

        self.gridLayout_5.addWidget(self.doubleSpinBox_h_c, 1, 3, 1, 1)

        self.label_7 = QLabel(self.groupBox_3)
        self.label_7.setObjectName(u"label_7")
        self.label_7.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout_5.addWidget(self.label_7, 1, 0, 1, 1)

        self.doubleSpinBox_w_c = QDoubleSpinBox(self.groupBox_3)
        self.doubleSpinBox_w_c.setObjectName(u"doubleSpinBox_w_c")

        self.gridLayout_5.addWidget(self.doubleSpinBox_w_c, 1, 1, 1, 1)

        self.widget_4 = QWidget(self.groupBox_3)
        self.widget_4.setObjectName(u"widget_4")
        self.widget_4.setMinimumSize(QSize(0, 40))

        self.gridLayout_5.addWidget(self.widget_4, 0, 0, 1, 4)


        self.verticalLayout_fileinfo.addWidget(self.groupBox_3)

        self.verticalSpacer_2 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.verticalLayout_fileinfo.addItem(self.verticalSpacer_2)

        self.toolBox.addItem(self.device_page, u"Device")
        self.target_page = QWidget()
        self.target_page.setObjectName(u"target_page")
        self.target_page.setGeometry(QRect(0, 0, 360, 501))
        self.verticalLayout_target = QVBoxLayout(self.target_page)
        self.verticalLayout_target.setObjectName(u"verticalLayout_target")
        self.verticalLayout_target.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_targetlist = QGridLayout()
        self.gridLayout_targetlist.setObjectName(u"gridLayout_targetlist")
        self.pushButton_targetadd = QPushButton(self.target_page)
        self.pushButton_targetadd.setObjectName(u"pushButton_targetadd")

        self.gridLayout_targetlist.addWidget(self.pushButton_targetadd, 1, 0, 1, 1)

        self.pushButton_targetdelete = QPushButton(self.target_page)
        self.pushButton_targetdelete.setObjectName(u"pushButton_targetdelete")

        self.gridLayout_targetlist.addWidget(self.pushButton_targetdelete, 1, 1, 1, 1)

        self.listView_targets = QListView(self.target_page)
        self.listView_targets.setObjectName(u"listView_targets")
        self.targetsmodel = QStandardItemModel()
        self.listView_targets.setModel(self.targetsmodel)

        self.gridLayout_targetlist.addWidget(self.listView_targets, 0, 0, 1, 2)
        

        self.verticalLayout_target.addLayout(self.gridLayout_targetlist)

        self.groupBox_targetinfo = QGroupBox(self.target_page)
        self.groupBox_targetinfo.setObjectName(u"groupBox_targetinfo")
        self.groupBox_targetinfo.setAlignment(Qt.AlignCenter)
        self.verticalLayout_targetinfo = QVBoxLayout(self.groupBox_targetinfo)
        self.verticalLayout_targetinfo.setObjectName(u"verticalLayout_targetinfo")
        self.verticalLayout_targetinfo.setContentsMargins(4, 4, 4, 4)
        self.gridLayout_targetinfo = QGridLayout()
        self.gridLayout_targetinfo.setObjectName(u"gridLayout_targetinfo")
        self.label_targetnote = QLabel(self.groupBox_targetinfo)
        self.label_targetnote.setObjectName(u"label_targetnote")

        self.gridLayout_targetinfo.addWidget(self.label_targetnote, 4, 0, 1, 1)

        self.label_color = QLabel(self.groupBox_targetinfo)
        self.label_color.setObjectName(u"label_color")

        self.gridLayout_targetinfo.addWidget(self.label_color, 1, 0, 1, 1)
        self.pushButton_targetcolor = QPushButton(self.groupBox_targetinfo)
        self.pushButton_targetcolor.setObjectName(u"pushButton_targetcolor")
        self.pushButton_targetcolor.setMinimumSize(QSize(0, 24))
        self.pushButton_targetcolor.setStyleSheet(u"background-color: rgb(255, 0, 0);\n"
"border: none;")
        self.pushButton_targetcolor.setFlat(True)

        self.gridLayout_targetinfo.addWidget(self.pushButton_targetcolor, 1, 1, 1, 1)

        self.label_targetsize = QLabel(self.groupBox_targetinfo)
        self.label_targetsize.setObjectName(u"label_targetsize")

        self.gridLayout_targetinfo.addWidget(self.label_targetsize, 3, 0, 1, 1)

        self.doubleSpinBox_targetsize = QDoubleSpinBox(self.groupBox_targetinfo)
        self.doubleSpinBox_targetsize.setObjectName(u"doubleSpinBox_targetsize")
        self.doubleSpinBox_targetsize.setDecimals(3)
        self.doubleSpinBox_targetsize.setMinimum(0.001000000000000)
        self.doubleSpinBox_targetsize.setMaximum(100.000000000000000)
        self.doubleSpinBox_targetsize.setSingleStep(0.100000000000000)
        self.doubleSpinBox_targetsize.setValue(1.000000000000000)

        self.gridLayout_targetinfo.addWidget(self.doubleSpinBox_targetsize, 3, 1, 1, 1)

        self.doubleSpinBox_targetconcentration = QDoubleSpinBox(self.groupBox_targetinfo)
        self.doubleSpinBox_targetconcentration.setObjectName(u"doubleSpinBox_targetconcentration")
        self.doubleSpinBox_targetconcentration.setDecimals(3)
        self.doubleSpinBox_targetconcentration.setValue(1.000000000000000)

        self.gridLayout_targetinfo.addWidget(self.doubleSpinBox_targetconcentration, 2, 1, 1, 1)

        self.label_targetconcentration = QLabel(self.groupBox_targetinfo)
        self.label_targetconcentration.setObjectName(u"label_targetconcentration")

        self.gridLayout_targetinfo.addWidget(self.label_targetconcentration, 2, 0, 1, 1)

        self.plainTextEdit_targetnote = QPlainTextEdit(self.groupBox_targetinfo)
        self.plainTextEdit_targetnote.setObjectName(u"plainTextEdit_targetnote")

        self.gridLayout_targetinfo.addWidget(self.plainTextEdit_targetnote, 5, 0, 1, 2)

        self.pushButton_targetsave = QPushButton(self.groupBox_targetinfo)
        self.pushButton_targetsave.setObjectName(u"pushButton_targetsave")

        self.gridLayout_targetinfo.addWidget(self.pushButton_targetsave, 4, 1, 1, 1)


        self.verticalLayout_targetinfo.addLayout(self.gridLayout_targetinfo)


        self.verticalLayout_target.addWidget(self.groupBox_targetinfo)

        self.toolBox.addItem(self.target_page, u"Target")
        self.wavelet_page = QWidget()
        self.wavelet_page.setObjectName(u"wavelet_page")
        self.wavelet_page.setGeometry(QRect(0, 0, 360, 501))
        self.verticalLayout_wavelet = QVBoxLayout(self.wavelet_page)
        self.verticalLayout_wavelet.setObjectName(u"verticalLayout_wavelet")
        self.verticalLayout_wavelet.setContentsMargins(0, 0, 0, 0)
        self.widget_plot_wavelet = pg.GraphicsLayoutWidget(parent=self.wavelet_page, show=True)
        self.widget_plot_wavelet.setAntialiasing(True)
        self.widget_plot_wavelet.setBackground(None)
        self.waveletplot_line = self.widget_plot_wavelet.addPlot(parrent=self.widget_plot_wavelet)
        # self.plot_line.vb.sigXRangeChanged.connect(self.setYRange)
        self.waveletplot_line.hideAxis('left')
        self.waveletplot_line.hideAxis('bottom')
        self.waveletplot_line.hideButtons()
        # self.plot_line.enableAutoRange(self.plot_line.vb.YAxis,True)
        self.waveletplot_line.enableAutoRange()

        self.verticalLayout_wavelet.addWidget(self.widget_plot_wavelet)

        self.groupBox_waveletparameters = QGroupBox(self.wavelet_page)
        self.groupBox_waveletparameters.setObjectName(u"groupBox_waveletparameters")
        self.groupBox_waveletparameters.setAlignment(Qt.AlignCenter)
        self.verticalLayout_waveletparameters = QVBoxLayout(self.groupBox_waveletparameters)
        self.verticalLayout_waveletparameters.setObjectName(u"verticalLayout_waveletparameters")
        self.verticalLayout_waveletparameters.setContentsMargins(4, 4, 4, 4)
        self.gridLayout_waveletparameters = QGridLayout()
        self.gridLayout_waveletparameters.setObjectName(u"gridLayout_waveletparameters")
        self.comboBox_target = QComboBox(self.groupBox_waveletparameters)
        self.comboBox_target.setObjectName(u"comboBox_target")
        self.comboBox_target.setEditable(False)
        self.comboBox_target.setModel(self.targetsmodel)

        self.gridLayout_waveletparameters.addWidget(self.comboBox_target, 0, 1, 1, 2)

        self.label_wavelet = QLabel(self.groupBox_waveletparameters)
        self.label_wavelet.setObjectName(u"label_wavelet")

        self.gridLayout_waveletparameters.addWidget(self.label_wavelet, 1, 0, 1, 1)

        self.label_target = QLabel(self.groupBox_waveletparameters)
        self.label_target.setObjectName(u"label_target")

        self.gridLayout_waveletparameters.addWidget(self.label_target, 0, 0, 1, 1)

        self.comboBox_wavelet = QComboBox(self.groupBox_waveletparameters)
        self.comboBox_wavelet.setObjectName(u"comboBox_wavelet")
        sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(3)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.comboBox_wavelet.sizePolicy().hasHeightForWidth())
        self.comboBox_wavelet.setSizePolicy(sizePolicy)
        [self.comboBox_wavelet.addItem(w) for w in ['Multi-spot Gaussian', 'Multi-spot Gaussian (encoded)','Morlet', 'Morlet_Complex', 'Ricker']]

        self.gridLayout_waveletparameters.addWidget(self.comboBox_wavelet, 1, 1, 1, 1)

        self.lineEdit_N = QLineEdit(self.groupBox_waveletparameters)
        self.lineEdit_N.setObjectName(u"lineEdit_N")
        sizePolicy1 = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        sizePolicy1.setHorizontalStretch(1)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.lineEdit_N.sizePolicy().hasHeightForWidth())
        self.lineEdit_N.setSizePolicy(sizePolicy1)

        self.gridLayout_waveletparameters.addWidget(self.lineEdit_N, 1, 2, 1, 1)

        self.groupBox_wavelet_customize = QGroupBox(self.groupBox_waveletparameters)
        self.groupBox_wavelet_customize.setObjectName(u"groupBox_wavelet_customize")
        self.groupBox_wavelet_customize.setCheckable(True)
        self.groupBox_wavelet_customize.setChecked(False)
        self.verticalLayout = QVBoxLayout(self.groupBox_wavelet_customize)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(4, 4, 4, 4)
        self.gridLayout = QGridLayout()
        self.gridLayout.setObjectName(u"gridLayout")
        self.doubleSpinBox_mod = QDoubleSpinBox(self.groupBox_wavelet_customize)
        self.doubleSpinBox_mod.setObjectName(u"doubleSpinBox_mod")
        self.doubleSpinBox_mod.setMaximum(1.000000000000000)
        self.doubleSpinBox_mod.setSingleStep(0.100000000000000)

        self.gridLayout.addWidget(self.doubleSpinBox_mod, 0, 3, 1, 1)

        self.label_mod = QLabel(self.groupBox_wavelet_customize)
        self.label_mod.setObjectName(u"label_mod")

        self.gridLayout.addWidget(self.label_mod, 0, 0, 1, 1)

        self.label_shift = QLabel(self.groupBox_wavelet_customize)
        self.label_shift.setObjectName(u"label_shift")

        self.gridLayout.addWidget(self.label_shift, 1, 0, 1, 1)

        self.doubleSpinBox_shift = QDoubleSpinBox(self.groupBox_wavelet_customize)
        self.doubleSpinBox_shift.setObjectName(u"doubleSpinBox_shift")
        self.doubleSpinBox_shift.setSingleStep(0.100000000000000)

        self.gridLayout.addWidget(self.doubleSpinBox_shift, 1, 3, 1, 1)

        self.label_skewness = QLabel(self.groupBox_wavelet_customize)
        self.label_skewness.setObjectName(u"label_skewness")

        self.gridLayout.addWidget(self.label_skewness, 2, 0, 1, 1)

        self.doubleSpinBox_skewness = QDoubleSpinBox(self.groupBox_wavelet_customize)
        self.doubleSpinBox_skewness.setObjectName(u"doubleSpinBox_skewness")

        self.gridLayout.addWidget(self.doubleSpinBox_skewness, 2, 3, 1, 1)

        self.horizontalSlider_skewness = QSlider(self.groupBox_wavelet_customize)
        self.horizontalSlider_skewness.setObjectName(u"horizontalSlider_skewness")
        self.horizontalSlider_skewness.setMaximum(100)
        self.horizontalSlider_skewness.setValue(50)
        self.horizontalSlider_skewness.setOrientation(Qt.Horizontal)

        self.gridLayout.addWidget(self.horizontalSlider_skewness, 2, 1, 1, 2)

        self.horizontalSlider_shift = QSlider(self.groupBox_wavelet_customize)
        self.horizontalSlider_shift.setObjectName(u"horizontalSlider_shift")
        self.horizontalSlider_shift.setMaximum(100)
        self.horizontalSlider_shift.setValue(50)
        self.horizontalSlider_shift.setOrientation(Qt.Horizontal)

        self.gridLayout.addWidget(self.horizontalSlider_shift, 1, 1, 1, 2)

        self.horizontalSlider_mod = QSlider(self.groupBox_wavelet_customize)
        self.horizontalSlider_mod.setObjectName(u"horizontalSlider_mod")
        self.horizontalSlider_mod.setMaximum(100)
        self.horizontalSlider_mod.setValue(50)
        self.horizontalSlider_mod.setOrientation(Qt.Horizontal)

        self.gridLayout.addWidget(self.horizontalSlider_mod, 0, 1, 1, 2)


        self.verticalLayout.addLayout(self.gridLayout)


        self.gridLayout_waveletparameters.addWidget(self.groupBox_wavelet_customize, 2, 0, 1, 3)

        self.verticalLayout_waveletparameters.addLayout(self.gridLayout_waveletparameters)


        self.verticalLayout_wavelet.addWidget(self.groupBox_waveletparameters)

        self.verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.verticalLayout_wavelet.addItem(self.verticalSpacer)

        self.toolBox.addItem(self.wavelet_page, u"Wavelet")
        self.analyze_page = QWidget()
        self.analyze_page.setObjectName(u"analyze_page")
        self.analyze_page.setGeometry(QRect(0, 0, 360, 501))
        self.verticalLayout_analyze = QVBoxLayout(self.analyze_page)
        self.verticalLayout_analyze.setObjectName(u"verticalLayout_analyze")
        self.verticalLayout_analyze.setContentsMargins(0, 0, 0, 0)
        self.groupBox_analyze_cwt = QGroupBox(self.analyze_page)
        self.groupBox_analyze_cwt.setObjectName(u"groupBox_analyze_cwt")
        self.groupBox_analyze_cwt.setAlignment(Qt.AlignCenter)
        self.verticalLayout_analyze_cwt = QVBoxLayout(self.groupBox_analyze_cwt)
        self.verticalLayout_analyze_cwt.setObjectName(u"verticalLayout_analyze_cwt")
        self.verticalLayout_analyze_cwt.setContentsMargins(4, 4, 4, 4)
        self.gridLayout_analyze_cwt = QGridLayout()
        self.gridLayout_analyze_cwt.setObjectName(u"gridLayout_analyze_cwt")
        self.doubleSpinBox_selectivity = QDoubleSpinBox(self.groupBox_analyze_cwt)
        self.doubleSpinBox_selectivity.setObjectName(u"doubleSpinBox_selectivity")
        self.doubleSpinBox_selectivity.setMaximum(100.000000000000000)
        self.doubleSpinBox_selectivity.setSingleStep(0.100000000000000)
        self.doubleSpinBox_selectivity.setValue(self.params['cwt']['selectivity'])

        self.gridLayout_analyze_cwt.addWidget(self.doubleSpinBox_selectivity, 3, 2, 1, 1)

        self.doubleSpinBox_extent = QDoubleSpinBox(self.groupBox_analyze_cwt)
        self.doubleSpinBox_extent.setObjectName(u"doubleSpinBox_extent")
        self.doubleSpinBox_extent.setMaximum(100.000000000000000)
        self.doubleSpinBox_extent.setSingleStep(0.100000000000000)
        self.doubleSpinBox_extent.setValue(self.params['cwt']['extent'])

        self.gridLayout_analyze_cwt.addWidget(self.doubleSpinBox_extent, 3, 3, 1, 1)

        self.doubleSpinBox_threshold_cwt = QDoubleSpinBox(self.groupBox_analyze_cwt)
        self.doubleSpinBox_threshold_cwt.setObjectName(u"doubleSpinBox_threshold_cwt")
        self.doubleSpinBox_threshold_cwt.setMaximum(10000.000000000000000)
        self.doubleSpinBox_threshold_cwt.setValue(self.params['cwt']['threshold'])

        self.gridLayout_analyze_cwt.addWidget(self.doubleSpinBox_threshold_cwt, 2, 2, 1, 2)

        self.label_threshold_cwt = QLabel(self.groupBox_analyze_cwt)
        self.label_threshold_cwt.setObjectName(u"label_threshold_cwt")

        self.gridLayout_analyze_cwt.addWidget(self.label_threshold_cwt, 2, 0, 1, 2)

        self.spinBox_scales_count = QSpinBox(self.groupBox_analyze_cwt)
        self.spinBox_scales_count.setObjectName(u"spinBox_scales_count")
        self.spinBox_scales_count.setMinimum(1)
        self.spinBox_scales_count.setMaximum(1000)
        self.spinBox_scales_count.setValue(self.params['cwt']['scales']['count'])

        self.gridLayout_analyze_cwt.addWidget(self.spinBox_scales_count, 1, 3, 1, 1)

        self.label_selectivity = QLabel(self.groupBox_analyze_cwt)
        self.label_selectivity.setObjectName(u"label_selectivity")

        self.gridLayout_analyze_cwt.addWidget(self.label_selectivity, 3, 0, 1, 2)

        self.checkBox_store_cwt = QCheckBox(self.groupBox_analyze_cwt)
        self.checkBox_store_cwt.setObjectName(u"checkBox_store_cwt")

        self.gridLayout_analyze_cwt.addWidget(self.checkBox_store_cwt, 4, 0, 1, 2)

        self.doubleSpinBox_scales_max = QDoubleSpinBox(self.groupBox_analyze_cwt)
        self.doubleSpinBox_scales_max.setObjectName(u"doubleSpinBox_scales_max")
        self.doubleSpinBox_scales_max.setDecimals(3)
        self.doubleSpinBox_scales_max.setMinimum(0.001000000000000)
        self.doubleSpinBox_scales_max.setMaximum(1000.000000000000000)
        self.doubleSpinBox_scales_max.setSingleStep(0.100000000000000)
        self.doubleSpinBox_scales_max.setValue(self.params['cwt']['scales']['max']*1e3)

        self.gridLayout_analyze_cwt.addWidget(self.doubleSpinBox_scales_max, 1, 2, 1, 1)

        self.checkBox_store_events = QCheckBox(self.groupBox_analyze_cwt)
        self.checkBox_store_events.setObjectName(u"checkBox_store_events")
        self.checkBox_store_events.setChecked(True)

        self.gridLayout_analyze_cwt.addWidget(self.checkBox_store_events, 5, 0, 1, 2)

        self.pushButton_cwt = QPushButton(self.groupBox_analyze_cwt)
        self.pushButton_cwt.setObjectName(u"pushButton_cwt")
        sizePolicy2 = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.pushButton_cwt.sizePolicy().hasHeightForWidth())
        self.pushButton_cwt.setSizePolicy(sizePolicy2)
        self.pushButton_cwt.setMinimumSize(QSize(0, 64))

        self.gridLayout_analyze_cwt.addWidget(self.pushButton_cwt, 4, 2, 2, 2)

        self.comboBox_showcwt = QComboBox(self.groupBox_analyze_cwt)
        self.comboBox_showcwt.setObjectName(u"comboBox_showcwt")
        self.comboBox_showcwt.setModel(self.targetsmodel)

        self.gridLayout_analyze_cwt.addWidget(self.comboBox_showcwt, 6, 2, 1, 2)

        self.label_showcwt = QLabel(self.groupBox_analyze_cwt)
        self.label_showcwt.setObjectName(u"label_showcwt")

        self.gridLayout_analyze_cwt.addWidget(self.label_showcwt, 6, 0, 1, 2)

        self.label_scales = QLabel(self.groupBox_analyze_cwt)
        self.label_scales.setObjectName(u"label_scales")

        self.gridLayout_analyze_cwt.addWidget(self.label_scales, 0, 0, 1, 4)

        self.doubleSpinBox_scales_min = QDoubleSpinBox(self.groupBox_analyze_cwt)
        self.doubleSpinBox_scales_min.setObjectName(u"doubleSpinBox_scales_min")
        self.doubleSpinBox_scales_min.setDecimals(3)
        self.doubleSpinBox_scales_min.setMinimum(0.001000000000000)
        self.doubleSpinBox_scales_min.setMaximum(1000.000000000000000)
        self.doubleSpinBox_scales_min.setSingleStep(0.010000000000000)
        self.doubleSpinBox_scales_min.setValue(self.params['cwt']['scales']['min']*1e3)

        self.gridLayout_analyze_cwt.addWidget(self.doubleSpinBox_scales_min, 1, 1, 1, 1)


        self.verticalLayout_analyze_cwt.addLayout(self.gridLayout_analyze_cwt)


        self.verticalLayout_analyze.addWidget(self.groupBox_analyze_cwt)

        self.groupBox_analyze_shiftmultiply = QGroupBox(self.analyze_page)
        self.groupBox_analyze_shiftmultiply.setObjectName(u"groupBox_analyze_shiftmultiply")
        self.groupBox_analyze_shiftmultiply.setAlignment(Qt.AlignCenter)
        self.verticalLayout_analyze_shiftmultiply = QVBoxLayout(self.groupBox_analyze_shiftmultiply)
        self.verticalLayout_analyze_shiftmultiply.setObjectName(u"verticalLayout_analyze_shiftmultiply")
        self.verticalLayout_analyze_shiftmultiply.setContentsMargins(4, 4, 4, 4)
        self.gridLayout_analyze_shiftmultiply = QGridLayout()
        self.gridLayout_analyze_shiftmultiply.setObjectName(u"gridLayout_analyze_shiftmultiply")
        self.label_minseparation = QLabel(self.groupBox_analyze_shiftmultiply)
        self.label_minseparation.setObjectName(u"label_minseparation")

        self.gridLayout_analyze_shiftmultiply.addWidget(self.label_minseparation, 1, 0, 1, 1)

        self.label_threshold_shiftmultiply = QLabel(self.groupBox_analyze_shiftmultiply)
        self.label_threshold_shiftmultiply.setObjectName(u"label_threshold_shiftmultiply")

        self.gridLayout_analyze_shiftmultiply.addWidget(self.label_threshold_shiftmultiply, 0, 0, 1, 1)

        self.pushButton_shiftmultiply_classify = QPushButton(self.groupBox_analyze_shiftmultiply)
        self.pushButton_shiftmultiply_classify.setObjectName(u"pushButton_shiftmultiply_classify")
        self.pushButton_shiftmultiply_classify.setMinimumSize(QSize(0, 64))

        self.gridLayout_analyze_shiftmultiply.addWidget(self.pushButton_shiftmultiply_classify, 2, 1, 1, 1)

        self.doubleSpinBox_minseparation = QDoubleSpinBox(self.groupBox_analyze_shiftmultiply)
        self.doubleSpinBox_minseparation.setObjectName(u"doubleSpinBox_minseparation")
        self.doubleSpinBox_minseparation.setDecimals(3)
        self.doubleSpinBox_minseparation.setMaximum(100.000000000000000)
        self.doubleSpinBox_minseparation.setSingleStep(0.100000000000000)
        self.doubleSpinBox_minseparation.setValue(1.000000000000000)

        self.gridLayout_analyze_shiftmultiply.addWidget(self.doubleSpinBox_minseparation, 1, 1, 1, 1)

        self.doubleSpinBox_threshold_shiftmultiply = QDoubleSpinBox(self.groupBox_analyze_shiftmultiply)
        self.doubleSpinBox_threshold_shiftmultiply.setObjectName(u"doubleSpinBox_threshold_shiftmultiply")
        self.doubleSpinBox_threshold_shiftmultiply.setMaximum(100.000000000000000)
        self.doubleSpinBox_threshold_shiftmultiply.setValue(10.000000000000000)

        self.gridLayout_analyze_shiftmultiply.addWidget(self.doubleSpinBox_threshold_shiftmultiply, 0, 1, 1, 1)

        self.pushButton_shiftmultiply_find = QPushButton(self.groupBox_analyze_shiftmultiply)
        self.pushButton_shiftmultiply_find.setObjectName(u"pushButton_shiftmultiply_find")
        self.pushButton_shiftmultiply_find.setMinimumSize(QSize(0, 64))

        self.gridLayout_analyze_shiftmultiply.addWidget(self.pushButton_shiftmultiply_find, 2, 0, 1, 1)


        self.verticalLayout_analyze_shiftmultiply.addLayout(self.gridLayout_analyze_shiftmultiply)


        self.verticalLayout_analyze.addWidget(self.groupBox_analyze_shiftmultiply)

        self.verticalSpacer_analyze = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        
        self.verticalLayout_analyze.addItem(self.verticalSpacer_analyze)

        self.toolBox.addItem(self.analyze_page, u"Analyze")
        self.results_page = QWidget()
        self.results_page.setObjectName(u"results_page")
        self.results_page.setGeometry(QRect(0, 0, 286, 501))
        self.verticalLayout_2 = QVBoxLayout(self.results_page)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.verticalLayout_2.setContentsMargins(0, 6, 0, 0)
        self.tabWidget = QTabWidget(self.results_page)
        self.tabWidget.setObjectName(u"tabWidget")
        self.tabWidget.setTabPosition(QTabWidget.North)
        self.tabWidget.setTabShape(QTabWidget.Triangular)
        self.tabWidget.setMovable(True)
        self.tab_summary = QWidget()
        self.tab_summary.setObjectName(u"tab_summary")
        self.verticalLayout_7 = QVBoxLayout(self.tab_summary)
        self.verticalLayout_7.setObjectName(u"verticalLayout_7")
        self.verticalLayout_7.setContentsMargins(0, 6, 0, 0)
        # self.widget_plotcounts = pg.plot(parent=self.tab_summary, show=True)
        # self.widget_plotcounts.setAntialiasing(True)
        # self.widget_plotcounts.setBackground(None)
        # self.widget_plotcounts.hideAxis('left')
        # self.widget_plotcounts.hideAxis('bottom')
        # self.widget_plotcounts.hideButtons()
        # self.widget_plotcounts.setMenuEnabled(False)
        # self.widget_plotcounts.setMouseEnabled(x=False, y=False)
        # self.widget_plotcounts.getViewBox().setMouseEnabled(False)
        # self.widget_plotcounts.setMinimumHeight(100)
        # self.widget_plotcounts.setMaximumHeight(100)
        # self.widget_plotrate = pg.plot(parent=self.tab_summary, show=True)
        # bg2 = pg.BarGraphItem(x0=[0]*3, x1=mcount, y=range(3), height=1, brushes=pen)
        # bgt = [pg.TextItem(f'target{n}: {mcount[n]}', anchor=(0,n)) for n in range(3)]
        # # bg3 = pg.BarGraphItem(x=0, width=5, y=2, height=1, pen='b')
        # # [self.widget_plotcounts.addItem(b) for b in bg]
        # self.widget_plotcounts.addItem(bg2)
        # [self.widget_plotcounts.addItem(_bgt) for _bgt in bgt]
        self.canvas_plotsummary = FigureCanvas(Figure(figsize=(1, 1),constrained_layout=True))
        self.canvas_plotdist = FigureCanvas(Figure(figsize=(1, 1),constrained_layout=True))
        self.canvas_plotjointdist = FigureCanvas(Figure(figsize=(1, 1),constrained_layout=True))
        # self.widget_plotrate.addItem(bg1)
        # self.widget_plotrate.addItem(bg2)
        # self.widget_plotrate.addItem(bg3)

        self.verticalLayout_7.addWidget(self.canvas_plotsummary)
        # self.verticalLayout_7.addWidget(self.widget_plotrate)

        self.tabWidget.addTab(self.tab_summary, "")
        self.tab_dist = QWidget()
        self.tab_dist.setObjectName(u"tab_dist")
        self.verticalLayout_4 = QVBoxLayout(self.tab_dist)
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.verticalLayout_4.setContentsMargins(0, 6, 0, 0)
        self.verticalLayout_4.addWidget(self.canvas_plotdist)
        self.verticalLayout_4.addWidget(self.canvas_plotjointdist)
        
        self.tabWidget.addTab(self.tab_dist, "")
        self.tab_list = QWidget()
        self.tab_list.setObjectName(u"tab_list")
        self.verticalLayout_3 = QVBoxLayout(self.tab_list)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.verticalLayout_3.setContentsMargins(0, 6, 0, 0)
        self.tableView_events = QTableView(self.tab_list)
        self.tableView_events.setObjectName(u"tableView_events")
        self.tableView_events.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tableView_events.setTabKeyNavigation(False)
        self.tableView_events.setProperty("showDropIndicator", False)
        self.tableView_events.setAlternatingRowColors(True)
        self.tableView_events.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.tableView_events.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tableView_events.setShowGrid(False)
        self.tableView_events.setSortingEnabled(True)
        self.tableView_events.horizontalHeader().setMinimumSectionSize(20)
        self.tableView_events.horizontalHeader().setDefaultSectionSize(46)
        self.tableView_events.horizontalHeader().setStretchLastSection(True)

        self.verticalLayout_3.addWidget(self.tableView_events)

        self.pushButton_save_events = QPushButton(self.tab_list)
        self.pushButton_save_events.setObjectName(u"pushButton_save_events")

        self.verticalLayout_3.addWidget(self.pushButton_save_events)

        self.tabWidget.addTab(self.tab_list, "")

        self.verticalLayout_2.addWidget(self.tabWidget)

        self.toolBox.addItem(self.results_page, u"Results")

        self.verticalLayout_binningandwindow_2.addWidget(self.toolBox)

        self.groupBox_binning_and_window = QGroupBox(self.dockWidget_controlpanel_Contents)
        self.groupBox_binning_and_window.setObjectName(u"groupBox_binning_and_window")
        self.groupBox_binning_and_window.setAlignment(Qt.AlignCenter)
        self.verticalLayout_binningandwindow = QVBoxLayout(self.groupBox_binning_and_window)
        self.verticalLayout_binningandwindow.setObjectName(u"verticalLayout_binningandwindow")
        self.verticalLayout_binningandwindow.setContentsMargins(4, 4, 4, 4)
        self.gridLayout_binningandwindow = QGridLayout()
        self.gridLayout_binningandwindow.setObjectName(u"gridLayout_binningandwindow")
        self.label_window = QLabel(self.groupBox_binning_and_window)
        self.label_window.setObjectName(u"label_window")

        self.gridLayout_binningandwindow.addWidget(self.label_window, 1, 0, 1, 1)

        self.doubleSpinBox_window_r = QDoubleSpinBox(self.groupBox_binning_and_window)
        self.doubleSpinBox_window_r.setObjectName(u"doubleSpinBox_window_r")
        self.doubleSpinBox_window_r.setDecimals(3)
        self.doubleSpinBox_window_r.setMaximum(100000)

        self.gridLayout_binningandwindow.addWidget(self.doubleSpinBox_window_r, 1, 2, 1, 1)

        self.label_binsize = QLabel(self.groupBox_binning_and_window)
        self.label_binsize.setObjectName(u"label_binsize")

        self.gridLayout_binningandwindow.addWidget(self.label_binsize, 0, 0, 1, 1)

        self.doubleSpinBox_binsize = QDoubleSpinBox(self.groupBox_binning_and_window)
        self.doubleSpinBox_binsize.setObjectName(u"doubleSpinBox_binsize")
        self.doubleSpinBox_binsize.setDecimals(3)
        self.doubleSpinBox_binsize.setMaximum(1000000.000000000000000)
        self.doubleSpinBox_binsize.setSingleStep(0.010000000000000)
        self.doubleSpinBox_binsize.setValue(0.05)

        self.gridLayout_binningandwindow.addWidget(self.doubleSpinBox_binsize, 0, 1, 1, 2)

        self.doubleSpinBox_window_l = QDoubleSpinBox(self.groupBox_binning_and_window)
        self.doubleSpinBox_window_l.setObjectName(u"doubleSpinBox_window_l")
        self.doubleSpinBox_window_l.setDecimals(3)
        self.doubleSpinBox_window_l.setMaximum(100000)

        self.gridLayout_binningandwindow.addWidget(self.doubleSpinBox_window_l, 1, 1, 1, 1)


        self.verticalLayout_binningandwindow.addLayout(self.gridLayout_binningandwindow)


        self.verticalLayout_binningandwindow_2.addWidget(self.groupBox_binning_and_window)

        self.dockWidget_controlpanel.setWidget(self.dockWidget_controlpanel_Contents)
        MainWindow.addDockWidget(Qt.LeftDockWidgetArea, self.dockWidget_controlpanel)
        
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)
        MainWindow.statusBar().showMessage('Idle')
        self.progressBar = QProgressBar(MainWindow.statusBar())
        MainWindow.statusBar().addPermanentWidget(self.progressBar)
        self.progressBar.setMaximum(100)
        self.progressBar.setVisible(False)

        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuHelp.menuAction())
        self.menuFile.addAction(self.actionLoad)
        self.menuFile.addAction(self.actionSave)
        self.menuFile.addAction(self.actionSave_As)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionExit)
        self.menuHelp.addAction(self.actionHelp)
        self.menuHelp.addAction(self.actionAbout)

        self.retranslateUi(MainWindow)

        self.toolBox.setCurrentIndex(0)
        self.toolBox.layout().setSpacing(4)
        self.tabWidget.setCurrentIndex(0)


        QMetaObject.connectSlotsByName(MainWindow)
        
        self.actionLoad.triggered.connect(self.config_load)
        self.actionSave.triggered.connect(self.config_save)
        self.actionSave_As.triggered.connect(self.config_save_as)
        self.actionExit.triggered.connect(MainWindow.close)
        self.treeView_explorer.clicked.connect(self.get_file_list)
        self.plot_line.vb.sigXRangeChanged.connect(self.setYRange)
        self.image_cwt_ax.vb.sigYRangeChanged.connect(self.setYRange_cwt_vb)
        self.image_cwt_ax.vb.sigResized.connect(self.update_cwt_vb)
        self.image_cwt_ax.vb.sigResized.connect(self.update_cwt_vb)
        self.comboBox_showcwt.currentTextChanged.connect(self.toggle_cwt_image)
        self.pushButton_targetadd.clicked.connect(self.target_add)
        self.pushButton_targetdelete.clicked.connect(lambda: self.target_delete(self.listView_targets.currentIndex().row()))
        self.listView_targets.clicked.connect(self.target_load)
        self.targetsmodel.itemChanged.connect(self.target_state_update)
        self.pushButton_targetcolor.clicked.connect(lambda: self.color_selected(QColorDialog.getColor()))
        self.pushButton_targetsave.clicked.connect(self.target_update)
        self.comboBox_wavelet.currentTextChanged.connect(self.wavelet_update)
        self.comboBox_wavelet.setCurrentIndex(0)
        self.comboBox_target.currentTextChanged.connect(self.wavelet_load)
        self.doubleSpinBox_mod.valueChanged.connect(lambda x: self.horizontalSlider_mod.setValue(100*float(x)))
        self.horizontalSlider_mod.valueChanged.connect(lambda x: self.doubleSpinBox_mod.setValue(float(x)/100))
        self.horizontalSlider_mod.valueChanged.connect(self.wavelet_update)
        self.doubleSpinBox_shift.valueChanged.connect(lambda x: self.horizontalSlider_shift.setValue(100*float(x)))
        self.horizontalSlider_shift.valueChanged.connect(lambda x: self.doubleSpinBox_shift.setValue(float(x)/100))
        self.horizontalSlider_shift.valueChanged.connect(self.wavelet_update)
        self.doubleSpinBox_skewness.valueChanged.connect(lambda x: self.horizontalSlider_skewness.setValue(100*float(x)))
        self.horizontalSlider_skewness.valueChanged.connect(lambda x: self.doubleSpinBox_skewness.setValue(float(x)/100))
        self.horizontalSlider_skewness.valueChanged.connect(self.wavelet_update)
        self.lineEdit_N.editingFinished.connect(self.wavelet_update)

        self.doubleSpinBox_scales_min.valueChanged.connect(self.update_params)
        self.doubleSpinBox_scales_max.valueChanged.connect(self.update_params)
        self.spinBox_scales_count.valueChanged.connect(self.update_params)
        self.doubleSpinBox_threshold_cwt.valueChanged.connect(self.update_params)
        self.doubleSpinBox_selectivity.valueChanged.connect(self.update_params)
        self.doubleSpinBox_extent.valueChanged.connect(self.update_params)

        self.pushButton_save_events.clicked.connect(lambda: self.save_events(self.eventsmodel._data))
        
        # self.doubleSpinBox_mod.setValue(0.5)
        # self.doubleSpinBox_shift.setValue(1)
        # self.doubleSpinBox_skewness.setValue(0.4)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"Single Molecule Detection: Analysis", None))
        self.actionLoad.setText(QCoreApplication.translate("MainWindow", u"Load", None))
        self.actionSave.setText(QCoreApplication.translate("MainWindow", u"Save", None))
        self.actionSave_As.setText(QCoreApplication.translate("MainWindow", u"Save As...", None))
        self.actionExit.setText(QCoreApplication.translate("MainWindow", u"Exit", None))
        self.actionHelp.setText(QCoreApplication.translate("MainWindow", u"Help", None))
        self.actionAbout.setText(QCoreApplication.translate("MainWindow", u"About", None))
        self.menuFile.setTitle(QCoreApplication.translate("MainWindow", u"File", None))
        self.menuHelp.setTitle(QCoreApplication.translate("MainWindow", u"Help", None))
        self.dockWidget_controlpanel.setWindowTitle(QCoreApplication.translate("MainWindow", u"Control Panel", None))
        self.toolBox.setItemText(self.toolBox.indexOf(self.explorer_page), QCoreApplication.translate("MainWindow", u"File Explorer", None))
        self.groupBox.setTitle(QCoreApplication.translate("MainWindow", u"Excitation Profile", None))
        self.label_3.setText(QCoreApplication.translate("MainWindow", u"# of modes", None))
        self.label_4.setText(QCoreApplication.translate("MainWindow", u"V-offset [um]", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u"W [um]", None))
        self.label_2.setText(QCoreApplication.translate("MainWindow", u"H [um]", None))
        self.groupBox_2.setTitle(QCoreApplication.translate("MainWindow", u"Channel Dimensions", None))
        self.label_6.setText(QCoreApplication.translate("MainWindow", u"H [um]", None))
        self.label_5.setText(QCoreApplication.translate("MainWindow", u"W [um]", None))
        self.groupBox_3.setTitle(QCoreApplication.translate("MainWindow", u"Collection Profile", None))
        self.label_8.setText(QCoreApplication.translate("MainWindow", u"H [um]", None))
        self.label_7.setText(QCoreApplication.translate("MainWindow", u"W [um]", None))
        self.toolBox.setItemText(self.toolBox.indexOf(self.device_page), QCoreApplication.translate("MainWindow", u"Device", None))
        self.pushButton_targetadd.setText(QCoreApplication.translate("MainWindow", u"Add", None))
        self.pushButton_targetdelete.setText(QCoreApplication.translate("MainWindow", u"Delete", None))
        self.groupBox_targetinfo.setTitle(QCoreApplication.translate("MainWindow", u"Information", None))
        self.label_targetnote.setText(QCoreApplication.translate("MainWindow", u"Additional note", None))
        self.label_color.setText(QCoreApplication.translate("MainWindow", u"Color", None))
        self.pushButton_targetcolor.setText("")
        self.label_targetsize.setText(QCoreApplication.translate("MainWindow", u"Size [um]", None))
        self.doubleSpinBox_targetconcentration.setSuffix(QCoreApplication.translate("MainWindow", u"e6", None))
        self.label_targetconcentration.setText(QCoreApplication.translate("MainWindow", u"Concentration [/mL]", None))
        self.pushButton_targetsave.setText(QCoreApplication.translate("MainWindow", u"Save", None))
        self.toolBox.setItemText(self.toolBox.indexOf(self.target_page), QCoreApplication.translate("MainWindow", u"Target", None))
        self.groupBox_waveletparameters.setTitle(QCoreApplication.translate("MainWindow", u"Wavelet Parameters", None))
        self.label_wavelet.setText(QCoreApplication.translate("MainWindow", u"Wavelet", None))
        self.label_target.setText(QCoreApplication.translate("MainWindow", u"Target", None))
        self.lineEdit_N.setText(QCoreApplication.translate("MainWindow", u"6", None))
        self.groupBox_wavelet_customize.setTitle(QCoreApplication.translate("MainWindow", u"Customize", None))
        self.label_mod.setText(QCoreApplication.translate("MainWindow", u"Mod", None))
        self.label_shift.setText(QCoreApplication.translate("MainWindow", u"Shift", None))
        self.label_skewness.setText(QCoreApplication.translate("MainWindow", u"Skewness", None))
        self.toolBox.setItemText(self.toolBox.indexOf(self.wavelet_page), QCoreApplication.translate("MainWindow", u"Wavelet", None))
        self.groupBox_analyze_cwt.setTitle(QCoreApplication.translate("MainWindow", u"CWT", None))
        self.label_threshold_cwt.setText(QCoreApplication.translate("MainWindow", u"Threshold", None))
        self.label_selectivity.setText(QCoreApplication.translate("MainWindow", u"Selectivity, extent", None))
        self.checkBox_store_cwt.setText(QCoreApplication.translate("MainWindow", u"Store CWT", None))
        self.checkBox_store_events.setText(QCoreApplication.translate("MainWindow", u"Store Events", None))
        self.pushButton_cwt.setText(QCoreApplication.translate("MainWindow", u"Detect Events", None))
        self.label_showcwt.setText(QCoreApplication.translate("MainWindow", u"Show CWT for", None))
        self.label_scales.setText(QCoreApplication.translate("MainWindow", u"Scales [ms]: min, max, count", None))
        self.groupBox_analyze_shiftmultiply.setTitle(QCoreApplication.translate("MainWindow", u"Shift Multiply", None))
        self.label_minseparation.setText(QCoreApplication.translate("MainWindow", u"Min Separation [ms]", None))
        self.label_threshold_shiftmultiply.setText(QCoreApplication.translate("MainWindow", u"Threshold [cnts/bin]", None))
        self.pushButton_shiftmultiply_classify.setText(QCoreApplication.translate("MainWindow", u"Classify", None))
        self.pushButton_shiftmultiply_find.setText(QCoreApplication.translate("MainWindow", u"Find Events", None))
        self.toolBox.setItemText(self.toolBox.indexOf(self.analyze_page), QCoreApplication.translate("MainWindow", u"Analyze", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_summary), QCoreApplication.translate("MainWindow", u"Summary", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_dist), QCoreApplication.translate("MainWindow", u"Distributions", None))
        self.pushButton_save_events.setText(QCoreApplication.translate("MainWindow", u"Save As", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_list), QCoreApplication.translate("MainWindow", u"Events List", None))
        self.toolBox.setItemText(self.toolBox.indexOf(self.results_page), QCoreApplication.translate("MainWindow", u"Results", None))
        self.groupBox_binning_and_window.setTitle(QCoreApplication.translate("MainWindow", u"Binning && Window", None))
        self.label_window.setText(QCoreApplication.translate("MainWindow", u"Window [s]", None))
        self.label_binsize.setText(QCoreApplication.translate("MainWindow", u"Bin size [ms]", None))
    # retranslateUi
	
    def config_init(self):
        if os.path.exists("config.json"):
            with open("config.json") as cf:
                self.params = json.load(cf)
        else:
            self.params = {
                'currentdir': '',
                'binsize': 1e-3,
                'window': {
                    'l': 0,
                    'r': 0
                    },
                'targets': {
                    'name':[],
                    'color':[],
                    'size':[],
                    'concentration':[],
                    'active':[],
                    'note':[],
                    'wavelet':{
                        'name':[],
                        'parameters':[]
                        }
                },
                'cwt': {
                    'resolution': 10,
                    'scales':{
                        'min': 0.1e-3,
                        'max': 1.5e-3,
                        'count': 30
                    },
                    'window': {
                        'l': 0,
                        'r': 0
                    },
                    'threshold': 5,
                    'selectivity': 3,
                    'extent': 1,
                    'chunksize': 100000,
                    'log': True,
                    'refine': True,
                }
            }
            self.config_save()
    def config_load(self):
        filename = QFileDialog.getOpenFileName(None,'Load Configurations', self.params['currentdir'], 'JSON Files (*.json)')
        if filename[0] is not '':
            with open(filename[0], 'r') as cf:
                self.params = json.load(cf)
                self.load_params()
    def config_save(self):
        with open('config.json', 'w') as cf:
            json.dump(self.params, cf, sort_keys=True, indent=4)
    def config_save_as(self):
        filename = QFileDialog.getSaveFileName(None,'Save Configurations As', self.params['currentdir'], 'JSON Files (*.json)')
        if filename[0] is not '':
            with open(filename[0], 'w') as cf:
                json.dump(self.params, cf, sort_keys=True, indent=4)

    def update_params(self,*value):
        self.params['cwt']['scales']['min'] = self.doubleSpinBox_scales_min.value()*1e-3
        self.params['cwt']['scales']['max'] = self.doubleSpinBox_scales_max.value()*1e-3
        self.params['cwt']['scales']['count'] = self.spinBox_scales_count.value()
        self.params['cwt']['threshold'] = self.doubleSpinBox_threshold_cwt.value()
        self.params['cwt']['selectivity'] = self.doubleSpinBox_selectivity.value()
        self.params['cwt']['extent'] = self.doubleSpinBox_extent.value()
        print(self.params['cwt'])

    def load_params(self):
        self.doubleSpinBox_scales_min.setValue(self.params['cwt']['scales']['min']*1e3)
        self.doubleSpinBox_scales_max.setValue(self.params['cwt']['scales']['max']*1e3)
        self.spinBox_scales_count.setValue(self.params['cwt']['scales']['count'])
        self.doubleSpinBox_threshold_cwt.setValue(self.params['cwt']['threshold'])
        self.doubleSpinBox_selectivity.setValue(self.params['cwt']['selectivity'])
        self.doubleSpinBox_extent.setValue(self.params['cwt']['extent'])

    def get_file_list(self, signal):
        self.params['currentdir'] = self.dirmodel.filePath(signal)+'/'
        files = QDir(self.dirmodel.filePath(signal),"*.ptu *.PTU").entryList()
        self.listWidget_files.clear()
        [self.listWidget_files.addItem(f) for f in files]

    def setYRange(self, *arg):
        self.plot_line.enableAutoRange(axis='y')
        self.plot_line.setAutoVisible(y=True)

    def setYRange_cwt_vb(self, *arg):
        self.image_cwt_vb.setYRange(arg[-1][0],arg[-1][1])

    def update_cwt_vb(self):
        self.image_cwt_vb.setGeometry(self.image_cwt_ax.vb.sceneBoundingRect())
        self.scatter_events_vb.setGeometry(self.image_cwt_ax.vb.sceneBoundingRect())

    def target_add(self):
        default = {
            'name':f"target_{len(self.params['targets']['name'])+1}",
            'color':'rgb(255, 0, 0)',
            'concentration':1,
            'size':0.1,
            'active':True,
            'note':'',
            'wavelet':{
                'name':'Multi-spot Gaussian',
                'parameters':{'N':6, 'pattern':'6', 'mod':0.5,'shift':1,'skewness':0.5}
            }
        }
        for key,value in default.items():
            if type(value) is dict:
                for k,v in value.items():
                    self.params['targets'][key][k].append(v)
            else:
                self.params['targets'][key].append(value)
        self.target_update_list()
    
    def target_update_list(self):
        self.targetsmodel.clear()
        for t in self.params['targets']['name']:
            item = QStandardItem(t)
            item.setCheckable(True)
            item.setCheckState(Qt.CheckState.Checked)
            self.targetsmodel.appendRow(item)
    
    def target_delete(self,item):
        if len(self.params['targets']['name']) > 1:
            for key, value in self.params['targets'].items():
                if type(value) is dict:
                    for k,v in value.items():
                        del v[item]
                else:
                    del value[item]
            self.targetsmodel.removeRow(item)
            self.target_update_list()

    def target_load(self,item):
        if item.data() in self.params['targets']['name']:
            n = item.row()
            # print(n)
            self.pushButton_targetcolor.setStyleSheet(f"background-color: {self.params['targets']['color'][n]};\n"
"border: none;")
            self.doubleSpinBox_targetconcentration.setValue(self.params['targets']['concentration'][n])
            self.doubleSpinBox_targetsize.setValue(self.params['targets']['size'][n])
            self.plainTextEdit_targetnote.setPlainText(self.params['targets']['note'][n])
        print(self.params['targets'])

    def color_selected(self,color):
        self.pushButton_targetcolor.setStyleSheet(f"background-color: rgb{color.getRgb()};\n"
"border: none;")

    def target_update(self,*item):
        n = self.listView_targets.currentIndex().row()
        self.params['targets']['color'][n] = self.pushButton_targetcolor.styleSheet().split('background-color: ')[1].split(';')[0]
        self.params['targets']['concentration'][n] = self.doubleSpinBox_targetconcentration.value()
        self.params['targets']['size'][n] = self.doubleSpinBox_targetsize.value()
        self.params['targets']['note'][n] = self.plainTextEdit_targetnote.toPlainText()

    def target_state_update(self,*item):
        # print(item)
        # print(self.pushButton_targetcolor.styleSheet().split('background-color: ')[1].split(';')[0])
        n = item[0].row()
        self.params['targets']['name'][n] = self.targetsmodel.item(n).text()
        if self.targetsmodel.item(n).checkState() is Qt.CheckState.Checked:
            self.params['targets']['active'][n] = True
        else:
            self.params['targets']['active'][n] = False

    def wavelet_load(self,*sig):
        import wavelet as wlt
        n = self.comboBox_target.currentIndex()
        # print(n)
        self.lineEdit_N.setText(self.params['targets']['wavelet']['parameters'][n]['pattern'])
        self.doubleSpinBox_mod.setValue(self.params['targets']['wavelet']['parameters'][n]['mod'])
        self.doubleSpinBox_shift.setValue(self.params['targets']['wavelet']['parameters'][n]['shift'])
        self.doubleSpinBox_skewness.setValue(self.params['targets']['wavelet']['parameters'][n]['skewness'])
        self.comboBox_wavelet.currentTextChanged.disconnect()
        self.comboBox_wavelet.setCurrentText(self.params['targets']['wavelet']['name'][n])
        self.wavelet_update()
        self.comboBox_wavelet.currentTextChanged.connect(self.wavelet_update)

    def wavelet_update(self,*sig):
        import wavelet as wlt
        n = self.comboBox_target.currentIndex()
        wavelet_name = self.comboBox_wavelet.currentText()
        pattern = self.lineEdit_N.text()
        if wavelet_name == 'Multi-spot Gaussian (encoded)':
            N = len(pattern)
        else:
            N = int(pattern)
        if N > 10:
            print('N is too big')
            return
        mod = self.doubleSpinBox_mod.value()
        shift = self.doubleSpinBox_shift.value()
        skewness = self.doubleSpinBox_skewness.value()
        self.params['targets']['wavelet']['name'][n] = wavelet_name
        self.params['targets']['wavelet']['parameters'][n] = {'N':N, 'pattern':pattern, 'mod':mod, 'shift':shift, 'skewness':skewness}
        wavelet = getattr(wlt, wavelet_name.lower().replace('-','_').replace(' ','_').replace('(','').replace(')',''))(N=N,pattern=pattern,mod=mod,shift=shift,skewness=skewness)
        if np.iscomplexobj(wavelet):
            self.waveletplot_line.plot(np.real(wavelet),pen=pg.mkPen(width=1,color='k'),clear=True)
            self.waveletplot_line.plot(np.imag(wavelet),pen=pg.mkPen(width=1,color=(30,30,30),style=QtCore.Qt.DashLine))
        else:
            self.waveletplot_line.plot(wavelet,pen=pg.mkPen(width=1,color='k'),clear=True)

    def toggle_cwt_image(self,*sig):
        name = self.comboBox_showcwt.currentText()
        self.image_cwt_vb.clear()
        if name in self.image_cwt:
            # print(self.image_cwt)
            [self.image_cwt_vb.addItem(img) for img in self.image_cwt[name]]
            
    def gen_summary(self,events):
        import pandas as pd
        self.canvas_plotsummary.figure.clf()
        self.gs_summaryplot = self.canvas_plotsummary.figure.add_gridspec(3, 2, height_ratios=[1,2,2])
        self.ax_plotcounts = self.canvas_plotsummary.figure.add_subplot(self.gs_summaryplot[0,:])
        self.ax_rate = self.canvas_plotsummary.figure.add_subplot(self.gs_summaryplot[1,:])
        self.ax_velocity = self.canvas_plotsummary.figure.add_subplot(self.gs_summaryplot[2,:])
        names = self.params['targets']['name']
        self.df = pd.DataFrame(events)
        self.df['velocity'] = 5e-6/self.df['scale']*1e2
        self.df['intensity'] = self.df['coeff']
        # df['intensity'] = df['coeff']/df['scale']/df['N']
        # print(df.describe())
        _counts = []
        colors = []
        ticks = ['']*len(names)
        for n,c in enumerate(self.params['targets']['color']):
            colors.append([float(c)/255 for c in eval(c.split('rgb')[1])])
            _counts.append(np.count_nonzero(events['label'] == n))
        for n,name in enumerate(names):
            ticks[n] = f'{name} [{_counts[n]}]'

        self.ax_plotcounts.barh(range(len(_counts)), _counts, align='center', color=colors)
        # [self.ax_plotcounts.text(y=y, s=f' {c} ', va='center', bbox=dict(boxstyle="round",ec=(0., 0., 0.),fc=(0., 0., 0.)), **text[y]) for y,c in enumerate(_counts)]
        self.ax_plotcounts.set_yticks(range(len(_counts)))
        self.ax_plotcounts.set_yticklabels(ticks)
        self.ax_plotcounts.set_xlabel('Count')
        self.ax_rate.hist([events[events['label']==n]['time'] for n in range(len(_counts))], rwidth=0.8, 
                            bins=20, density=False, histtype='bar', stacked=True, color=colors, label=names)
        self.ax_rate.set_xlabel('Time [s]')
        self.ax_rate.set_ylabel(f"Events/{(np.max(events['time'])-np.min(events['time']))/20:.3f}")
        self.ax_rate.legend()
        bins,B = pd.cut(self.df['time'], 20, retbins=True, right=False)
        V = self.df.groupby(bins)['velocity'].agg(['mean', 'std']).fillna(0)
        # print(V['mean'].to_numpy())
        self.ax_velocity.errorbar(B[:-1], V['mean'].to_numpy(), color='b', yerr=V['std'].to_numpy(),label="Velocity")
        self.ax_velocity.set_xlabel('Time [s]')
        # self.ax_velocity.set_ylabel(u"\u0394t [ms]", color='b')
        self.ax_velocity.set_ylabel("Velocity [cm/s]", color='b')
        self.ax_intensity = self.ax_velocity.twinx()
        I = self.df.groupby(bins)['intensity'].agg(['mean', 'std']).fillna(0)
        # print(I)
        self.ax_intensity.errorbar(B[:-1], I['mean'].to_numpy(), color='r', linestyle='--', yerr=I['std'].to_numpy(),label="Intensity")
        # self.ax_intensity.set_xlabel('Time [s]')
        self.ax_intensity.set_ylabel("Intensity [a.u.]", color='r')
        # self.ax_velocity.legend()
        self.ax_velocity.tick_params(axis='y', labelcolor='b')
        self.ax_intensity.tick_params(axis='y', labelcolor='r')
        self.canvas_plotsummary.draw()
    
    def gen_dist(self,events):
        # import pandas as pd
        # import seaborn as sns
        self.canvas_plotdist.figure.clf()
        self.canvas_plotjointdist.figure.clf()
        self.ax_dist = self.canvas_plotdist.figure.subplots(2,1)
        # self.df = pd.DataFrame(events)
        # self.df['velocity'] = 5e-6/self.df['scale']*1e2
        # self.df['intensity'] = self.df['coeff']
        # self.df['class'] = self.df['label']
        self.df['scale'] = self.df['scale']*1e3
        self.df['label'] = np.array(self.params['targets']['name'])[self.df['label']]
        colors = []
        names = self.params['targets']['name']
        for n,c in enumerate(self.params['targets']['color']):
            colors.append([float(c)/255 for c in eval(c.split('rgb')[1])])
        self.ax_dist[0].hist([self.df[self.df['label']==n]['intensity'] for n in range(len(names))], rwidth=0.8, 
                        bins=20, density=False, histtype='bar', stacked=False, color=colors, label=names)
        self.ax_dist[1].hist([self.df[self.df['label']==n]['velocity'] for n in range(len(names))], rwidth=0.8, 
                        bins=20, density=False, histtype='bar',stacked=False, color=colors, label=names)
        self.ax_dist[0].set_xlabel('Intensity [a.u.]')
        self.ax_dist[1].set_xlabel('Velocity [cm/s]')
        self.ax_dist[0].legend()
        self.ax_dist[1].legend()
        self.ax_scatter = self.canvas_plotjointdist.figure.subplots()
        if self.params['cwt']['log']:
            xbins = np.logspace(np.log10(self.params['cwt']['scales']['min']*1e3), np.log10(self.params['cwt']['scales']['max']*1e3), self.params['cwt']['scales']['count'], dtype=np.float64)
        else:
            xbins = np.linspace(self.params['cwt']['scales']['min']*1e3, self.params['cwt']['scales']['max']*1e3, self.params['cwt']['scales']['count'], dtype=np.float64)
        # hist, xbins = np.histogram(self.df['scale'], bins=self.params['cwt']['scales']['count'])
        # xlogbins = np.logspace(np.log10(xbins[0]),np.log10(xbins[-1]),len(xbins))
        hist, ybins = np.histogram(self.df['intensity'], bins=self.params['cwt']['scales']['count'])
        ylogbins = np.logspace(np.log10(ybins[0]),np.log10(ybins[-1]),len(ybins))
        self.ax_scatter.hist2d(self.df['scale'],self.df['intensity'],cmap='hot',bins=[xbins,ylogbins])
        # for n,name in enumerate(names):
        #     self.ax_scatter.scatter(self.df[self.df['class']==n]['scale'],self.df[self.df['class']==n]['intensity'],alpha=0.2,marker='.',color=colors[n],label=name)
        self.ax_scatter.set_xlabel(u"\u0394t [ms]")
        self.ax_scatter.set_ylabel('Intensity [a.u.]')
        self.ax_scatter.set_yscale('log')
        self.ax_scatter.set_xscale('log')
        plt.xlim(self.params['cwt']['scales']['min']*1e3, self.params['cwt']['scales']['max']*1e3)
        # self.ax_scatter.legend()
        # self.verticalLayout_4.removeWidget(self.canvas_plotjointdist)
        # self.canvas_plotjointdist = FigureCanvas(fig)
        # self.verticalLayout_4.addWidget(self.canvas_plotjointdist)
        self.canvas_plotdist.draw()
        self.canvas_plotjointdist.draw()

    def save_events(self,events):
        if events is not None:
            filename = QFileDialog.getSaveFileName(None,'Save Events As', self.params['currentdir'], 'Numpy Array (*.npy);;CSV (*.csv)')[0]
            if filename is not '':
                if os.path.splitext(filename)[1] == 'npy':
                    with open(filename, 'wb') as f:
                        np.save(f, events)
                else:
                    with open(filename, 'wb') as f:
                        np.savetxt(f, events, delimiter=',', comments='', header=','.join(events.dtype.names))

class Events_Model(QAbstractTableModel):
    def __init__(self, data):
        super(Events_Model, self).__init__()
        if data is not None:
            self._data = data
            self._header = ['time[s]','scale[ms]','coeff','N','label']
        else:
            self._data = np.zeros((0,0))
            self._header = []

    def data(self, index, role):
        if role == Qt.DisplayRole:
            # See below for the nested-list data structure.
            # .row() indexes into the outer list,
            # .column() indexes into the sub-list
            value = self._data[index.row()][index.column()]
            if index.column() in [0,2]:
                return f"{value:.3f}"
            if index.column() is 1:
                return f"{value*1e3:.3f}"
            if index.column() in [3,4]:
                return f"{value:.0f}"
            return str(value)

    def rowCount(self, index):
        # The length of the outer list.
        return len(self._data)

    def columnCount(self, index):
        # The following takes the first sub-list, and returns
        # the length (only works if all rows are an equal length)
        return len(self._data[0])
        
    def headerData(self, section, orientation, role):
        # section is the index of the column/row.
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return str(self._header[section])