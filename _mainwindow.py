# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'mainwindow.ui'
##
## Created by: Qt User Interface Compiler version 5.15.0
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import (QCoreApplication, QDate, QDateTime, QMetaObject,
    QObject, QPoint, QRect, QSize, QTime, QUrl, Qt)
from PySide2.QtGui import (QBrush, QColor, QConicalGradient, QCursor, QFont,
    QFontDatabase, QIcon, QKeySequence, QLinearGradient, QPalette, QPainter,
    QPixmap, QRadialGradient)
from PySide2.QtWidgets import *


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1200, 800)
        icon = QIcon()
        iconThemeName = u"icon.ico"
        if QIcon.hasThemeIcon(iconThemeName):
            icon = QIcon.fromTheme(iconThemeName)
        else:
            icon.addFile(u".", QSize(), QIcon.Normal, QIcon.Off)
        
        MainWindow.setWindowIcon(icon)
        self.actionLoad = QAction(MainWindow)
        self.actionLoad.setObjectName(u"actionLoad")
        self.actionSave = QAction(MainWindow)
        self.actionSave.setObjectName(u"actionSave")
        self.actionSave_As = QAction(MainWindow)
        self.actionSave_As.setObjectName(u"actionSave_As")
        self.actionExit = QAction(MainWindow)
        self.actionExit.setObjectName(u"actionExit")
        self.actionHelp = QAction(MainWindow)
        self.actionHelp.setObjectName(u"actionHelp")
        self.actionAbout = QAction(MainWindow)
        self.actionAbout.setObjectName(u"actionAbout")
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.verticalLayout_centralwidget = QVBoxLayout(self.centralwidget)
        self.verticalLayout_centralwidget.setObjectName(u"verticalLayout_centralwidget")
        self.verticalLayout_centralwidget.setContentsMargins(0, 0, 0, 0)
        self.widget_plot_timetrace = QWidget(self.centralwidget)
        self.widget_plot_timetrace.setObjectName(u"widget_plot_timetrace")
        self.widget_plot_timetrace.setCursor(QCursor(Qt.CrossCursor))

        self.verticalLayout_centralwidget.addWidget(self.widget_plot_timetrace)

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
        self.explorer_page.setGeometry(QRect(0, 0, 256, 518))
        self.verticalLayout_explorer = QVBoxLayout(self.explorer_page)
        self.verticalLayout_explorer.setObjectName(u"verticalLayout_explorer")
        self.verticalLayout_explorer.setContentsMargins(0, 0, 0, 0)
        self.splitter_explorer = QSplitter(self.explorer_page)
        self.splitter_explorer.setObjectName(u"splitter_explorer")
        self.splitter_explorer.setOrientation(Qt.Vertical)
        self.splitter_explorer.setChildrenCollapsible(False)
        self.treeView_explorer = QTreeView(self.splitter_explorer)
        self.treeView_explorer.setObjectName(u"treeView_explorer")
        self.treeView_explorer.setIndentation(10)
        self.treeView_explorer.setWordWrap(True)
        self.splitter_explorer.addWidget(self.treeView_explorer)
        self.treeView_explorer.header().setVisible(False)
        self.listWidget_files = QListWidget(self.splitter_explorer)
        self.listWidget_files.setObjectName(u"listWidget_files")
        self.listWidget_files.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.splitter_explorer.addWidget(self.listWidget_files)

        self.verticalLayout_explorer.addWidget(self.splitter_explorer)

        self.toolBox.addItem(self.explorer_page, u"File Explorer")
        self.device_page = QWidget()
        self.device_page.setObjectName(u"device_page")
        self.device_page.setGeometry(QRect(0, 0, 267, 518))
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

        self.widget_2 = QWidget(self.groupBox)
        self.widget_2.setObjectName(u"widget_2")
        self.widget_2.setMinimumSize(QSize(0, 40))

        self.gridLayout_3.addWidget(self.widget_2, 0, 0, 1, 4)

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
        self.target_page.setGeometry(QRect(0, 0, 268, 518))
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
        self.wavelet_page.setGeometry(QRect(0, 0, 266, 518))
        self.verticalLayout_wavelet = QVBoxLayout(self.wavelet_page)
        self.verticalLayout_wavelet.setObjectName(u"verticalLayout_wavelet")
        self.verticalLayout_wavelet.setContentsMargins(0, 0, 0, 0)
        self.widget = QWidget(self.wavelet_page)
        self.widget.setObjectName(u"widget")

        self.verticalLayout_wavelet.addWidget(self.widget)

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
        self.comboBox_target.setEditable(True)

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
        self.groupBox_wavelet_customize.setChecked(True)
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

        self.horizontalSlider_skewness = QSlider(self.groupBox_wavelet_customize)
        self.horizontalSlider_skewness.setObjectName(u"horizontalSlider_skewness")
        self.horizontalSlider_skewness.setMaximum(100)
        self.horizontalSlider_skewness.setValue(50)
        self.horizontalSlider_skewness.setOrientation(Qt.Horizontal)

        self.gridLayout.addWidget(self.horizontalSlider_skewness, 2, 1, 1, 2)

        self.doubleSpinBox_skewness = QDoubleSpinBox(self.groupBox_wavelet_customize)
        self.doubleSpinBox_skewness.setObjectName(u"doubleSpinBox_skewness")

        self.gridLayout.addWidget(self.doubleSpinBox_skewness, 2, 3, 1, 1)

        self.doubleSpinBox_shift = QDoubleSpinBox(self.groupBox_wavelet_customize)
        self.doubleSpinBox_shift.setObjectName(u"doubleSpinBox_shift")
        self.doubleSpinBox_shift.setSingleStep(0.100000000000000)

        self.gridLayout.addWidget(self.doubleSpinBox_shift, 1, 3, 1, 1)

        self.label_skewness = QLabel(self.groupBox_wavelet_customize)
        self.label_skewness.setObjectName(u"label_skewness")

        self.gridLayout.addWidget(self.label_skewness, 2, 0, 1, 1)

        self.horizontalSlider_mod = QSlider(self.groupBox_wavelet_customize)
        self.horizontalSlider_mod.setObjectName(u"horizontalSlider_mod")
        self.horizontalSlider_mod.setMaximum(100)
        self.horizontalSlider_mod.setValue(50)
        self.horizontalSlider_mod.setOrientation(Qt.Horizontal)

        self.gridLayout.addWidget(self.horizontalSlider_mod, 0, 1, 1, 2)

        self.label_mod = QLabel(self.groupBox_wavelet_customize)
        self.label_mod.setObjectName(u"label_mod")

        self.gridLayout.addWidget(self.label_mod, 0, 0, 1, 1)

        self.label_shift = QLabel(self.groupBox_wavelet_customize)
        self.label_shift.setObjectName(u"label_shift")

        self.gridLayout.addWidget(self.label_shift, 1, 0, 1, 1)

        self.horizontalSlider_shift = QSlider(self.groupBox_wavelet_customize)
        self.horizontalSlider_shift.setObjectName(u"horizontalSlider_shift")
        self.horizontalSlider_shift.setMaximum(100)
        self.horizontalSlider_shift.setValue(50)
        self.horizontalSlider_shift.setOrientation(Qt.Horizontal)

        self.gridLayout.addWidget(self.horizontalSlider_shift, 1, 1, 1, 2)


        self.verticalLayout.addLayout(self.gridLayout)


        self.gridLayout_waveletparameters.addWidget(self.groupBox_wavelet_customize, 2, 0, 1, 3)


        self.verticalLayout_waveletparameters.addLayout(self.gridLayout_waveletparameters)


        self.verticalLayout_wavelet.addWidget(self.groupBox_waveletparameters)

        self.verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.verticalLayout_wavelet.addItem(self.verticalSpacer)

        self.toolBox.addItem(self.wavelet_page, u"Wavelet")
        self.analyze_page = QWidget()
        self.analyze_page.setObjectName(u"analyze_page")
        self.analyze_page.setGeometry(QRect(0, 0, 253, 518))
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
        self.doubleSpinBox_extent = QDoubleSpinBox(self.groupBox_analyze_cwt)
        self.doubleSpinBox_extent.setObjectName(u"doubleSpinBox_extent")
        self.doubleSpinBox_extent.setMaximum(100.000000000000000)
        self.doubleSpinBox_extent.setSingleStep(0.100000000000000)
        self.doubleSpinBox_extent.setValue(1.000000000000000)

        self.gridLayout_analyze_cwt.addWidget(self.doubleSpinBox_extent, 3, 3, 1, 1)

        self.spinBox_scales_count = QSpinBox(self.groupBox_analyze_cwt)
        self.spinBox_scales_count.setObjectName(u"spinBox_scales_count")
        self.spinBox_scales_count.setMinimum(1)
        self.spinBox_scales_count.setMaximum(1000)
        self.spinBox_scales_count.setValue(10)

        self.gridLayout_analyze_cwt.addWidget(self.spinBox_scales_count, 1, 3, 1, 1)

        self.label_scales = QLabel(self.groupBox_analyze_cwt)
        self.label_scales.setObjectName(u"label_scales")

        self.gridLayout_analyze_cwt.addWidget(self.label_scales, 0, 0, 1, 4)

        self.doubleSpinBox_scales_min = QDoubleSpinBox(self.groupBox_analyze_cwt)
        self.doubleSpinBox_scales_min.setObjectName(u"doubleSpinBox_scales_min")
        self.doubleSpinBox_scales_min.setDecimals(3)
        self.doubleSpinBox_scales_min.setMinimum(0.001000000000000)
        self.doubleSpinBox_scales_min.setMaximum(1000.000000000000000)
        self.doubleSpinBox_scales_min.setSingleStep(0.100000000000000)
        self.doubleSpinBox_scales_min.setValue(0.100000000000000)

        self.gridLayout_analyze_cwt.addWidget(self.doubleSpinBox_scales_min, 1, 1, 1, 1)

        self.checkBox_store_cwt = QCheckBox(self.groupBox_analyze_cwt)
        self.checkBox_store_cwt.setObjectName(u"checkBox_store_cwt")

        self.gridLayout_analyze_cwt.addWidget(self.checkBox_store_cwt, 4, 0, 1, 2)

        self.label_threshold_cwt = QLabel(self.groupBox_analyze_cwt)
        self.label_threshold_cwt.setObjectName(u"label_threshold_cwt")

        self.gridLayout_analyze_cwt.addWidget(self.label_threshold_cwt, 2, 0, 1, 2)

        self.doubleSpinBox_scales_max = QDoubleSpinBox(self.groupBox_analyze_cwt)
        self.doubleSpinBox_scales_max.setObjectName(u"doubleSpinBox_scales_max")
        self.doubleSpinBox_scales_max.setDecimals(3)
        self.doubleSpinBox_scales_max.setMinimum(0.001000000000000)
        self.doubleSpinBox_scales_max.setMaximum(1000.000000000000000)
        self.doubleSpinBox_scales_max.setSingleStep(0.100000000000000)
        self.doubleSpinBox_scales_max.setValue(1.000000000000000)

        self.gridLayout_analyze_cwt.addWidget(self.doubleSpinBox_scales_max, 1, 2, 1, 1)

        self.checkBox_store_events = QCheckBox(self.groupBox_analyze_cwt)
        self.checkBox_store_events.setObjectName(u"checkBox_store_events")
        self.checkBox_store_events.setChecked(True)

        self.gridLayout_analyze_cwt.addWidget(self.checkBox_store_events, 5, 0, 1, 2)

        self.doubleSpinBox_selectivity = QDoubleSpinBox(self.groupBox_analyze_cwt)
        self.doubleSpinBox_selectivity.setObjectName(u"doubleSpinBox_selectivity")
        self.doubleSpinBox_selectivity.setMaximum(100.000000000000000)
        self.doubleSpinBox_selectivity.setSingleStep(0.100000000000000)
        self.doubleSpinBox_selectivity.setValue(3.000000000000000)

        self.gridLayout_analyze_cwt.addWidget(self.doubleSpinBox_selectivity, 3, 2, 1, 1)

        self.doubleSpinBox_threshold_cwt = QDoubleSpinBox(self.groupBox_analyze_cwt)
        self.doubleSpinBox_threshold_cwt.setObjectName(u"doubleSpinBox_threshold_cwt")
        self.doubleSpinBox_threshold_cwt.setValue(5.000000000000000)

        self.gridLayout_analyze_cwt.addWidget(self.doubleSpinBox_threshold_cwt, 2, 2, 1, 2)

        self.label_selectivity = QLabel(self.groupBox_analyze_cwt)
        self.label_selectivity.setObjectName(u"label_selectivity")

        self.gridLayout_analyze_cwt.addWidget(self.label_selectivity, 3, 0, 1, 2)

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

        self.gridLayout_analyze_cwt.addWidget(self.comboBox_showcwt, 6, 2, 1, 2)

        self.label_showcwt = QLabel(self.groupBox_analyze_cwt)
        self.label_showcwt.setObjectName(u"label_showcwt")

        self.gridLayout_analyze_cwt.addWidget(self.label_showcwt, 6, 0, 1, 2)


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
        self.results_page.setGeometry(QRect(0, 0, 262, 518))
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
        self.widget_summaryplots = QWidget(self.tab_summary)
        self.widget_summaryplots.setObjectName(u"widget_summaryplots")

        self.verticalLayout_7.addWidget(self.widget_summaryplots)

        self.tabWidget.addTab(self.tab_summary, "")
        self.tab_dist = QWidget()
        self.tab_dist.setObjectName(u"tab_dist")
        self.verticalLayout_4 = QVBoxLayout(self.tab_dist)
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.verticalLayout_4.setContentsMargins(0, 6, 0, 0)
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
        self.tableView_events.horizontalHeader().setDefaultSectionSize(40)
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
        self.verticalLayout_binningandwindow.setContentsMargins(4, 4, 4, 10)
        self.gridLayout_binningandwindow = QGridLayout()
        self.gridLayout_binningandwindow.setObjectName(u"gridLayout_binningandwindow")
        self.label_window = QLabel(self.groupBox_binning_and_window)
        self.label_window.setObjectName(u"label_window")

        self.gridLayout_binningandwindow.addWidget(self.label_window, 1, 0, 1, 1)

        self.doubleSpinBox_window_r = QDoubleSpinBox(self.groupBox_binning_and_window)
        self.doubleSpinBox_window_r.setObjectName(u"doubleSpinBox_window_r")
        self.doubleSpinBox_window_r.setDecimals(3)
        self.doubleSpinBox_window_r.setMaximum(100000.000000000000000)

        self.gridLayout_binningandwindow.addWidget(self.doubleSpinBox_window_r, 1, 2, 1, 1)

        self.label_binsize = QLabel(self.groupBox_binning_and_window)
        self.label_binsize.setObjectName(u"label_binsize")

        self.gridLayout_binningandwindow.addWidget(self.label_binsize, 0, 0, 1, 1)

        self.doubleSpinBox_window_l = QDoubleSpinBox(self.groupBox_binning_and_window)
        self.doubleSpinBox_window_l.setObjectName(u"doubleSpinBox_window_l")
        self.doubleSpinBox_window_l.setDecimals(3)
        self.doubleSpinBox_window_l.setMaximum(100000.000000000000000)

        self.gridLayout_binningandwindow.addWidget(self.doubleSpinBox_window_l, 1, 1, 1, 1)

        self.doubleSpinBox_binsize = QDoubleSpinBox(self.groupBox_binning_and_window)
        self.doubleSpinBox_binsize.setObjectName(u"doubleSpinBox_binsize")
        self.doubleSpinBox_binsize.setDecimals(3)
        self.doubleSpinBox_binsize.setMaximum(100000.000000000000000)
        self.doubleSpinBox_binsize.setSingleStep(0.010000000000000)
        self.doubleSpinBox_binsize.setValue(1.000000000000000)

        self.gridLayout_binningandwindow.addWidget(self.doubleSpinBox_binsize, 0, 1, 1, 2)


        self.verticalLayout_binningandwindow.addLayout(self.gridLayout_binningandwindow)


        self.verticalLayout_binningandwindow_2.addWidget(self.groupBox_binning_and_window)

        self.dockWidget_controlpanel.setWidget(self.dockWidget_controlpanel_Contents)
        MainWindow.addDockWidget(Qt.LeftDockWidgetArea, self.dockWidget_controlpanel)
        self.statusBar = QStatusBar(MainWindow)
        self.statusBar.setObjectName(u"statusBar")
        MainWindow.setStatusBar(self.statusBar)

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

        self.toolBox.layout().setSpacing(4)
        self.tabWidget.setCurrentIndex(0)


        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"Event Detector", None))
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
        self.label_skewness.setText(QCoreApplication.translate("MainWindow", u"Skewness", None))
        self.label_mod.setText(QCoreApplication.translate("MainWindow", u"Mod", None))
        self.label_shift.setText(QCoreApplication.translate("MainWindow", u"Shift", None))
        self.toolBox.setItemText(self.toolBox.indexOf(self.wavelet_page), QCoreApplication.translate("MainWindow", u"Wavelet", None))
        self.groupBox_analyze_cwt.setTitle(QCoreApplication.translate("MainWindow", u"CWT", None))
        self.label_scales.setText(QCoreApplication.translate("MainWindow", u"Scales [ms]: min, max, count", None))
        self.checkBox_store_cwt.setText(QCoreApplication.translate("MainWindow", u"Store CWT", None))
        self.label_threshold_cwt.setText(QCoreApplication.translate("MainWindow", u"Threshold", None))
        self.checkBox_store_events.setText(QCoreApplication.translate("MainWindow", u"Store Events", None))
        self.label_selectivity.setText(QCoreApplication.translate("MainWindow", u"Selectivity, extent", None))
        self.pushButton_cwt.setText(QCoreApplication.translate("MainWindow", u"Detect Events", None))
        self.label_showcwt.setText(QCoreApplication.translate("MainWindow", u"Show CWT for", None))
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

