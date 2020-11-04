from PySide2.QtCore import (QCoreApplication, QDate, QDateTime, QMetaObject,
    QObject, QPoint, QRect, QSize, QTime, QUrl, Qt, Signal, Slot)
from PySide2.QtGui import (QBrush, QColor, QConicalGradient, QCursor, QFont,
    QFontDatabase, QIcon, QKeySequence, QLinearGradient, QPalette, QPainter,
    QPixmap, QRadialGradient)
from PySide2.QtWidgets import *

from concurrent.futures import ProcessPoolExecutor as Executor
from concurrent.futures import as_completed
import numpy as np

from matplotlib.backends.qt_compat import QtCore, QtWidgets, is_pyqt5
if QtCore.qVersion():
    from matplotlib.backends.backend_qt5agg import (
        FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
else:
    from matplotlib.backends.backend_qt4agg import (
        FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure
from matplotlib import cm
import matplotlib.pyplot as plt
import threading
from math import (floor, ceil)
from scipy.stats import entropy

class Ui_Info(QWidget):
    progress = Signal(int)
    started = Signal(bool)
    message = Signal(str)
    futures = ['# of events', 'mean', 'std', 'entropy']
    def setupUi(self, Info, parent=None):
        if not Info.objectName():
            Info.setObjectName(u"Info")
        self.hide()
        self.parent = parent
        Info.resize(800, 600)
        icon = QIcon(u"icon.ico")
        Info.setWindowIcon(icon)
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(Info.sizePolicy().hasHeightForWidth())
        Info.setSizePolicy(sizePolicy)
        self.horizontalLayout = QHBoxLayout(Info)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.verticalLayout = QVBoxLayout()
        self.verticalLayout.setObjectName(u"verticalLayout")

        self.canvas_avg = FigureCanvas(Figure(figsize=(1, 1),constrained_layout=True))
        self.canvas_wavelet = FigureCanvas(Figure(figsize=(1, 1),constrained_layout=True))

        self.verticalLayout.addWidget(self.canvas_avg)

        self.groupBox = QGroupBox(Info)
        self.groupBox.setObjectName(u"groupBox")
        sizePolicy1 = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        sizePolicy1.setHorizontalStretch(1)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.groupBox.sizePolicy().hasHeightForWidth())
        self.groupBox.setSizePolicy(sizePolicy1)
        self.groupBox.setAlignment(Qt.AlignCenter)
        self.formLayout = QFormLayout(self.groupBox)
        self.formLayout.setObjectName(u"formLayout")
        self.label = QLabel(self.groupBox)
        self.label.setObjectName(u"label")

        self.formLayout.setWidget(0, QFormLayout.LabelRole, self.label)

        self.comboBox_target = QComboBox(self.groupBox)
        self.comboBox_target.setObjectName(u"comboBox_target")

        self.formLayout.setWidget(0, QFormLayout.FieldRole, self.comboBox_target)

        self.tableWidget_stat = QTableWidget(len(self.futures),1,self.groupBox)
        self.tableWidget_stat.setVerticalHeaderLabels(['']*len(self.futures))
        [self.tableWidget_stat.setItem(r, 0, QTableWidgetItem(f)) for r, f in enumerate(self.futures)]
        self.tableWidget_stat.setObjectName(u"tableWidget_stat")

        self.formLayout.setWidget(1, QFormLayout.SpanningRole, self.tableWidget_stat)


        self.verticalLayout.addWidget(self.groupBox)

        self.verticalSpacer_2 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.verticalLayout.addItem(self.verticalSpacer_2)

        self.verticalLayout.setStretch(0, 1)
        self.verticalLayout.setStretch(1, 1)

        self.horizontalLayout.addLayout(self.verticalLayout)

        self.verticalLayout_2 = QVBoxLayout()
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")

        self.verticalLayout_2.addWidget(self.canvas_wavelet)

        self.groupBox_2 = QGroupBox(Info)
        self.groupBox_2.setObjectName(u"groupBox_2")
        sizePolicy1.setHeightForWidth(self.groupBox_2.sizePolicy().hasHeightForWidth())
        self.groupBox_2.setSizePolicy(sizePolicy1)
        self.groupBox_2.setAlignment(Qt.AlignCenter)
        self.formLayout_2 = QFormLayout(self.groupBox_2)
        self.formLayout_2.setObjectName(u"formLayout_2")
        self.label_2 = QLabel(self.groupBox_2)
        self.label_2.setObjectName(u"label_2")

        self.formLayout_2.setWidget(0, QFormLayout.LabelRole, self.label_2)

        self.lineEdit_wavelet_name = QLineEdit(self.groupBox_2)
        self.lineEdit_wavelet_name.setObjectName(u"lineEdit_wavelet_name")

        self.formLayout_2.setWidget(0, QFormLayout.FieldRole, self.lineEdit_wavelet_name)

        self.groupBox_3 = QGroupBox(self.groupBox_2)
        self.groupBox_3.setObjectName(u"groupBox_3")
        self.groupBox_3.setCheckable(True)
        self.formLayout_3 = QFormLayout(self.groupBox_3)
        self.formLayout_3.setObjectName(u"formLayout_3")
        self.label_3 = QLabel(self.groupBox_3)
        self.label_3.setObjectName(u"label_3")

        self.formLayout_3.setWidget(0, QFormLayout.LabelRole, self.label_3)

        self.doubleSpinBox_shift = QDoubleSpinBox(self.groupBox_3)
        self.doubleSpinBox_shift.setObjectName(u"doubleSpinBox_shift")

        self.formLayout_3.setWidget(0, QFormLayout.FieldRole, self.doubleSpinBox_shift)

        self.label_4 = QLabel(self.groupBox_3)
        self.label_4.setObjectName(u"label_4")

        self.formLayout_3.setWidget(1, QFormLayout.LabelRole, self.label_4)

        self.doubleSpinBox_width = QDoubleSpinBox(self.groupBox_3)
        self.doubleSpinBox_width.setObjectName(u"doubleSpinBox_width")

        self.formLayout_3.setWidget(1, QFormLayout.FieldRole, self.doubleSpinBox_width)

        self.label_5 = QLabel(self.groupBox_3)
        self.label_5.setObjectName(u"label_5")

        self.formLayout_3.setWidget(2, QFormLayout.LabelRole, self.label_5)

        self.doubleSpinBox_skewness = QDoubleSpinBox(self.groupBox_3)
        self.doubleSpinBox_skewness.setObjectName(u"doubleSpinBox_skewness")

        self.formLayout_3.setWidget(2, QFormLayout.FieldRole, self.doubleSpinBox_skewness)


        self.formLayout_2.setWidget(2, QFormLayout.SpanningRole, self.groupBox_3)

        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.pushButton_save = QPushButton(self.groupBox_2)
        self.pushButton_save.setObjectName(u"pushButton_save")

        self.horizontalLayout_2.addWidget(self.pushButton_save)

        self.pushButton_cancel = QPushButton(self.groupBox_2)
        self.pushButton_cancel.setObjectName(u"pushButton_cancel")

        self.horizontalLayout_2.addWidget(self.pushButton_cancel)


        self.formLayout_2.setLayout(3, QFormLayout.SpanningRole, self.horizontalLayout_2)


        self.verticalLayout_2.addWidget(self.groupBox_2)

        self.verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.verticalLayout_2.addItem(self.verticalSpacer)

        self.verticalLayout_2.setStretch(0, 1)

        self.horizontalLayout.addLayout(self.verticalLayout_2)


        self.retranslateUi(Info)

        QMetaObject.connectSlotsByName(Info)
    # setupUi

    def retranslateUi(self, Info):
        Info.setWindowTitle(QCoreApplication.translate("Info", u"Events' Information", None))
        self.groupBox.setTitle(QCoreApplication.translate("Info", u"Events Statistics", None))
        self.label.setText(QCoreApplication.translate("Info", u"Target:", None))
        self.groupBox_2.setTitle(QCoreApplication.translate("Info", u"Wavelet (Filter) Generator", None))
        self.label_2.setText(QCoreApplication.translate("Info", u"Name", None))
        self.groupBox_3.setTitle(QCoreApplication.translate("Info", u"Side Peaks", None))
        self.label_3.setText(QCoreApplication.translate("Info", u"Shift", None))
        self.label_4.setText(QCoreApplication.translate("Info", u"Width", None))
        self.label_5.setText(QCoreApplication.translate("Info", u"Skewness", None))
        self.pushButton_save.setText(QCoreApplication.translate("Info", u"Save", None))
        self.pushButton_cancel.setText(QCoreApplication.translate("Info", u"Cancel", None))
    # retranslateUi

    def update_stats_call(self, item):
        self.update_stats_thread = threading.Thread(target=self.update_stats, args=(item,), daemon=False)
        self.update_stats_thread.start()

    def update_stats(self, target):
        try:
            events = self.parent.eventdetector.selected_events
            if len(events) == 0: return
        except Exception as excpt:
            print(excpt)
            return
        ts = self.parent.ts
        # print(target)
        resolution = self.parent.ui.params['cwt']['resolution']
        N = max(events['N'])
        if len(events) == 0:
            return
        events.sort(order='scale')
        scales = np.unique(events['scale'])
        _events = [events[events['scale']==scale] for scale in scales]
        total = len(scales)
        n, progress = 0, 0
        self.message.emit('Collecting info...')
        self.progress.emit(progress)
        self.started.emit(True)
        if ts.type == 'nanopore':
            ts.dt = ts.nanopore_globres
            ts.active = True
            ts.load(plot=False, relim=False)
        events_normalized = []
        for event in _events:
            events_normalized += trim_event(event, ts, resolution, N)
            # print(_sig_avg)
            n += 1
            progress = 100*n/total
            self.progress.emit(progress)
        # with Executor() as e:
        #     _futures = [e.submit(trim_event, event, ts, resolution, N) for event in events]
        #     for _f in as_completed(_futures):
        #         _result = _f.result()
        #         print(_result)
        #         n += 1
        #         progress = 100*n/total
        #         self.progress.emit(progress)
            events_normalized = [event for event in events_normalized if type(event) != None]
        events_normalized = np.stack(events_normalized, axis=0)
        _label = np.unique(events['label'])
        self.events_normalized = {_l:events_normalized[events['label']==_l,:] for _l in _label}
        self.show_stats(target)

    def show_stats(self, target):
        # print(target)
        target_label = self.parent.ui.params['targets']['name'].index(target)
        try:
            events_target = self.events_normalized[target_label]
            # print(len(events_target))
        except Exception as excpt:
            self.canvas_avg.figure.clf()
            self.ax_avg = self.canvas_avg.figure.add_subplot()
            self.canvas_avg.draw()
            self.message.emit('Idle')
            self.started.emit(False)
            return
        self.canvas_avg.figure.clf()
        self.ax_avg = self.canvas_avg.figure.add_subplot()
        _avg, _std = np.mean(events_target, axis=0), np.std(events_target, axis=0)
        self.ax_avg.fill_between(np.arange(len(_avg)), _avg-_std, _avg+_std, color='gray', alpha=0.3)
        self.ax_avg.plot(_avg, color='black')
        self.canvas_avg.draw()
        for c in range(len(self.parent.ui.params['targets']['name'])):
            try:
                events_target = self.events_normalized[c]
                _avg = np.mean(events_target, axis=0)
                futures = [len(events_target), np.mean(_avg), np.std(_avg), entropy(_avg)]
            except Exception as excpt:
                print(excpt)
                futures = [0, 0, 0, 0]
            print(futures)
            [self.tableWidget_stat.setItem(r,c+1,QTableWidgetItem(f'{f:.3g}')) for r, f in enumerate(futures)]
        self.message.emit('Idle')
        self.started.emit(False)

    def show_window(self):
        self.comboBox_target.setModel(self.parent.ui.targetsmodel)
        self.tableWidget_stat.setColumnCount(self.comboBox_target.count()+1)
        self.tableWidget_stat.setHorizontalHeaderLabels(['feature']+self.parent.ui.params['targets']['name'])
        header = self.tableWidget_stat.horizontalHeader()
        for n in range(self.comboBox_target.count()+1):
            if n == 0 or n == self.comboBox_target.count():
                header.setSectionResizeMode(n, QtWidgets.QHeaderView.Stretch)
            else:
                header.setSectionResizeMode(n, QtWidgets.QHeaderView.Interactive)
        self.comboBox_target.currentTextChanged.connect(self.show_stats)
        self.show()
        self.update_stats_call(self.comboBox_target.currentText())
        # self.update_stats()

def normalize(event):
    if len(event) == 0:
        return
    event -= np.min(event)
    return event/np.max(event)
    # return event

def trim_event(event, ts, resolution, N):
    scale = event[0]['scale']
    # print(scale)
    dt = scale/resolution
    _time, _trace = ts.resample(dt)
    # print(len(_time),len(_trace))
    _window_half = int(N*resolution) if N>1 else int(10*N*resolution)
    _t0 = _time[0]
    # print([normalize(_trace[int((e['time']-_time[0])/dt-_window_half):int((e['time']-_time[0])/dt+_window_half)]) for e in event])
    # sig_avg = np.mean([normalize(_trace[int((e['time']-_time[0])/dt-_window_half):int((e['time']-_time[0])/dt+_window_half)]) for e in event])
    event_normalized = [normalize(_trace[int(floor((e['time']-_t0)/dt)-_window_half):int(floor((e['time']-_t0)/dt)+_window_half)]) for e in event]
    return event_normalized