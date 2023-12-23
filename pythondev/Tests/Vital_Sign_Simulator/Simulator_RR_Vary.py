from PyQt5 import QtCore, QtWidgets, QtMultimedia, QtGui
from PyQt5.QtCore import *
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import lfilter
from scipy.io.wavfile import write
import serial
import time

class MyWindow(QtWidgets.QWidget):

    def __init__(self, parent=None):
        # QtWidgets.QWidget.__init__(self, parent, flags=QtCore.Qt.Window | QtCore.Qt.MSWindowsFixedSizeDialogHint)
        QtWidgets.QWidget.__init__(self, parent, flags=QtCore.Qt.Window)
        app.setStyle('Fusion')
        self.setWindowTitle("Neteera AudioWave Player")
        self.setWindowIcon(QtGui.QIcon('Neteera logo.png'))
        p = self.palette()
        p.setColor(QtGui.QPalette.Window, QtCore.Qt.lightGray)
        self.setPalette(p)

        # Создаем сам проигрыватель
        self.playlist = QtMultimedia.QMediaPlaylist()
        self.mplPlayer = QtMultimedia.QMediaPlayer()
        self.mplPlayer.setVolume(50)
        self.mplPlayer.mediaStatusChanged.connect(self.initPlayer)
        self.mplPlayer.stateChanged.connect(self.setPlayerState)
        vbox = QtWidgets.QVBoxLayout()
        # Создаем кнопку открытия файла
        btnOpen = QtWidgets.QPushButton("&Open File...")
        btnOpen.clicked.connect(self.openFile)
        vbox.addWidget(btnOpen)

        # Создаем компоненты для управления воспроизведением. Делаем их изначально недоступными
        self.sldPosition = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sldPosition.setMinimum(0)
        self.mplPlayer.positionChanged.connect(self.sldPosition.setValue)
        self.sldPosition.setEnabled(False)
        vbox.addWidget(self.sldPosition)
        hbox = QtWidgets.QHBoxLayout()
        self.btnPlay = QtWidgets.QPushButton("S&tart")
        self.btnPlay.clicked.connect(self.mplPlayer.play)
        self.btnPlay.setEnabled(False)
        hbox.addWidget(self.btnPlay)
        self.btnPause = QtWidgets.QPushButton("&Pause")
        self.btnPause.clicked.connect(self.mplPlayer.pause)
        self.btnPause.setEnabled(False)
        hbox.addWidget(self.btnPause)
        self.btnStop = QtWidgets.QPushButton("&Stop")
        self.btnStop.clicked.connect(self.mplPlayer.stop)
        self.btnStop.setEnabled(False)
        hbox.addWidget(self.btnStop)
        self.btnLoop = QtWidgets.QPushButton("&Loop")
        self.btnLoop.setCheckable(True)
        self.btnLoop.clicked.connect(self.loop)
        self.btnLoop.setEnabled(False)
        hbox.addWidget(self.btnLoop)
        vbox.addLayout(hbox)
        # Создаем компоненты для управления громкостью
        hbox = QtWidgets.QHBoxLayout()
        lblVolume = QtWidgets.QLabel("&Volume")
        hbox.addWidget(lblVolume)
        sldVolume = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        sldVolume.setRange(0, 100)
        sldVolume.setTickPosition(QtWidgets.QSlider.TicksAbove)
        sldVolume.setTickInterval(10)
        sldVolume.setValue(50)
        lblVolume.setBuddy(sldVolume)
        sldVolume.valueChanged.connect(self.mplPlayer.setVolume)
        hbox.addWidget(sldVolume)
        btnMute = QtWidgets.QPushButton("&Mute!")
        btnMute.setCheckable(True)
        btnMute.toggled.connect(self.mplPlayer.setMuted)
        hbox.addWidget(btnMute)
        vbox.addLayout(hbox)
        #  Simulator
        hbox = QtWidgets.QHBoxLayout()
        vbox.addSpacing(20)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.lblHeartBeats = QtWidgets.QLabel("HeartBeats From, bpm:", self)
        self.lblHeartBeats.setFont(font)
        self.lblHeartBeats.setFrameStyle(
            QtWidgets.QFrame.WinPanel | QtWidgets.QFrame.Sunken)  # (QtWidgets.QFrame.Box | QtWidgets.QFrame.Plain)
        self.lblHeartBeats.move(60, 60)
        hbox.addWidget(self.lblHeartBeats)
        self.spinBox_HR = QtWidgets.QSpinBox(self)
        self.spinBox_HR.setMinimum(1)
        self.spinBox_HR.setMaximum(180)
        self.spinBox_HR.setWrapping(True)
        self.spinBox_HR.setProperty("value", 73)
        self.spinBox_HR.setDisplayIntegerBase(100)
        self.spinBox_HR.setObjectName("spinBox_HR")
        hbox.addWidget(self.spinBox_HR)

        self.lblRespFrom = QtWidgets.QLabel("Respiration From, bpm:", self)
        self.lblRespFrom.setFont(font)
        self.lblRespFrom.setFrameStyle(
            QtWidgets.QFrame.WinPanel | QtWidgets.QFrame.Sunken)  # (QtWidgets.QFrame.WinPanel | QtWidgets.QFrame.Raised)
        hbox.addWidget(self.lblRespFrom)
        self.spinBox_RR_From = QtWidgets.QSpinBox(self)
        self.spinBox_RR_From.setMinimum(1)
        self.spinBox_RR_From.setMaximum(50)
        self.spinBox_RR_From.setWrapping(True)
        self.spinBox_RR_From.setProperty("value", 12)
        self.spinBox_RR_From.setDisplayIntegerBase(100)
        self.spinBox_RR_From.setObjectName("spinBox_RR_From")
        hbox.addWidget(self.spinBox_RR_From)
        vbox.addLayout(hbox)

        hbox = QtWidgets.QHBoxLayout()
        self.lblHeartBeatsTill = QtWidgets.QLabel("HeartBeats Till, bpm:  ", self)
        self.lblHeartBeatsTill.setFont(font)
        self.lblHeartBeatsTill.setFrameStyle(
            QtWidgets.QFrame.WinPanel | QtWidgets.QFrame.Sunken)  # (QtWidgets.QFrame.Box | QtWidgets.QFrame.Plain)
        self.lblHeartBeatsTill.move(60, 60)
        hbox.addWidget(self.lblHeartBeatsTill)
        self.spinBox_HR_Till = QtWidgets.QSpinBox(self)
        self.spinBox_HR_Till.setMinimum(1)
        self.spinBox_HR_Till.setMaximum(180)
        self.spinBox_HR_Till.setWrapping(True)
        self.spinBox_HR_Till.setProperty("value", 90)
        self.spinBox_HR_Till.setDisplayIntegerBase(100)
        self.spinBox_HR_Till.setObjectName("spinBox_HR_Till")
        hbox.addWidget(self.spinBox_HR_Till)

        self.lblRespTill = QtWidgets.QLabel("Respiration Till, bpm:  ", self)
        self.lblRespTill.setFont(font)
        self.lblRespTill.setFrameStyle(QtWidgets.QFrame.WinPanel | QtWidgets.QFrame.Sunken)
        hbox.addWidget(self.lblRespTill)
        self.spinBox_RR_Till = QtWidgets.QSpinBox(self)
        self.spinBox_RR_Till.setAlignment(Qt.AlignBottom)
        self.spinBox_RR_Till.setMinimum(0)
        self.spinBox_RR_Till.setMaximum(50)
        self.spinBox_RR_Till.setWrapping(True)
        self.spinBox_RR_Till.setProperty("value", 20)
        self.spinBox_RR_Till.setDisplayIntegerBase(100)
        self.spinBox_RR_Till.setObjectName("spinBox_RR_Till")
        hbox.addWidget(self.spinBox_RR_Till)
        vbox.addLayout(hbox)

        # Amplitude of Heart Rate
        hbox = QtWidgets.QHBoxLayout()
        self.lblHRAmpl = QtWidgets.QLabel("HR Amplitude           ", self)
        self.lblHRAmpl.setFont(font)
        hbox.addWidget(self.lblHRAmpl)
        self.spinBox_HRAm = QtWidgets.QSpinBox(self)
        self.spinBox_HRAm.setMinimum(1)
        self.spinBox_HRAm.setMaximum(50)
        self.spinBox_HRAm.setWrapping(True)
        self.spinBox_HRAm.setProperty("value", 1)
        self.spinBox_HRAm.setDisplayIntegerBase(10)
        self.spinBox_HRAm.setObjectName("spinBox_HRAm")
        hbox.addWidget(self.spinBox_HRAm)

        # Amplitude of Respiration
        self.lblRRAm = QtWidgets.QLabel("RR Ampitude             ", self)
        self.lblRRAm.setFont(font)
        hbox.addWidget(self.lblRRAm)
        self.spinBox_RRAm = QtWidgets.QSpinBox(self)
        self.spinBox_RRAm.setMinimum(1)
        self.spinBox_RRAm.setMaximum(50)
        self.spinBox_RRAm.setWrapping(True)
        self.spinBox_RRAm.setProperty("value", 5)
        self.spinBox_RRAm.setDisplayIntegerBase(10)
        self.spinBox_RRAm.setObjectName("spinBox_RRAm")
        hbox.addWidget(self.spinBox_RRAm)
        vbox.addLayout(hbox)

        # Duration
        hbox = QtWidgets.QHBoxLayout()
        self.lblDurS = QtWidgets.QLabel("Duration, sec:           ", self)
        self.lblDurS.setFont(font)
        hbox.addWidget(self.lblDurS)
        self.spinBox_DurS = QtWidgets.QSpinBox(self)
        self.spinBox_DurS.setMinimum(1)
        self.spinBox_DurS.setMaximum(900)
        self.spinBox_DurS.setWrapping(True)
        self.spinBox_DurS.setProperty("value", 300)
        self.spinBox_DurS.setDisplayIntegerBase(10)
        self.spinBox_DurS.setObjectName("spinBox_DurS")
        hbox.addWidget(self.spinBox_DurS)

        self.lblDurPart = QtWidgets.QLabel("Duration Part, (0 - 1):", self)
        self.lblDurPart.setFont(font)
        hbox.addWidget(self.lblDurPart)
        self.spinBox_Dur_Part = QtWidgets.QDoubleSpinBox(self)
        self.spinBox_Dur_Part.setAlignment(Qt.AlignBottom)
        self.spinBox_Dur_Part.setRange(0, 1)
        self.spinBox_Dur_Part.setSingleStep(0.1)
        self.spinBox_Dur_Part.setWrapping(True)
        self.spinBox_Dur_Part.setProperty("value", 0.2)
        self.spinBox_Dur_Part.setObjectName("spinBox_Dur_Part")
        hbox.addWidget(self.spinBox_Dur_Part)
        vbox.addLayout(hbox)

        hbox = QtWidgets.QHBoxLayout()
        self.lblNoiseAmp = QtWidgets.QLabel("Noise Amplitude:           ", self)
        self.lblNoiseAmp.setFont(font)
        hbox.addWidget(self.lblNoiseAmp)
        self.spinBox_NoiseAmp = QtWidgets.QSpinBox(self)
        self.spinBox_NoiseAmp.setMinimum(1)
        self.spinBox_NoiseAmp.setMaximum(100)
        self.spinBox_NoiseAmp.setWrapping(True)
        self.spinBox_NoiseAmp.setProperty("value", 5)
        self.spinBox_NoiseAmp.setDisplayIntegerBase(10)
        self.spinBox_DurS.setObjectName("spinBox_NoiseAmp")
        hbox.addWidget(self.spinBox_NoiseAmp)

        self.lblFr = QtWidgets.QLabel("Sampling frequency, Hz ", self)
        self.lblFr.setFont(font)
        hbox.addWidget(self.lblFr)
        self.spinBox_Fr = QtWidgets.QSpinBox(self)
        self.spinBox_Fr.setButtonSymbols(2)
        self.spinBox_Fr.setStyleSheet("margin: 1px; padding: 2px; background-color: rgba(255,255,0,255);\
                                 color: rgba(0,0,255,255); border-style: solid; border-radius: 5px; \
                                 border-width: 2px; border-color: rgba(0,0,100,255);")
        self.spinBox_Fr.setMinimum(20)
        self.spinBox_Fr.setMaximum(48000)
        self.spinBox_Fr.setProperty("value", 50)
        self.spinBox_Fr.setDisplayIntegerBase(10)
        self.spinBox_Fr.setObjectName("spinBox_Fr")
        hbox.addWidget(self.spinBox_Fr)
        vbox.addLayout(hbox)

        # Plot and WAV File
        hbox = QtWidgets.QHBoxLayout()
        self.lblPlt = QtWidgets.QLabel("Show plot:   ", self)
        self.lblPlt.setFont(font)
        hbox.addWidget(self.lblPlt)
        self.checkBox_Pl = QtWidgets.QCheckBox(self)
        self.checkBox_Pl.setObjectName("checkBox_Pl")
        self.checkBox_Pl.setChecked(True)
        hbox.addWidget(self.checkBox_Pl)

        self.lblNois = QtWidgets.QLabel("Add White Noise:   ", self)
        self.lblNois.setFont(font)
        hbox.addWidget(self.lblNois)
        self.checkBox_Ns = QtWidgets.QCheckBox(self)
        self.checkBox_Ns.setObjectName("checkBox_Ns")
        self.checkBox_Ns.setChecked(False)
        hbox.addWidget(self.checkBox_Ns)

        self.lblMotor = QtWidgets.QLabel("Motor Play:   ", self)
        self.lblMotor.setFont(font)
        hbox.addWidget(self.lblMotor)
        self.checkBox_Mtr = QtWidgets.QCheckBox(self)
        self.checkBox_Mtr.setObjectName("checkBox_Mtr")
        self.checkBox_Mtr.setChecked(False)
        hbox.addWidget(self.checkBox_Mtr)

        self.lblWav = QtWidgets.QLabel("Generate WAV:    ", self)
        self.lblWav.setFont(font)
        hbox.addWidget(self.lblWav)
        self.checkBox_Wav = QtWidgets.QCheckBox(self)
        self.checkBox_Wav.setObjectName("checkBox_Wav")
        hbox.addWidget(self.checkBox_Wav)
        vbox.addLayout(hbox)

        hbox = QtWidgets.QHBoxLayout()
        self.btnRun = QtWidgets.QPushButton("&Run")
        self.btnRun.clicked.connect(self.Run_clicked)
        self.btnRun.setEnabled(True)
        hbox.addWidget(self.btnRun)
        vbox.addLayout(hbox)

        self.setLayout(vbox)
        self.resize(400, 200)

    def Run_clicked(self):
        bpm = self.spinBox_HR.value()  # Heart beat frequency From
        Bpm_Till = self.spinBox_HR_Till.value()  # Heart beat frequency end
        rpm = self.spinBox_RR_From.value()  # Respiration beat frequency start
        Rpm_Till = self.spinBox_RR_Till.value()  # Respiration beat frequency end
        eta = self.spinBox_HRAm.value()  # Amplitude HR
        Kb = self.spinBox_RRAm.value()  # Amplitude RR
        dur = self.spinBox_DurS.value()  # Duration in sec
        Part_Tm = self.spinBox_Dur_Part.value()  # Part_Tm: part time from where changing starting From 0.1 t0 1
        khz = self.spinBox_Fr.value()  # Sampling frequency
        RRange = 1  # RR amplitude gate variance
        Normal = 1000   # for audio 25750

        Tau = 1.5
        c, omega, gamma, b = 0.025, 1.05, 1, 0.6
        eta, Kb = int(eta), int(Kb)
        bbi = 60 / int(bpm)
        inhale, exhale = (60 / int(rpm) / 2), (60 / int(rpm) / 2)
        sample_rate = int(khz)    # * 1e3)
        windowSize = 5  # Window Size for filter, was 100 for 48kHz

        # generating the breath signal
        t_breath = np.arange(sample_rate * (inhale + exhale)) / sample_rate  # one breath duration
        one_w = wave(t_breath, inhale, exhale, Kb, Tau, 0)
        num_breaths = int((dur * rpm / 60) + 1)
        if Part_Tm != 0:
            Num_Br_Eq = int((dur * Part_Tm * rpm / 60) + 1)
            Num_Br_Vary = num_breaths - Num_Br_Eq
            Breath_Vary = BrWave(dur, Part_Tm, rpm, Rpm_Till, Kb, Tau, sample_rate, windowSize, RRange, Normal)  # Part_Tm (0.1 : 1) time from where changing starting
        else:
            Num_Br_Eq = num_breaths

        breath_signal = lfilter(np.full((windowSize,), 1 / windowSize), 1, np.tile(one_w, Num_Br_Eq))
        breath_signal = Normalize_Signal(breath_signal, -Normal, Normal)
        if Kb > 0:
            if sample_rate < 1000:
                parser = (np.abs(breath_signal.astype(int))).argmin()
            else:
                parser = np.where(breath_signal.astype(int) == 0)[0][0]
        else:
            parser = np.where(breath_signal.astype(int) == -1)[0][0]

        breath_signal = np.concatenate((breath_signal[parser:], breath_signal[:parser]))[:len(one_w)]
        breath_signals = np.tile(breath_signal, Num_Br_Eq)
        Breath_Common = np.array([*breath_signals, *Breath_Vary])

        # generating the HR signal
        t_hb = np.arange(sample_rate * bbi) / sample_rate  # time for one Heart beat
        one_hb = xhs(t_hb, c, omega, gamma, eta, b)
        num_hbs = int(int(dur) * int(bpm) / 60)
        if Part_Tm != 0:
            Num_HR_Eq = int((dur * Part_Tm * bpm / 60) + 1)
            duration = int(dur * Part_Tm)
            Heart_R_Vary = HrWave(dur, Part_Tm, bpm, Bpm_Till, eta, Kb, sample_rate, windowSize, RRange, c, omega, gamma, b, Normal)  # Part_Tm (0.1 : 1) time from where changing starting
        else:
            Num_HR_Eq = num_hbs
            duration = dur

        hb_signal = np.tile(one_hb, Num_HR_Eq + 1)
        num = Normal * (eta / Kb)  # Scaling factor
        hb_signal = Normalize_Signal(hb_signal, -num, num)

        win_dow = np.log(np.arange(0.999, 0, -1 / ((int(duration) / 4) * sample_rate)))
        win_dow -= np.min(win_dow)
        win_dow /= np.max(win_dow)
        hb_signal = np.concatenate((hb_signal[:-win_dow.shape[0]], hb_signal[-win_dow.shape[0]:] * win_dow))
        hb_signal = hb_signal[np.where(hb_signal.astype(int) == 0)[0][0]:]
        Heart_Common = np.array([*hb_signal, *Heart_R_Vary])  # HR const part + HR vary part

        # combining all together
        if Heart_Common.shape[0] < Breath_Common.shape[0]:
            Breath_Common = Breath_Common[:Heart_Common.shape[0]]
        else:
            Heart_Common = Heart_Common[:Breath_Common.shape[0]]

        Heart_Common = lfilter(np.full((windowSize,), 1 / windowSize), 1, Heart_Common)
        ttHeart = np.arange(len(Heart_Common)) / sample_rate
        combined_signal = Normalize_Signal(Heart_Common + Breath_Common, -Normal, Normal)
        zero_crossings = np.where(np.diff(np.sign(combined_signal)))[0]  # zero  crossing
        combined_signal = np.int32(combined_signal[zero_crossings[0]:zero_crossings[-1]])
        tt = np.arange(len(combined_signal)) / sample_rate
        if self.checkBox_Ns.isChecked():  # Noise addition
            rho = self.spinBox_NoiseAmp.value()
            Noisy_Sig = white_noise(combined_signal, rho, sample_rate, mu=0)
            fig = plt.figure(2, figsize=(10, 6))
            ax1 = fig.add_subplot(211)
            ax1.set_title('Breath and Heart Combined with White Noise')
            ax1.plot(tt, Noisy_Sig, '--r', '-b', lw=1.5)
            ax1.plot(tt, combined_signal, '-b', lw=1.0)
            ax1.set_xlabel("Time, sec", fontsize=14, fontweight='bold')
            ax1.set_ylabel("Amlitude, arb.u.", fontsize=14, fontweight='bold')
            ax1.grid(True)
            ax1 = fig.add_subplot(212)
            ax1.plot(tt, Noisy_Sig, '--r', '-b', lw=1.5)
            ax1.plot(tt, combined_signal, '-b', lw=1.0)
            plt.xlim(25, 50)
            ax1.set_xlabel("Time, sec", fontsize=14, fontweight='bold')
            ax1.set_ylabel("Amlitude, arb.u.", fontsize=14, fontweight='bold')
            ax1.grid(True)
            fig.show()

        # tx = ''
        # for num in combined_signal.astype(int):
        #    tx += str(num) + ','
        # with open('HR' + str(bpm) + '_RR' + str(rpm) + '_Amp' + str(eta) + '_' + str(Kb) + '.txt', 'w') as f:
        #    f.write(tx[:-1] + '\n\n')
        #    f.write(str(combined_signal.shape[0]) + '\n\n')
        #    f.write(str(rpm) + ',' + str(bpm))
        if self.checkBox_Wav.isChecked():  # if_to_wav:
            make_wav_file(combined_signal, bpm, rpm, eta, Kb, sample_rate)
            print('Wav is completed')
        if self.checkBox_Pl.isChecked():  # if_to_plot:
            fig = plt.figure(figsize=(12, 7), constrained_layout=True)
            gs = gridspec.GridSpec(ncols=2, nrows=2, figure=fig)
            fig_ax_1 = fig.add_subplot(gs[0, :])
            plt.plot(tt, combined_signal, label='Breath Combined with Heart', color='blue', linewidth=1.5)
            plt.title('Breath Combined with Heart')
            plt.xlabel('Time, sec')
            # plt.legend(loc=2)
            plt.grid(True)
            fig_ax_2 = fig.add_subplot(gs[1, 0])
            plt.plot(Breath_Common, '-m', label='Breath Common', linewidth=1.5)
            plt.grid(True)
            plt.title('Breath Common')
            fig_ax_3 = fig.add_subplot(gs[1, 1])
            plt.plot(ttHeart[0:1000], Heart_Common[0:1000], '-r', label='Heart Beats', linewidth=1)
            plt.title('Heart Beats')
            plt.grid(True)
            plt.show()
            plt.pause(0.5)
            fig.savefig('HR' + str(bpm) + '_RR' + str(rpm) + '_Amp' + str(eta) + '_' + str(Kb) + '.jpg', dpi=fig.dpi)

        if self.checkBox_Mtr.isChecked():   #  Motor run
            [Ard_COM_Port, tiempo] = Port_Init(self)
            while True:
                for step in combined_signal:
                    x = str(step)
                    Ard_COM_Port.write(bytes(x, 'utf-8'))
                    # index += 1
                    data = Ard_COM_Port.readline()
                    print(x, data)  # printing the value
                    time.sleep(0.001)


    # To open the file, use the getOpenFileUrl () method of the class
    # QFileDialog, because to create an instance of the class
    # QMediaContent we need the path to the file, given as an instance of the QUrl class
    def openFile(self):
        file = QtWidgets.QFileDialog.getOpenFileUrl(parent=self, caption="Select a sound file", filter="Sound files (*.mp3 *.wav)")
        self.mplPlayer.setMedia(QtMultimedia.QMediaContent(file[0]))
        # fn = QtCore.QUrl.fromLocalFile(file[0])
        # self.playlist.addMedia(QtMultimedia.QMediaContent(file))
        # self.playlist.setCurrentIndex(0)
        # self.mplPlayer.setPlaylist(self.playlist)
        # if file != None:
        # self.currentPlaylist.addMedia(QMediaContent(file[0]))

    def initPlayer(self, state):
        if state == QtMultimedia.QMediaPlayer.LoadedMedia:
            # После загрузки файла подготавливаем проигрыватель для его воспроизведения
            self.mplPlayer.stop()
            self.btnPlay.setEnabled(True)
            self.btnPause.setEnabled(False)
            self.btnLoop.setEnabled(True)
            self.sldPosition.setEnabled(True)
            self.sldPosition.setMaximum(self.mplPlayer.duration())
        elif state == QtMultimedia.QMediaPlayer.EndOfMedia:
            # По окончании воспроизведения файла возвращаем проигрыватель в изначальное состояние
            self.mplPlayer.stop()
            self.sldPosition.setValue(0)
            self.sldPosition.setEnabled(False)
            self.btnPlay.setEnabled(False)
            self.btnPause.setEnabled(False)
            self.btnStop.setEnabled(False)
        elif state == QtMultimedia.QMediaPlayer.NoMedia or state == QtMultimedia.QMediaPlayer.InvalidMedia:
            # Если файл не был загружен, отключаем компоненты, управляющие воспроизведением
            self.sldPosition.setValue(0)
            self.sldPosition.setEnabled(False)
            self.btnPlay.setEnabled(False)
            self.btnPause.setEnabled(False)
            self.btnStop.setEnabled(False)

    # В зависимости от того, воспроизводится ли файл, поставлен
    # ли он на паузу или остановлен, делаем соответствующие кнопки
    # доступными или недоступными
    def setPlayerState(self, state):
        if state == QtMultimedia.QMediaPlayer.StoppedState:
            self.sldPosition.setValue(0)
            self.btnPlay.setEnabled(True)
            self.btnPause.setEnabled(False)
            self.btnStop.setEnabled(False)
        elif state == QtMultimedia.QMediaPlayer.PlayingState:
            self.btnPlay.setEnabled(False)
            self.btnPause.setEnabled(True)
            self.btnStop.setEnabled(True)
        elif state == QtMultimedia.QMediaPlayer.PausedState:
            self.btnPlay.setEnabled(True)
            self.btnPause.setEnabled(False)
            self.btnStop.setEnabled(True)

    def loop(self):
        if self.btnLoop.isChecked():
            # self.playlist.setPlaybackMode(QtMultimedia.QMediaPlaylist.Loop)
            self.playlist.setPlaybackMode(QtMultimedia.QMediaPlaylist.CurrentItemInLoop)
        else:
            self.playlist.setPlaybackMode(QtMultimedia.QMediaPlaylist.Sequential)


def wave(t, Ti, Te, kb, tau, bias):
    T = Te + Ti
    peak = np.where(t >= Ti)[0][0]
    ti = t[:peak]
    te = t[peak:]
    wi = (-kb * ti ** 2) / (Ti * Te) + (kb * T * ti) / (Ti * Te)
    we = (kb / (1 - np.exp(-Te / tau))) * (np.exp(-(te - t[peak]) / tau) - np.exp(-Te / tau))
    return [*wi + bias, *we + bias]


def BrWave(dur, Tm_Part, RPM_From, RPM_Till, Kb, Tau, signal_freq, windowSize, RRange, Nrm):
    Br_Vary = np.zeros(1)
    tm, i = 0, 0
    RPM_Lin = np.arange(RPM_From, RPM_Till + 1, 1)
    Tm_Needs = np.sum(60 / RPM_Lin)
    Tm_Remain = dur - dur * Tm_Part
    if (int(Tm_Remain / Tm_Needs) == 0) or (int(Tm_Remain / Tm_Needs) == 1):
        RPM_Step = 1
    else:
        RPM_Step = round(Tm_Remain / Tm_Needs)
    if (Tm_Needs * RPM_Step > Tm_Remain):
        Tm_Remain = Tm_Needs * RPM_Step

    while tm < Tm_Remain:  # time issue ??? Sum buggs
        inhale, exhale = 60 / RPM_From / 2, 60 / RPM_From / 2
        t_breath = np.arange(signal_freq * (inhale + exhale)) / signal_freq
        for n in range(0, RPM_Step):
            RR_Ampl_Rnd = Random_RR_Ampl(RRange, Kb)
            RPM_One = wave(t_breath, inhale, exhale, RR_Ampl_Rnd, Tau, 0)
            RPM_One = lfilter(np.full((windowSize,), 1 / windowSize), 1, np.tile(RPM_One, 1))
            Br_Vary = [*Br_Vary, *RPM_One]
        tm += RPM_Step * 60 / RPM_Lin[i]
        i += 1
        RPM_From += 1

    Br_Vary = Normalize_Signal(Br_Vary, -Nrm, Nrm)
    if Kb > 0:
        parser = np.where(Br_Vary.astype(int) == 0)[0][0]
    else:
        parser = np.where(Br_Vary.astype(int) == -1)[0][0]
    return [*Br_Vary[parser + 1:], *Br_Vary[:parser]]


def HrWave(dur, Tm_Part, HR_From, HR_Till, eta, Kb, signal_freq, windowSize, HRange, c, omega, gamma, b, Nrm):
    Hr_Vary = np.zeros(1)
    tm, i = 0, 0
    norm_fact = Nrm * (eta / Kb)  # Scaling factor
    HR_Lin = np.arange(HR_From, HR_Till + 1, 1)
    Tm_Needs = np.sum(60 / HR_Lin)
    Tm_Remain = dur - dur * Tm_Part
    if (int(Tm_Remain / Tm_Needs) == 0) or (int(Tm_Remain / Tm_Needs) == 1):
        RPM_Step = 1
    else:
        HR_Step = round(Tm_Remain / Tm_Needs)
    if (Tm_Needs * HR_Step > Tm_Remain):
        Tm_Remain = Tm_Needs * HR_Step

    while tm < (Tm_Remain-1):
        t_heart = np.arange(signal_freq * (60 / HR_From)) / signal_freq
        for n in range(0, HR_Step):
            HR_Ampl_Rnd = Random_RR_Ampl(HRange, eta)
            HR_One = xhs(t_heart, c, omega, gamma, eta, b)
            HR_One = lfilter(np.full((windowSize,), 1 / windowSize), 1, np.tile(HR_One, 1))
            Hr_Vary = [*Hr_Vary, *HR_One]
        tm += HR_Step * 60 / HR_Lin[i]
        i += 1
        HR_From += 1
    Hr_Vary = Normalize_Signal(Hr_Vary,  -norm_fact,  norm_fact)
    if eta > 0:
        if signal_freq < 1000:
            parser = (np.abs(Hr_Vary.astype(int))).argmin()
        else:
            parser = np.where(Hr_Vary.astype(int) == 0)[0][0]
    else:
        parser = np.where(Hr_Vary.astype(int) == -1)[0][0]
    return [*Hr_Vary[parser + 1:], *Hr_Vary[:parser]]


def Random_RR_Ampl(gate, Kb):
    if gate == 0:
        return Kb
    Low = Kb - gate
    High = Kb + gate + 1
    return np.random.randint(low=Low, high=High, size=(1,))


def white_noise(signal_pure, rho, sample_rate, mu=0):
    sigma = rho * np.sqrt(sample_rate / 2)
    noise1 = np.random.normal(mu, sigma, signal_pure.shape)
    return signal_pure + noise1


def Normalize_Signal(sig, min_val, max_val):
    min_sig = np.min(sig)
    max_sig = np.max(sig)
    sig = min_val + ((sig - min_sig) / (max_sig - min_sig)) * (max_val - min_val)
    # return sig.astype(int)
    return sig.astype(float)


def xhs(t, _c, _omega, _gamma, _eta, _b):
    within_the_cos = _omega * t + _gamma * np.sin(2 * t)
    return _eta * np.cos(within_the_cos) * np.exp(-((t - _b) ** 2) / _c)


def make_wav_file(data, _bpm, _rpm, _eta, _Kb, _khz):
    # wav_file = wav.open('HR'+str(_bpm)+'_RR'+str(_rpm)+'_Amp'+str(_eta)+'_'+str(_Kb)+'.wav', "w")
    wav_file = 'HR' + str(_bpm) + '_RR' + str(_rpm) + '_Amp' + str(_eta) + '_' + str(_Kb) + '.wav'
    nchannels, sampwidth = 1, 2
    n_frames = len(data)
    comptype, compname = "NONE", "not compressed"
    # wav_file.setparams((nchannels, sampwidth, int(_khz) * 1000, n_frames, comptype, compname))
    # for sample in data:
    # wav_file.writeframes(st.pack('h', int(sample)))
    write(wav_file, _khz, data)
    # wav_file.close()


def Tmp():  # part noise to signal in debbuging
    # ax2 = fig.add_subplot(212)
    # ax2.set_title('Noise added to part of the signal')
    # random_indices = np.random.randint(0, tt.size, int(tt.size * 0.05))
    # pr = combined_signal.copy().astype(float)
    # pr[random_indices] += Noisy_Sig[random_indices]
    # ax2.plot(tt[random_indices], pr[random_indices], '-r', label='signal+noise')
    # ax2.grid(True)
    plt.show()
    plt.figure(3)
    random_indices = np.random.randint(0, tt.size, int(tt.size * 1))
    pr = combined_signal.copy().astype(float)
    pr[random_indices] += Noisy_Sig[random_indices]
    plt.title('Noise added to part of the signal')
    plt.plot(tt[random_indices], pr[random_indices], '-r', label='signal+noise')
    plt.xlabel("frequency (Hz)")
    plt.ylabel("psd (arb.u./SQRT(Hz))")
    plt.legend()

def Port_Init(self):
    prt = 'COM10'
    brte = 115200  # 9600
    tout = 0.001  # Miliseconds
    Ard_COM_Port = serial.Serial(port=prt, baudrate=brte, timeout=tout)
    tiempo = (1 / brte) * 1000  # Time to update the data of the sensor signal Rs=9600baud T=1/Rs
    return [Ard_COM_Port, tiempo]


app = QtWidgets.QApplication(sys.argv)
window = MyWindow()
window.show()
sys.exit(app.exec_())
