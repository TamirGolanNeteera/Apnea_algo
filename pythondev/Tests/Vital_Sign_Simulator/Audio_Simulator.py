from PyQt5 import QtCore, QtWidgets, QtMultimedia, QtGui
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter
import wave as wav
import struct as st

class MyWindow(QtWidgets.QWidget):

    def __init__(self, parent=None):
        #QtWidgets.QWidget.__init__(self, parent, flags=QtCore.Qt.Window | QtCore.Qt.MSWindowsFixedSizeDialogHint)
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
        # Создаем компоненты для управления воспроизведением.
        # Делаем их изначально недоступными
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
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.lblHeartBeats = QtWidgets.QLabel("HeartBeats per minute", self)
        self.lblHeartBeats.setFont(font)
        self.lblHeartBeats.move(60, 60)
        hbox.addWidget(self.lblHeartBeats)
        self.spinBox_HR = QtWidgets.QSpinBox(self)
        self.spinBox_HR.setMinimum(1)
        self.spinBox_HR.setMaximum(180)
        self.spinBox_HR.setProperty("value", 60)
        self.spinBox_HR.setDisplayIntegerBase(100)
        self.spinBox_HR.setObjectName("spinBox_HR")
        hbox.addWidget(self.spinBox_HR)

        self.lblRespiration = QtWidgets.QLabel("Respiration per minute", self)
        self.lblRespiration.setFont(font)
        hbox.addWidget(self.lblRespiration)
        self.spinBox_RR = QtWidgets.QSpinBox(self)
        self.spinBox_RR.setMinimum(1)
        self.spinBox_RR.setMaximum(50)
        self.spinBox_RR.setProperty("value", 10)
        self.spinBox_RR.setDisplayIntegerBase(100)
        self.spinBox_RR.setObjectName("spinBox_RR")
        hbox.addWidget(self.spinBox_RR)
        vbox.addLayout(hbox)
        # Amlitude
        hbox = QtWidgets.QHBoxLayout()
        self.lblHRAmpl = QtWidgets.QLabel("HR Amplitude               ", self)
        self.lblHRAmpl.setFont(font)
        hbox.addWidget(self.lblHRAmpl)
        self.spinBox_HRAm = QtWidgets.QSpinBox(self)
        self.spinBox_HRAm.setMinimum(1)
        self.spinBox_HRAm.setMaximum(50)
        self.spinBox_HRAm.setProperty("value", 1)
        self.spinBox_HRAm.setDisplayIntegerBase(10)
        self.spinBox_HRAm.setObjectName("spinBox_HRAm")
        hbox.addWidget(self.spinBox_HRAm)

        self.lblRRAm = QtWidgets.QLabel("RR Ampitude                 ", self)
        self.lblRRAm.setFont(font)
        hbox.addWidget(self.lblRRAm)
        self.spinBox_RRAm = QtWidgets.QSpinBox(self)
        self.spinBox_RRAm.setMinimum(1)
        self.spinBox_RRAm.setMaximum(50)
        self.spinBox_RRAm.setProperty("value", 5)
        self.spinBox_RRAm.setDisplayIntegerBase(10)
        self.spinBox_RRAm.setObjectName("spinBox_RRAm")
        hbox.addWidget(self.spinBox_RRAm)
        vbox.addLayout(hbox)
        # Duration
        hbox = QtWidgets.QHBoxLayout()
        self.lblDurS = QtWidgets.QLabel("Duration in sec                   ", self)
        self.lblDurS.setFont(font)
        hbox.addWidget(self.lblDurS)
        self.spinBox_DurS = QtWidgets.QSpinBox(self)
        self.spinBox_DurS.setMinimum(1)
        self.spinBox_DurS.setMaximum(180)
        self.spinBox_DurS.setProperty("value", 30)
        self.spinBox_DurS.setDisplayIntegerBase(10)
        self.spinBox_DurS.setObjectName("spinBox_DurS")
        hbox.addWidget(self.spinBox_DurS)

        self.lblFr = QtWidgets.QLabel("Sampling frequency, kHz ", self)
        self.lblFr.setFont(font)
        hbox.addWidget(self.lblFr)
        self.spinBox_Fr = QtWidgets.QSpinBox(self)
        self.spinBox_Fr.setMinimum(32)
        self.spinBox_Fr.setMaximum(120)
        self.spinBox_Fr.setProperty("value", 48)
        self.spinBox_Fr.setDisplayIntegerBase(10)
        self.spinBox_Fr.setObjectName("spinBox_Fr")
        hbox.addWidget(self.spinBox_Fr)
        vbox.addLayout(hbox)
        # Plot and WAV File
        hbox = QtWidgets.QHBoxLayout()
        self.lblPlt = QtWidgets.QLabel("Show plot       ", self)
        self.lblPlt.setFont(font)
        hbox.addWidget(self.lblPlt)
        self.checkBox_Pl = QtWidgets.QCheckBox(self)
        self.checkBox_Pl.setObjectName("checkBox_Pl")
        hbox.addWidget(self.checkBox_Pl)

        # Duration
        self.lblWav = QtWidgets.QLabel("Generate WAV      ", self)
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
        bpm = self.spinBox_HR.value()  # Heart beat frequency
        rpm = self.spinBox_RR.value()  # Respiration beat frequency
        eta = self.spinBox_HRAm.value()  # Amplitude
        Kb = self.spinBox_RRAm.value()   # Amplitude
        dur = self.spinBox_DurS.value()  # Duration in sec
        khz = self.spinBox_Fr.value()  # Sampling frequency

        Tau = 1.5
        A = 0.2
        b = 0.6
        c, omega, gamma = 0.025, 1.05, 1
        eta, Kb = int(eta), int(Kb)
        bbi = 60 / int(bpm)
        inhale, exhale = (60 / int(rpm) / 2), (60 / int(rpm) / 2)
        signal_freq = int(khz) * 1000

        # generating the breath signal
        t_breath = np.arange(signal_freq * (inhale + exhale)) / signal_freq
        num_breaths = int((int(dur) / 60) * int(rpm)) + 1
        one_w = w(t_breath, inhale, exhale, Kb, Tau, 0)
        breath_signal = lfilter(np.full((4000,), 1 / 4000), 1, np.tile(one_w, 2))
        breath_signal = normalize_signal(breath_signal, -25750, 25750)
        if Kb > 0:
            parser = np.where(breath_signal.astype(int) == 0)[0][0]
        else:
            parser = np.where(breath_signal.astype(int) == -1)[0][0]

        breath_signal = np.concatenate((breath_signal[parser:], breath_signal[:parser]))[:one_w.shape[0]]
        breath_signals = np.tile(breath_signal, num_breaths)

        # generating the hb signal
        t_hb = np.arange(signal_freq * bbi) / signal_freq
        one_hb = xhs(t_hb, c, omega, gamma, eta, b)
        num_hbs = int(int(dur) * int(bpm) / 60)
        hb_signal = np.tile(one_hb, num_hbs + 1)
        num = 25750 * (eta / Kb)
        hb_signal = normalize_signal(hb_signal, -num, num)
        hb_signal = hb_signal[:len(breath_signals)]
        win_dow = np.log(np.arange(0.999, 0, -1 / ((int(dur) / 4) * signal_freq)))
        win_dow -= np.min(win_dow)
        win_dow /= np.max(win_dow)
        hb_signal = np.concatenate((hb_signal[:-len(win_dow)], hb_signal[-len(win_dow):] * win_dow))
        hb_signal = hb_signal[np.where(hb_signal == 0)[0][0]:]

        # combining all together
        breath_signals = breath_signals[:len(hb_signal)]

        fig, ax = plt.subplots()
        hb_signal = lfilter(np.full((4000,), 1 / 4000), 1, hb_signal)
        combined_signal = normalize_signal(hb_signal + breath_signals, -25750, 25750)
        mod = (len(combined_signal)) % (np.floor(signal_freq * (inhale + exhale)).astype(int))
        parsers = np.where(combined_signal.astype(int) == 0)[0]

        idx = (np.abs(parsers - (len(combined_signal) - mod))).argmin()
        combined_signal = combined_signal[:parsers[idx]]
        l, = plt.plot(np.arange(len(combined_signal) / 100) / signal_freq, combined_signal.astype(int)[::100])
        plt.grid()
        ax.margins(x=0)
        tx = ''
        for num in combined_signal.astype(int):
            tx += str(num) + ','
        with open('HR' + str(bpm) + '_RR' + str(rpm) + '_Amp' + str(eta) + '_' + str(Kb) + '.txt', 'w') as f:
            f.write(tx[:-1] + '\n\n')
            f.write(str(combined_signal.shape[0]) + '\n\n')
            f.write(str(rpm) + ',' + str(bpm))
        if self.checkBox_Wav.isChecked():  # if_to_wav:
            make_wav_file(combined_signal, bpm, rpm, eta, Kb, khz)
            print('Wav is completed')
        if self.checkBox_Pl.isChecked():  # if_to_plot:
            plt.show()
        fig.savefig('HR' + str(bpm) + '_RR' + str(rpm) + '_Amp' + str(eta) + '_' + str(Kb) + '.jpg', dpi=fig.dpi)


    # To open the file, use the getOpenFileUrl () method of the class
    # QFileDialog, because to create an instance of the class
    # QMediaContent we need the path to the file, given as an instance of the QUrl class
    def openFile(self):
        file = QtWidgets.QFileDialog.getOpenFileUrl(parent=self, caption="Select a sound file", filter="Sound files (*.mp3 *.wav)")
        self.mplPlayer.setMedia(QtMultimedia.QMediaContent(file[0]))
       # if file != None:
       #     self.currentPlaylist.addMedia(QMediaContent(file[0]))

    def initPlayer(self, state):
        if state == QtMultimedia.QMediaPlayer.LoadedMedia:
            # После загрузки файла подготавливаем проигрыватель
            # для его воспроизведения
            self.mplPlayer.stop()
            self.btnPlay.setEnabled(True)
            self.btnPause.setEnabled(False)
            self.btnLoop.setEnabled(True)
            self.sldPosition.setEnabled(True)    #was True
            self.sldPosition.setMaximum(self.mplPlayer.duration())
        elif state == QtMultimedia.QMediaPlayer.EndOfMedia:
            # По окончании воспроизведения файла возвращаем
            # проигрыватель в изначальное состояние
            self.mplPlayer.stop()
            self.sldPosition.setValue(0)
            self.sldPosition.setEnabled(False)
            self.btnPlay.setEnabled(False)
            self.btnPause.setEnabled(False)
            self.btnStop.setEnabled(False)
        elif state == QtMultimedia.QMediaPlayer.NoMedia or state == QtMultimedia.QMediaPlayer.InvalidMedia:
            # Если файл не был загружен, отключаем компоненты,
            # управляющие воспроизведением
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
            self.playlist.setPlaybackMode(QtMultimedia.QMediaPlaylist.Loop)
        else:
            self.playlist.setPlaybackMode(QtMultimedia.QMediaPlaylist.Sequential)

def w(t, Ti, Te, kb, tau, bias):
        T = Te + Ti
        peak = np.where(t >= Ti)[0][0]
        ti = t[:peak]
        te = t[peak:]
        wi = (-kb * ti ** 2) / (Ti * Te) + (kb * T * ti) / (Ti * Te)
        we = (kb / (1 - np.exp(-Te / tau))) * (np.exp(-(te - t[peak]) / tau) - np.exp(-Te / tau))
        return np.concatenate((wi, we)) + bias

def normalize_signal(sig, min_val, max_val):
    min_sig = np.min(sig)
    max_sig = np.max(sig)
    sig = (sig - min_sig) / (max_sig - min_sig)
    sig *= (max_val - min_val)
    sig += min_val
    return sig.astype(int)

def xhs(t, _c, _omega, _gamma, _eta, _b):
    within_the_cos = _omega*t + _gamma*np.sin(2*t)
    return _eta * np.cos(within_the_cos) * np.exp(-((t-_b)**2) / _c)

def make_wav_file(data, _bpm, _rpm, _eta, _Kb, _khz):
    wav_file = wav.open('HR'+str(_bpm)+'_RR'+str(_rpm)+'_Amp'+str(_eta)+'_'+str(_Kb)+'.wav', "w")
    nchannels, sampwidth = 1, 2
    n_frames = len(data)
    comptype, compname = "NONE", "not compressed"
    wav_file.setparams((nchannels, sampwidth, int(_khz) * 1000, n_frames, comptype, compname))
    for sample in data:
        wav_file.writeframes(st.pack('h', int(sample)))
    wav_file.close()

app = QtWidgets.QApplication(sys.argv)
window = MyWindow()
window.show()
sys.exit(app.exec_())
