import sys
import threading
import time
import numpy as np
import sounddevice as sd
import wave
import matplotlib.pyplot as plt
from collections import deque
from PyQt5.QtCore import pyqtSignal, QObject
import json
from PyQt5.QtWidgets import QStatusBar, QMainWindow, QLabel, QPushButton, QTextEdit, QVBoxLayout, QWidget, QDialog, QFormLayout, QLineEdit, QDialogButtonBox
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QTextEdit, QVBoxLayout, QWidget
from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QPushButton, QTextEdit, QLabel
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QFont
import whisper
from ascii_colors import ASCIIColors, trace_exception
from lollms_client import LollmsClient, LollmsDiscussion, LollmsXTTS
import gc

class TranscriptionSignal(QObject):
    new_user_transcription = pyqtSignal(str, str)
    new_lollms_transcription = pyqtSignal(str, str)
    update_status = pyqtSignal(str)

class AudioRecorder:
    def __init__(self, lollms_address="http://localhost:9600", cond="Act as a helpful AI assistant called lollms.", threshold=500, silence_duration=2, sound_threshold_percentage=10, gain=1.0, rate=44100, channels=1, buffer_size=10, model="small.en"):
        self.lc = LollmsClient(lollms_address)
        self.tts = LollmsXTTS(self.lc)
        self.cond = cond
        self.rate = rate
        self.channels = channels
        self.threshold = threshold
        self.silence_duration = silence_duration
        self.buffer_size = buffer_size
        self.gain = gain
        self.sound_threshold_percentage = sound_threshold_percentage

        self.frames = []
        self.silence_counter = 0
        self.current_silence_duration = 0
        self.longest_silence_duration = 0
        self.sound_frames = 0
        self.audio_values = []

        self.max_audio_value = 0
        self.min_audio_value = 0
        self.total_frames = 0  # Initialize total_frames

        self.file_index = 0
        self.recording = False
        self.stop_flag = False

        self.buffer = deque(maxlen=buffer_size)
        self.transcribed_files = deque()
        self.buffer_lock = threading.Condition()
        self.transcribed_lock = threading.Condition()
        self.transcription_signal = TranscriptionSignal()
        ASCIIColors.info("Loading whisper...",end="",flush=True)

        self.model = model
        self.whisper = whisper.load_model(model)
        ASCIIColors.success("OK")
        self.discussion = LollmsDiscussion(self.lc)
        

    def start_recording(self):
        self.recording = True
        self.stop_flag = False

        threading.Thread(target=self._record).start()
        threading.Thread(target=self._process_files).start()

    def stop_recording(self):
        self.stop_flag = True

    def _record(self):
        with sd.InputStream(channels=self.channels, samplerate=self.rate, callback=self.callback, dtype='int16'):
            while not self.stop_flag:
                time.sleep(0.1)

        if self.frames:
            self._save_wav(self.frames)
        self.recording = False

        self._save_histogram(self.audio_values)

    def callback(self, indata, frames, time, status):
        audio_data = np.frombuffer(indata, dtype=np.int16)
        max_value = np.max(audio_data)
        min_value = np.min(audio_data)

        if max_value > self.max_audio_value:
            self.max_audio_value = max_value
        if min_value < self.min_audio_value:
            self.min_audio_value = min_value

        self.audio_values.extend(audio_data)

        self.total_frames += frames
        if max_value < self.threshold:
            self.silence_counter += 1
            self.current_silence_duration += frames
        else:
            self.silence_counter = 0
            self.current_silence_duration = 0
            self.sound_frames += frames

        if self.current_silence_duration > self.longest_silence_duration:
            self.longest_silence_duration = self.current_silence_duration

        if self.silence_counter > (self.rate / frames * self.silence_duration):
            trimmed_frames = self._trim_silence(self.frames)
            sound_percentage = self._calculate_sound_percentage(trimmed_frames)
            if sound_percentage >= self.sound_threshold_percentage:
                self._save_wav(self.frames)
            self.frames = []
            self.silence_counter = 0
            self.total_frames = 0
            self.sound_frames = 0
        else:
            self.frames.append(indata.copy())

    def _apply_gain(self, frames):
        audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
        audio_data = audio_data * self.gain
        audio_data = np.clip(audio_data, -32768, 32767)
        return audio_data.astype(np.int16).tobytes()

    def _trim_silence(self, frames):
        audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
        non_silent_indices = np.where(np.abs(audio_data) >= self.threshold)[0]

        if non_silent_indices.size:
            start_index = max(non_silent_indices[0] - self.rate, 0)
            end_index = min(non_silent_indices[-1] + self.rate, len(audio_data))
            trimmed_data = audio_data[start_index:end_index]
        else:
            trimmed_data = np.array([], dtype=np.int16)

        return trimmed_data.tobytes()

    def _calculate_sound_percentage(self, frames):
        audio_data = np.frombuffer(frames, dtype=np.int16)
        num_bins = len(audio_data) // self.rate
        sound_count = 0

        for i in range(num_bins):
            bin_data = audio_data[i * self.rate: (i + 1) * self.rate]
            if np.max(bin_data) >= self.threshold:
                sound_count += 1

        sound_percentage = (sound_count / num_bins) * 100 if num_bins > 0 else 0
        return sound_percentage

    def _save_wav(self, frames):
        ASCIIColors.green("<<SEGMENT_RECOVERED>>")
        self.transcription_signal.update_status.emit("Segment detected and saved")
        filename = f"recording_{self.file_index}.wav"
        self.file_index += 1

        amplified_frames = self._apply_gain(frames)
        trimmed_frames = self._trim_silence([amplified_frames])

        wf = wave.open("logs/"+ filename, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(2)
        wf.setframerate(self.rate)
        wf.writeframes(trimmed_frames)
        wf.close()

        with self.buffer_lock:
            while len(self.buffer) >= self.buffer.maxlen:
                self.buffer_lock.wait()
            self.buffer.append(filename)
            self.buffer_lock.notify()

    def _save_histogram(self, audio_values):
        plt.hist(audio_values, bins=50, edgecolor='black')
        plt.title('Histogram of Audio Values')
        plt.xlabel('Audio Value')
        plt.ylabel('Frequency')
        plt.savefig('audio_values_histogram.png')
        plt.close()

    def _process_files(self):
        while not self.stop_flag or len(self.buffer) > 0:
            with self.buffer_lock:
                while not self.buffer and not self.stop_flag:
                    self.buffer_lock.wait()
                if self.buffer:
                    filename = self.buffer.popleft()
                    self.buffer_lock.notify()

            if filename:
                self.transcription_signal.update_status.emit("Transcribing")
                ASCIIColors.green("<<TRANSCRIBING>>")
                result = self.whisper.transcribe("logs/"+filename)
                transcription_fn = "logs/"+ filename + ".txt"
                with open(transcription_fn, "w", encoding="utf-8") as f:
                    f.write(result["text"])

                with self.transcribed_lock:
                    self.transcribed_files.append((filename, result["text"]))
                    self.transcribed_lock.notify()
                self.transcription_signal.new_user_transcription.emit(filename, result["text"])
                self.discussion.add_message("user",result["text"])
                if result["text"]!="":
                    discussion = self.discussion.format_discussion(4096)
                    print(discussion)
                    self.transcription_signal.update_status.emit("Generating answer")
                    lollms_text = self.lc.generate_text('!@>system:' + self.cond + discussion+"\n!@>lollms:", personality=0)
                    self.discussion.add_message("lollms",lollms_text)
                    print(lollms_text)
                    self.transcription_signal.update_status.emit("Listening")
                    self.transcription_signal.new_lollms_transcription.emit(filename, lollms_text)
                    self.tts.text2Audio(lollms_text)
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.recorder = AudioRecorder()
        self.recorder.transcription_signal.new_user_transcription.connect(self.display_user_transcription)
        self.recorder.transcription_signal.new_lollms_transcription.connect(self.display_lollms_transcription)
        self.recorder.transcription_signal.update_status.connect(self.update_status_bar)
        
        self.load_settings()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Talk to LoLLMs')
        self.setGeometry(100, 100, 600, 400)

        # Title Label
        title_label = QLabel('Talk to LoLLMs')
        title_font = QFont('Arial', 24, QFont.Bold)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)

        # Record Button
        self.record_button = QPushButton('Start Recording')
        self.record_button.setFont(QFont('Arial', 14))
        self.record_button.setStyleSheet("background-color: #4CAF50; color: white; padding: 10px; border-radius: 5px;")
        self.record_button.clicked.connect(self.toggle_recording)

        # Settings Button
        self.settings_button = QPushButton()
        self.settings_button.setIcon(QIcon('assets/settings.svg'))  # Load the SVG icon
        self.settings_button.setIconSize(QSize(24, 24))
        self.settings_button.clicked.connect(self.show_settings_dialog)

        # Text Edit for Transcriptions
        self.text_edit = QTextEdit()
        self.text_edit.setFont(QFont('Arial', 12))
        self.text_edit.setStyleSheet("background-color: #f0f0f0; padding: 10px; border-radius: 5px;")

        # Status Bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        layout = QVBoxLayout()
        layout.addWidget(title_label)
        layout.addWidget(self.record_button)
        layout.addWidget(self.settings_button)
        layout.addWidget(self.text_edit)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def toggle_recording(self):
        if self.recorder.recording:
            self.recorder.stop_recording()
            self.record_button.setText('Start Recording')
            self.record_button.setStyleSheet("background-color: #4CAF50; color: white; padding: 10px; border-radius: 5px;")
            self.update_status_bar("Recording stopped")
        else:
            self.recorder.start_recording()
            self.record_button.setText('Stop Recording')
            self.record_button.setStyleSheet("background-color: #f44336; color: white; padding: 10px; border-radius: 5px;")
            self.update_status_bar("Recording started")

    def show_settings_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Settings")
        form_layout = QFormLayout()

        cond_input =  QTextEdit(self.recorder.cond)
        lollms_address_input = QLineEdit(self.recorder.lc.host_address)
        threshold_input = QLineEdit(str(self.recorder.threshold))
        silence_duration_input = QLineEdit(str(self.recorder.silence_duration))
        sound_threshold_percentage_input = QLineEdit(str(self.recorder.sound_threshold_percentage))
        gain_input = QLineEdit(str(self.recorder.gain))
        rate_input = QLineEdit(str(self.recorder.rate))
        channels_input = QLineEdit(str(self.recorder.channels))
        buffer_size_input = QLineEdit(str(self.recorder.buffer_size))
        model_input =  QLineEdit(self.recorder.model)

        
        form_layout.addRow("LoLLMs Conditionning:", cond_input)
        form_layout.addRow("LoLLMs Address:", lollms_address_input)
        form_layout.addRow("Threshold:", threshold_input)
        form_layout.addRow("Silence Duration:", silence_duration_input)
        form_layout.addRow("Sound Threshold Percentage:", sound_threshold_percentage_input)
        form_layout.addRow("Gain:", gain_input)
        form_layout.addRow("Rate:", rate_input)
        form_layout.addRow("Channels:", channels_input)
        form_layout.addRow("Buffer Size:", buffer_size_input)
        form_layout.addRow("Whisper Model:", model_input)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(lambda: self.save_settings_and_close(
            dialog,
            cond_input.toPlainText(),
            lollms_address_input.text(),
            int(threshold_input.text()),
            int(silence_duration_input.text()),
            int(sound_threshold_percentage_input.text()),
            float(gain_input.text()),
            int(rate_input.text()),
            int(channels_input.text()),
            int(buffer_size_input.text()),
            model_input.text(),
            
        ))
        button_box.rejected.connect(dialog.reject)

        form_layout.addWidget(button_box)
        dialog.setLayout(form_layout)
        dialog.exec_()

    def save_settings_and_close(self, dialog, cond, lollms_address, threshold, silence_duration, sound_threshold_percentage, gain, rate, channels, buffer_size, model):
        
        self.recorder.cond = cond
        self.recorder.lc.host_address = lollms_address
        self.recorder.threshold = threshold
        self.recorder.silence_duration = silence_duration
        self.recorder.sound_threshold_percentage = sound_threshold_percentage
        self.recorder.gain = gain
        self.recorder.rate = rate
        self.recorder.channels = channels
        self.recorder.buffer_size = buffer_size
        if self.recorder.model != model:
            self.recorder.whisper = None
            gc.collect
            self.recorder.whisper = whisper.load_model(model)
            self.recorder.model = model

        settings = {
            'cond': cond,
            'lollms_address': lollms_address,
            'threshold': threshold,
            'silence_duration': silence_duration,
            'sound_threshold_percentage': sound_threshold_percentage,
            'gain': gain,
            'rate': rate,
            'channels': channels,
            'buffer_size': buffer_size
        }

        with open('settings.json', 'w') as f:
            json.dump(settings, f)
        
        self.update_status_bar("Settings saved")
        dialog.accept()

    def load_settings(self):
        try:
            with open('settings.json', 'r') as f:
                settings = json.load(f)
                self.recorder.cond = settings['cond']
                self.recorder.lc.host_address = settings['lollms_address']
                self.recorder.threshold = settings['threshold']
                self.recorder.silence_duration = settings['silence_duration']
                self.recorder.sound_threshold_percentage = settings['sound_threshold_percentage']
                self.recorder.gain = settings['gain']
                self.recorder.rate = settings['rate']
                self.recorder.channels = settings['channels']
                self.recorder.buffer_size = settings['buffer_size']
        except FileNotFoundError:
            pass

    def display_user_transcription(self, filename, transcription:str):
        transcription = transcription.replace('\n','<br>')
        self.text_edit.append(f"<b>User:</b><br>{transcription}<br>")

    def display_lollms_transcription(self, filename, transcription:str):
        transcription = transcription.replace('\n','<br>')
        self.text_edit.append(f"<b>LoLLMs:</b><br>{transcription}<br>")

    def update_status_bar(self, message:str):
        self.status_bar.showMessage(message)  # Display message for 5 seconds



if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
