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
from PyQt5.QtWidgets import QSpinBox, QCheckBox, QMessageBox, QHBoxLayout, QFileDialog, QComboBox, QStatusBar, QMainWindow, QLabel, QPushButton, QTextEdit, QVBoxLayout, QWidget, QDialog, QFormLayout, QLineEdit, QDialogButtonBox
from PyQt5.QtGui import QFont, QIcon, QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QTextEdit, QVBoxLayout, QWidget
from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QPushButton, QTextEdit, QLabel
from PyQt5.QtCore import Qt, QTimer, QSize
from PyQt5.QtGui import QFont
import whisper
from ascii_colors import ASCIIColors, trace_exception
from lollms_client import LollmsClient, LollmsDiscussion, LollmsXTTS, TasksLibrary, FunctionCalling_Library 
import gc
import requests
from pathlib import Path
import re
import importlib
import cv2
import math
from datetime import datetime

class TranscriptionSignal(QObject):
    new_user_transcription = pyqtSignal(str, str)
    new_lollms_transcription = pyqtSignal(str, str)
    update_status = pyqtSignal(str)

class AudioRecorder:
    def __init__(self, ui, lollms_address="http://localhost:9600", cond="Act as a helpful AI assistant called lollms.", threshold=1000, silence_duration=2, sound_threshold_percentage=10, gain=1.0, rate=44100, channels=1, buffer_size=10, model="small.en", snd_device=None, logs_folder="logs", voice=None, block_while_talking=True, context_size=4096):
        self.ui = ui
        self.lc = LollmsClient(lollms_address)
        self.tts = LollmsXTTS(self.lc)
        self.tl = TasksLibrary(self.lc)
        self.fn = FunctionCalling_Library(self.tl)

        self.fn.register_function(
                                        "calculator_function", 
                                        self.calculator_function, 
                                        "returns the result of a calculation passed through the expression string parameter",
                                        [{"name": "expression", "type": "str"}]
                                    )
        self.fn.register_function(
                                        "get_date_time", 
                                        self.get_date_time, 
                                        "returns the current date and time",
                                        []
                                    )
        self.fn.register_function(
                                        "take_a_photo", 
                                        self.take_a_photo, 
                                        "Takes a photo and returns the status",
                                        []
                                    )

        self.block_listening = False
        if not voice:
            voices = self.get_voices(lollms_address)
            voice = voices[0]
        self.voice = voice
        self.context_size = context_size
        self.cond = cond
        self.rate = rate
        self.channels = channels
        self.threshold = threshold
        self.silence_duration = silence_duration
        self.buffer_size = buffer_size
        self.gain = gain
        self.sound_threshold_percentage = sound_threshold_percentage
        self.block_while_talking = block_while_talking
        self.image_shot = None

        if snd_device is None:
            devices = sd.query_devices()
            snd_device = [device['name'] for device in devices][0]

        self.snd_device = snd_device
        self.logs_folder = logs_folder

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
    
    def get_date_time(self):
        now = datetime.now()
        return now.strftime("%Y-%m-%d %H:%M:%S")        
    
    def calculator_function(self, expression: str) -> float:
        try:
            # Add the math module functions to the local namespace
            allowed_names = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
            
            # Evaluate the expression safely using the allowed names
            result = eval(expression, {"__builtins__": None}, allowed_names)
            return result
        except Exception as e:
            return str(e)
        
    def take_a_photo(self):
        if self.ui.camera_available:
            self.image_shot = self.logs_folder+"/shot.png"
            cv2.imwrite(self.image_shot, self.ui.frame)
            return "Photo taken!"
        return "Couldn't take a photo"


    def start_recording(self):
        self.recording = True
        self.stop_flag = False

        threading.Thread(target=self._record).start()
        threading.Thread(target=self._process_files).start()

    def stop_recording(self):
        self.stop_flag = True

    def _record(self):
        sd.default.device = self.snd_device
        with sd.InputStream(channels=self.channels, samplerate=self.rate, callback=self.callback, dtype='int16'):
            while not self.stop_flag:
                time.sleep(0.1)

        if self.frames:
            self._save_wav(self.frames)
        self.recording = False

        # self._save_histogram(self.audio_values)

    def callback(self, indata, frames, time, status):
        if not self.block_listening:
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
        else:
            self.frames = []
            self.silence_counter = 0
            self.current_silence_duration = 0
            self.longest_silence_duration = 0
            self.sound_frames = 0
            self.audio_values = []

            self.max_audio_value = 0
            self.min_audio_value = 0
            self.total_frames = 0  # Initialize total_frames

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
        logs_file = Path(self.logs_folder)/filename
        logs_file.parent.mkdir(exist_ok=True, parents=True)

        wf = wave.open(str(logs_file), 'wb')
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

    def fix_string_for_xtts(self, input_string):
        # Remove excessive exclamation marks
        fixed_string = input_string.rstrip('!')
        
        return fixed_string
    
    def _process_files(self):
        while not self.stop_flag:
            with self.buffer_lock:
                while not self.buffer and not self.stop_flag:
                    self.buffer_lock.wait()
                if self.buffer:
                    filename = self.buffer.popleft()
                    self.buffer_lock.notify()
            if self.block_while_talking:
                self.block_listening = True
            try:
                if filename:
                    self.transcription_signal.update_status.emit("Transcribing")
                    ASCIIColors.green("<<TRANSCRIBING>>")
                    result = self.whisper.transcribe(str(Path(self.logs_folder)/filename))
                    transcription_fn = str(Path(self.logs_folder)/filename) + ".txt"
                    with open(transcription_fn, "w", encoding="utf-8") as f:
                        f.write(result["text"])

                    with self.transcribed_lock:
                        self.transcribed_files.append((filename, result["text"]))
                        self.transcribed_lock.notify()
                    if result["text"]!="":
                        self.transcription_signal.new_user_transcription.emit(filename, result["text"])
                        self.discussion.add_message("user",result["text"])
                        discussion = self.discussion.format_discussion(self.context_size)
                        full_context = '!@>system:' + self.cond +"\n" + discussion+"\n!@>lollms:"
                        ASCIIColors.red(" ---------------- Discussion ---------------------")
                        ASCIIColors.yellow(full_context)
                        ASCIIColors.red(" -------------------------------------------------")
                        self.transcription_signal.update_status.emit("Generating answer")
                        ASCIIColors.green("<<RESPONDING>>")
                        lollms_text, function_calls =self.fn.generate_with_functions(full_context)
                        if len(function_calls)>0:
                            responses = self.fn.execute_function_calls(function_calls=function_calls)
                            if self.image_shot:
                                lollms_text = self.lc.generate_with_images(full_context+"!@>lollms: "+ lollms_text + "\n!@>functions outputs:\n"+ "\n".join(responses) +"!@>lollms:", [self.image_shot])
                            else:
                                lollms_text = self.lc.generate(full_context+"!@>lollms: "+ lollms_text + "\n!@>functions outputs:\n"+ "\n".join(responses) +"!@>lollms:")
                        lollms_text = self.fix_string_for_xtts(lollms_text)
                        self.discussion.add_message("lollms",lollms_text)
                        ASCIIColors.red(" -------------- LOLLMS answer -------------------")
                        ASCIIColors.yellow(lollms_text)
                        ASCIIColors.red(" -------------------------------------------------")
                        self.transcription_signal.new_lollms_transcription.emit(filename, lollms_text)
                        self.transcription_signal.update_status.emit("Talking")
                        ASCIIColors.green("<<TALKING>>")
                        self.tts.text2Audio(lollms_text.replace("!",".").replace("###"," ").replace("###"," "), voice=self.voice)
            except Exception as ex:
                trace_exception(ex)
            self.block_listening = False
            ASCIIColors.green("<<LISTENING>>")
            self.transcription_signal.update_status.emit("Listening")

    def get_voices(self, host_address):
        endpoint = f"{host_address}/list_voices"
        try:
            response = requests.get(endpoint)
            response.raise_for_status()  # Raise an error for bad status codes
            voices = response.json()  # Assuming the response is in JSON format
            return voices["voices"]
        except requests.exceptions.RequestException as e:
            print(f"Couldn't list voices: {e}")
            return ["main_voice"]

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.recorder = AudioRecorder(self)
        self.recorder.transcription_signal.new_user_transcription.connect(self.display_user_transcription)
        self.recorder.transcription_signal.new_lollms_transcription.connect(self.display_lollms_transcription)
        self.recorder.transcription_signal.update_status.connect(self.update_status_bar)
        
        self.load_settings()
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.camera_available = False
            self.cap.release()
        else:
            self.camera_available = True
            self.timer = QTimer()
            self.timer.timeout.connect(self.update_frame)

        self.init_ui()
        if self.camera_available :
            self.timer.start(20)        

    def init_ui(self):
        self.setWindowTitle('Talk to LoLLMs')
        self.setGeometry(100, 100, 800, 600)

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

        # Mute Button
        self.mute_button = QPushButton('Mute')
        self.mute_button.setFont(QFont('Arial', 14))
        self.mute_button.setStyleSheet("background-color: #FFA500; color: white; padding: 10px; border-radius: 5px;")
        self.mute_button.clicked.connect(self.toggle_mute)

        # Settings Button
        self.settings_button = QPushButton()
        # Get the path to the SVG file
        try:
            icon_path = importlib.resources.files('talk_to_lollms.assets').joinpath('settings.svg')
            # Load the SVG icon
            self.settings_button.setIcon(QIcon(str(icon_path)))
        except:
            try:
                self.settings_button.setIcon(QIcon("talk_to_lollms/assets/settings.svg"))
            except:
                pass
        self.settings_button.setIconSize(QSize(24, 24))
        self.settings_button.clicked.connect(self.show_settings_dialog)

        # Text Edit for Transcriptions
        self.text_edit = QTextEdit()
        self.text_edit.setFont(QFont('Arial', 12))
        self.text_edit.setStyleSheet("background-color: #ffffff; font-family: Arial, sans-serif; font-size: 14px; padding: 10px;")

        # Layouts
        layout = QVBoxLayout()
        layout.addWidget(title_label)
        
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.record_button)
        button_layout.addWidget(self.mute_button)
        button_layout.addWidget(self.settings_button)
        layout.addLayout(button_layout)

        main_layout = QHBoxLayout()
        
        left_layout = QVBoxLayout()
        if self.camera_available:
            # Camera Feed Label
            self.camera_feed = QLabel()
            self.camera_feed.setFixedSize(320, 240)
            left_layout.addWidget(self.camera_feed)

            # Image Feed Label
            self.image_feed = QLabel()
            self.image_feed.setFixedSize(320, 240)
            left_layout.addWidget(self.image_feed)

        # Another Whiteboard
        self.whiteboard = QLabel()
        self.whiteboard.setFixedSize(320, 240)
        self.whiteboard.setStyleSheet("background-color: #ffffff; border: 1px solid #000000;")
        left_layout.addWidget(self.whiteboard)
        
        main_layout.addLayout(left_layout)
        main_layout.addWidget(self.text_edit)

        layout.addLayout(main_layout)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        
        # Status Bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)


    def toggle_mute(self):
        self.recorder.block_listening = self.recorder.block_listening
        
    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.frame=frame
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            convert_to_qt_format = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            p = convert_to_qt_format.scaled(self.camera_feed.width(), self.camera_feed.height(), Qt.KeepAspectRatio)
            self.camera_feed.setPixmap(QPixmap.fromImage(p))

    def closeEvent(self, event):
        if self.camera_available:
            self.cap.release()
        event.accept()

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
        devices_list = QComboBox(self)
        devices = sd.query_devices()
        device_names = [device['name'] for device in devices]
        devices_list.addItems(device_names)        
        devices_list.setCurrentText(str(self.recorder.snd_device))

        logs_folder_input = QLineEdit(self.recorder.logs_folder)
        logs_folder_input.setText(self.recorder.logs_folder)
        logs_folder_button = QPushButton("...")

        voices_list = QComboBox(self)
        voices_list_values = self.recorder.get_voices(self.recorder.lc.host_address)
        voices_list.addItems(voices_list_values)        
        voices_list.setCurrentText(str(self.recorder.snd_device))

        block_while_talking_input = QCheckBox()
        block_while_talking_input.setChecked(self.recorder.block_while_talking)

        context_size_input = QSpinBox()
        context_size_input.setMaximum(100000000)
        context_size_input.setValue(self.recorder.context_size if type(self.recorder.context_size)==int else 4096)
        

        def open_folder_dialog():
            folder_path = QFileDialog.getExistingDirectory(dialog, "Select Logs Folder")
            if folder_path:
                logs_folder_input.setText(folder_path)
        
        logs_folder_button.clicked.connect(open_folder_dialog)
        logs_folder_layout = QHBoxLayout()
        logs_folder_layout.addWidget(logs_folder_input)
        logs_folder_layout.addWidget(logs_folder_button)        

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
        form_layout.addRow("Audio Devices:", devices_list)
        form_layout.addRow("Logs Folder:", logs_folder_layout)
        form_layout.addRow("Voices List:", voices_list)
        form_layout.addRow("Block While Talking:", block_while_talking_input)        
        form_layout.addRow("Context Size:", context_size_input)
        
        

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
            devices_list.currentText(),
            logs_folder_input.text(),
            voices_list.currentText(),
            block_while_talking_input.isChecked(),
            context_size_input.value()
        ))
        button_box.rejected.connect(dialog.reject)

        form_layout.addWidget(button_box)
        dialog.setLayout(form_layout)
        dialog.exec_()

    def save_settings_and_close(self, dialog, cond, lollms_address, threshold, silence_duration, sound_threshold_percentage, gain, rate, channels, buffer_size, model, snd_device, logs_folder, voice, block_while_talking, context_size):
        
        self.recorder.cond = cond
        self.recorder.lc.host_address = lollms_address
        self.recorder.threshold = threshold
        self.recorder.silence_duration = silence_duration
        self.recorder.sound_threshold_percentage = sound_threshold_percentage
        self.recorder.gain = gain
        self.recorder.rate = rate
        self.recorder.channels = channels
        self.recorder.buffer_size = buffer_size
        self.recorder.snd_device = snd_device
        self.recorder.logs_folder = logs_folder
        self.recorder.voice = voice
        self.recorder.block_while_talking = block_while_talking
        self.context_size = context_size
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
            'buffer_size': buffer_size,
            'snd_device': snd_device,
            'logs_folder': logs_folder,
            'voice': voice,
            'block_while_talking': block_while_talking,
            'context_size': context_size
        }

        with open('settings.json', 'w') as f:
            json.dump(settings, f)
        
        self.update_status_bar("Settings saved")
        dialog.accept()
        try:
            sd.default.device = snd_device
        except Exception as e:
            QMessageBox.critical(dialog, "Error", f"Failed to set sound device: {e}")
            return       

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
                self.recorder.snd_device = settings.get('snd_device',"")
                self.recorder.logs_folder = settings.get('logs_folder',"")
                self.recorder.voice = settings.get('voice',"")
                self.recorder.context_size = settings.get('context_size',"")
                
                
        except FileNotFoundError:
            pass

    def display_user_transcription(self, filename, transcription: str):
        transcription = transcription.replace('\n', '<br>')
        user_html = f"""
        <div style="background-color: #e1f5fe; border: 1px solid #0277bd; border-radius: 10px; padding: 10px; margin: 5px 0; max-width: 70%; word-wrap: break-word;">
            <b style="color: #0277bd;">User:</b><br>
            <span>{transcription}</span>
        </div>
        """
        self.text_edit.append(user_html)

    def display_lollms_transcription(self, filename, transcription: str):
        transcription = transcription.replace('\n', '<br>')
        lollms_html = f"""
        <div style="background-color: #e8f5e9; border: 1px solid #388e3c; border-radius: 10px; padding: 10px; margin: 5px 0; max-width: 70%; word-wrap: break-word; align-self: flex-end; text-align: right;">
            <b style="color: #388e3c;">LoLLMs:</b><br>
            <span>{transcription}</span>
        </div>
        """
        self.text_edit.append(f'<div style="display: flex; justify-content: flex-end;">{lollms_html}</div>')

    def update_status_bar(self, message:str):
        self.status_bar.showMessage(message)  # Display message for 5 seconds



if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
