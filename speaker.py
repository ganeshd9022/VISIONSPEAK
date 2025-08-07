import pyttsx3
import threading
import queue

class VoiceSpeaker:
    def __init__(self, rate=150, enable_voice=True):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', rate)
        self.enable_voice = enable_voice
        self.speech_queue = queue.Queue()
        self.thread = threading.Thread(target=self._speak_worker, daemon=True)
        self.thread.start()

    def _speak_worker(self):
        while True:
            text = self.speech_queue.get()
            if text is None:
                break
            self.engine.say(text)
            self.engine.runAndWait()
            self.speech_queue.task_done()

    def speak(self, text):
        if self.enable_voice:
            self.speech_queue.put(text)

    def stop(self):
        self.speech_queue.put(None)
        self.thread.join()
        self.engine.stop()
