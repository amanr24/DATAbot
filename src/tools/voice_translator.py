import tempfile
import os
import torchaudio
from faster_whisper import WhisperModel
from deep_translator import GoogleTranslator
from gtts import gTTS

class VoiceTranslator:
    def __init__(self):
        self.model = WhisperModel("small", device="cpu", compute_type="int8")
        self.source_lang = None
        self.audio_path = None

    def audio_to_english(self, audio_file) -> str:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_file.read())
            self.audio_path = tmp.name

        try:
            waveform, sr = torchaudio.load(self.audio_path)
            if sr != 16000:
                waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(waveform)
                torchaudio.save(self.audio_path, waveform, 16000)

            segments, info = self.model.transcribe(self.audio_path, task="translate")
            self.source_lang = info.language
            english_text = " ".join([seg.text for seg in segments])
            return english_text

        finally:
            try:
                os.remove(self.audio_path)
            except Exception as e:
                print(f"⚠️ Failed to delete temp file: {e}")

    def english_to_original_language(self, text: str) -> str:
        if not self.source_lang:
            raise ValueError("Source language not detected yet.")
        return GoogleTranslator(source="en", target=self.source_lang).translate(text)


    def get_detected_language(self):
        return self.source_lang
