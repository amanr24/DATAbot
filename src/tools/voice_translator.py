'''
voice_translator.py

Utility for converting spoken audio to English text and back-translating responses to the original language.
Uses WhisperModel for transcription (with translation) and GoogleTranslator for text translation.
'''

import tempfile
import os
import torch
import torchaudio  # Audio I/O and transformations
from faster_whisper import WhisperModel  # Whisper for STT and translation
from deep_translator import GoogleTranslator  # Text translation
from gtts import gTTS  # Optional: for text-to-speech output if needed


class VoiceTranslator:
    '''
    Handles speech-to-text (STT) and back-translation.

    Attributes:
        model (WhisperModel): Whisper model instance for transcription/translation.
        source_lang (str): Language code detected in the last transcription.
        audio_path (str): Path to the temporary WAV file of uploaded audio.
    '''
    def __init__(self):
        # Choose GPU if available, else CPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Load Whisper model in int8 mode for efficiency
        self.model = WhisperModel("small", device=device, compute_type="int8")
        self.source_lang = None
        self.audio_path = None

    def audio_to_english(self, audio_file) -> str:
        '''
        Transcribes and translates an uploaded audio file to English.

        Steps:
        1. Save uploaded bytes to a temporary .wav file.
        2. Load waveform and ensure 16 kHz sampling.
        3. Use WhisperModel to transcribe (translate task).
        4. Delete temp file and return combined text.

        Args:
            audio_file: File-like object with .read() method (e.g., Streamlit UploadedFile).

        Returns:
            str: Transcribed English text.
        '''
        # Write bytes to a named temp WAV file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_file.read())
            self.audio_path = tmp.name

        try:
            # Load the WAV file
            waveform, sr = torchaudio.load(self.audio_path)
            if sr != 16000:
                waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(waveform)
                torchaudio.save(self.audio_path, waveform, 16000)

            # Transcribe with translation to English
            segments, info = self.model.transcribe(self.audio_path, task="translate")
            # Save detected source language for back-translation
            self.source_lang = info.language
            # Concatenate segment texts
            english_text = " ".join(seg.text for seg in segments)
            return english_text

        finally:
            # Clean up temporary file
            try:
                os.remove(self.audio_path)
            except Exception as e:
                print(f"⚠️ Failed to delete temp file: {e}")

    def english_to_original_language(self, text: str) -> str:
        '''
        Translates English text back into the originally detected source language.

        Args:
            text (str): English text to translate.

        Returns:
            str: Text translated to the original language.

        Raises:
            ValueError: If no source language was detected yet.
        '''
        if not self.source_lang:
            raise ValueError("Source language not detected yet.")
        return GoogleTranslator(source="en", target=self.source_lang).translate(text)

    def get_detected_language(self) -> str:
        '''
        Returns the language code detected during the last transcription.

        Returns:
            str: ISO language code (e.g., 'en', 'hi').
        '''
        return self.source_lang
