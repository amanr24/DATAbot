from streamlit.components.v1 import html
import html as html_escape
import streamlit as st
import re

class TextToSpeechRenderer:
    def __init__(self, lang="en-US"):
        self.lang = lang

    def render(self, message: str, id_suffix: str):
        '''
        Display a chat message and inject HTML/JS for TTS playback.

        - Escapes and sanitizes text for safe embedding in JS.
        - Creates a button that toggles speech on/off.
        - Uses SpeechSynthesisUtterance with the configured language.

        Args:
            message (str): The text content to display and speak.
            id_suffix (str): Unique suffix to differentiate multiple buttons.
        '''
        # Escape HTML to prevent injection, remove markdown symbols and newlines
        clean_text = html_escape.escape(message.replace("*", "").replace("-", "").replace("\n", " "))

        st.markdown(message)
        html_code = f"""
        <div style='margin-bottom: 1em;'>
            <button id='btn_{id_suffix}' onclick="toggleSpeak_{id_suffix}()" style="background: transparent; border: none; font-size: 16px; cursor: pointer; color: inherit;">🔊 </button>
            <script>
                var isSpeaking_{id_suffix} = false;
                var speakMsg_{id_suffix} = new SpeechSynthesisUtterance(`{clean_text}`);
                speakMsg_{id_suffix}.lang = '{self.lang}';

                function toggleSpeak_{id_suffix}() {{
                    if (isSpeaking_{id_suffix}) {{
                        window.speechSynthesis.cancel();
                        isSpeaking_{id_suffix} = false;
                        document.getElementById('btn_{id_suffix}').innerText = '🔊';
                        return;
                    }}

                    function selectVoiceAndSpeak() {{
                        const voices = speechSynthesis.getVoices();
                        console.log("🗣️ Available voices:");
                        voices.forEach(v => console.log(v.lang + " - " + v.name));
                        let selected = voices.find(v => v.lang.toLowerCase().includes('{self.lang}'.toLowerCase()));

                        if (!selected) {{
                            console.warn("⚠️ No matching voice found for '{self.lang}', falling back to 'en-US'");
                            selected = voices.find(v => v.lang === 'en-US') || voices[0];
                        }}

                        if (selected) {{
                            speakMsg_{id_suffix}.voice = selected;
                            console.log("✅ Using voice: " + selected.name + " | " + selected.lang);
                            speechSynthesis.speak(speakMsg_{id_suffix});
                            isSpeaking_{id_suffix} = true;
                            document.getElementById('btn_{id_suffix}').innerText = '⏹️';
                        }} else {{
                            alert("❌ No voice found. Try changing your system TTS settings.");
                        }}
                    }}

                    speakMsg_{id_suffix}.onend = function() {{
                        isSpeaking_{id_suffix} = false;
                        document.getElementById('btn_{id_suffix}').innerText = '🔊';
                    }};

                    if (speechSynthesis.getVoices().length === 0) {{
                        speechSynthesis.addEventListener('voiceschanged', selectVoiceAndSpeak);
                    }} else {{
                        selectVoiceAndSpeak();
                    }}
                }}
            </script>
        </div>
        """
        html(html_code, height=120)
