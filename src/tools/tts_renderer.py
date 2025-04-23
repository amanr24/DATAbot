from streamlit.components.v1 import html
import html as html_escape
import streamlit as st
import re

class TextToSpeechRenderer:
    def __init__(self, lang="en-US"):
        self.lang = lang

    def render(self, message: str, id_suffix: str):
        
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
                        const selected = voices.find(v => v.lang.toLowerCase().includes('{self.lang}'.toLowerCase()));

                        if (selected) {{
                            speakMsg_{id_suffix}.voice = selected;
                            console.log("✅ Using voice: " + selected.name + " | " + selected.lang);
                        }} else {{
                            console.warn("⚠️ No matching voice found for {self.lang}. Using default.");
                        }}

                        speechSynthesis.speak(speakMsg_{id_suffix});
                        isSpeaking_{id_suffix} = true;
                        document.getElementById('btn_{id_suffix}').innerText = '⏹️';
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
