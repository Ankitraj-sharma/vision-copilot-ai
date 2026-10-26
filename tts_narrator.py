"""
Client-Side Voice Narration & Audio Feedback Engine for Vision Copilot.
Uses the HTML5 Web Speech API to vocalize visual analysis findings,
dosage reminders, nutritional facts, and accessibility navigation cues.
"""

import streamlit.components.v1 as components


def speak_copilot_narration(text: str, rate: float = 1.0, pitch: float = 1.0) -> None:
    """
    Renders client-side JavaScript to speak the narration aloud via browser speechSynthesis.
    Zero cloud API latency, completely private, and works offline in modern browsers.
    """
    if not text:
        return

    # Escape quotes and backslashes
    clean_text = (
        text.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("'", "\\'")
        .replace("\n", " ")
        .replace("\r", "")
    )

    html_code = f"""
    <script>
    (function() {{
        try {{
            if ('speechSynthesis' in window) {{
                window.speechSynthesis.cancel(); // Stop any pending utterance
                var utterance = new SpeechSynthesisUtterance("{clean_text}");
                utterance.rate = {rate};
                utterance.pitch = {pitch};
                utterance.volume = 1.0;
                window.speechSynthesis.speak(utterance);
            }}
        }} catch(err) {{
            console.error('SpeechSynthesis error: ', err);
        }}
    }})();
    </script>
    """
    components.html(html_code, height=0, width=0)
