import sounddevice as sd
import wave
import os
import asyncio
import json
import time
from openai import AsyncOpenAI
from langdetect import detect, detect_langs

# Set OpenAI API key

OPENAI_API_KEY = 'sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxx'

# Audio settings

SAMPLE_RATE = 16000

DURATION = 7  # Duration of each audio chunk in seconds

ENABLE_PRINT = False  # Flag to control printing to the console


def conditional_print(message):
    if ENABLE_PRINT:
        print(message)

def record_audio(duration, sample_rate):
    conditional_print("Recording...")
    print(">")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()  # Wait until the recording is finished
    conditional_print("Done recording.")
    return audio


def save_audio_to_disk(audio, sample_rate, filename):
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)

        wf.setsampwidth(2)  # Two bytes per sample

        wf.setframerate(sample_rate)

        wf.writeframes(audio.tobytes())


def detect_language(text):
    try:

        lang = detect(text)

        return lang

    except Exception as e:

        return "unknown"


async def transcribe_audio(client, audio_file_path):
    conditional_print("Transcribing...")
    with open(audio_file_path, 'rb') as audio_file:
        response = await client.audio.transcriptions.create(
            file=(audio_file_path, audio_file.read(), 'audio/wav'),
            model="whisper-1"
        )

    return response


async def translate_text(client, text, target_language="en"):
    conditional_print("Translating text...")

    response = await client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Translate the following text to English:\n\n{text}"}
        ],
        temperature=0.5,
        max_tokens=300
    )

    return response


async def main():
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)

    audio_filename = "temp_audio.wav"

    try:
        while True:
            # Capture audio from the microphone
            audio_data = record_audio(DURATION, SAMPLE_RATE)
            # Save audio to a file on disk
            save_audio_to_disk(audio_data, SAMPLE_RATE, audio_filename)
            # Transcribe the audio file
            response = await transcribe_audio(client, audio_filename)
            if hasattr(response, 'text'):
                transcription = response.text
                # Detect the language of the transcribed text
                language = detect_language(transcription)
                conditional_print("Detected Language: " + language)
                detected_language = language  # response.language
            else:
                response_dict = json.loads(response)
                transcription = response_dict.get("text", "")
                detected_language = response_dict.get("language", "")
                conditional_print("Transcription: " + transcription)
                conditional_print("Detected Language: " + detected_language)

            if detected_language and detected_language != "en":
                translation_response = await translate_text(client, transcription)
                try:
                    translation = translation_response.choices[0].message.content
                except Exception:
                    translation = ""
                conditional_print("English Translation: " + translation)
                print("[" + detected_language + "]:" + translation)
            else:
                print(transcription)

            # time.sleep(1)  # Wait 1 second before recording the next chunk


    except KeyboardInterrupt:
        conditional_print("Terminated by user.")
    finally:
        if os.path.exists(audio_filename):
            os.remove(audio_filename)


if __name__ == "__main__":
    asyncio.run(main())
