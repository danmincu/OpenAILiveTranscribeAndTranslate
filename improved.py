# Import necessary libraries
import sounddevice as sd  # Library for recording audio from microphone
import wave  # Library for saving/reading WAV files
import os  # Library for file operations
import asyncio  # Library for asynchronous programming
import json  # Library for JSON processing
import time  # Library for time-related functions
import numpy as np  # Library for numerical operations
import queue  # Library for queue data structure
import threading  # Library for threading support
from openai import AsyncOpenAI  # OpenAI's asynchronous client for API access
from langdetect import detect, detect_langs  # Libraries for language detection

# Set OpenAI API key - This is a placeholder key and should be replaced with a valid one
# WARNING: API keys should generally not be hardcoded in scripts
OPENAI_API_KEY = 'sk-xxx'

###
### THIS VERSION IS IMPROVED AS IT LISTENS ALL THE TIME VERSUS THE INITIAL BACK AND FORTH BETWEEN LISTENING AND TRANSLATION
###


# Audio recording configuration
SAMPLE_RATE = 16000  # Audio sample rate in Hz (16kHz is good for speech)
CHUNK_DURATION = 7  # Duration of each audio chunk in seconds
ENABLE_PRINT = False  # Flag to control verbose console output

# Global queue for audio chunks
audio_queue = queue.Queue()
# Flag to control recording
is_recording = True


def conditional_print(message):
    """
    Print messages only if ENABLE_PRINT is True.
    Acts as a simple logging mechanism.
    """
    if ENABLE_PRINT:
        print(message)


def audio_callback(indata, frames, time, status):
    """
    Callback function for the audio stream.
    This is called for each audio block captured by the stream.

    Args:
        indata: Recorded audio data as numpy array
        frames: Number of frames
        time: Timing information
        status: Status of the recording
    """
    if status:
        conditional_print(f"Recording status: {status}")

    # Add the audio chunk to the queue
    audio_queue.put(indata.copy())


def continuous_record():
    """
    Function to continuously record audio in a separate thread.
    Uses a callback approach to ensure no gaps in recording.
    """
    conditional_print("Starting continuous recording...")

    # Calculate the block size based on sample rate and chunk duration
    block_size = int(SAMPLE_RATE * CHUNK_DURATION)

    # Start the input stream with the callback function
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                        callback=audio_callback,
                        blocksize=block_size,
                        dtype='int16'):
        while is_recording:
            # Just keep the stream running
            time.sleep(0.1)


def save_audio_to_disk(audio, sample_rate, filename):
    """
    Save recorded audio data to a WAV file.

    Args:
        audio: Numpy array containing audio data
        sample_rate: Audio sample rate in Hz
        filename: Path to save the WAV file
    """
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)  # Set to mono audio
        wf.setsampwidth(2)  # Two bytes per sample (16 bits)
        wf.setframerate(sample_rate)  # Set the sample rate
        wf.writeframes(audio.tobytes())  # Convert numpy array to bytes and write


def detect_language(text):
    """
    Detect the language of a given text.

    Args:
        text: Text string to analyze

    Returns:
        ISO language code (e.g., 'en' for English) or 'unknown' if detection fails
    """
    try:
        lang = detect(text)  # Attempt to detect the language
        return lang
    except Exception as e:
        conditional_print(f"Language detection error: {e}")
        return "unknown"  # Return 'unknown' if detection fails


async def transcribe_audio(client, audio_file_path):
    """
    Transcribe audio using OpenAI's Whisper API.

    Args:
        client: OpenAI client instance
        audio_file_path: Path to the audio file

    Returns:
        Transcription response from the API
    """
    conditional_print("Transcribing...")
    try:
        with open(audio_file_path, 'rb') as audio_file:
            # Send the audio file to OpenAI's Whisper model for transcription
            response = await client.audio.transcriptions.create(
                file=(audio_file_path, audio_file.read(), 'audio/wav'),
                model="whisper-1"
            )
        return response
    except Exception as e:
        conditional_print(f"Transcription error: {e}")
        return None


async def translate_text(client, text, target_language="en"):
    """
    Translate text to English using OpenAI's GPT model.

    Args:
        client: OpenAI client instance
        text: Text to translate
        target_language: Target language code (default is English)

    Returns:
        Translation response from the API
    """
    conditional_print("Translating text...")
    try:
        # Use ChatGPT to translate the text
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Translate the following text to English:\n\n{text}"}
            ],
            temperature=0.5,  # Lower temperature for more deterministic outputs
            max_tokens=300  # Limit response length
        )
        return response
    except Exception as e:
        conditional_print(f"Translation error: {e}")
        return None


async def process_audio_chunk(client, chunk_id):
    """
    Process a single audio chunk: save, transcribe, detect language, and translate if needed.

    Args:
        client: OpenAI client instance
        chunk_id: Identifier for this audio chunk

    Returns:
        None
    """
    # Skip processing if queue is empty
    if audio_queue.empty():
        return

    # Get the next audio chunk from the queue
    audio_data = audio_queue.get()

    # Create a unique filename for this chunk
    audio_filename = f"temp_audio_{chunk_id}.wav"

    try:
        # Save the audio chunk to disk
        save_audio_to_disk(audio_data, SAMPLE_RATE, audio_filename)

        # Transcribe the audio file
        response = await transcribe_audio(client, audio_filename)

        if response is None:
            conditional_print(f"No transcription for chunk {chunk_id}")
            return

        # Process the transcription response
        if hasattr(response, 'text'):
            # Handle the response as an object with attributes
            transcription = response.text
            if not transcription.strip():  # Skip empty transcriptions
                return

            # Detect the language of the transcribed text
            language = detect_language(transcription)
            conditional_print(f"Chunk {chunk_id} - Detected Language: {language}")
            detected_language = language
        else:
            # Handle the response as a dictionary (fallback)
            try:
                response_dict = json.loads(response)
                transcription = response_dict.get("text", "")
                detected_language = response_dict.get("language", "")
                if not transcription.strip():  # Skip empty transcriptions
                    return
                conditional_print(f"Chunk {chunk_id} - Transcription: {transcription}")
                conditional_print(f"Chunk {chunk_id} - Detected Language: {detected_language}")
            except Exception as e:
                conditional_print(f"Error processing response: {e}")
                return

        # Print the result based on language
        if detected_language and detected_language != "en":
            # If text is not in English, translate it
            translation_response = await translate_text(client, transcription)
            if translation_response:
                try:
                    translation = translation_response.choices[0].message.content
                    # Print result with language tag
                    print(f"[{detected_language}]: {translation}")
                except Exception as e:
                    conditional_print(f"Translation parsing error: {e}")
        else:
            # If text is already in English, just print the transcription
            print(transcription)

    except Exception as e:
        conditional_print(f"Error processing chunk {chunk_id}: {e}")
    finally:
        # Clean up the temporary audio file
        if os.path.exists(audio_filename):
            os.remove(audio_filename)

        # Mark this task as done in the queue
        audio_queue.task_done()


async def main():
    """
    Main function that manages continuous recording and asynchronous processing.
    """
    # Initialize the OpenAI client with the API key
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)

    # Start continuous recording in a separate thread
    recording_thread = threading.Thread(target=continuous_record)
    recording_thread.daemon = True  # Thread will exit when main program exits
    recording_thread.start()

    print("Listening continuously. Press Ctrl+C to stop.")

    try:
        chunk_id = 0
        # Main processing loop
        while True:
            # If we have audio to process, process it
            if not audio_queue.empty():
                # Process one audio chunk at a time
                await process_audio_chunk(client, chunk_id)
                chunk_id += 1
            else:
                # Small sleep to prevent CPU hogging when queue is empty
                await asyncio.sleep(0.1)

    except KeyboardInterrupt:
        # Handle user termination (Ctrl+C)
        print("\nTerminated by user.")
    finally:
        # Stop the recording thread
        global is_recording
        is_recording = False

        # Clean up any temporary files that might be left
        for filename in os.listdir('.'):
            if filename.startswith('temp_audio_') and filename.endswith('.wav'):
                os.remove(filename)


# Entry point: Run the main function when script is executed directly
if __name__ == "__main__":
    asyncio.run(main())  # Run the async main function using asyncio