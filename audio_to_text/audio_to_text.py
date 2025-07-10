import whisper
import os


def audio_transcription(audio_file_path):
    # Specify the full path to the ffmpeg executable
    ffmpeg_path = 'C:/Users/James Muthama/ICS Project 1/ICS Project 1/audio_to_text/ffmpeg.exe'

    # Set the environment variable for ffmpeg
    os.environ['PATH'] = f"{os.environ['PATH']};{os.path.dirname(ffmpeg_path)}"

    # Load the "base" model from the whisper library
    model = whisper.load_model("base")

    # Perform audio transcription using the loaded model
    # fp16=False specifies not to use half-precision (FP16) floating-point format
    result = model.transcribe(audio_file_path, fp16=False)

    # Retrieve the transcribed text from the result dictionary
    transcribed_text = result["text"]

    # Return the transcribed text
    return transcribed_text
