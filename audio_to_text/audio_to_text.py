import whisper
import os


def audio_transcription(audio_file_path):
    # Specify the full path to the ffmpeg executable
    ffmpeg_path = r'C:\Users\james\PycharmProjects\CustomerInsightAI\Customer-Insight-AI\audio_to_text\ffmpeg.exe'

    # Check if ffmpeg exists at the specified path
    if not os.path.exists(ffmpeg_path):
        print(f"Warning: ffmpeg not found at {ffmpeg_path}")
        print("Trying to use system ffmpeg...")
    else:
        # Add ffmpeg directory to PATH
        ffmpeg_dir = os.path.dirname(ffmpeg_path)
        if ffmpeg_dir not in os.environ['PATH']:
            os.environ['PATH'] = f"{ffmpeg_dir};{os.environ['PATH']}"

    # Load the "base" model from the whisper library
    model = whisper.load_model("base")

    # Perform audio transcription using the loaded model
    # fp16=False specifies not to use half-precision (FP16) floating-point format
    result = model.transcribe(audio_file_path, fp16=False)

    # Retrieve the transcribed text from the result dictionary
    transcribed_text = result["text"]

    # Return the transcribed text
    return transcribed_text


# Main execution
#if __name__ == "__main__":
#    # Path to the audio file to transcribe
#    audio_path = r"C:\Users\james\Downloads\Customer Service Sample Call - Product Refund.mp3"
#
#    # Check if the file exists
#    if os.path.exists(audio_path):
#        print("Starting transcription...")
#        transcription = audio_transcription(audio_path)
#        print("\nTranscription complete!")
#        print("-" * 50)
#        print(transcription)
#        print("-" * 50)
#    else:
#        print(f"Error: Audio file not found at {audio_path}")
#        print("Please check the file path and try again.")