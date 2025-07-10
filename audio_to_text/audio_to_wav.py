from pydub import AudioSegment
import os


def mp3_to_wav(mp3_filepath):
    """
    Convert an MP3 file to WAV format and save it in the same directory.

    Args:
        mp3_filepath (str): Path to the input MP3 file

    Returns:
        str: Path to the output WAV file
    """

    # Set explicit paths for ffmpeg and ffprobe
    ffmpeg_path = r'C:\Users\james\PycharmProjects\CustomerInsightAI\Customer-Insight-AI\audio_to_text\ffmpeg.exe'
    ffprobe_path = r'C:\Users\james\PycharmProjects\CustomerInsightAI\Customer-Insight-AI\audio_to_text\ffprobe.exe'

    # Check if ffmpeg exists at the specified path
    if not os.path.exists(ffmpeg_path):
        print(f"Warning: ffmpeg not found at {ffmpeg_path}")
        print("Trying to use system ffmpeg...")
    else:
        # Add ffmpeg directory to PATH
        ffmpeg_dir = os.path.dirname(ffmpeg_path)
        if ffmpeg_dir not in os.environ['PATH']:
            os.environ['PATH'] = f"{ffmpeg_dir};{os.environ['PATH']}"

    # Configure pydub to use the specified ffmpeg and ffprobe
    AudioSegment.converter = ffmpeg_path
    AudioSegment.ffprobe = ffprobe_path

    audio = AudioSegment.from_file(mp3_filepath, format="mp3",
                                   parameters=["-probesize", "10000000", "-analyzeduration", "10000000"])
    wav_filepath = mp3_filepath.replace(".mp3", ".wav")
    audio.export(wav_filepath, format="wav")
    return wav_filepath

## Example usage
#input_mp3 = r"C:\Users\james\Downloads\CustomerService.mp3"
#output_wav = mp3_to_wav(input_mp3)
#print(f"WAV file saved at: {output_wav}")