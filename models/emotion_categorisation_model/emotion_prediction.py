import torch
import torchaudio
import torchaudio.transforms as T
import os
import tempfile
import logging
from transformers import Wav2Vec2Model, Wav2Vec2Processor
import torch.nn as nn
from pydub import AudioSegment
from collections import Counter
import queue

# =============================================================================
# CONFIGURATION AND SETUP
# =============================================================================

# Set explicit paths for ffmpeg and ffprobe executables
# These are required by pydub for audio processing operations
FFMPEG_PATH = r"C:\Users\james\PycharmProjects\CustomerInsightAI\Customer-Insight-AI\models\emotion_categorisation_model\ffmpeg.exe"
FFPROBE_PATH = r"/Customer-Insight-AI/audio_to_text/ffprobe.exe"  # Corrected path

# Configure pydub to use the specified ffmpeg and ffprobe executables
# This ensures consistent audio processing across different environments
AudioSegment.converter = FFMPEG_PATH
AudioSegment.ffprobe = FFPROBE_PATH

# Set up logging configuration for debugging and monitoring
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# NEURAL NETWORK MODEL DEFINITION
# =============================================================================

class EmotionClassifier(nn.Module):
    """
    A neural network model for emotion classification from audio.

    The model uses a pre-trained Wav2Vec2 model as a feature extractor,
    followed by adaptive pooling and a linear classifier to predict emotions.

    Architecture:
    1. Wav2Vec2 feature extraction (facebook/wav2vec2-base)
    2. Adaptive average pooling to create fixed-size representations
    3. Linear layer for classification
    4. Softmax activation for probability distribution
    """

    def __init__(self, encoder_dim=768, out_n_neurons=4):
        """
        Initialize the EmotionClassifier model.

        Args:
            encoder_dim (int): Dimension of the Wav2Vec2 encoder output (default: 768)
            out_n_neurons (int): Number of emotion classes to predict (default: 4)
        """
        super().__init__()

        # Load pre-trained Wav2Vec2 model for audio feature extraction
        # This model converts raw audio waveforms into meaningful representations
        self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")

        # Adaptive average pooling to convert variable-length sequences to fixed size
        # This ensures consistent input size for the classification layer
        self.avg_pool = nn.AdaptiveAvgPool1d(1)

        # Linear layer for emotion classification
        # Maps from encoder dimension to number of emotion classes
        self.output_mlp = nn.Linear(encoder_dim, out_n_neurons, bias=False)

        # Softmax activation to convert logits to probability distribution
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        Forward pass through the emotion classifier.

        Args:
            x (torch.Tensor): Input audio tensor

        Returns:
            tuple: (probabilities, logits) - probability distribution and raw logits
        """
        # Extract features using Wav2Vec2 encoder
        # Shape: (batch_size, sequence_length, encoder_dim)
        features = self.wav2vec2(x).last_hidden_state

        # Apply adaptive pooling to get fixed-size representation
        # Transpose to (batch_size, encoder_dim, sequence_length) for pooling
        # Then squeeze to remove the sequence dimension
        pooled = self.avg_pool(features.transpose(1, 2)).squeeze(-1)

        # Generate logits through linear classification layer
        logits = self.output_mlp(pooled)

        # Convert logits to probability distribution
        probs = self.softmax(logits)

        return probs, logits


# =============================================================================
# MODEL INITIALIZATION AND LOADING
# =============================================================================

def initialize_emotion_classifier():
    """
    Initialize the EmotionClassifier model and load pre-trained weights.

    This function:
    1. Locates the pre-trained model files
    2. Initializes the EmotionClassifier architecture
    3. Loads saved model weights (both custom MLP and Wav2Vec2)
    4. Loads emotion labels from the label encoder
    5. Initializes the Wav2Vec2 processor for input preprocessing

    Returns:
        tuple: (model, processor, emotion_labels)
            - model: Initialized EmotionClassifier with loaded weights
            - processor: Wav2Vec2Processor for audio preprocessing
            - emotion_labels: List of emotion class names

    Raises:
        FileNotFoundError: If model files are not found
        ImportError: If required libraries are missing
        Exception: For other initialization errors
    """
    try:
        # Define the base path where the pre-trained model is stored
        base_model_path = r"C:\Users\james\PycharmProjects\CustomerInsightAI\Customer-Insight-AI\models\emotion_categorisation_model\models--speechbrain--emotion-recognition-wav2vec2-IEMOCAP"
        snapshots_dir = os.path.join(base_model_path, "snapshots")

        # Find available model snapshots (different versions/checkpoints)
        snapshot_dirs = [d for d in os.listdir(snapshots_dir) if os.path.isdir(os.path.join(snapshots_dir, d))]
        if not snapshot_dirs:
            raise FileNotFoundError("No snapshot directories found")

        # Use the first available snapshot directory
        local_model_path = os.path.join(snapshots_dir, snapshot_dirs[0])
        logger.info(f"Using model path: {local_model_path}")

        # Initialize the emotion classifier model with default architecture
        logger.info("Initializing emotion classifier model...")
        model = EmotionClassifier()
        logger.info("Model initialized successfully!")

        # Define paths to the saved model checkpoints
        model_ckpt_path = os.path.join(local_model_path, "model.ckpt")  # Custom MLP weights
        wav2vec2_ckpt_path = os.path.join(local_model_path, "wav2vec2.ckpt")  # Wav2Vec2 fine-tuned weights

        logger.info(f"Looking for model checkpoint at: {model_ckpt_path}")
        logger.info(f"Looking for wav2vec2 checkpoint at: {wav2vec2_ckpt_path}")

        # Load the custom MLP classification layer weights
        if os.path.exists(model_ckpt_path):
            logger.info("Loading model checkpoint...")
            model_state = torch.load(model_ckpt_path, map_location='cpu')

            # Handle different checkpoint formats
            if 'model' in model_state:
                # If weights are nested under 'model' key
                model.output_mlp.load_state_dict(model_state['model']['0'])
                logger.info("MLP weights loaded from 'model.0'")
            else:
                # If weights are stored directly
                model.output_mlp.load_state_dict(model_state)
                logger.info("MLP weights loaded directly")
        else:
            logger.warning(f"Model checkpoint not found: {model_ckpt_path}")

        # Load the fine-tuned Wav2Vec2 feature extractor weights
        if os.path.exists(wav2vec2_ckpt_path):
            logger.info("Loading wav2vec2 checkpoint...")
            try:
                wav2vec2_state = torch.load(wav2vec2_ckpt_path, map_location='cpu')
                model.wav2vec2.load_state_dict(wav2vec2_state)
                logger.info("Wav2Vec2 weights loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load wav2vec2 checkpoint: {e}")
        else:
            logger.warning(f"Wav2Vec2 checkpoint not found: {wav2vec2_ckpt_path}")

        # Load emotion labels from the label encoder file
        label_encoder_path = os.path.join(local_model_path, "label_encoder.txt")
        # Default emotion labels (commonly used in emotion recognition)
        emotion_labels = ['angry', 'happy', 'neutral', 'sad']

        if os.path.exists(label_encoder_path):
            logger.info("Loading label encoder...")
            try:
                # Read emotion labels from file (one label per line)
                with open(label_encoder_path, 'r') as f:
                    lines = f.readlines()
                    emotion_labels = [line.strip() for line in lines if line.strip()]
            except Exception as e:
                logger.warning(f"Failed to load label encoder: {e}. Using default labels.")
        else:
            logger.warning(f"Label encoder not found: {label_encoder_path}. Using default labels.")

        logger.info(f"Emotion labels: {emotion_labels}")

        # Set model to evaluation mode (disable dropout, batch norm training mode)
        model.eval()

        # Initialize the Wav2Vec2 processor for audio preprocessing
        logger.info("Loading wav2vec2 processor...")
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
        logger.info("Processor loaded successfully!")

        return model, processor, emotion_labels

    except ImportError as e:
        logger.error(f"Import error: {e}")
        raise
    except FileNotFoundError as e:
        logger.error(f"File not found error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        raise


# =============================================================================
# AUDIO PROCESSING UTILITIES
# =============================================================================

def split_wav_to_chunks(wav_path, target_size_kb=100):
    """
    Split a WAV audio file into smaller chunks for processing.

    Large audio files are split into manageable chunks to:
    1. Reduce memory usage during processing
    2. Enable parallel processing of audio segments
    3. Provide more granular emotion analysis

    Args:
        wav_path (str): Path to the input WAV file
        target_size_kb (int): Target size for each chunk in kilobytes (default: 100KB)

    Returns:
        queue.Queue: Queue containing file paths to temporary WAV chunk files

    Raises:
        FileNotFoundError: If the input WAV file doesn't exist
        Exception: For audio processing errors
    """
    try:
        logger.info(f"Splitting WAV file: {wav_path}")

        # Verify that the input file exists
        if not os.path.exists(wav_path):
            raise FileNotFoundError(f"WAV file not found: {wav_path}")

        # Load the audio file using pydub
        audio = AudioSegment.from_wav(wav_path)

        # Extract audio properties for chunk size calculation
        sample_rate = audio.frame_rate  # Samples per second
        channels = audio.channels  # Number of audio channels (mono=1, stereo=2)
        sample_width = audio.sample_width  # Bytes per sample

        # Calculate chunk duration to achieve target file size
        # Formula: Size (bytes) = sample_rate * channels * sample_width * duration (seconds)
        # Rearranging: duration = Size / (sample_rate * channels * sample_width)
        target_size_bytes = target_size_kb * 1024
        duration_ms = (target_size_bytes * 1000) // (sample_rate * channels * sample_width)
        logger.info(f"Calculated chunk duration: {duration_ms} ms")

        # Initialize queue to store chunk file paths
        chunk_queue = queue.Queue()
        total_duration_ms = len(audio)  # Total audio duration in milliseconds
        chunk_index = 0

        # Split audio into chunks of calculated duration
        for start_ms in range(0, total_duration_ms, duration_ms):
            # Calculate end time, ensuring we don't exceed total duration
            end_ms = min(start_ms + duration_ms, total_duration_ms)

            # Extract audio chunk
            chunk = audio[start_ms:end_ms]

            # Create temporary file for the chunk
            temp_chunk = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            chunk_path = temp_chunk.name

            # Export chunk to temporary WAV file
            chunk.export(chunk_path, format='wav')

            # Add chunk path to processing queue
            chunk_queue.put(chunk_path)
            chunk_index += 1

        logger.info(f"Total chunks created: {chunk_index}")
        return chunk_queue

    except FileNotFoundError as e:
        logger.error(f"File not found error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in split_wav_to_chunks: {e}")
        import traceback
        traceback.print_exc()
        raise


# =============================================================================
# EMOTION PREDICTION FUNCTIONS
# =============================================================================

def predict_emotion(wav_path, model, processor, emotion_labels):
    """
    Predict emotion from a single WAV audio file.

    This function processes a single audio file through the complete pipeline:
    1. Load and preprocess the audio (resampling, mono conversion)
    2. Extract features using Wav2Vec2 processor
    3. Run inference through the emotion classifier
    4. Return prediction results with confidence scores

    Args:
        wav_path (str): Path to the WAV audio file
        model (EmotionClassifier): Pre-initialized emotion classifier model
        processor (Wav2Vec2Processor): Pre-initialized Wav2Vec2 processor
        emotion_labels (list): List of emotion class names

    Returns:
        dict: Dictionary containing:
            - predicted_emotion (str): Most likely emotion class
            - confidence (float): Confidence score for the prediction
            - class_index (int): Index of the predicted class
            - probabilities (dict): Probability scores for all emotion classes

    Raises:
        FileNotFoundError: If the audio file doesn't exist
        Exception: For audio processing or model inference errors
    """
    try:
        # Verify that the audio file exists
        if not os.path.exists(wav_path):
            raise FileNotFoundError(f"Audio file not found: {wav_path}")

        # Load audio file using torchaudio
        # signal: audio waveform tensor, fs: sample rate
        signal, fs = torchaudio.load(wav_path)

        # Resample audio to 16kHz if necessary (Wav2Vec2 requirement)
        if fs != 16000:
            resampler = T.Resample(fs, 16000)
            signal = resampler(signal)

        # Convert stereo to mono by averaging channels
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)

        # Convert tensor to numpy array for processor
        audio_array = signal.squeeze().numpy()

        # Preprocess audio using Wav2Vec2 processor
        # This handles normalization, padding, and tensor conversion
        inputs = processor(audio_array, sampling_rate=16000, return_tensors="pt", padding=True)

        # Run inference through the emotion classifier
        with torch.no_grad():  # Disable gradient computation for inference
            probs, logits = model(inputs.input_values)

        # Extract prediction results
        predicted_idx = torch.argmax(probs, dim=-1).item()  # Index of most likely class
        confidence = torch.max(probs, dim=-1).values.item()  # Confidence score
        predicted_emotion = emotion_labels[predicted_idx]  # Emotion label

        # Create probability dictionary for all emotions
        probabilities = {emotion: prob.item() for emotion, prob in zip(emotion_labels, probs[0])}

        return {
            'predicted_emotion': predicted_emotion,
            'confidence': confidence,
            'class_index': predicted_idx,
            'probabilities': probabilities
        }

    except FileNotFoundError as e:
        logger.error(f"File not found error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        raise


def predict_emotion_from_wav(wav_path, model, processor, emotion_labels):
    """
    Analyze emotion from a WAV file by processing it in chunks.

    This function provides comprehensive emotion analysis for longer audio files:
    1. Splits the audio into manageable chunks
    2. Processes each chunk individually for emotion prediction
    3. Aggregates results to determine overall emotion distribution
    4. Identifies the most frequent emotion across all chunks
    5. Cleans up temporary files

    Args:
        wav_path (str): Path to the WAV audio file
        model (EmotionClassifier): Pre-initialized emotion classifier model
        processor (Wav2Vec2Processor): Pre-initialized Wav2Vec2 processor
        emotion_labels (list): List of emotion class names

    Returns:
        dict: Dictionary containing:
            - emotions_found (list): List of emotions detected in each chunk
            - most_frequent_emotion (str): Most common emotion across all chunks
            - chunk_results (list): Detailed results for each processed chunk
            - emotion_counts (dict): Count of each emotion type found

    Raises:
        FileNotFoundError: If the WAV file doesn't exist
        Exception: For audio processing or prediction errors
    """
    try:
        logger.info(f"Loading WAV file: {wav_path}")

        # Verify that the input file exists
        if not os.path.exists(wav_path):
            raise FileNotFoundError(f"WAV file not found: {wav_path}")

        # Split the WAV file into 100KB chunks for processing
        chunk_queue = split_wav_to_chunks(wav_path, target_size_kb=100)

        # Initialize lists to store results
        emotions_found = []  # List of emotions detected in each chunk
        chunk_results = []  # Detailed results for each chunk

        # Process each audio chunk
        while not chunk_queue.empty():
            chunk_path = chunk_queue.get()
            logger.info(f"Processing chunk: {chunk_path}")

            try:
                # Predict emotion for this chunk
                result = predict_emotion(chunk_path, model, processor, emotion_labels)

                # Store results
                emotions_found.append(result['predicted_emotion'])
                chunk_results.append(result)

                logger.info(f"Emotion for chunk {chunk_path}: {result['predicted_emotion']}")

            except Exception as e:
                logger.error(f"Failed to process chunk {chunk_path}: {e}")

            finally:
                # Clean up temporary chunk file
                if os.path.exists(chunk_path):
                    os.unlink(chunk_path)
                    logger.info(f"Removed chunk file: {chunk_path}")

        # Clean up the main WAV file
        if os.path.exists(wav_path):
            os.unlink(wav_path)
            logger.info(f"Removed main WAV file: {wav_path}")

        # Analyze emotion distribution across all chunks
        if emotions_found:
            # Count occurrences of each emotion
            emotion_counts = Counter(emotions_found)

            # Find the most frequent emotion
            most_common = emotion_counts.most_common(1)
            emotion = most_common[0][0]  # Most frequent emotion
        else:
            emotion = None
            logger.warning("No emotions found in any chunks")
            print("\nNo emotions detected in the audio file.")

        return {
            'emotions_found': emotions_found,
            'most_frequent_emotion': emotion,
            'chunk_results': chunk_results,
            'emotion_counts': dict(emotion_counts)
        }

    except FileNotFoundError as e:
        logger.error(f"File not found error: {e}")
        # Clean up WAV file if it exists
        if 'wav_path' in locals() and os.path.exists(wav_path):
            os.unlink(wav_path)
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()

        # Clean up files in case of error
        if 'wav_path' in locals() and os.path.exists(wav_path):
            os.unlink(wav_path)

        # Clean up any remaining chunk files
        while 'chunk_queue' in locals() and not chunk_queue.empty():
            chunk_path = chunk_queue.get()
            if os.path.exists(chunk_path):
                os.unlink(chunk_path)
        raise


# =============================================================================
# MAIN EMOTION PREDICTION INTERFACE
# =============================================================================

def get_emotion_prediction(wav_path):
    """
    Main interface function for emotion prediction from WAV files.

    This is the primary function that external code should call to analyze emotions
    in audio files. It handles the complete pipeline:
    1. Initialize the emotion classification model
    2. Process the audio file in chunks
    3. Aggregate results and calculate statistics
    4. Return comprehensive emotion analysis

    Args:
        wav_path (str): Path to the WAV audio file to analyze

    Returns:
        dict: Comprehensive emotion analysis results containing:
            - emotion_counts (dict): Raw count of each emotion detected
            - most_frequent_emotion (str): Most common emotion in the audio
            - total_emotions (int): Total number of chunks processed
            - emotion_percentage (dict): Percentage distribution of emotions

    Raises:
        FileNotFoundError: If the WAV file doesn't exist
        Exception: For model initialization or processing errors
    """
    try:
        # Initialize the emotion classification model and required components
        model, processor, emotion_labels = initialize_emotion_classifier()

        # Verify that the input file exists
        if not os.path.exists(wav_path):
            logger.error(f"WAV file does not exist: {wav_path}")
            raise FileNotFoundError(f"WAV file does not exist: {wav_path}")

        # Process the WAV file and get emotion predictions
        result_mp3 = predict_emotion_from_wav(wav_path, model, processor, emotion_labels)
        logger.info(f"Emotion prediction results: {result_mp3['emotion_counts']}")

        # Calculate statistics from the results
        total_emotions = sum(result_mp3['emotion_counts'].values())

        # Calculate the percentage distribution of emotions
        emotion_percentage = {
            emotion: (count / total_emotions * 100)
            for emotion, count in result_mp3['emotion_counts'].items()
        }

        # Return comprehensive results
        return {
            'emotion_counts': result_mp3['emotion_counts'],
            'most_frequent_emotion': result_mp3['most_frequent_emotion'],
            'total_emotions': total_emotions,
            'emotion_percentage': emotion_percentage
        }

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise  # Re-raise to let the caller handle it
    except Exception as e:
        logger.error(f"Main execution failed: {e}")
        import traceback
        traceback.print_exc()
        raise  # Re-raise to let the caller handle it

# =============================================================================
# EXAMPLE USAGE (COMMENTED OUT)
# =============================================================================

# Example of how to use the emotion prediction system:
# if __name__ == '__main__':
#     # Path to your audio file
#     wav_path = r"C:\Users\james\Downloads\CustomerService.wav"
#
#     # Get emotion prediction results
#     emotion_results = get_emotion_prediction(wav_path)
#
#     # Print results
#     print(f"Most frequent emotion: {emotion_results['most_frequent_emotion']}")
#     print(f"Emotion distribution: {emotion_results['emotion_percentage']}")
#     print(f"Total chunks processed: {emotion_results['total_emotions']}")