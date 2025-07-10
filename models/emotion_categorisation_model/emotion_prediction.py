import torch
import torchaudio
import torchaudio.transforms as T
import tempfile
import os
import pickle
from transformers import Wav2Vec2Model, Wav2Vec2Processor
import torch.nn as nn
import numpy as np


class EmotionClassifier(nn.Module):
    def __init__(self, encoder_dim=768, out_n_neurons=4):
        super().__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.output_mlp = nn.Linear(encoder_dim, out_n_neurons, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # Get wav2vec2 features
        features = self.wav2vec2(x).last_hidden_state
        # Apply average pooling
        pooled = self.avg_pool(features.transpose(1, 2)).squeeze(-1)
        # Apply MLP
        logits = self.output_mlp(pooled)
        # Apply softmax
        probs = self.softmax(logits)
        return probs, logits


try:
    # Model path
    base_model_path = r"C:\Users\james\.cache\huggingface\hub\models--speechbrain--emotion-recognition-wav2vec2-IEMOCAP"
    snapshots_dir = os.path.join(base_model_path, "snapshots")

    # Get the snapshot directory
    snapshot_dirs = [d for d in os.listdir(snapshots_dir) if os.path.isdir(os.path.join(snapshots_dir, d))]
    if not snapshot_dirs:
        raise FileNotFoundError("No snapshot directories found")

    local_model_path = os.path.join(snapshots_dir, snapshot_dirs[0])
    print(f"Using model path: {local_model_path}")

    # Initialize the model
    print("Initializing emotion classifier model...")
    model = EmotionClassifier()
    print("Model initialized successfully!")

    # Load the pretrained weights
    model_ckpt_path = os.path.join(local_model_path, "model.ckpt")
    wav2vec2_ckpt_path = os.path.join(local_model_path, "wav2vec2.ckpt")

    print(f"Looking for model checkpoint at: {model_ckpt_path}")
    print(f"Looking for wav2vec2 checkpoint at: {wav2vec2_ckpt_path}")

    if os.path.exists(model_ckpt_path):
        print("Loading model checkpoint...")
        model_state = torch.load(model_ckpt_path, map_location='cpu')
        # Load only the MLP weights
        if 'model' in model_state:
            model.output_mlp.load_state_dict(model_state['model']['0'])
        else:
            model.output_mlp.load_state_dict(model_state)

    if os.path.exists(wav2vec2_ckpt_path):
        print("Loading wav2vec2 checkpoint...")
        wav2vec2_state = torch.load(wav2vec2_ckpt_path, map_location='cpu')
        model.wav2vec2.load_state_dict(wav2vec2_state)

    # Load label encoder
    label_encoder_path = os.path.join(local_model_path, "label_encoder.txt")
    emotion_labels = ['angry', 'happy', 'neutral', 'sad']  # Default IEMOCAP labels

    if os.path.exists(label_encoder_path):
        print("Loading label encoder...")
        try:
            with open(label_encoder_path, 'r') as f:
                lines = f.readlines()
                emotion_labels = [line.strip() for line in lines if line.strip()]
        except:
            print("Using default emotion labels")

    print(f"Emotion labels: {emotion_labels}")

    # Set model to evaluation mode
    model.eval()

    # Initialize wav2vec2 processor
    print("Loading wav2vec2 processor...")
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    print("Processor loaded successfully!")

    # Load the audio file
    audio_path = r"C:\Users\james\PycharmProjects\CustomerInsightAI\Customer-Insight-AI\Customer Service Sample Call - Product Refund.wav"

    print(f"Loading audio file: {audio_path}")
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    signal, fs = torchaudio.load(audio_path)
    print(f"Original sample rate: {fs} Hz")
    print(f"Audio shape: {signal.shape}")

    # Resample to 16000 Hz if necessary
    if fs != 16000:
        print("Resampling to 16000 Hz...")
        resampler = T.Resample(fs, 16000)
        signal = resampler(signal)

    # Convert to mono if stereo
    if signal.shape[0] > 1:
        print("Converting to mono...")
        signal = torch.mean(signal, dim=0, keepdim=True)

    # Convert to numpy for processor
    audio_array = signal.squeeze().numpy()

    # Process audio
    print("Processing audio with wav2vec2 processor...")
    inputs = processor(audio_array, sampling_rate=16000, return_tensors="pt", padding=True)
    print(f"Processed input shape: {inputs.input_values.shape}")

    # Perform inference
    print("Performing emotion classification...")
    with torch.no_grad():
        probs, logits = model(inputs.input_values)

    print("Inference completed!")

    # Get predictions
    predicted_idx = torch.argmax(probs, dim=-1).item()
    confidence = torch.max(probs, dim=-1).values.item()

    # Display results
    print("\n" + "=" * 50)
    print("EMOTION RECOGNITION RESULTS")
    print("=" * 50)
    print(f"Predicted emotion: {emotion_labels[predicted_idx]}")
    print(f"Confidence score: {confidence:.4f}")
    print(f"Class index: {predicted_idx}")

    # Display all emotion probabilities
    print("\nAll emotion probabilities:")
    for i, (emotion, prob) in enumerate(zip(emotion_labels, probs[0])):
        print(f"  {emotion}: {prob.item():.4f}")

except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you have the required packages installed:")
    print("  pip install torch torchaudio transformers")

except FileNotFoundError as e:
    print(f"File not found error: {e}")
    print("Please check that both the model path and audio file path are correct.")

except Exception as e:
    print(f"Unexpected error: {e}")
    import traceback

    traceback.print_exc()