import numpy as np
import torch
import torchaudio
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import os
from transformers import Wav2Vec2Processor


class SpeechEmotionDataset(Dataset):
    def __init__(self, df, processor, max_length=44100, label_encoder=None):
        """
        Initialize the SpeechEmotionDataset

        Args:
            df (pd.DataFrame): DataFrame with 'Path' and 'Emotions' columns
            processor: Wav2Vec2Processor for audio preprocessing
            max_length (int): Maximum length of audio samples (at 16kHz sampling rate)
            label_encoder (LabelEncoder): Optional pre-fitted label encoder
        """
        self.df = df.copy()
        self.processor = processor
        self.max_length = max_length
        self.target_sr = 16000  # Wav2Vec2 requires 16kHz

        # Handle label encoding
        if label_encoder is None:
            self.label_encoder = LabelEncoder()
            self.df['label'] = self.label_encoder.fit_transform(self.df['Emotions'])
        else:
            self.label_encoder = label_encoder
            self.df['label'] = self.label_encoder.transform(self.df['Emotions'])

        # Store emotion mapping for reference
        self.emotion_to_label = dict(zip(self.label_encoder.classes_,
                                         self.label_encoder.transform(self.label_encoder.classes_)))
        self.label_to_emotion = {v: k for k, v in self.emotion_to_label.items()}

        print(f"Dataset initialized with {len(self.df)} samples")
        print(f"Emotion mapping: {self.emotion_to_label}")
        print(
            f"Target audio length: {max_length} samples at {self.target_sr}Hz ({max_length / self.target_sr:.2f} seconds)")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        audio_path = self.df.iloc[idx]['Path']
        label = self.df.iloc[idx]['label']
        emotion = self.df.iloc[idx]['Emotions']

        try:
            # Load the audio file
            speech, sr = torchaudio.load(audio_path)
            speech = speech.squeeze()

            # Resample to 16kHz if needed (Wav2Vec2 requirement)
            if sr != self.target_sr:
                resampler = torchaudio.transforms.Resample(sr, self.target_sr)
                speech = resampler(speech)

            # Convert to numpy
            speech = speech.numpy()

            # Truncate or pad to target length BEFORE passing to processor
            if len(speech) > self.max_length:
                # Truncate if too long
                speech = speech[:self.max_length]
            else:
                # Pad if too short
                speech = np.pad(speech, (0, self.max_length - len(speech)), 'constant')

            # Use the Wav2Vec2 processor - FIXED: Remove truncation parameter
            # Since we already handled truncation/padding above
            inputs = self.processor(
                speech,
                sampling_rate=self.target_sr,
                return_tensors="pt",
                padding=False  # We already padded, so no need for processor padding
            )

            input_values = inputs['input_values'].squeeze()
            labels = torch.tensor(label, dtype=torch.long)

            return {
                'input_values': input_values,
                'labels': labels,
                'emotion': emotion,
                'audio_path': audio_path
            }

        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            # Return a dummy sample in case of error
            dummy_speech = np.zeros(self.max_length)
            inputs = self.processor(
                dummy_speech,
                sampling_rate=self.target_sr,
                return_tensors="pt",
                padding=False
            )
            return {
                'input_values': inputs['input_values'].squeeze(),
                'labels': torch.tensor(0, dtype=torch.long),
                'emotion': 'unknown',
                'audio_path': audio_path
            }

    def get_class_weights(self):
        """Calculate class weights for handling imbalanced data"""
        emotion_counts = self.df['Emotions'].value_counts()
        total_samples = len(self.df)
        weights = []

        for emotion in self.label_encoder.classes_:
            weight = total_samples / (len(self.label_encoder.classes_) * emotion_counts[emotion])
            weights.append(weight)

        return torch.FloatTensor(weights)


def load_and_prepare_data(csv_path, test_size=0.2, random_state=42):
    """
    Load and prepare the emotion dataset from CSV

    Args:
        csv_path (str): Path to the CSV file with emotion data
        test_size (float): Proportion of data to use for testing
        random_state (int): Random state for reproducibility

    Returns:
        tuple: (train_df, test_df, label_encoder)
    """
    from sklearn.model_selection import train_test_split

    # Load the data
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} samples from {csv_path}")
    print(f"Emotions distribution:\n{df['Emotions'].value_counts()}")

    # Verify all audio files exist
    valid_files = []
    for idx, row in df.iterrows():
        if os.path.exists(row['Path']):
            valid_files.append(idx)
        else:
            print(f"Warning: File not found - {row['Path']}")

    df = df.loc[valid_files].reset_index(drop=True)
    print(f"Found {len(df)} valid audio files")

    # Split the data
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df['Emotions']
    )

    # Create label encoder
    label_encoder = LabelEncoder()
    label_encoder.fit(df['Emotions'])

    print(f"Training samples: {len(train_df)}")
    print(f"Testing samples: {len(test_df)}")

    return train_df, test_df, label_encoder


def create_data_loaders(train_df, test_df, label_encoder, batch_size=16, max_length=16000):
    """
    Create PyTorch DataLoaders for training and testing

    Args:
        train_df (pd.DataFrame): Training data
        test_df (pd.DataFrame): Testing data
        label_encoder (LabelEncoder): Fitted label encoder
        batch_size (int): Batch size for data loaders
        max_length (int): Maximum audio length in samples (at 16kHz)

    Returns:
        tuple: (train_loader, test_loader, train_dataset, test_dataset)
    """
    # Load processor
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

    # Create datasets
    train_dataset = SpeechEmotionDataset(train_df, processor, max_length, label_encoder)
    test_dataset = SpeechEmotionDataset(test_df, processor, max_length, label_encoder)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 for Windows compatibility
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    return train_loader, test_loader, train_dataset, test_dataset


def main():
    """
    Main function demonstrating complete usage of the SpeechEmotionDataset
    """
    print("Setting up Speech Emotion Recognition Dataset...")

    # Path to your CSV file
    csv_path = r"C:\Users\james\PycharmProjects\CustomerInsightAI\Customer-Insight-AI\models\emotion_categorisation_model\data_path.csv"

    try:
        # Load and prepare data
        train_df, test_df, label_encoder = load_and_prepare_data(csv_path)

        # Create data loaders
        train_loader, test_loader, train_dataset, test_dataset = create_data_loaders(
            train_df, test_df, label_encoder, batch_size=8, max_length=16000  # 1 second at 16kHz
        )

        print("\nDataset setup complete!")
        print(f"Number of classes: {len(label_encoder.classes_)}")
        print(f"Classes: {list(label_encoder.classes_)}")

        # Calculate class weights for handling imbalanced data
        class_weights = train_dataset.get_class_weights()
        print(f"Class weights: {class_weights}")

        # Test the data loader
        print("\nTesting data loader...")
        for i, batch in enumerate(train_loader):
            print(f"Batch {i + 1}:")
            print(f"  Input shape: {batch['input_values'].shape}")
            print(f"  Labels shape: {batch['labels'].shape}")
            print(f"  Sample emotions: {batch['emotion'][:3]}")  # First 3 emotions in batch

            if i >= 2:  # Only show first 3 batches
                break

        print("\nDataset is ready for training!")

        return train_loader, test_loader, train_dataset, test_dataset, label_encoder

    except Exception as e:
        print(f"Error setting up dataset: {e}")
        return None


if __name__ == "__main__":
    # Run the main function
    result = main()

    if result:
        train_loader, test_loader, train_dataset, test_dataset, label_encoder = result

        # Example of how to use in a training loop
        print("\nExample training loop structure:")
        print("""
        for epoch in range(num_epochs):
            for batch in train_loader:
                input_values = batch['input_values']
                labels = batch['labels']

                # Forward pass
                outputs = model(input_values)
                loss = criterion(outputs, labels)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        """)