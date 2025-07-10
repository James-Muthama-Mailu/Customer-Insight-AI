import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import json
from datetime import datetime
import warnings
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
from transformers import TrainingArguments, Trainer
import librosa
import soundfile as sf

warnings.filterwarnings('ignore')


# Import your dataset class
# from dataset_class import create_data_loaders, load_and_prepare_data


class SpeechBrainEmotionModel(nn.Module):
    """
    Wrapper for SpeechBrain emotion recognition model
    """

    def __init__(self, model_path, num_classes=6, fine_tune_layers=2):
        super().__init__()

        self.model_path = model_path
        self.num_classes = num_classes

        # Load the pre-trained model and processor
        print(f"Loading model from: {model_path}")
        try:
            # Try loading as Wav2Vec2 model first
            self.processor = Wav2Vec2Processor.from_pretrained(model_path)
            self.model = Wav2Vec2ForSequenceClassification.from_pretrained(
                model_path,
                num_labels=num_classes,
                ignore_mismatched_sizes=True
            )
        except Exception as e:
            print(f"Error loading as Wav2Vec2: {e}")
            # Fallback to manual loading
            self._load_speechbrain_model()

        # Freeze most layers for faster training
        self._setup_fine_tuning(fine_tune_layers)

    def _load_speechbrain_model(self):
        """Fallback method to load SpeechBrain model manually"""
        try:
            # Look for model files in the directory
            model_files = []
            for root, dirs, files in os.walk(self.model_path):
                for file in files:
                    if file.endswith(('.pt', '.pth', '.bin', '.ckpt')):
                        model_files.append(os.path.join(root, file))

            print(f"Found model files: {model_files}")

            # Load the most recent model file
            if model_files:
                model_file = model_files[0]  # Take first available
                print(f"Loading model from: {model_file}")

                # Create a basic Wav2Vec2 model structure
                from transformers import Wav2Vec2Config
                config = Wav2Vec2Config(
                    vocab_size=32,
                    num_labels=self.num_classes,
                    hidden_size=768,
                    num_hidden_layers=12,
                    num_attention_heads=12,
                    intermediate_size=3072,
                )

                self.model = Wav2Vec2ForSequenceClassification(config)
                self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

                # Try to load weights
                try:
                    checkpoint = torch.load(model_file, map_location='cpu')
                    if 'model' in checkpoint:
                        self.model.load_state_dict(checkpoint['model'], strict=False)
                    elif 'state_dict' in checkpoint:
                        self.model.load_state_dict(checkpoint['state_dict'], strict=False)
                    else:
                        self.model.load_state_dict(checkpoint, strict=False)
                    print("Model weights loaded successfully!")
                except Exception as e:
                    print(f"Could not load weights: {e}")
                    print("Using randomly initialized weights")
            else:
                # Create from scratch
                print("No model files found, creating from scratch")
                self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
                self.model = Wav2Vec2ForSequenceClassification.from_pretrained(
                    "facebook/wav2vec2-base-960h",
                    num_labels=self.num_classes,
                    ignore_mismatched_sizes=True
                )

        except Exception as e:
            print(f"Error in fallback loading: {e}")
            # Last resort - use base model
            self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
            self.model = Wav2Vec2ForSequenceClassification.from_pretrained(
                "facebook/wav2vec2-base-960h",
                num_labels=self.num_classes,
                ignore_mismatched_sizes=True
            )

    def _setup_fine_tuning(self, fine_tune_layers):
        """Setup which layers to fine-tune"""
        # Freeze the feature extractor
        if hasattr(self.model, 'wav2vec2'):
            self.model.wav2vec2.feature_extractor._freeze_parameters()

            # Freeze most transformer layers
            total_layers = len(self.model.wav2vec2.encoder.layers)
            layers_to_freeze = max(0, total_layers - fine_tune_layers)

            for i in range(layers_to_freeze):
                for param in self.model.wav2vec2.encoder.layers[i].parameters():
                    param.requires_grad = False

            print(f"Frozen {layers_to_freeze}/{total_layers} transformer layers")

        # Keep classifier layers trainable
        if hasattr(self.model, 'classifier'):
            for param in self.model.classifier.parameters():
                param.requires_grad = True

    def forward(self, input_values, attention_mask=None):
        """Forward pass"""
        return self.model(input_values=input_values, attention_mask=attention_mask)

    def get_trainable_parameters(self):
        """Get count of trainable parameters"""
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        return trainable, total


class SpeechBrainEmotionTrainer:
    """
    Trainer for SpeechBrain emotion model with fast training optimizations
    """

    def __init__(self, model, train_loader, test_loader, device, num_classes=6,
                 class_weights=None, label_encoder=None, learning_rate=1e-4):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.num_classes = num_classes
        self.label_encoder = label_encoder

        # Setup loss function with class weights
        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        else:
            self.criterion = nn.CrossEntropyLoss()

        # Setup optimizer with lower learning rate for pre-trained model
        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.model.parameters()),
            lr=learning_rate,
            weight_decay=0.01,
            eps=1e-8
        )

        # Setup learning rate scheduler
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=learning_rate * 3,
            steps_per_epoch=len(train_loader),
            epochs=10,  # Fewer epochs for pre-trained model
            pct_start=0.1,
            anneal_strategy='cos'
        )

        # Training history
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []

        # Mixed precision training
        self.scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

        # Print model info
        trainable, total = model.get_trainable_parameters()
        print(f"Trainable parameters: {trainable:,} / {total:,} ({100 * trainable / total:.1f}%)")

    def preprocess_batch(self, input_values):
        """Preprocess audio for the model"""
        processed_inputs = []
        attention_masks = []

        for audio in input_values:
            # Convert to numpy if tensor
            if torch.is_tensor(audio):
                audio = audio.cpu().numpy()

            # Ensure audio is 1D
            if audio.ndim > 1:
                audio = audio.flatten()

            # Resample if needed (assuming 16kHz target)
            if len(audio) > 160000:  # Truncate if too long (10 seconds at 16kHz)
                audio = audio[:160000]
            elif len(audio) < 1600:  # Pad if too short (0.1 seconds at 16kHz)
                audio = np.pad(audio, (0, 1600 - len(audio)))

            # Process with the processor
            inputs = self.model.processor(
                audio,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=160000
            )

            processed_inputs.append(inputs.input_values.squeeze(0))
            if hasattr(inputs, 'attention_mask') and inputs.attention_mask is not None:
                attention_masks.append(inputs.attention_mask.squeeze(0))

        # Stack and pad sequences
        max_len = max(x.size(0) for x in processed_inputs)
        padded_inputs = []
        padded_masks = []

        for i, inp in enumerate(processed_inputs):
            if inp.size(0) < max_len:
                pad_len = max_len - inp.size(0)
                inp = torch.cat([inp, torch.zeros(pad_len)])
            padded_inputs.append(inp)

            # Create attention mask
            mask = torch.ones(inp.size(0))
            if inp.size(0) < max_len:
                mask = torch.cat([mask, torch.zeros(max_len - inp.size(0))])
            padded_masks.append(mask)

        return torch.stack(padded_inputs), torch.stack(padded_masks)

    def train_epoch(self):
        """Train for one epoch"""
        self.model.model.train()
        total_loss = 0
        num_batches = 0

        progress_bar = tqdm(self.train_loader, desc="Training")

        for batch in progress_bar:
            try:
                # Get data
                input_values = batch['input_values']
                labels = batch['labels'].to(self.device, non_blocking=True)

                # Preprocess inputs
                processed_inputs, attention_mask = self.preprocess_batch(input_values)
                processed_inputs = processed_inputs.to(self.device, non_blocking=True)
                attention_mask = attention_mask.to(self.device, non_blocking=True)

                self.optimizer.zero_grad()

                # Mixed precision forward pass
                if self.scaler is not None:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(processed_inputs, attention_mask=attention_mask)
                        logits = outputs.logits
                        loss = self.criterion(logits, labels)

                    # Mixed precision backward pass
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.model.parameters(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    outputs = self.model(processed_inputs, attention_mask=attention_mask)
                    logits = outputs.logits
                    loss = self.criterion(logits, labels)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.model.parameters(), 1.0)
                    self.optimizer.step()

                # Update scheduler
                self.scheduler.step()

                total_loss += loss.item()
                num_batches += 1

                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'lr': f'{self.scheduler.get_last_lr()[0]:.2e}'
                })

            except Exception as e:
                print(f"Error in batch processing: {e}")
                continue

        if num_batches == 0:
            return float('inf')

        return total_loss / num_batches

    def evaluate(self):
        """Evaluate the model"""
        self.model.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Evaluating", leave=False):
                try:
                    # Get data
                    input_values = batch['input_values']
                    labels = batch['labels'].to(self.device, non_blocking=True)

                    # Preprocess inputs
                    processed_inputs, attention_mask = self.preprocess_batch(input_values)
                    processed_inputs = processed_inputs.to(self.device, non_blocking=True)
                    attention_mask = attention_mask.to(self.device, non_blocking=True)

                    # Forward pass
                    if self.scaler is not None:
                        with torch.cuda.amp.autocast():
                            outputs = self.model(processed_inputs, attention_mask=attention_mask)
                            logits = outputs.logits
                            loss = self.criterion(logits, labels)
                    else:
                        outputs = self.model(processed_inputs, attention_mask=attention_mask)
                        logits = outputs.logits
                        loss = self.criterion(logits, labels)

                    total_loss += loss.item()
                    num_batches += 1

                    # Get predictions
                    predictions = torch.argmax(logits, dim=1)
                    all_predictions.extend(predictions.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

                except Exception as e:
                    print(f"Error in evaluation batch: {e}")
                    continue

        if num_batches == 0:
            return float('inf'), 0.0, [], []

        avg_loss = total_loss / num_batches
        accuracy = accuracy_score(all_labels, all_predictions) if all_labels else 0.0

        return avg_loss, accuracy, all_predictions, all_labels

    def train(self, num_epochs, save_dir="./speechbrain_emotion_checkpoints"):
        """Training loop"""
        os.makedirs(save_dir, exist_ok=True)

        best_accuracy = 0
        best_model_path = None

        print(f"Starting SpeechBrain emotion model training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Mixed Precision: {self.scaler is not None}")

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 30)

            # Train
            train_loss = self.train_epoch()

            if train_loss == float('inf'):
                print("Training failed - no valid batches processed")
                break

            # Evaluate
            val_loss, val_accuracy, predictions, labels = self.evaluate()

            # Store history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_accuracy)

            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Val Accuracy: {val_accuracy:.4f}")

            # Save best model
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_model_path = os.path.join(save_dir, f"best_speechbrain_model_epoch_{epoch + 1}.pth")
                self.save_model(best_model_path, epoch, val_accuracy)
                print(f"New best model saved! Accuracy: {val_accuracy:.4f}")

        print(f"\nTraining completed!")
        print(f"Best validation accuracy: {best_accuracy:.4f}")

        return best_model_path

    def save_model(self, path, epoch, accuracy):
        """Save model checkpoint"""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'accuracy': accuracy,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'label_encoder': self.label_encoder,
            'model_config': {
                'num_classes': self.num_classes,
                'model_path': self.model.model_path
            }
        }, path)


def main():
    """Main training function"""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Model path
    model_path = r"C:\Users\james\PycharmProjects\CustomerInsightAI\Customer-Insight-AI\models\emotion_categorisation_model\models--speechbrain--emotion-recognition-wav2vec2-IEMOCAP"

    # Path to your CSV file
    csv_path = r"C:\Users\james\PycharmProjects\CustomerInsightAI\Customer-Insight-AI\models\emotion_categorisation_model\emotion_dataset.csv"

    # Load and prepare data
    print("Loading data...")
    train_df, test_df, label_encoder = load_and_prepare_data(csv_path)

    # Create data loaders
    train_loader, test_loader, train_dataset, test_dataset = create_data_loaders(
        train_df, test_df, label_encoder,
        batch_size=8,  # Smaller batch size for large model
        max_length=160000  # 10 seconds at 16kHz
    )

    # Get class weights
    class_weights = train_dataset.get_class_weights()
    print(f"Class weights: {class_weights}")

    # Initialize model
    num_classes = len(label_encoder.classes_)
    model = SpeechBrainEmotionModel(
        model_path=model_path,
        num_classes=num_classes,
        fine_tune_layers=2  # Only fine-tune last 2 layers
    )

    print(f"Model initialized with {num_classes} classes")
    print(f"Classes: {list(label_encoder.classes_)}")

    # Initialize trainer
    trainer = SpeechBrainEmotionTrainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        num_classes=num_classes,
        class_weights=class_weights,
        label_encoder=label_encoder,
        learning_rate=1e-4  # Lower learning rate for pre-trained model
    )

    # Train the model
    num_epochs = 1000   # Fewer epochs for pre-trained model
    save_dir = "./speechbrain_emotion_checkpoints"

    best_model_path = trainer.train(num_epochs=num_epochs, save_dir=save_dir)

    print(f"\nTraining completed successfully!")
    print(f"Best model saved at: {best_model_path}")

    return trainer, best_model_path


if __name__ == "__main__":
    # Make sure to import your dataset functions
    from dataset_class import create_data_loaders, load_and_prepare_data

    # Run training
    trainer, best_model_path = main()

    print("\nSpeechBrain emotion model training completed!")