# pre_processing_dataset.py
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import warnings


def process_ravdess_dataset(ravdess_path):
    """
    Process RAVDESS dataset and extract emotion labels from filenames.

    RAVDESS filename format: Modality-VocalChannel-Emotion-Intensity-Statement-Repetition-Actor.wav
    Example: 03-01-06-01-02-01-12.wav
    - Modality (01 = full-AV, 02 = video-only, 03 = audio-only)
    - Vocal channel (01 = speech, 02 = song)
    - Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised)
    - Emotional intensity (01 = normal, 02 = strong)
    - Statement (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door")
    - Repetition (01 = 1st repetition, 02 = 2nd repetition)
    - Actor (01 to 24. Odd numbered actors are male, even numbered actors are female)

    Args:
        ravdess_path (str): Path to RAVDESS dataset folder

    Returns:
        pd.DataFrame: DataFrame with 'Emotions' and 'Path' columns
    """
    print("Processing RAVDESS dataset...")

    if not os.path.exists(ravdess_path):
        print(f"RAVDESS path not found: {ravdess_path}")
        return pd.DataFrame()

    file_emotion = []
    file_path = []

    # Check if there's an audio_speech_actors folder
    audio_speech_path = os.path.join(ravdess_path, "audio_speech_actors_01-24")
    if os.path.exists(audio_speech_path):
        print(f"Found audio_speech_actors folder: {audio_speech_path}")
        ravdess_directory_list = os.listdir(audio_speech_path)
        base_path = audio_speech_path
    else:
        print(f"Using direct RAVDESS path: {ravdess_path}")
        ravdess_directory_list = os.listdir(ravdess_path)
        base_path = ravdess_path

    print(f"Found {len(ravdess_directory_list)} actor directories")

    for actor_dir in ravdess_directory_list:
        actor_path = os.path.join(base_path, actor_dir)

        # Check if it's a directory (actor folder)
        if os.path.isdir(actor_path):
            print(f"Processing actor directory: {actor_dir}")

            # List all files in the actor directory
            actor_files = os.listdir(actor_path)

            for file in actor_files:
                # Process both .wav and .mp4 files (but we'll focus on audio)
                if file.endswith(('.wav', '.mp4')):
                    try:
                        # Split filename to extract emotion information
                        filename_without_ext = file.split('.')[0]
                        parts = filename_without_ext.split('-')

                        # Check if we have the expected number of parts (7 parts in RAVDESS format)
                        if len(parts) == 7:
                            modality = int(parts[0])  # Should be 03 for audio-only
                            vocal_channel = int(parts[1])  # 01 = speech, 02 = song
                            emotion = int(parts[2])  # This is what we want
                            intensity = int(parts[3])
                            statement = int(parts[4])
                            repetition = int(parts[5])
                            actor = int(parts[6])

                            # Only process audio-only files (modality = 03) and speech (vocal_channel = 01)
                            if modality == 3 and vocal_channel == 1:
                                file_emotion.append(emotion)
                                file_path.append(os.path.join(actor_path, file))
                            elif file.endswith('.wav'):  # Include all .wav files regardless of modality
                                file_emotion.append(emotion)
                                file_path.append(os.path.join(actor_path, file))
                        else:
                            print(f"Unexpected filename format: {file} (has {len(parts)} parts, expected 7)")

                    except (ValueError, IndexError) as e:
                        print(f"Error processing file {file}: {e}")
                        continue

    if not file_emotion:
        print("No valid RAVDESS files found!")
        return pd.DataFrame()

    # Create dataframes
    emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])
    path_df = pd.DataFrame(file_path, columns=['Path'])
    ravdess_df = pd.concat([emotion_df, path_df], axis=1)

    # Map emotion numbers to emotion names according to RAVDESS specification
    # Only include the emotions we want: neutral, happy, sad, angry
    emotion_mapping = {
        1: 'neutral',
        3: 'happy',
        4: 'sad',
        5: 'angry'
        # Removed: 2: 'calm', 6: 'fear', 7: 'disgust', 8: 'surprise'
    }

    ravdess_df['Emotions'] = ravdess_df['Emotions'].map(emotion_mapping)

    # Remove any rows where emotion mapping failed (this will filter out unwanted emotions)
    ravdess_df = ravdess_df.dropna(subset=['Emotions'])

    print(f"RAVDESS: Found {len(ravdess_df)} files")
    print(f"RAVDESS emotions found: {ravdess_df['Emotions'].value_counts().to_dict()}")

    return ravdess_df


def process_crema_dataset(crema_path):
    """
    Process CREMA dataset and extract emotion labels from filenames.

    Args:
        crema_path (str): Path to CREMA dataset folder

    Returns:
        pd.DataFrame: DataFrame with 'Emotions' and 'Path' columns
    """
    print("Processing CREMA dataset...")

    if not os.path.exists(crema_path):
        print(f"CREMA path not found: {crema_path}")
        return pd.DataFrame()

    crema_directory_list = os.listdir(crema_path)

    file_emotion = []
    file_path = []

    for file in crema_directory_list:
        if file.endswith('.wav'):  # Only process audio files
            # storing file paths
            file_path.append(os.path.join(crema_path, file))
            # storing file emotions
            part = file.split('_')
            if len(part) >= 3:
                if part[2] == 'SAD':
                    file_emotion.append('sad')
                elif part[2] == 'ANG':
                    file_emotion.append('angry')
                elif part[2] == 'HAP':
                    file_emotion.append('happy')
                elif part[2] == 'NEU':
                    file_emotion.append('neutral')
                else:
                    # Skip other emotions: DIS (disgust), FEA (fear)
                    file_emotion.append('skip')
            else:
                file_emotion.append('skip')

    # dataframe for emotion of files
    emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])

    # dataframe for path of files.
    path_df = pd.DataFrame(file_path, columns=['Path'])
    crema_df = pd.concat([emotion_df, path_df], axis=1)

    # Remove rows with 'skip' emotions
    crema_df = crema_df[crema_df['Emotions'] != 'skip']

    print(f"CREMA: Found {len(crema_df)} files")
    print(f"CREMA emotions found: {crema_df['Emotions'].value_counts().to_dict()}")

    return crema_df


def process_tess_dataset(tess_path):
    """
    Process TESS dataset and extract emotion labels from filenames.

    Args:
        tess_path (str): Path to TESS dataset folder

    Returns:
        pd.DataFrame: DataFrame with 'Emotions' and 'Path' columns
    """
    print("Processing TESS dataset...")

    if not os.path.exists(tess_path):
        print(f"TESS path not found: {tess_path}")
        return pd.DataFrame()

    tess_directory_list = os.listdir(tess_path)

    file_emotion = []
    file_path = []

    for dir in tess_directory_list:
        dir_path = os.path.join(tess_path, dir)
        if os.path.isdir(dir_path):
            directories = os.listdir(dir_path)
            for file in directories:
                if file.endswith('.wav'):  # Only process audio files
                    part = file.split('.')[0]
                    part = part.split('_')
                    if len(part) >= 3:
                        emotion_part = part[2]
                        # Only keep the emotions we want
                        if emotion_part in ['happy', 'sad', 'angry', 'neutral']:
                            file_emotion.append(emotion_part)
                            file_path.append(os.path.join(dir_path, file))
                        # Skip other emotions like 'ps' (surprise), 'disgust', 'fear'

    # dataframe for emotion of files
    emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])

    # dataframe for path of files.
    path_df = pd.DataFrame(file_path, columns=['Path'])
    tess_df = pd.concat([emotion_df, path_df], axis=1)

    print(f"TESS: Found {len(tess_df)} files")
    print(f"TESS emotions found: {tess_df['Emotions'].value_counts().to_dict()}")

    return tess_df


def process_savee_dataset(savee_path):
    """
    Process SAVEE dataset and extract emotion labels from filenames.

    Args:
        savee_path (str): Path to SAVEE dataset folder

    Returns:
        pd.DataFrame: DataFrame with 'Emotions' and 'Path' columns
    """
    print("Processing SAVEE dataset...")

    if not os.path.exists(savee_path):
        print(f"SAVEE path not found: {savee_path}")
        return pd.DataFrame()

    savee_directory_list = os.listdir(savee_path)

    file_emotion = []
    file_path = []

    for file in savee_directory_list:
        if file.endswith('.wav'):  # Only process audio files
            part = file.split('_')
            if len(part) >= 2:
                ele = part[1][:-6] if len(part[1]) > 6 else part[1]
                # Only keep the emotions we want
                if ele == 'a':
                    file_emotion.append('angry')
                    file_path.append(os.path.join(savee_path, file))
                elif ele == 'h':
                    file_emotion.append('happy')
                    file_path.append(os.path.join(savee_path, file))
                elif ele == 'n':
                    file_emotion.append('neutral')
                    file_path.append(os.path.join(savee_path, file))
                elif ele == 'sa':
                    file_emotion.append('sad')
                    file_path.append(os.path.join(savee_path, file))
                # Skip other emotions: 'd' (disgust), 'f' (fear), 'su' (surprise)

    # dataframe for emotion of files
    emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])

    # dataframe for path of files.
    path_df = pd.DataFrame(file_path, columns=['Path'])
    savee_df = pd.concat([emotion_df, path_df], axis=1)

    print(f"SAVEE: Found {len(savee_df)} files")
    print(f"SAVEE emotions found: {savee_df['Emotions'].value_counts().to_dict()}")

    return savee_df


def visualize_dataset_distribution(data_path):
    """
    Create visualizations for the emotion dataset distribution.

    Args:
        data_path (pd.DataFrame): Combined dataset with emotion labels and paths
    """
    print("\nCreating dataset visualizations...")

    # Set up the plotting style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Overall emotion distribution
    emotion_counts = data_path['Emotions'].value_counts()

    axes[0, 0].bar(emotion_counts.index, emotion_counts.values, color='skyblue')
    axes[0, 0].set_title('Overall Emotion Distribution', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Emotions')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].tick_params(axis='x', rotation=45)

    # 2. Pie chart
    axes[0, 1].pie(emotion_counts.values, labels=emotion_counts.index, autopct='%1.1f%%', startangle=90)
    axes[0, 1].set_title('Emotion Distribution (%)', fontsize=14, fontweight='bold')

    # 3. Dataset source distribution (if we can infer from paths)
    dataset_sources = []
    for path in data_path['Path']:
        if 'Ravdess' in path or 'ravdess' in path.lower():
            dataset_sources.append('RAVDESS')
        elif 'Crema' in path or 'crema' in path.lower():
            dataset_sources.append('CREMA')
        elif 'Tess' in path or 'tess' in path.lower():
            dataset_sources.append('TESS')
        elif 'Savee' in path or 'savee' in path.lower():
            dataset_sources.append('SAVEE')
        else:
            dataset_sources.append('Unknown')

    data_path['Dataset'] = dataset_sources
    dataset_counts = data_path['Dataset'].value_counts()

    axes[1, 0].bar(dataset_counts.index, dataset_counts.values, color='lightcoral')
    axes[1, 0].set_title('Dataset Source Distribution', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Dataset')
    axes[1, 0].set_ylabel('Count')

    # 4. Emotion distribution by dataset (stacked bar)
    emotion_by_dataset = pd.crosstab(data_path['Dataset'], data_path['Emotions'])
    emotion_by_dataset.plot(kind='bar', stacked=True, ax=axes[1, 1], colormap='Set3')
    axes[1, 1].set_title('Emotion Distribution by Dataset', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Dataset')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].legend(title='Emotions', bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1, 1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()

    # Print summary statistics
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"Total audio files: {len(data_path)}")
    print(f"Unique emotions: {data_path['Emotions'].nunique()}")
    print(f"Emotions found: {list(data_path['Emotions'].unique())}")
    print("\nEmotion distribution:")
    print(data_path['Emotions'].value_counts())
    print("\nDataset source distribution:")
    print(data_path['Dataset'].value_counts())


def main():
    """
    Main function to process all emotion datasets and create a combined dataset.
    Only keeps the emotions: happy, sad, neutral, angry
    """
    warnings.filterwarnings('ignore')

    print("Starting Speech Emotion Dataset Preprocessing...")
    print("Filtering for emotions: happy, sad, neutral, angry")
    print("=" * 60)

    # Define base path - update this to your actual path
    base_path = r"C:\Users\james\PycharmProjects\CustomerInsightAI\Customer-Insight-AI\models\emotion_categorisation_model\speech-emotion-recognition-en\versions\1"

    # Define paths for each dataset
    ravdess_path = os.path.join(base_path, "Ravdess")
    crema_path = os.path.join(base_path, "Crema")
    tess_path = os.path.join(base_path, "Tess")
    savee_path = os.path.join(base_path, "Savee")

    print(f"Base path: {base_path}")
    print(f"Looking for datasets in:")
    print(f"  - RAVDESS: {ravdess_path}")
    print(f"  - CREMA: {crema_path}")
    print(f"  - TESS: {tess_path}")
    print(f"  - SAVEE: {savee_path}")
    print("-" * 60)

    # Process each dataset
    dataframes = []

    # Process RAVDESS
    ravdess_df = process_ravdess_dataset(ravdess_path)
    if not ravdess_df.empty:
        dataframes.append(ravdess_df)

    # Process CREMA
    crema_df = process_crema_dataset(crema_path)
    if not crema_df.empty:
        dataframes.append(crema_df)

    # Process TESS
    tess_df = process_tess_dataset(tess_path)
    if not tess_df.empty:
        dataframes.append(tess_df)

    # Process SAVEE
    savee_df = process_savee_dataset(savee_path)
    if not savee_df.empty:
        dataframes.append(savee_df)

    # Combine all datasets
    if dataframes:
        print("\nCombining all datasets...")
        data_path = pd.concat(dataframes, axis=0, ignore_index=True)

        # Final filter to ensure only desired emotions are kept
        desired_emotions = ['happy', 'sad', 'neutral', 'angry']
        data_path = data_path[data_path['Emotions'].isin(desired_emotions)]

        print(f"\nFinal dataset - Total samples: {len(data_path)}")
        print("Final emotion distribution:")
        emotion_counts = data_path['Emotions'].value_counts()
        print(emotion_counts.to_dict())

        # Save to CSV
        output_file = "emotion_dataset_filtered.csv"
        data_path.to_csv(output_file, index=False)
        print(f"Filtered dataset saved to: {output_file}")
        print(f"Full path: {os.path.abspath(output_file)}")

        # Create visualizations
        visualize_dataset_distribution(data_path)

        # Display sample data
        print("\nSample data:")
        print(data_path.head(10))

        return data_path
    else:
        print("No datasets were successfully processed!")
        return None


if __name__ == '__main__':
    df = main()