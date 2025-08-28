from abc import ABC, abstractmethod
import soundfile as sf
import os
from typing import Dict, Tuple

class SoundProcessor(ABC):
    """Abstract factory for sound processing"""
    @abstractmethod
    def process_file(self, file_path: str) -> Tuple[str, str]:
        """Process a sound file and return instrument type and color"""
        pass

class GeneralSoundProcessor(SoundProcessor):
    def __init__(self, classifier):
        self.classifier = classifier
        # Color map for different sound categories
        self.color_map = {
            # Drums and Percussion
            'Drums': '#FFD700',          # Yellow
            'Snare drum': '#FFA500',     # Orange
            'Bass drum': '#FF4500',      # Red-Orange
            'Cymbal': '#FFE4B5',         # Moccasin
            'Hi-hat': '#FFDAB9',         # Peach
            'Percussion': '#DEB887',     # Burlywood
            
            # String instruments
            'Guitar': '#98FB98',         # Pale Green
            'Electric guitar': '#90EE90', # Light Green
            'Bass guitar': '#32CD32',    # Lime Green
            'Acoustic guitar': '#228B22', # Forest Green
            'Piano': '#00FF00',          # Green
            'Violin': '#7CFC00',         # Lawn Green
            
            # Wind instruments
            'Saxophone': '#87CEEB',      # Sky Blue
            'Trumpet': '#00BFFF',        # Deep Sky Blue
            'Flute': '#1E90FF',         # Dodger Blue
            'Clarinet': '#4169E1',      # Royal Blue
            
            # Electronic
            'Synthesizer': '#FF00FF',    # Magenta
            'Electronic': '#FF69B4',     # Hot Pink
            'Sample': '#DA70D6',         # Orchid
            
            # Voice
            'Speech': '#FF1493',         # Deep Pink
            'Male speech': '#DB7093',    # Pale Violet Red
            'Female speech': '#FFB6C1',  # Light Pink
            'Singing': '#FF69B4',        # Hot Pink
            'Vocals': '#FFC0CB',         # Pink
            
            # Default for other categories
            'Music': '#FFFFFF',          # White
            'Generic': '#CCCCCC'         # Gray
        }
    
    def process_file(self, file_path: str) -> Tuple[str, str]:
        """
        Process a wav file and return the detected sound type and color
        Returns: Tuple[sound_type: str, color_code: str]
        """
        try:
            # Load the audio file with specific parameters
            audio, sample_rate = sf.read(file_path, dtype='float32', always_2d=True)
            
            # If stereo, convert to mono by averaging channels
            if audio.shape[1] > 1:
                audio = audio.mean(axis=1)
            else:
                audio = audio.flatten()
                
            # Ensure we have enough samples (at least 0.1 seconds)
            min_samples = int(sample_rate * 0.1)
            if len(audio) < min_samples:
                raise ValueError(f"Audio file is too short. Needs at least {min_samples} samples.")
            
            # Get classifications
            classifications = self.classifier.classify(audio, sample_rate)
            
            # Filter out low confidence scores and 'Music' category unless it's the only one
            threshold = 0.15  # Minimum confidence threshold
            filtered_classifications = {
                k: v for k, v in classifications.items()
                if (v > threshold and k != 'Music') or (k == 'Music' and all(v2 <= threshold for k2, v2 in classifications.items() if k2 != 'Music'))
            }
            
            if not filtered_classifications:
                filtered_classifications = {'Music': 0.1}
            
            # Find the highest scoring type
            instrument_type = max(filtered_classifications.items(), key=lambda x: x[1])[0]
            
            # Get the corresponding color
            color = self.color_map.get(instrument_type, '#FFFFFF')  # Default to white if not found
            
            return instrument_type, color
            
        except Exception as e:
            raise Exception(f"Error processing audio file: {str(e)}. Make sure the file is a valid WAV file with sufficient length.")
