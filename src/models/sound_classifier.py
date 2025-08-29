import numpy as np
import resampy
import os
import librosa
from scipy import stats
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging

try:
    import tensorflow as tf
    import tensorflow_hub as hub
    import crepe
except ImportError as e:
    raise ImportError(
        "Failed to import required packages. "
        "Please make sure you have installed the correct versions from requirements.txt"
    ) from e

class SoundClassifier:
    """Strategy pattern for sound classification"""
    def __init__(self):
        self.model = None
        tf.get_logger().setLevel('ERROR')  # Reduce TensorFlow logging
        self._load_model()
        
    def _load_model(self):
        """Load YAMNet model"""
        try:
            self.model = hub.load('https://tfhub.dev/google/yamnet/1')
        except Exception as e:
            raise Exception(
                "Failed to load YAMNet model. Please check your internet connection "
                "and make sure you have the correct versions of TensorFlow and TensorFlow Hub."
            ) from e
        
    def _initialize_sound_classes(self):
        """Initialize the dictionary of sound classes with zero confidence"""
        return {
            # Drums and Percussion
            'Drums': 0.0, 'Snare drum': 0.0, 'Bass drum': 0.0,
            'Cymbal': 0.0, 'Hi-hat': 0.0, 'Percussion': 0.0,
            # String instruments
            'Guitar': 0.0, 'Electric guitar': 0.0, 'Bass guitar': 0.0,
            'Acoustic guitar': 0.0, 'Piano': 0.0,
            # Wind instruments
            'Saxophone': 0.0, 'Trumpet': 0.0, 'Flute': 0.0, 'Clarinet': 0.0,
            # Electronic
            'Synthesizer': 0.0, 'Electronic': 0.0, 'Sample': 0.0,
            # Voice
            'Speech': 0.0, 'Male speech': 0.0, 'Female speech': 0.0,
            'Singing': 0.0, 'Vocals': 0.0,
            # Generic
            'Music': 0.0
        }

    def classify(self, waveform, sample_rate):
        """Classify the audio using multiple analysis techniques"""
        try:
            # Ensure waveform is float32 and in the correct range (-1 to 1)
            waveform = waveform.astype(np.float32)
            if waveform.max() > 1.0 or waveform.min() < -1.0:
                waveform = waveform / 32768.0

            # Get initial YAMNet classification
            if sample_rate != 16000:
                yamnet_waveform = resampy.resample(waveform, sample_rate, 16000)
            else:
                yamnet_waveform = waveform

            # Run YAMNet model
            scores, embeddings, mel_spectrogram = self.model(yamnet_waveform)
            yamnet_scores = scores.numpy().mean(axis=0)

            # Initialize sound classes
            sound_classes = self._initialize_sound_classes()

            # Extract audio features
            spectral_centroids = librosa.feature.spectral_centroid(y=waveform, sr=sample_rate)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=waveform, sr=sample_rate)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=waveform, sr=sample_rate)[0]
            zero_crossing = librosa.feature.zero_crossing_rate(waveform)[0]
            mfcc = librosa.feature.mfcc(y=waveform, sr=sample_rate, n_mfcc=13)

            # Analyze spectral features for instrument identification
            spec_mean = np.mean(spectral_centroids)
            spec_std = np.std(spectral_centroids)
            zcr_mean = np.mean(zero_crossing)
            
            # Detect percussion instruments
            if zcr_mean > 0.1 and spec_mean > 2000:
                if spec_mean > 5000:
                    sound_classes['Cymbal'] = 0.9
                    sound_classes['Hi-hat'] = 0.8
                elif spec_mean < 1000:
                    sound_classes['Bass drum'] = 0.9
                else:
                    sound_classes['Snare drum'] = 0.85
                sound_classes['Drums'] = 0.9

            # Detect guitars
            if 500 < spec_mean < 3000 and spec_std > 500:
                if np.mean(spectral_bandwidth) > 2000:
                    sound_classes['Electric guitar'] = 0.9
                else:
                    sound_classes['Acoustic guitar'] = 0.9
                sound_classes['Guitar'] = 0.85

            # Detect vocals
            mfcc_std = np.std(mfcc, axis=1)
            if np.mean(mfcc_std[1:5]) > 15:
                sound_classes['Vocals'] = 0.9
                if spec_mean > 1800:
                    sound_classes['Female speech'] = 0.8
                else:
                    sound_classes['Male speech'] = 0.8

            # Detect electronic sounds
            if zcr_mean > 0.2 and spec_std < 500:
                sound_classes['Electronic'] = 0.85
                sound_classes['Synthesizer'] = 0.8

            # Combine with YAMNet predictions
            class_map = self._load_class_map()
            for i, class_name in enumerate(class_map):
                score = yamnet_scores[i]
                if score > 0.3:
                    class_name = class_name.lower()
                    for sound_type in sound_classes.keys():
                        if sound_type.lower() in class_name:
                            sound_classes[sound_type] = max(sound_classes[sound_type], score)

            # Only use Music category if no other instrument is detected with confidence
            if any(v > 0.3 for k, v in sound_classes.items() if k != 'Music'):
                sound_classes['Music'] = 0.0

            return sound_classes

        except Exception as e:
            raise Exception(f"Error classifying audio: {str(e)}")

    def _load_class_map(self):
        """Load YAMNet class names"""
        try:
            import csv
            import urllib.request
            
            class_map_url = "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv"
            response = urllib.request.urlopen(class_map_url)
            lines = [l.decode('utf-8') for l in response.readlines()]
            reader = csv.reader(lines)
            next(reader)  # Skip header
            return [row[2] for row in reader]  # Return display names
        except Exception:
            # Fallback to basic class names if unable to load
            return [
                "Speech", "Male speech", "Female speech", "Vocals", "Singing",
                "Drums", "Snare drum", "Bass drum", "Hi-hat", "Cymbal",
                "Guitar", "Electric guitar", "Acoustic guitar", "Piano",
                "Synthesizer", "Electronic music", "Music"
            ]
            
            # Post-processing to improve detection
            # 1. If we detect electric guitar with high confidence, boost related categories
            if sound_classes['Electric guitar'] > 0.3:
                sound_classes['Guitar'] = max(sound_classes['Guitar'], sound_classes['Electric guitar'] * 0.8)
            
            # 2. If we detect any percussion with high confidence, boost Drums category
            percussion_confidence = max(
                sound_classes['Snare drum'],
                sound_classes['Bass drum'],
                sound_classes['Cymbal'],
                sound_classes['Hi-hat'],
                sound_classes['Percussion']
            )
            if percussion_confidence > 0.3:
                sound_classes['Drums'] = max(sound_classes['Drums'], percussion_confidence * 0.8)
            
            # 3. Voice detection improvements
            if sound_classes['Singing'] > 0.3:
                sound_classes['Vocals'] = max(sound_classes['Vocals'], sound_classes['Singing'] * 0.9)
            
            # Only use Music as fallback if no other category has significant confidence
            significant_detection = any(score > 0.15 for category, score in sound_classes.items() if category != 'Music')
            if not significant_detection:
                sound_classes['Music'] = 0.1
            else:
                sound_classes['Music'] = 0.0  # Zero out generic music if we have specific detections
                
            return sound_classes
            
        except Exception as e:
            raise Exception(f"Error classifying audio: {str(e)}")
            
        except Exception as e:
            raise Exception(f"Error classifying audio: {str(e)}")
            
    def _load_class_map(self):
        """Load YAMNet class names"""
        try:
            import csv
            import tensorflow_hub as hub
            import urllib.request
            
            # YAMNet class map URL
            class_map_url = "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv"
            
            # Download and read the class map
            response = urllib.request.urlopen(class_map_url)
            lines = [l.decode('utf-8') for l in response.readlines()]
            reader = csv.reader(lines)
            next(reader)  # Skip header
            
            return [row[2] for row in reader]  # Return display names
            
        except Exception as e:
            # Fallback to a minimal set of drum-related classes if unable to load
            return [
                "Drums", "Snare drum", "Bass drum", "Drum kit",
                "Tabla", "Cymbal", "Hi-hat", "Bass drum", "Percussion"
            ]
