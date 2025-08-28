import numpy as np
import resampy
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging

try:
    import tensorflow as tf
    import tensorflow_hub as hub
except ImportError as e:
    raise ImportError(
        "Failed to import TensorFlow or TensorFlow Hub. "
        "Please make sure you're using Python 3.10 or 3.11 and have installed "
        "the correct versions from requirements.txt in a virtual environment."
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
        
    def classify(self, waveform, sample_rate):
        """Classify the audio and return detected sound types and their confidence scores"""
        try:
            # Ensure waveform is float32 and in the correct range (-1 to 1)
            waveform = waveform.astype(np.float32)
            if waveform.max() > 1.0 or waveform.min() < -1.0:
                waveform = waveform / 32768.0  # Convert from int16
            
            # Ensure we have enough samples for resampling (at least 0.1 seconds)
            min_samples = int(sample_rate * 0.1)
            if len(waveform) < min_samples:
                raise ValueError(f"Audio is too short. Need at least {min_samples} samples.")
            
            # Resample if needed (YAMNet expects 16kHz)
            if sample_rate != 16000:
                waveform = resampy.resample(waveform, sample_rate, 16000)
            
            # Run the model, get the scores
            scores, embeddings, mel_spectrogram = self.model(waveform)
            scores = scores.numpy()
            
            # Get frame-level predictions
            frame_predictions = scores.argmax(axis=1)
            unique_predictions, counts = np.unique(frame_predictions, return_counts=True)
            
            # Load YAMNet class names
            class_map = self._load_class_map()
            
            # Initialize categories with specific mapping
            specific_mappings = {
                # Drums and Percussion
                'drum': ['Drums'],
                'snare': ['Snare drum'],
                'bass drum': ['Bass drum'],
                'cymbal': ['Cymbal'],
                'hi-hat': ['Hi-hat'],
                'tambourine': ['Percussion'],
                'bongo': ['Percussion'],
                'conga': ['Percussion'],
                
                # Guitar categories
                'electric guitar': ['Electric guitar'],
                'acoustic guitar': ['Acoustic guitar'],
                'bass guitar': ['Bass guitar'],
                'guitar': ['Guitar'],  # Generic guitar if not specific
                
                # Piano
                'piano': ['Piano'],
                'keyboard': ['Piano'],
                
                # Voice categories
                'speech': ['Speech'],
                'male voice': ['Male speech'],
                'female voice': ['Female speech'],
                'male speech': ['Male speech'],
                'female speech': ['Female speech'],
                'singing': ['Singing'],
                'vocal': ['Vocals'],
                'voice': ['Speech'],
                
                # Electronic
                'synthesizer': ['Synthesizer'],
                'synth': ['Synthesizer'],
                'electronic': ['Electronic'],
                'techno': ['Electronic'],
                'sample': ['Sample']
            }
            
            # Initialize sound classes
            sound_classes = {
                'Drums': 0.0, 'Snare drum': 0.0, 'Bass drum': 0.0,
                'Cymbal': 0.0, 'Hi-hat': 0.0, 'Percussion': 0.0,
                'Guitar': 0.0, 'Electric guitar': 0.0, 'Bass guitar': 0.0,
                'Acoustic guitar': 0.0, 'Piano': 0.0,
                'Synthesizer': 0.0, 'Electronic': 0.0, 'Sample': 0.0,
                'Speech': 0.0, 'Male speech': 0.0, 'Female speech': 0.0,
                'Singing': 0.0, 'Vocals': 0.0,
                'Music': 0.0
            }
            
            # Process each frame's predictions
            for pred_idx, count in zip(unique_predictions, counts):
                class_name = class_map[pred_idx].lower()
                confidence = count / len(frame_predictions)  # Normalize by total frames
                
                # Check specific mappings
                for key, categories in specific_mappings.items():
                    if key in class_name:
                        for category in categories:
                            sound_classes[category] = max(
                                sound_classes[category],
                                confidence * scores[:, pred_idx].mean()
                            )
            
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
