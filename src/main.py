import os
import sys
# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.sound_classifier import SoundClassifier
from core.sound_processor import GeneralSoundProcessor
from utils.fl_studio_integrator import FLStudioIntegrator
from utils.gui import StemStampGUI

class StemStamp:
    """Main application class"""
    
    def __init__(self):
        # Initialize components
        self.classifier = SoundClassifier()
        self.processor = GeneralSoundProcessor(self.classifier)
        self.fl_studio = None
        self.gui = StemStampGUI(self.process_file)
        
    def process_file(self, file_path: str):
        """Process an audio file"""
        try:
            # Ensure FL Studio integrator is initialized with current directory
            if not self.fl_studio:
                fl_studio_path = self.gui.get_fl_studio_dir()
                if not fl_studio_path:
                    raise Exception("FL Studio directory not set")
                self.fl_studio = FLStudioIntegrator(fl_studio_path)
            
            # Process the sound file
            instrument_type, color = self.processor.process_file(file_path)
            
            # Move to FL Studio and apply color
            self.fl_studio.process_and_move_file(file_path, instrument_type, color)
            
            # Log the results through GUI
            self.gui.log(f"Detected instrument: {instrument_type}")
            self.gui.log(f"Assigned color: {color}")
            
        except Exception as e:
            self.gui.log(f"Error processing {file_path}: {str(e)}")
            raise e
    
    def run(self):
        """Start the application"""
        self.gui.run()

def main():
    # Create and start the application
    app = StemStamp()
    app.run()

if __name__ == "__main__":
    main()
