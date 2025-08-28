import os
import shutil
from pathlib import Path
from typing import Dict, Tuple

class FLStudioIntegrator:
    """Handles integration with FL Studio"""
    
    def __init__(self, fl_studio_path: str):
        self.fl_studio_path = fl_studio_path
        
    def process_and_move_file(self, source_path: str, instrument_type: str, color: str) -> None:
        """
        Process a file and move it to FL Studio directory with appropriate naming and color
        Args:
            source_path: Path to the source wav file
            instrument_type: Type of instrument detected
            color: Color code for the instrument
        """
        # Get the file name and create new name with instrument type
        file_name = Path(source_path).stem
        new_name = f"{instrument_type.lower()}_{file_name}.wav"
        
        # Create the destination path
        dest_path = os.path.join(self.fl_studio_path, new_name)
        
        # Copy the file to FL Studio directory
        shutil.copy2(source_path, dest_path)
        
        # Here you would typically interface with FL Studio's API to set the color
        # Since FL Studio doesn't have a direct Python API, you might need to:
        # 1. Create a configuration file that FL Studio reads
        # 2. Use Windows COM automation if on Windows
        # 3. Or provide instructions for manual color setting
        print(f"Moved file to: {dest_path}")
        print(f"Please set color to: {color} in FL Studio")
