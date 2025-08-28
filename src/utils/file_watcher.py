from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import os
from typing import Callable

class AudioFileHandler(FileSystemEventHandler):
    """Handles file system events for audio files"""
    
    def __init__(self, callback: Callable[[str], None]):
        self.callback = callback
        
    def on_created(self, event):
        if not event.is_directory and event.src_path.lower().endswith('.wav'):
            self.callback(event.src_path)

class WatchdogManager:
    """Manages the file system watching functionality"""
    
    def __init__(self, watch_path: str, callback: Callable[[str], None]):
        self.watch_path = watch_path
        self.event_handler = AudioFileHandler(callback)
        self.observer = Observer()
        
    def start_watching(self):
        """Start watching the directory for new wav files"""
        self.observer.schedule(self.event_handler, self.watch_path, recursive=False)
        self.observer.start()
        
    def stop_watching(self):
        """Stop watching the directory"""
        self.observer.stop()
        self.observer.join()
