import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
from typing import Callable
import threading
from tkinter.scrolledtext import ScrolledText

class StemStampGUI:
    def __init__(self, process_callback: Callable[[str], None]):
        self.process_callback = process_callback
        self.root = tk.Tk()
        self.root.title("STEM STAMP")
        self.root.geometry("800x600")
        
        # Set FL Studio directory
        self.fl_studio_dir = ""
        
        self._create_widgets()
        
    def _create_widgets(self):
        # Create main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # FL Studio directory selection
        ttk.Label(main_frame, text="FL Studio Samples Directory:").grid(row=0, column=0, sticky=tk.W)
        self.fl_dir_label = ttk.Label(main_frame, text="Not selected", wraplength=500)
        self.fl_dir_label.grid(row=1, column=0, columnspan=2, sticky=tk.W)
        ttk.Button(main_frame, text="Select Directory", 
                   command=self._select_fl_directory).grid(row=0, column=1, sticky=tk.E)
        
        # File selection frame
        file_frame = ttk.LabelFrame(main_frame, text="WAV File Selection", padding="20")
        file_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        file_frame.grid_columnconfigure(0, weight=1)
        file_frame.grid_rowconfigure(0, weight=1)
        
        # File selection button
        select_button = ttk.Button(file_frame, 
                                text="Select WAV Files",
                                command=self._select_files)
        select_button.grid(row=0, column=0, pady=20)
        
        # Log area
        log_frame = ttk.LabelFrame(main_frame, text="Processing Log", padding="5")
        log_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        log_frame.grid_columnconfigure(0, weight=1)
        log_frame.grid_rowconfigure(0, weight=1)
        
        self.log_text = ScrolledText(log_frame, height=10, wrap=tk.WORD)
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_rowconfigure(2, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(0, weight=1)
        
    def _select_fl_directory(self):
        directory = filedialog.askdirectory(title="Select FL Studio Samples Directory")
        if directory:
            self.fl_studio_dir = directory
            self.fl_dir_label.config(text=directory)
            self.log("FL Studio directory set to: " + directory)
            
    def _select_files(self):
        files = filedialog.askopenfilenames(
            title="Select WAV Files",
            filetypes=[("WAV files", "*.wav")]
        )
        if files:
            self._process_files(files)
            
    def _process_files(self, files):
        if not self.fl_studio_dir:
            messagebox.showerror("Error", "Please select FL Studio samples directory first!")
            return
            
        for file in files:
            if not file.lower().endswith('.wav'):
                self.log(f"Skipping non-WAV file: {file}")
                continue
                
            self.log(f"Processing: {file}")
            # Process in a separate thread to keep GUI responsive
            threading.Thread(target=self._process_single_file,
                           args=(file,),
                           daemon=True).start()
                
    def _process_single_file(self, file):
        try:
            self.process_callback(file)
            self.log(f"Successfully processed: {os.path.basename(file)}")
        except Exception as e:
            self.log(f"Error processing {os.path.basename(file)}: {str(e)}")
            
    def log(self, message: str):
        """Add message to log area"""
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        
    def get_fl_studio_dir(self) -> str:
        """Return the selected FL Studio directory"""
        return self.fl_studio_dir
        
    def run(self):
        """Start the GUI main loop"""
        self.root.mainloop()
