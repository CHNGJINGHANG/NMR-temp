# gui/video_tab.py
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import numpy as np
from typing import List, Dict, Any, Optional
import logging
import tkinter.simpledialog
# Import from core module
from core.Video_generator import SpectrumVideoGenerator
from core.ssh_utils import SSHConnection


class VideoTab:
    """GUI for 2D spectrum-to-video export"""
    
    def __init__(self, parent_notebook, main_app):
        self.parent = parent_notebook
        self.main_app = main_app
        self.logger = logging.getLogger(__name__)
        
        # Create the main frame (don't add to notebook)
        self.frame = ttk.Frame(parent_notebook)
        
        # State variables
        self.loaded_spectra = []
        self.loaded_metadata = []
        self.output_path = tk.StringVar(value="nmr_video.mp4")
        self.fps = tk.IntVar(value=24)
        self.progress_var = tk.DoubleVar(value=0.0)
        self.status_var = tk.StringVar(value="Ready")
        
        self._create_widgets()
        self._setup_layout()
            
    def _create_widgets(self):
        """Create all GUI widgets"""
        
        # File loading section
        self.file_frame = ttk.LabelFrame(self.frame, text="2D Spectrum Files", padding=10)
        
        self.load_btn = ttk.Button(
            self.file_frame, 
            text="Load 2D NPY Files", 
            command=self._load_npy_files
        )        
        
        # File list
        self.file_listbox = tk.Listbox(self.file_frame, height=6, selectmode=tk.EXTENDED)
        self.file_scrollbar = ttk.Scrollbar(self.file_frame, orient=tk.VERTICAL)
        self.file_listbox.config(yscrollcommand=self.file_scrollbar.set)
        self.file_scrollbar.config(command=self.file_listbox.yview)
        
        self.clear_files_btn = ttk.Button(
            self.file_frame,
            text="Clear Files",
            command=self._clear_files
        )
        
        self.remove_btn = ttk.Button(
            self.file_frame,
            text="Remove Selected",
            command=self._remove_selected_files
        )
        
        # Video settings section
        self.settings_frame = ttk.LabelFrame(self.frame, text="Video Settings", padding=10)
        
        # FPS setting - Row 0
        self.fps_label = ttk.Label(self.settings_frame, text="FPS:")
        self.fps_spin = tk.Spinbox(
            self.settings_frame,
            from_=1, to=120,
            textvariable=self.fps,
            width=10
        )
        
        # Output path - Row 1
        self.output_label = ttk.Label(self.settings_frame, text="Output File:")
        self.output_entry = ttk.Entry(self.settings_frame, textvariable=self.output_path, width=30)
        self.browse_btn = ttk.Button(
            self.settings_frame,
            text="Browse...",
            command=self._browse_output_file
        )
        
        # Save frames folder option - Row 2
        self.frames_label = ttk.Label(self.settings_frame, text="Save Frames To:")
        self.frames_folder_var = tk.StringVar(value="")
        self.frames_folder_entry = ttk.Entry(self.settings_frame, textvariable=self.frames_folder_var, width=25)
        self.frames_browse_btn = ttk.Button(
            self.settings_frame, 
            text="Browse...", 
            command=self._browse_frames_folder
        )
        
        # Export section
        self.export_frame = ttk.LabelFrame(self.frame, text="Export", padding=10)

        self.test_btn = ttk.Button(
            self.export_frame,
            text="Test Environment",
            command=self._test_video_environment
        )
                
        self.export_btn = ttk.Button(
            self.export_frame,
            text="Export Video",
            command=self._export_video
        )

                
        # Progress section
        self.progress_frame = ttk.Frame(self.export_frame)
        
        self.progress_bar = ttk.Progressbar(
            self.progress_frame,
            variable=self.progress_var,
            maximum=100,
            length=300
        )
        
        self.status_label = ttk.Label(self.progress_frame, textvariable=self.status_var)
        
        self.cancel_btn = ttk.Button(
            self.progress_frame,
            text="Cancel",
            state=tk.DISABLED,
            command=self._cancel_export
        )
        
    def _setup_layout(self):
        """Setup widget layout"""
        
        # File loading section
        self.file_frame.pack(fill=tk.X, padx=10, pady=5)
        
        button_frame = ttk.Frame(self.file_frame)
        button_frame.pack(fill=tk.X, pady=(0, 10))
        self.load_btn.pack(side=tk.LEFT, padx=(0, 5))
        self.clear_files_btn.pack(side=tk.RIGHT)
        self.remove_btn.pack(side=tk.RIGHT, padx=(5, 0))
        
        # File listbox with scrollbar
        list_frame = ttk.Frame(self.file_frame)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        self.file_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.file_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Settings section
        self.settings_frame.pack(fill=tk.X, padx=10, pady=5)

        # FPS setting - Row 0
        self.fps_label.grid(row=0, column=0, sticky=tk.W, pady=2, padx=(0, 10))
        self.fps_spin.grid(row=0, column=1, sticky=tk.W, pady=2)

        # Output file - Row 1
        self.output_label.grid(row=1, column=0, sticky=tk.W, pady=2, padx=(0, 10))
        self.output_entry.grid(row=1, column=1, sticky=tk.W, pady=2, padx=(0, 5))
        self.browse_btn.grid(row=1, column=2, pady=2)

        # Frames folder - Row 2
        self.frames_label.grid(row=2, column=0, sticky=tk.W, pady=2, padx=(0, 10))
        self.frames_folder_entry.grid(row=2, column=1, sticky=tk.W, pady=2, padx=(0, 5))
        self.frames_browse_btn.grid(row=2, column=2, pady=2)
        
        # Export section
        self.export_frame.pack(fill=tk.X, padx=10, pady=5)
        
        button_frame = ttk.Frame(self.export_frame)
        button_frame.pack(pady=(0, 10))

        self.test_btn.pack(side=tk.LEFT, padx=(0, 10))
        self.export_btn.pack(side=tk.LEFT)

        # Progress section
        self.progress_frame.pack(fill=tk.X)
        self.progress_bar.pack(fill=tk.X, pady=(0, 5))
        
        status_frame = ttk.Frame(self.progress_frame)
        status_frame.pack(fill=tk.X)
        
        self.status_label.pack(side=tk.LEFT)
        self.cancel_btn.pack(side=tk.RIGHT)
        
    def _load_npy_files(self):
        """Load 2D NPY spectrum files"""
        file_paths = filedialog.askopenfilenames(
            title="Select 2D NPY Spectrum Files",
            filetypes=[("NumPy files", "*.npy"), ("All files", "*.*")]
        )
        
        if file_paths:
            try:
                with SpectrumVideoGenerator() as generator:
                    new_spectra, filenames = generator.load_2d_spectra_from_npy(file_paths)
                
                # Add to existing lists
                for i, (spectrum, filename) in enumerate(zip(new_spectra, filenames)):
                    enhanced_metadata = {
                        'title': f'2D: {filename}',
                        'source': 'video_load',
                        'path': file_paths[i],
                        'filename': filename
                    }
                    self.loaded_spectra.append(spectrum)
                    self.loaded_metadata.append(enhanced_metadata)
                
                # Update file list display
                self.file_listbox.delete(0, tk.END)
                for i, metadata in enumerate(self.loaded_metadata):
                    dimensions = f"{self.loaded_spectra[i].shape[0]}×{self.loaded_spectra[i].shape[1]}"
                    self.file_listbox.insert(tk.END, f"{i+1}. {metadata['filename']} [{dimensions}]")
                
                self.status_var.set(f"Total: {len(self.loaded_spectra)} 2D spectra loaded")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load files: {str(e)}")
                self.logger.error(f"Failed to load NPY files: {e}")
    
    def _clear_files(self):
        """Clear loaded files"""
        self.file_listbox.delete(0, tk.END)
        self.loaded_spectra.clear()
        self.loaded_metadata.clear()
        self.status_var.set("Ready")

    def _remove_selected_files(self):
        """Remove selected files from the list"""
        selected_indices = self.file_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("Warning", "Please select files to remove")
            return
        
        # Remove in reverse order to maintain indices
        for idx in reversed(selected_indices):
            self.loaded_spectra.pop(idx)
            self.loaded_metadata.pop(idx)
        
        # Update file list display
        self.file_listbox.delete(0, tk.END)
        for i, metadata in enumerate(self.loaded_metadata):
            dimensions = f"{self.loaded_spectra[i].shape[0]}×{self.loaded_spectra[i].shape[1]}"
            self.file_listbox.insert(tk.END, f"{i+1}. {metadata['filename']} [{dimensions}]")
        
        self.status_var.set(f"Total: {len(self.loaded_spectra)} 2D spectra loaded")
    
    def _browse_output_file(self):
        """Browse for output file location"""
        filename = filedialog.asksaveasfilename(
            title="Save Video As",
            defaultextension=".mp4",
            filetypes=[("MP4 files", "*.mp4"), ("AVI files", "*.avi"), ("All files", "*.*")]
        )
        
        if filename:
            self.output_path.set(filename)

    def _browse_frames_folder(self):
        """Browse for frames output folder"""
        folder = filedialog.askdirectory(title="Select Folder to Save Frame Images")
        if folder:
            self.frames_folder_var.set(folder)            
    
    def _export_video(self):
        """Export 2D heatmap video"""
        if not self.loaded_spectra:
            messagebox.showwarning("Warning", "Please load 2D spectrum files first")
            return
        
        if not self.output_path.get():
            messagebox.showwarning("Warning", "Please specify output file path")
            return
        
        # Check all spectra are 2D
        non_2d_count = sum(1 for spec in self.loaded_spectra if spec.ndim != 2)
        if non_2d_count > 0:
            messagebox.showerror("Error", f"{non_2d_count} spectra are not 2D. Please load only 2D spectra.")
            return
        
        # Disable export button
        self.export_btn.config(state=tk.DISABLED)
        self.progress_var.set(0)
        self.status_var.set("Creating video...")
        self.frame.update()
        
        try:
            # Create titles from metadata
            titles = [metadata['title'] for metadata in self.loaded_metadata]
            
            # Get frames folder if specified
            frames_folder = self.frames_folder_var.get() if self.frames_folder_var.get() else None
            
            # Create video
            with SpectrumVideoGenerator() as generator:
                success = generator.create_2d_heatmap_video(
                    spectra_2d=self.loaded_spectra,
                    output_path=self.output_path.get(),
                    fps=self.fps.get(),
                    titles=titles,
                    colormap='gray',  # Fixed to grayscale
                    save_frames_folder=frames_folder
                )                
                
                self._export_completed(success)
                
        except Exception as e:
            self.logger.error(f"Export failed: {e}")
            self._export_failed(str(e))
    
    def _export_completed(self, success: bool):
        """Handle export completion"""
        self.export_btn.config(state=tk.NORMAL)
        self.cancel_btn.config(state=tk.DISABLED)
        self.progress_var.set(100)
        
        if success:
            self.status_var.set("Video export completed successfully!")
            messagebox.showinfo("Success", f"Video saved to: {self.output_path.get()}")
        else:
            self.status_var.set("Video export failed")
            messagebox.showerror("Error", "Failed to create video")
    
    def _export_failed(self, error_msg: str):
        """Handle export failure"""
        self.export_btn.config(state=tk.NORMAL)
        self.cancel_btn.config(state=tk.DISABLED)
        self.progress_var.set(0)
        self.status_var.set("Export failed")
        messagebox.showerror("Error", f"Export failed: {error_msg}")
    
    def _cancel_export(self):
        """Cancel ongoing export"""
        self.status_var.set("Cancelling...")
        messagebox.showinfo("Info", "Export cancellation requested")

    def _test_video_environment(self):
        """Test FFmpeg environment"""
        try:
            from core.Video_generator import diagnose_ffmpeg_environment
            diagnose_ffmpeg_environment()
        except Exception as e:
            messagebox.showerror("Diagnostic Error", f"Failed to run diagnostics: {e}")


