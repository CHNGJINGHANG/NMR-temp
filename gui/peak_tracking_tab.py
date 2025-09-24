"""
Enhanced Peak Tracking Tab for NMR Data Analysis
Now powered by YOLOv11 + SAHI for AI-based peak detection and tracking
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import os
import json
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import pandas as pd
from threading import Thread
from core.yolo_peak_tracker import YOLOv11PeakTracker, AIPeak


class PeakTrackingTab:
    def __init__(self, parent, main_app):
        self.parent = parent
        self.main_app = main_app
        self.yolo_tracker = None
        self.loaded_sequences = {}  # Store sequences of NPY files
        self.current_data = None
        self.current_peaks = []
        self.tracking_results = None
        self.model_path = None
        
        # Create the main frame
        self.frame = ttk.Frame(parent)
        self.create_widgets()
    
    def create_widgets(self):
        """Create all widgets for the peak tracking tab"""
        # Title
        title_frame = ttk.Frame(self.frame)
        title_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(title_frame, text="AI-Powered NMR Peak Detection & Tracking (YOLOv11 + SAHI)", 
                 font=('Arial', 14, 'bold')).pack()
        
        # Model configuration section
        self.create_model_section()
        
        # File loading section
        self.create_file_section()
        
        # AI Parameters section
        self.create_ai_parameter_section()
        
        # Processing section
        self.create_processing_section()
        
        # Results section
        self.create_results_section()
    
    def create_model_section(self):
        """Create AI model configuration section"""
        model_frame = ttk.LabelFrame(self.frame, text="YOLOv11 Model Configuration")
        model_frame.pack(fill=tk.X, padx=10, pady=5)
        
        button_frame = ttk.Frame(model_frame)
        button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(button_frame, text="Load Custom Model", 
                  command=self.load_custom_model).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Use Default YOLOv11n", 
                  command=self.use_default_model).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Initialize Tracker", 
                  command=self.initialize_tracker).pack(side=tk.LEFT, padx=5)
        
        self.model_info_label = ttk.Label(button_frame, text="No model loaded")
        self.model_info_label.pack(side=tk.LEFT, padx=20)
        
        # Device selection
        device_frame = ttk.Frame(model_frame)
        device_frame.pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Label(device_frame, text="Device:").pack(side=tk.LEFT, padx=5)
        self.device_var = tk.StringVar(value="auto")
        device_combo = ttk.Combobox(device_frame, textvariable=self.device_var,
                                  values=["auto", "cpu", "cuda"], width=10)
        device_combo.pack(side=tk.LEFT, padx=5)
        device_combo.state(['readonly'])
    
    def create_file_section(self):
        """Create file loading section"""
        file_frame = ttk.LabelFrame(self.frame, text="Data Loading")
        file_frame.pack(fill=tk.X, padx=10, pady=5)
        
        button_frame = ttk.Frame(file_frame)
        button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(button_frame, text="Load Single NPY", 
                  command=self.load_single_npy).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Load NPY Sequence", 
                  command=self.load_npy_sequence).pack(side=tk.LEFT, padx=5)
        
        self.file_info_label = ttk.Label(button_frame, text="No files loaded")
        self.file_info_label.pack(side=tk.LEFT, padx=20)
    
    def create_ai_parameter_section(self):
        """Create AI detection parameter section"""
        param_frame = ttk.LabelFrame(self.frame, text="AI Detection Parameters")
        param_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Create parameter controls in a grid
        controls_frame = ttk.Frame(param_frame)
        controls_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Confidence threshold
        ttk.Label(controls_frame, text="Confidence Threshold:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.confidence_var = tk.DoubleVar(value=0.3)
        confidence_scale = ttk.Scale(controls_frame, from_=0.1, to=0.9, 
                                   variable=self.confidence_var, orient=tk.HORIZONTAL)
        confidence_scale.grid(row=0, column=1, sticky=tk.EW, padx=5)
        self.confidence_label = ttk.Label(controls_frame, text="0.3")
        self.confidence_label.grid(row=0, column=2, padx=5)
        confidence_scale.configure(command=self.update_confidence_label)
        
        # IoU threshold
        ttk.Label(controls_frame, text="IoU Threshold:").grid(row=1, column=0, sticky=tk.W, padx=5)
        self.iou_var = tk.DoubleVar(value=0.5)
        iou_scale = ttk.Scale(controls_frame, from_=0.1, to=0.9, 
                              variable=self.iou_var, orient=tk.HORIZONTAL)
        iou_scale.grid(row=1, column=1, sticky=tk.EW, padx=5)
        self.iou_label = ttk.Label(controls_frame, text="0.5")
        self.iou_label.grid(row=1, column=2, padx=5)
        iou_scale.configure(command=self.update_iou_label)
        
        # Max disappeared frames
        ttk.Label(controls_frame, text="Max Missing Frames:").grid(row=2, column=0, sticky=tk.W, padx=5)
        self.max_frames_var = tk.IntVar(value=3)
        frames_scale = ttk.Scale(controls_frame, from_=1, to=10, 
                                variable=self.max_frames_var, orient=tk.HORIZONTAL)
        frames_scale.grid(row=2, column=1, sticky=tk.EW, padx=5)
        self.frames_label = ttk.Label(controls_frame, text="3")
        self.frames_label.grid(row=2, column=2, padx=5)
        frames_scale.configure(command=self.update_frames_label)
        
        # SAHI slice size
        ttk.Label(controls_frame, text="SAHI Slice Size:").grid(row=3, column=0, sticky=tk.W, padx=5)
        self.slice_size_var = tk.IntVar(value=512)
        slice_combo = ttk.Combobox(controls_frame, textvariable=self.slice_size_var,
                                  values=[256, 512, 640, 768, 1024], width=10)
        slice_combo.grid(row=3, column=1, sticky=tk.EW, padx=5)
        slice_combo.state(['readonly'])
        
        # Configure grid weights
        controls_frame.columnconfigure(1, weight=1)
    
    def create_processing_section(self):
        """Create processing controls section"""
        process_frame = ttk.LabelFrame(self.frame, text="AI Processing")
        process_frame.pack(fill=tk.X, padx=10, pady=5)
        
        button_frame = ttk.Frame(process_frame)
        button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(button_frame, text="Detect Peaks (Single Frame)", 
                  command=self.detect_peaks_ai).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Track Full Sequence", 
                  command=self.track_sequence_ai).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Export Results", 
                  command=self.export_results).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Generate Report", 
                  command=self.generate_analysis_report).pack(side=tk.LEFT, padx=5)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(button_frame, variable=self.progress_var, 
                                          maximum=100)
        self.progress_bar.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(20, 5))
        
        # Status label
        self.status_label = ttk.Label(process_frame, text="Ready")
        self.status_label.pack(pady=2)
    
    def create_results_section(self):
        """Create results display section"""
        results_frame = ttk.LabelFrame(self.frame, text="AI Tracking Results")
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Create notebook for different result views
        results_notebook = ttk.Notebook(results_frame)
        results_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Visualization tab
        viz_frame = ttk.Frame(results_notebook)
        results_notebook.add(viz_frame, text="Visualization")
        
        # Create matplotlib figure
        self.fig = Figure(figsize=(12, 8), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Peak Statistics tab
        stats_frame = ttk.Frame(results_notebook)
        results_notebook.add(stats_frame, text="Peak Statistics")
        
        # Create treeview for peak statistics
        columns = ('Peak ID', 'Trajectory Length', 'Avg Confidence', 'Avg Intensity', 'Total Displacement')
        self.stats_tree = ttk.Treeview(stats_frame, columns=columns, show='headings')
        
        for col in columns:
            self.stats_tree.heading(col, text=col)
            self.stats_tree.column(col, width=120)
        
        stats_scroll = ttk.Scrollbar(stats_frame, orient=tk.VERTICAL, 
                                   command=self.stats_tree.yview)
        self.stats_tree.configure(yscrollcommand=stats_scroll.set)
        
        self.stats_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        stats_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Analysis tab
        analysis_frame = ttk.Frame(results_notebook)
        results_notebook.add(analysis_frame, text="Analysis")
        
        # Text widget for analysis results
        self.analysis_text = tk.Text(analysis_frame, wrap=tk.WORD, font=('Courier', 10))
        analysis_scroll = ttk.Scrollbar(analysis_frame, orient=tk.VERTICAL, 
                                       command=self.analysis_text.yview)
        self.analysis_text.configure(yscrollcommand=analysis_scroll.set)
        
        self.analysis_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        analysis_scroll.pack(side=tk.RIGHT, fill=tk.Y)
    
    # Label update functions
    def update_confidence_label(self, value):
        self.confidence_label.config(text=f"{float(value):.2f}")
    
    def update_iou_label(self, value):
        self.iou_label.config(text=f"{float(value):.2f}")
    
    def update_frames_label(self, value):
        self.frames_label.config(text=str(int(float(value))))
    
    # Model management functions
    def load_custom_model(self):
        """Load custom trained YOLOv11 model"""
        filepath = filedialog.askopenfilename(
            title="Select YOLOv11 model file",
            filetypes=[("PyTorch files", "*.pt"), ("All files", "*.*")]
        )
        
        if filepath:
            self.model_path = filepath
            filename = os.path.basename(filepath)
            self.model_info_label.config(text=f"Custom model: {filename}")
    
    def use_default_model(self):
        """Use default YOLOv11 nano model"""
        self.model_path = None
        self.model_info_label.config(text="Using YOLOv11n (default)")
    
    def initialize_tracker(self):
        """Initialize the YOLOv11 tracker"""
        try:
            self.status_label.config(text="Initializing AI tracker...")
            self.frame.update()
            
            device = self.device_var.get()
            self.yolo_tracker = YOLOv11PeakTracker(
                model_path=self.model_path,
                device=device
            )
            
            # Update tracker parameters
            self.yolo_tracker.confidence_threshold = self.confidence_var.get()
            self.yolo_tracker.iou_threshold = self.iou_var.get()
            self.yolo_tracker.max_disappeared_frames = self.max_frames_var.get()
            
            self.model_info_label.config(text=f"✓ Tracker ready ({device})")
            self.status_label.config(text="AI tracker initialized successfully")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to initialize tracker: {str(e)}")
            self.status_label.config(text="Tracker initialization failed")
    
    def load_single_npy(self):
        """Load single NPY file for peak detection"""
        filepath = filedialog.askopenfilename(
            title="Select NPY file",
            filetypes=[("NumPy files", "*.npy"), ("All files", "*.*")]
        )
        
        if filepath:
            try:
                self.current_data = np.load(filepath)
                filename = os.path.basename(filepath)
                self.file_info_label.config(text=f"Loaded: {filename} - Shape: {self.current_data.shape}")
                
                # Display the data
                self.display_data()
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file: {str(e)}")
    
    def load_npy_sequence(self):
        """Load sequence of NPY files for tracking"""
        folder_path = filedialog.askdirectory(title="Select folder containing NPY sequence")
        
        if folder_path:
            try:
                npy_files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]
                npy_files.sort()  # Sort to ensure proper sequence
                
                if not npy_files:
                    messagebox.showwarning("Warning", "No NPY files found in selected folder")
                    return
                
                self.loaded_sequences = {}
                for i, filename in enumerate(npy_files):
                    filepath = os.path.join(folder_path, filename)
                    self.loaded_sequences[i] = np.load(filepath)
                
                self.file_info_label.config(text=f"Loaded sequence: {len(npy_files)} files")
                
                # Display first frame
                if self.loaded_sequences:
                    self.current_data = self.loaded_sequences[0]
                    self.display_data()
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load sequence: {str(e)}")
    
    def display_data(self):
        """Display the current NMR data"""
        if self.current_data is not None:
            self.fig.clear()
            ax = self.fig.add_subplot(111)
            
            # Display the data as an image
            im = ax.imshow(np.abs(self.current_data), aspect='auto', origin='lower', cmap='viridis')
            ax.set_title('NMR Data')
            ax.set_xlabel('Chemical Shift')
            ax.set_ylabel('Time/Frame')
            self.fig.colorbar(im, ax=ax)
            
            self.canvas.draw()
    
    def detect_peaks_ai(self):
        """Detect peaks in current frame using YOLOv11 + SAHI"""
        if self.yolo_tracker is None:
            messagebox.showwarning("Warning", "Please initialize the AI tracker first")
            return
        
        if self.current_data is None:
            messagebox.showwarning("Warning", "Please load data first")
            return
        
        try:
            self.status_label.config(text="Running AI detection...")
            self.frame.update()
            
            # Update tracker parameters
            self.yolo_tracker.confidence_threshold = self.confidence_var.get()
            
            # Detect peaks
            self.current_peaks = self.yolo_tracker.detect_peaks_frame(
                self.current_data, frame_id=0
            )
            
            # Update visualization
            self.update_peak_visualization()
            
            # Update statistics for single frame
            self.update_single_frame_statistics()
            
            self.status_label.config(text=f"AI detection complete: {len(self.current_peaks)} peaks found")
            messagebox.showinfo("Success", f"AI detected {len(self.current_peaks)} peaks")
            
        except Exception as e:
            messagebox.showerror("Error", f"AI detection failed: {str(e)}")
            self.status_label.config(text="AI detection failed")
    
    def track_sequence_ai(self):
        """Track peaks across sequence using YOLOv11"""
        if self.yolo_tracker is None:
            messagebox.showwarning("Warning", "Please initialize the AI tracker first")
            return
        
        if not self.loaded_sequences:
            messagebox.showwarning("Warning", "Please load a sequence first")
            return
        
        def track_thread():
            try:
                self.status_label.config(text="Starting AI tracking...")
                
                # Convert loaded sequences to list format
                nmr_frames = [self.loaded_sequences[i] for i in sorted(self.loaded_sequences.keys())]
                
                # Update tracker parameters
                self.yolo_tracker.confidence_threshold = self.confidence_var.get()
                self.yolo_tracker.iou_threshold = self.iou_var.get()
                self.yolo_tracker.max_disappeared_frames = self.max_frames_var.get()
                
                # Process sequence with progress updates
                total_frames = len(nmr_frames)
                
                # Override the progress update in the tracker
                original_process = self.yolo_tracker.process_nmr_sequence
                
                def process_with_progress(frames):
                    results = {'frames': {}, 'trajectories': {}, 'statistics': {}}
                    
                    for i, frame_data in enumerate(frames):
                        self.progress_var.set((i / total_frames) * 100)
                        self.status_label.config(text=f"Processing frame {i+1}/{total_frames}")
                        self.frame.update()
                        
                        # Detect and track peaks
                        detected_peaks = self.yolo_tracker.detect_peaks_frame(frame_data, i)
                        tracked_peaks = self.yolo_tracker.track_peaks_ai(detected_peaks, i)
                        
                        # Store results
                        self.yolo_tracker.peaks_history[i] = tracked_peaks
                        results['frames'][i] = [peak.__dict__ for peak in tracked_peaks]
                    
                    # Generate final results
                    results['trajectories'] = self.yolo_tracker._generate_trajectory_data()
                    results['statistics'] = self.yolo_tracker._calculate_tracking_statistics()
                    
                    return results
                
                # Run tracking
                self.tracking_results = process_with_progress(nmr_frames)
                
                # Update GUI in main thread
                self.frame.after(0, self.update_tracking_results)
                
            except Exception as e:
                self.frame.after(0, lambda: self.handle_tracking_error(str(e)))
        
        # Start tracking in separate thread
        Thread(target=track_thread, daemon=True).start()
    
    def update_tracking_results(self):
        """Update GUI with tracking results"""
        try:
            # Update visualization
            self.update_trajectory_visualization()
            
            # Update statistics
            self.update_trajectory_statistics()
            
            # Update analysis
            self.update_analysis_text()
            
            # Reset progress
            self.progress_var.set(0)
            total_peaks = self.tracking_results['statistics']['total_unique_peaks']
            self.status_label.config(text=f"AI tracking complete: {total_peaks} unique peaks tracked")
            
            messagebox.showinfo("Success", f"AI tracking completed successfully!\n"
                                         f"Tracked {total_peaks} unique peaks across sequence")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to update results: {str(e)}")
    
    def handle_tracking_error(self, error_msg):
        """Handle tracking errors in main thread"""
        self.progress_var.set(0)
        self.status_label.config(text="AI tracking failed")
        messagebox.showerror("Error", f"AI tracking failed: {error_msg}")
    
    def update_peak_visualization(self):
        """Update visualization with detected peaks (single frame)"""
        if self.current_data is None or not self.current_peaks:
            return
        
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        
        # Display the data
        im = ax.imshow(np.abs(self.current_data), aspect='auto', origin='lower', cmap='viridis')
        
        # Overlay detected peaks
        for peak in self.current_peaks:
            centroid = peak.centroid
            bbox = peak.bbox
            
            # Draw bounding box
            rect = plt.Rectangle((bbox['x1'], bbox['y1']), 
                               bbox['x2'] - bbox['x1'], bbox['y2'] - bbox['y1'],
                               linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            
            # Draw centroid
            ax.plot(centroid[0], centroid[1], 'ro', markersize=6)
            
            # Add confidence score
            ax.text(centroid[0] + 5, centroid[1], f'{peak.confidence:.2f}', 
                   color='red', fontsize=8, fontweight='bold')
        
        ax.set_title(f'AI Peak Detection: {len(self.current_peaks)} peaks (YOLOv11 + SAHI)')
        ax.set_xlabel('Chemical Shift')
        ax.set_ylabel('Time/Frame')
        self.fig.colorbar(im, ax=ax)
        
        self.canvas.draw()
    
    def update_trajectory_visualization(self):
        """Update visualization with trajectory results"""
        if not self.tracking_results:
            return
        
        self.fig.clear()
        
        # Create subplots for comprehensive visualization
        gs = self.fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # Plot 1: Peak trajectories
        ax1 = self.fig.add_subplot(gs[0, :])
        trajectories = self.tracking_results['trajectories']
        
        # Sample up to 20 trajectories for clarity
        sample_trajs = list(trajectories.items())[:20]
        colors = plt.cm.tab20(np.linspace(0, 1, len(sample_trajs)))
        
        for i, (traj_id, traj_info) in enumerate(sample_trajs):
            if len(traj_info['positions']) > 1:
                positions = traj_info['positions']
                x_coords = [pos[0] for pos in positions]
                y_coords = [pos[1] for pos in positions]
                ax1.plot(x_coords, y_coords, 'o-', color=colors[i], 
                        label=f'Peak {traj_id[:4]}', markersize=3, linewidth=1.5)
        
        ax1.set_xlabel('X Position')
        ax1.set_ylabel('Y Position')
        ax1.set_title('Peak Trajectories (Sample)')
        ax1.grid(True, alpha=0.3)
        if sample_trajs:
            ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
        
        # Plot 2: Peak count per frame
        ax2 = self.fig.add_subplot(gs[1, 0])
        frame_ids = sorted([int(fid) for fid in self.tracking_results['frames'].keys()])
        peak_counts = [len(self.tracking_results['frames'][str(fid)]) for fid in frame_ids]
        
        ax2.plot(frame_ids, peak_counts, 'b-o', markersize=4)
        ax2.set_xlabel('Frame ID')
        ax2.set_ylabel('Peak Count')
        ax2.set_title('Peaks per Frame')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Confidence distribution
        ax3 = self.fig.add_subplot(gs[1, 1])
        all_confidences = []
        for frame_peaks in self.tracking_results['frames'].values():
            all_confidences.extend([peak['confidence'] for peak in frame_peaks])
        
        if all_confidences:
            ax3.hist(all_confidences, bins=20, alpha=0.7, color='orange', edgecolor='black')
            ax3.set_xlabel('Confidence Score')
            ax3.set_ylabel('Count')
            ax3.set_title('Detection Confidence')
            ax3.grid(True, alpha=0.3)
        
        self.canvas.draw()
    
    def update_single_frame_statistics(self):
        """Update statistics for single frame detection"""
        # Clear existing data
        for item in self.stats_tree.get_children():
            self.stats_tree.delete(item)
        
        # Add peak data
        for i, peak in enumerate(self.current_peaks):
            self.stats_tree.insert('', 'end', values=(
                peak.id[:8],  # Shortened ID
                "1 (single frame)",
                f"{peak.confidence:.3f}",
                f"{peak.intensity:.2f}",
                "N/A (single frame)"
            ))
    
    def update_trajectory_statistics(self):
        """Update trajectory statistics table"""
        if not self.tracking_results:
            return
        
        # Clear existing data
        for item in self.stats_tree.get_children():
            self.stats_tree.delete(item)
        
        # Add trajectory data
        trajectories = self.tracking_results['trajectories']
        for peak_id, traj_info in trajectories.items():
            avg_confidence = np.mean(traj_info['confidences'])
            avg_intensity = np.mean(traj_info['intensities'])
            
            # Calculate displacement
            positions = traj_info['positions']
            displacement = 0
            if len(positions) > 1:
                for i in range(1, len(positions)):
                    displacement += np.sqrt(
                        (positions[i][0] - positions[i-1][0])**2 +
                        (positions[i][1] - positions[i-1][1])**2
                    )
            
            self.stats_tree.insert('', 'end', values=(
                peak_id[:8],  # Shortened ID
                traj_info['length'],
                f"{avg_confidence:.3f}",
                f"{avg_intensity:.2f}",
                f"{displacement:.1f}"
            ))
    
    def update_analysis_text(self):
        """Update analysis text with detailed results"""
        if not self.tracking_results:
            return
        
        self.analysis_text.delete(1.0, tk.END)
        
        stats = self.tracking_results['statistics']
        
        analysis_report = f"""
AI-Powered NMR Peak Tracking Analysis Report
==========================================

DETECTION SUMMARY:
- Total Unique Peaks Detected: {stats['total_unique_peaks']}
- Average Trajectory Length: {stats['avg_trajectory_length']:.2f} frames
- Total Frames Processed: {stats['frame_count']}
- Average Detection Confidence: {stats['avg_confidence']:.3f}
- Average Peak Intensity: {stats['avg_intensity']:.2f}

TRAJECTORY ANALYSIS:
"""
        
        trajectories = self.tracking_results['trajectories']
        
        # Analyze trajectory lengths
        lengths = [info['length'] for info in trajectories.values()]
        if lengths:
            analysis_report += f"- Short trajectories (1-3 frames): {sum(1 for l in lengths if l <= 3)}\n"
            analysis_report += f"- Medium trajectories (4-10 frames): {sum(1 for l in lengths if 4 <= l <= 10)}\n"
            analysis_report += f"- Long trajectories (>10 frames): {sum(1 for l in lengths if l > 10)}\n"
        
        # Analyze confidence patterns
        all_confidences = []
        for info in trajectories.values():
            all_confidences.extend(info['confidences'])
        
        if all_confidences:
            analysis_report += f"\nCONFIDENCE ANALYSIS:\n"
            analysis_report += f"- High confidence peaks (>0.7): {sum(1 for c in all_confidences if c > 0.7)}\n"
            analysis_report += f"- Medium confidence peaks (0.3-0.7): {sum(1 for c in all_confidences if 0.3 <= c <= 0.7)}\n"
            analysis_report += f"- Low confidence peaks (<0.3): {sum(1 for c in all_confidences if c < 0.3)}\n"
        
        # Peak disappearance analysis
        if hasattr(self.yolo_tracker, 'get_peak_disappearance_analysis'):
            disappearance = self.yolo_tracker.get_peak_disappearance_analysis()
            analysis_report += f"\nPEAK DYNAMICS:\n"
            analysis_report += f"- Peaks by lifespan: {dict(list(disappearance['peaks_by_lifespan'].items())[:5])}\n"
        
        analysis_report += f"\nAI MODEL CONFIGURATION:\n"
        analysis_report += f"- Model: YOLOv11 + SAHI\n"
        analysis_report += f"- Confidence Threshold: {self.confidence_var.get():.2f}\n"
        analysis_report += f"- IoU Threshold: {self.iou_var.get():.2f}\n"
        analysis_report += f"- Max Missing Frames: {self.max_frames_var.get()}\n"
        analysis_report += f"- Device: {self.device_var.get()}\n"
        
        self.analysis_text.insert(tk.END, analysis_report)
    
    def export_results(self):
        """Export AI tracking results"""
        if not self.tracking_results and not self.current_peaks:
            messagebox.showwarning("Warning", "No results to export")
            return
        
        folder_path = filedialog.askdirectory(title="Select output folder")
        
        if folder_path:
            try:
                output_dir = Path(folder_path) / "ai_tracking_results"
                output_dir.mkdir(exist_ok=True)
                
                if self.tracking_results:
                    # Export full sequence results
                    self.yolo_tracker.export_results(self.tracking_results, str(output_dir))
                    
                    # Generate visualization
                    viz_fig = self.yolo_tracker.visualize_tracking_results(self.tracking_results)
                    viz_fig.savefig(output_dir / "tracking_visualization.png", dpi=300, bbox_inches='tight')
                    plt.close(viz_fig)
                    
                else:
                    # Export single frame results
                    single_frame_data = {
                        'frame_0': [peak.__dict__ for peak in self.current_peaks]
                    }
                    
                    with open(output_dir / 'single_frame_detection.json', 'w') as f:
                        json.dump(single_frame_data, f, indent=2, default=str)
                    
                    # Export as CSV
                    peak_data = []
                    for peak in self.current_peaks:
                        peak_data.append({
                            'peak_id': peak.id,
                            'centroid_x': peak.centroid[0],
                            'centroid_y': peak.centroid[1],
                            'confidence': peak.confidence,
                            'intensity': peak.intensity,
                            'bbox_x1': peak.bbox['x1'],
                            'bbox_y1': peak.bbox['y1'],
                            'bbox_x2': peak.bbox['x2'],
                            'bbox_y2': peak.bbox['y2']
                        })
                    
                    df = pd.DataFrame(peak_data)
                    df.to_csv(output_dir / 'single_frame_peaks.csv', index=False)
                
                messagebox.showinfo("Success", f"Results exported to {output_dir}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Export failed: {str(e)}")
    
    def generate_analysis_report(self):
        """Generate comprehensive analysis report"""
        if not self.tracking_results:
            messagebox.showwarning("Warning", "No tracking results available")
            return
        
        filepath = filedialog.asksaveasfilename(
            title="Save analysis report",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if filepath:
            try:
                with open(filepath, 'w') as f:
                    # Write the analysis text content
                    content = self.analysis_text.get(1.0, tk.END)
                    f.write(content)
                    
                    # Add additional detailed analysis
                    f.write("\n\nDETAILED TRAJECTORY DATA:\n")
                    f.write("=" * 50 + "\n")
                    
                    trajectories = self.tracking_results['trajectories']
                    for peak_id, info in trajectories.items():
                        f.write(f"\nPeak ID: {peak_id}\n")
                        f.write(f"  Frames: {info['start_frame']} - {info['end_frame']}\n")
                        f.write(f"  Length: {info['length']} frames\n")
                        f.write(f"  Avg Confidence: {np.mean(info['confidences']):.3f}\n")
                        f.write(f"  Avg Intensity: {np.mean(info['intensities']):.2f}\n")
                        
                        if len(info['positions']) > 1:
                            total_displacement = sum(
                                np.sqrt((info['positions'][i][0] - info['positions'][i-1][0])**2 +
                                       (info['positions'][i][1] - info['positions'][i-1][1])**2)
                                for i in range(1, len(info['positions']))
                            )
                            f.write(f"  Total Displacement: {total_displacement:.2f} pixels\n")
                
                messagebox.showinfo("Success", f"Analysis report saved to {filepath}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save report: {str(e)}")