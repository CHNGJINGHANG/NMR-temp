"""
Modern Peak Tracking GUI
Integrates with the modern trajectory tracking system using production-ready components
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import threading
import time
from pathlib import Path
import logging
import os

# Import the modern tracker
try:
    # This will work when integrated with the main application
    from core.Peak_tracker import ModernTrajectorySystem
except ImportError:
    try:
        # This will work for standalone testing
        from core.Peak_tracker import ModernTrajectorySystem
    except ImportError:
        # Fallback - create a mock system for testing
        class ModernTrajectorySystem:
            def __init__(self):
                self.frame_count = 0
                self.processing_results = []
                
            def load_video(self, path):
                return True
                
            def process_frame(self, frame):
                return {
                    'frame': frame,
                    'frame_number': self.frame_count,
                    'detections': [],
                    'tracks': []
                }
                
            def analyze_collected_trajectories(self):
                return {'status': 'no_data'}
                
            def export_results(self, path):
                return True
                
            def reset_session(self):
                pass
                
            def update_detector_params(self, **kwargs):
                pass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoPlayer:
    """Efficient video player with memory management"""
    
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = max(self.cap.get(cv2.CAP_PROP_FPS), 1.0)  # Avoid division by zero
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.current_frame = 0
        self.is_loaded = self.cap.isOpened()
        
        logger.info(f"Video loaded: {self.total_frames} frames, {self.fps:.2f} fps")
    
    def get_frame(self, frame_idx: int):
        """Get frame by index"""
        if not self.is_loaded:
            return None
            
        if frame_idx != self.current_frame:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            
        ret, frame = self.cap.read()
        if ret:
            self.current_frame = frame_idx
            return frame
        return None
    
    def get_next_frame(self):
        """Get next sequential frame"""
        return self.get_frame(self.current_frame + 1)
    
    def get_previous_frame(self):
        """Get previous frame"""
        return self.get_frame(max(0, self.current_frame - 1))
    
    def set_frame(self, frame_idx: int):
        """Set current frame position"""
        self.current_frame = max(0, min(frame_idx, self.total_frames - 1))
    
    def release(self):
        """Release video resources"""
        if self.cap:
            self.cap.release()

class TrajectoryVisualization:
    """Modern visualization for trajectory analysis"""
    
    def __init__(self, master):
        self.master = master
        
        # Create figure with better layout
        self.fig = Figure(figsize=(12, 8), dpi=100, tight_layout=True)
        self.canvas = FigureCanvasTkAgg(self.fig, master)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Color palette for consistent visualization
        self.colors = plt.cm.Set3(np.linspace(0, 1, 12))
        
        self.clear_plots()
    
    def clear_plots(self):
        """Clear all plots"""
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.text(0.5, 0.5, 'No data to display', ha='center', va='center', 
                transform=ax.transAxes, fontsize=14, color='gray')
        ax.set_title('Trajectory Visualization')
        ax.axis('off')
        self.canvas.draw()
    
    def plot_active_tracking(self, frame, tracks, detections=None):
        """Plot current frame with active tracks and detections"""
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        
        # Display frame
        if frame is not None:
            if len(frame.shape) == 3:
                ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                ax.imshow(frame, cmap='gray')
        
        # Draw detections if available
        if detections:
            for det in detections:
                bbox = det['bbox']
                rect = plt.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3],
                                   linewidth=1, edgecolor='yellow', facecolor='none',
                                   alpha=0.5, linestyle='--')
                ax.add_patch(rect)
        
        # Draw active tracks
        for i, track in enumerate(tracks):
            color = self.colors[i % len(self.colors)]
            
            # Draw bounding box
            bbox = track['bbox']
            rect = plt.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3],
                               linewidth=2, edgecolor=color, facecolor='none',
                               alpha=0.8)
            ax.add_patch(rect)
            
            # Draw centroid
            cx, cy = track['centroid']
            ax.plot(cx, cy, 'o', color=color, markersize=8)
            
            # Track label with length info
            label = f"ID:{track['id']} L:{track.get('trajectory_length', 0)}"
            ax.text(cx + 5, cy - 5, label, color=color, fontsize=9, 
                   fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", 
                   facecolor='white', alpha=0.8))
        
        ax.set_title(f'Active Tracking - {len(tracks)} tracks, {len(detections) if detections else 0} detections')
        ax.axis('off')
        self.canvas.draw()
    
    def plot_analysis_results(self, results):
        """Plot comprehensive trajectory analysis results"""
        self.fig.clear()
        
        if results.get('status') != 'success':
            ax = self.fig.add_subplot(111)
            status_msg = results.get('status', 'unknown')
            error_msg = results.get('error', '')
            ax.text(0.5, 0.5, f'Analysis Status: {status_msg}\n{error_msg}', 
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=12, color='red' if 'error' in status_msg else 'orange')
            ax.set_title('Trajectory Analysis')
            ax.axis('off')
            self.canvas.draw()
            return
        
        # Create subplot layout
        gs = self.fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # Plot 1: Cluster visualization
        ax1 = self.fig.add_subplot(gs[0, 0])
        
        reduced_features = np.array(results.get('reduced_features', []))
        cluster_labels = results.get('cluster_labels', [])
        
        if len(reduced_features) > 0 and len(cluster_labels) > 0:
            # Use first 2 components for visualization
            scatter = ax1.scatter(reduced_features[:, 0], reduced_features[:, 1], 
                                c=cluster_labels, cmap='tab10', alpha=0.7, s=50)
            ax1.set_xlabel('UMAP Component 1')
            ax1.set_ylabel('UMAP Component 2')
            ax1.set_title('Trajectory Clustering')
            plt.colorbar(scatter, ax=ax1, label='Cluster ID')
        
        # Plot 2: Cluster size distribution
        ax2 = self.fig.add_subplot(gs[0, 1])
        
        if cluster_labels:
            unique_clusters, counts = np.unique([c for c in cluster_labels if c != -1], 
                                               return_counts=True)
            if len(unique_clusters) > 0:
                bars = ax2.bar(range(len(unique_clusters)), counts, 
                              color='skyblue', alpha=0.7)
                ax2.set_xlabel('Cluster ID')
                ax2.set_ylabel('Number of Trajectories')
                ax2.set_title('Cluster Size Distribution')
                ax2.set_xticks(range(len(unique_clusters)))
                ax2.set_xticklabels(unique_clusters)
                
                # Add count labels on bars
                for bar, count in zip(bars, counts):
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height,
                           f'{count}', ha='center', va='bottom')
        
        # Plot 3: Feature importance visualization
        ax3 = self.fig.add_subplot(gs[1, :])
        
        # Summary statistics text
        stats_text = f"""Analysis Summary:

Total Trajectories: {results['n_trajectories']}
Clusters Found: {results['n_clusters']}
Noise Trajectories: {results['n_noise']}
Feature Matrix Shape: {results.get('feature_matrix_shape', 'N/A')}

Classification Rate: {((results['n_trajectories'] - results['n_noise']) / max(results['n_trajectories'], 1) * 100):.1f}%
Average Cluster Size: {(results['n_trajectories'] - results['n_noise']) / max(results['n_clusters'], 1):.1f}"""
        
        ax3.text(0.05, 0.95, stats_text, transform=ax3.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))
        
        # Add feature importance if available
        feature_names = ['Total Distance', 'Displacement', 'Tortuosity', 
                        'X Variance', 'Y Variance', 'X Mean', 'Y Mean',
                        'Velocity Mean', 'Velocity Std', 'Velocity Max',
                        'Angle Mean', 'Angle Std', 'Total Curvature']
        
        ax3.text(0.55, 0.95, 'Feature Categories:\n\n• Geometric: Distance, Displacement\n• Statistical: Position variance/means\n• Kinematic: Velocity characteristics\n• Shape: Curvature and angles', 
                transform=ax3.transAxes, fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgreen', alpha=0.8))
        
        ax3.axis('off')
        ax3.set_title('Trajectory Analysis Results', pad=20)
        
        self.canvas.draw()

class ModernPeakTrackingGUI:
    """Main GUI application using modern tracking components"""
    
    def __init__(self, parent_frame):
        self.parent_frame = parent_frame
        
        # Core components
        self.trajectory_system = ModernTrajectorySystem()
        self.video_player = None
        
        # GUI state
        self.is_playing = False
        self.play_thread = None
        self.stop_event = threading.Event()
        
        # Analysis results
        self.current_analysis = None
        self.processing_results = []
        
        # Create GUI
        self.create_widgets()
    
    def create_widgets(self):
        """Create the complete GUI layout"""
        # Main container with paned window
        main_paned = ttk.PanedWindow(self.parent_frame, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left panel: Controls
        self.create_control_panel(main_paned)
        
        # Right panel: Visualization
        self.create_visualization_panel(main_paned)
        
        # Bottom status bar
        self.create_status_bar()
    
    def create_control_panel(self, parent):
        """Create comprehensive control panel"""
        control_frame = ttk.Frame(parent, width=400)
        parent.add(control_frame, weight=1)
        
        # Video Controls Section
        video_group = ttk.LabelFrame(control_frame, text="Video Controls", padding=5)
        video_group.pack(fill=tk.X, padx=5, pady=5)
        
        # File loading
        file_frame = ttk.Frame(video_group)
        file_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(file_frame, text="Load Video", 
                  command=self.load_video, width=12).pack(side=tk.LEFT, padx=5)
        
        self.video_label = ttk.Label(file_frame, text="No video loaded", 
                                    foreground='gray')
        self.video_label.pack(side=tk.LEFT, padx=10)
        
        # Playback controls
        playback_frame = ttk.Frame(video_group)
        playback_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(playback_frame, text="◄◄", command=self.previous_frame, width=4).pack(side=tk.LEFT, padx=1)
        self.play_button = ttk.Button(playback_frame, text="▶", command=self.toggle_playback, width=4)
        self.play_button.pack(side=tk.LEFT, padx=1)
        ttk.Button(playback_frame, text="►►", command=self.next_frame, width=4).pack(side=tk.LEFT, padx=1)
        
        # Frame navigation
        nav_frame = ttk.Frame(video_group)
        nav_frame.pack(fill=tk.X, pady=5)
        
        self.frame_info = ttk.Label(nav_frame, text="Frame: 0/0")
        self.frame_info.pack()
        
        self.frame_var = tk.IntVar()
        self.frame_scale = ttk.Scale(nav_frame, from_=0, to=100, 
                                    orient=tk.HORIZONTAL, variable=self.frame_var,
                                    command=self.on_frame_change)
        self.frame_scale.pack(fill=tk.X, pady=5)
        
        # Detection Parameters Section
        detection_group = ttk.LabelFrame(control_frame, text="Detection Parameters", padding=5)
        detection_group.pack(fill=tk.X, padx=5, pady=5)
        
        # Confidence threshold for YOLO
        ttk.Label(detection_group, text="Confidence Threshold:").pack(anchor=tk.W)
        self.confidence_var = tk.DoubleVar(value=0.3)
        conf_frame = ttk.Frame(detection_group)
        conf_frame.pack(fill=tk.X, pady=2)
        ttk.Scale(conf_frame, from_=0.1, to=0.9, variable=self.confidence_var,
                 orient=tk.HORIZONTAL, command=self.update_detector_params).pack(fill=tk.X, side=tk.LEFT)
        ttk.Label(conf_frame, textvariable=self.confidence_var, width=6).pack(side=tk.RIGHT)
        
        # Fallback detector parameters (for gradient-based detection)
        ttk.Label(detection_group, text="Detection Threshold (fallback):").pack(anchor=tk.W, pady=(10,0))
        self.threshold_var = tk.DoubleVar(value=40.0)
        thresh_frame = ttk.Frame(detection_group)
        thresh_frame.pack(fill=tk.X, pady=2)
        ttk.Scale(thresh_frame, from_=10, to=100, variable=self.threshold_var,
                 orient=tk.HORIZONTAL, command=self.update_detector_params).pack(fill=tk.X, side=tk.LEFT)
        ttk.Label(thresh_frame, textvariable=self.threshold_var, width=6).pack(side=tk.RIGHT)
        
        ttk.Label(detection_group, text="Min/Max Area:").pack(anchor=tk.W, pady=(5,0))
        
        area_frame = ttk.Frame(detection_group)
        area_frame.pack(fill=tk.X, pady=2)
        
        self.min_area_var = tk.DoubleVar(value=3.0)
        self.max_area_var = tk.DoubleVar(value=500.0)
        
        ttk.Label(area_frame, text="Min:", width=4).pack(side=tk.LEFT)
        ttk.Scale(area_frame, from_=1, to=50, variable=self.min_area_var,
                 orient=tk.HORIZONTAL, command=self.update_detector_params, length=100).pack(side=tk.LEFT, padx=2)
        
        ttk.Label(area_frame, text="Max:", width=4).pack(side=tk.LEFT, padx=(10,0))
        ttk.Scale(area_frame, from_=100, to=2000, variable=self.max_area_var,
                 orient=tk.HORIZONTAL, command=self.update_detector_params, length=100).pack(side=tk.LEFT, padx=2)
        
        # Analysis Section
        analysis_group = ttk.LabelFrame(control_frame, text="Trajectory Analysis", padding=5)
        analysis_group.pack(fill=tk.X, padx=5, pady=5)
        
        # Analysis control buttons
        analysis_btn_frame = ttk.Frame(analysis_group)
        analysis_btn_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(analysis_btn_frame, text="Analyze Trajectories", 
                  command=self.run_analysis).pack(fill=tk.X, pady=2)
        
        ttk.Button(analysis_btn_frame, text="Clear Analysis", 
                  command=self.clear_analysis).pack(fill=tk.X, pady=2)
        
        # Analysis statistics
        self.analysis_info = ttk.Label(analysis_group, text="No analysis performed", 
                                      foreground='gray', wraplength=300)
        self.analysis_info.pack(pady=5)
        
        # Session Management Section
        session_group = ttk.LabelFrame(control_frame, text="Session Management", padding=5)
        session_group.pack(fill=tk.X, padx=5, pady=5)
        
        session_btn_frame = ttk.Frame(session_group)
        session_btn_frame.pack(fill=tk.X)
        
        ttk.Button(session_btn_frame, text="Export Results", 
                  command=self.export_results).pack(fill=tk.X, pady=2)
        
        ttk.Button(session_btn_frame, text="Reset Session", 
                  command=self.reset_session).pack(fill=tk.X, pady=2)
    
    def create_visualization_panel(self, parent):
        """Create visualization panel with tabs"""
        viz_frame = ttk.Frame(parent, width=800)
        parent.add(viz_frame, weight=2)
        
        # Notebook for different views
        self.notebook = ttk.Notebook(viz_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Video view tab
        self.video_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.video_tab, text="Live Tracking")
        
        # Video canvas for live tracking display
        self.video_canvas = tk.Canvas(self.video_tab, bg='black', width=640, height=480)
        self.video_canvas.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)
        
        # Analysis view tab
        self.analysis_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.analysis_tab, text="Analysis Results")
        
        # Create trajectory visualization
        self.trajectory_viz = TrajectoryVisualization(self.analysis_tab)
    
    def create_status_bar(self):
        """Create status bar at bottom"""
        status_frame = ttk.Frame(self.parent_frame)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=2)
        
        self.status_text = tk.StringVar(value="Ready - Load a video to start tracking")
        status_label = ttk.Label(status_frame, textvariable=self.status_text, 
                                relief=tk.SUNKEN, padding=5)
        status_label.pack(fill=tk.X)
    
    def load_video(self):
        """Load video file with modern file dialog"""
        filetypes = [
            ('Video files', '*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm'),
            ('MP4 files', '*.mp4'),
            ('AVI files', '*.avi'),
            ('All files', '*.*')
        ]
        
        filename = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=filetypes
        )
        
        if not filename:
            return
        
        try:
            # Load video
            self.video_player = VideoPlayer(filename)
            
            if not self.video_player.is_loaded:
                raise ValueError("Failed to open video file")
            
            # Load into trajectory system
            success = self.trajectory_system.load_video(filename)
            if not success:
                raise ValueError("Failed to initialize trajectory system")
            
            # Update UI
            video_name = os.path.basename(filename)
            self.video_label.config(text=f"Loaded: {video_name}", foreground='black')
            
            total_frames = self.video_player.total_frames
            self.frame_scale.config(to=total_frames - 1)
            self.update_frame_info()
            
            # Display first frame
            self.display_current_frame()
            
            self.status_text.set(f"Video loaded: {total_frames} frames, {self.video_player.fps:.1f} fps")
            logger.info(f"Successfully loaded video: {filename}")
            
        except Exception as e:
            error_msg = f"Failed to load video: {str(e)}"
            messagebox.showerror("Video Load Error", error_msg)
            self.status_text.set(f"Error: {error_msg}")
            logger.error(error_msg)
    
    def display_current_frame(self):
        """Display current frame with tracking overlay"""
        if not self.video_player:
            return
        
        frame = self.video_player.get_frame(self.video_player.current_frame)
        if frame is None:
            return
        
        try:
            # Process frame through trajectory system
            result = self.trajectory_system.process_frame(frame)
            tracks = result['tracks']
            detections = result['detections']
            
            # Update live tracking visualization
            self.trajectory_viz.plot_active_tracking(frame, tracks, detections)
            
            # Display frame on video canvas
            self.display_frame_on_canvas(frame, tracks, detections)
            
            # Update status
            self.status_text.set(
                f"Frame {self.video_player.current_frame + 1}/{self.video_player.total_frames} | "
                f"Tracks: {len(tracks)} | Detections: {len(detections)} | "
                f"Trajectories: {len(self.trajectory_system.get_trajectories())}"
            )
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            self.status_text.set(f"Processing error: {str(e)}")
    
    def display_frame_on_canvas(self, frame, tracks, detections):
        """Display frame with annotations on video canvas"""
        try:
            # Convert frame for display
            if len(frame.shape) == 3:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            
            # Draw tracking overlays directly on frame
            frame_annotated = frame_rgb.copy()
            
            # Draw detections in yellow
            for det in detections:
                bbox = det['bbox']
                cv2.rectangle(frame_annotated, (bbox[0], bbox[1]), 
                            (bbox[0] + bbox[2], bbox[1] + bbox[3]), 
                            (255, 255, 0), 1)
            
            # Draw tracks in different colors
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), 
                     (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0)]
            
            for i, track in enumerate(tracks):
                color = colors[i % len(colors)]
                bbox = track['bbox']
                centroid = track['centroid']
                
                # Draw bounding box
                cv2.rectangle(frame_annotated, (bbox[0], bbox[1]), 
                            (bbox[0] + bbox[2], bbox[1] + bbox[3]), color, 2)
                
                # Draw centroid
                cv2.circle(frame_annotated, (int(centroid[0]), int(centroid[1])), 5, color, -1)
                
                # Draw track ID
                label = f"ID:{track['id']}"
                cv2.putText(frame_annotated, label, (bbox[0], bbox[1] - 10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Resize to fit canvas
            canvas_width = self.video_canvas.winfo_width()
            canvas_height = self.video_canvas.winfo_height()
            
            if canvas_width > 1 and canvas_height > 1:
                h, w = frame_annotated.shape[:2]
                scale = min(canvas_width / w, canvas_height / h) * 0.95  # 95% to leave some margin
                new_w, new_h = int(w * scale), int(h * scale)
                
                frame_resized = cv2.resize(frame_annotated, (new_w, new_h))
                
                # Convert to PIL and display
                pil_image = Image.fromarray(frame_resized)
                self.photo = ImageTk.PhotoImage(pil_image)
                
                self.video_canvas.delete("all")
                self.video_canvas.create_image(canvas_width//2, canvas_height//2, 
                                             image=self.photo, anchor=tk.CENTER)
        
        except Exception as e:
            logger.error(f"Error displaying frame on canvas: {e}")
    
    def update_frame_info(self):
        """Update frame information display"""
        if not self.video_player:
            return
        
        current = self.video_player.current_frame + 1
        total = self.video_player.total_frames
        self.frame_info.config(text=f"Frame: {current}/{total}")
        self.frame_var.set(self.video_player.current_frame)
    
    def toggle_playback(self):
        """Toggle video playback"""
        if not self.video_player:
            messagebox.showwarning("No Video", "Please load a video first")
            return
        
        self.is_playing = not self.is_playing
        
        if self.is_playing:
            self.play_button.config(text="⏸")
            self.start_playback()
        else:
            self.play_button.config(text="▶")
            self.stop_playback()
    
    def start_playback(self):
        """Start video playback in separate thread"""
        if self.play_thread and self.play_thread.is_alive():
            return
        
        self.stop_event.clear()
        self.play_thread = threading.Thread(target=self.playback_worker, daemon=True)
        self.play_thread.start()
    
    def stop_playback(self):
        """Stop video playback"""
        self.stop_event.set()
    
    def playback_worker(self):
        """Playback worker thread"""
        while not self.stop_event.is_set() and self.is_playing:
            if not self.video_player:
                break
            
            # Move to next frame
            self.video_player.current_frame += 1
            
            if self.video_player.current_frame >= self.video_player.total_frames:
                # End of video
                self.is_playing = False
                self.parent_frame.after(0, lambda: self.play_button.config(text="▶"))
                break
            
            # Update display on main thread
            self.parent_frame.after(0, self.display_current_frame)
            self.parent_frame.after(0, self.update_frame_info)
            
            # Control playback speed
            time.sleep(1.0 / self.video_player.fps)
    
    def next_frame(self):
        """Move to next frame"""
        if not self.video_player:
            return
        
        if self.video_player.current_frame < self.video_player.total_frames - 1:
            self.video_player.current_frame += 1
            self.display_current_frame()
            self.update_frame_info()
    
    def previous_frame(self):
        """Move to previous frame"""
        if not self.video_player:
            return
        
        if self.video_player.current_frame > 0:
            self.video_player.current_frame -= 1
            self.display_current_frame()
            self.update_frame_info()
    
    def on_frame_change(self, value):
        """Handle frame slider change"""
        if not self.video_player:
            return
        
        frame_idx = int(float(value))
        self.video_player.set_frame(frame_idx)
        self.display_current_frame()
        self.update_frame_info()
    
    def update_detector_params(self, *args):
        """Update detector parameters"""
        try:
            self.trajectory_system.update_detector_params(
                confidence_threshold=self.confidence_var.get(),
                threshold=self.threshold_var.get(),
                min_area=self.min_area_var.get(),
                max_area=self.max_area_var.get()
            )
        except Exception as e:
            logger.error(f"Error updating detector params: {e}")
    
    def run_analysis(self):
        """Run trajectory analysis"""
        try:
            trajectories = self.trajectory_system.get_trajectories()
            
            if len(trajectories) < 2:
                messagebox.showwarning("Insufficient Data", 
                                     "Need at least 2 trajectories for analysis")
                return
            
            # Run analysis
            self.current_analysis = self.trajectory_system.analyze_collected_trajectories()
            
            if self.current_analysis['status'] == 'success':
                # Update visualization
                self.trajectory_viz.plot_analysis_results(self.current_analysis)
                
                # Switch to analysis tab
                self.notebook.select(self.analysis_tab)
                
                # Update info
                n_clusters = self.current_analysis['n_clusters']
                n_trajectories = self.current_analysis['n_trajectories']
                n_noise = self.current_analysis['n_noise']
                
                self.analysis_info.config(
                    text=f"Analysis complete: {n_clusters} clusters from {n_trajectories} trajectories "
                         f"({n_noise} noise points)",
                    foreground='green'
                )
                
                self.status_text.set(f"Analysis complete: {n_clusters} clusters found")
                
            else:
                error_msg = f"Analysis failed: {self.current_analysis['status']}"
                self.analysis_info.config(text=error_msg, foreground='red')
                messagebox.showerror("Analysis Error", error_msg)
        
        except Exception as e:
            error_msg = f"Analysis error: {str(e)}"
            logger.error(error_msg)
            messagebox.showerror("Analysis Error", error_msg)
            self.analysis_info.config(text=error_msg, foreground='red')
    
    def clear_analysis(self):
        """Clear analysis results"""
        self.current_analysis = None
        self.trajectory_viz.clear_plots()
        self.analysis_info.config(text="Analysis cleared", foreground='gray')
        self.status_text.set("Analysis cleared")
    
    def export_results(self):
        """Export tracking and analysis results"""
        if not self.trajectory_system.get_trajectories():
            messagebox.showwarning("No Data", "No trajectory data to export")
            return
        
        folder = filedialog.askdirectory(title="Select Export Folder")
        if not folder:
            return
        
        try:
            success = self.trajectory_system.export_results(folder)
            
            if success:
                messagebox.showinfo("Export Complete", f"Results exported to:\n{folder}")
                self.status_text.set(f"Results exported to {folder}")
            else:
                messagebox.showerror("Export Error", "Failed to export results")
        
        except Exception as e:
            error_msg = f"Export failed: {str(e)}"
            messagebox.showerror("Export Error", error_msg)
            logger.error(error_msg)
    
    def reset_session(self):
        """Reset tracking session"""
        result = messagebox.askyesno("Reset Session", 
                                   "This will clear all tracking data and analysis results.\n"
                                   "Continue?")
        if not result:
            return
        
        try:
            # Stop playback
            self.is_playing = False
            self.stop_event.set()
            
            # Reset trajectory system
            self.trajectory_system.reset_session()
            
            # Clear analysis
            self.current_analysis = None
            self.trajectory_viz.clear_plots()
            
            # Update UI
            self.analysis_info.config(text="No analysis performed", foreground='gray')
            self.status_text.set("Session reset - ready for new tracking")
            
            # If video is loaded, display current frame
            if self.video_player:
                self.display_current_frame()
            
            logger.info("Session reset complete")
        
        except Exception as e:
            error_msg = f"Reset failed: {str(e)}"
            messagebox.showerror("Reset Error", error_msg)
            logger.error(error_msg)

class PeakTrackingTab:
    """Peak tracking tab for main application integration"""
    
    def __init__(self, parent, main_app):
        self.parent = parent
        self.main_app = main_app
        
        # Create main frame
        self.frame = ttk.Frame(parent)
        
        try:
            # Initialize modern GUI
            self.tracking_gui = ModernPeakTrackingGUI(self.frame)
            logger.info("Modern peak tracking GUI initialized successfully")
            
        except Exception as e:
            # Fallback error display
            logger.error(f"Failed to initialize tracking GUI: {e}")
            
            error_frame = ttk.Frame(self.frame)
            error_frame.pack(fill=tk.BOTH, expand=True)
            
            title_label = ttk.Label(error_frame, 
                                   text="Peak Tracking & Trajectory Analysis", 
                                   font=('Arial', 16, 'bold'))
            title_label.pack(pady=20)
            
            error_label = ttk.Label(error_frame, 
                                   text=f"Initialization Error:\n{str(e)}\n\nPlease check that required packages are installed:\n- ultralytics (pip install ultralytics)\n- deep-sort-realtime (pip install deep-sort-realtime)\n- scikit-learn, opencv-python, matplotlib", 
                                   foreground='red',
                                   justify=tk.LEFT,
                                   wraplength=600)
            error_label.pack(pady=10, padx=20)
            
            # Installation instructions
            install_frame = ttk.LabelFrame(error_frame, text="Installation Instructions")
            install_frame.pack(fill=tk.X, padx=20, pady=10)
            
            install_text = """To install required packages:

1. Core packages:
   pip install ultralytics deep-sort-realtime

2. Supporting packages:
   pip install scikit-learn opencv-python matplotlib pillow numpy

3. For CUDA support (optional):
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

If YOLO models fail to download, the system will fall back to gradient-based detection."""
            
            install_label = ttk.Label(install_frame, text=install_text, 
                                     justify=tk.LEFT, font=('Courier', 9))
            install_label.pack(padx=10, pady=10)


def create_standalone_app():
    """Create standalone application for testing"""
    root = tk.Tk()
    root.title("Modern Peak Tracking System")
    root.geometry("1400x900")
    root.minsize(1200, 800)
    
    # Configure style
    style = ttk.Style()
    style.theme_use('clam')  # Modern looking theme
    
    # Main container
    main_frame = ttk.Frame(root)
    main_frame.pack(fill=tk.BOTH, expand=True)
    
    # Add title bar
    title_frame = ttk.Frame(main_frame)
    title_frame.pack(fill=tk.X, padx=10, pady=5)
    
    title_label = ttk.Label(title_frame, 
                           text="Modern Peak Tracking & Trajectory Analysis System", 
                           font=('Arial', 14, 'bold'))
    title_label.pack(side=tk.LEFT)
    
    version_label = ttk.Label(title_frame, 
                             text="v2.0 - Production Ready", 
                             font=('Arial', 10), 
                             foreground='gray')
    version_label.pack(side=tk.RIGHT)
    
    # Create separator
    ttk.Separator(main_frame, orient='horizontal').pack(fill=tk.X, padx=10, pady=2)
    
    # Create main content frame
    content_frame = ttk.Frame(main_frame)
    content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
    
    try:
        # Initialize the GUI
        app = ModernPeakTrackingGUI(content_frame)
        
        # Add menu bar
        menubar = tk.Menu(root)
        root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load Video", command=app.load_video)
        file_menu.add_separator()
        file_menu.add_command(label="Export Results", command=app.export_results)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=root.quit)
        
        # Analysis menu
        analysis_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Analysis", menu=analysis_menu)
        analysis_menu.add_command(label="Run Analysis", command=app.run_analysis)
        analysis_menu.add_command(label="Clear Analysis", command=app.clear_analysis)
        analysis_menu.add_separator()
        analysis_menu.add_command(label="Reset Session", command=app.reset_session)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=lambda: show_about_dialog(root))
        
        logger.info("Standalone application initialized successfully")
        
    except Exception as e:
        # Show error in the content frame
        error_label = ttk.Label(content_frame, 
                               text=f"Failed to initialize application:\n{str(e)}", 
                               foreground='red', 
                               font=('Arial', 12))
        error_label.pack(expand=True)
        logger.error(f"Failed to initialize standalone app: {e}")
    
    return root, app if 'app' in locals() else None


def show_about_dialog(parent):
    """Show about dialog"""
    about_window = tk.Toplevel(parent)
    about_window.title("About")
    about_window.geometry("500x400")
    about_window.resizable(False, False)
    
    # Make it modal
    about_window.transient(parent)
    about_window.grab_set()
    
    # Center the window
    about_window.geometry("+%d+%d" % (parent.winfo_rootx() + 50, parent.winfo_rooty() + 50))
    
    # Content
    content_frame = ttk.Frame(about_window, padding=20)
    content_frame.pack(fill=tk.BOTH, expand=True)
    
    title_label = ttk.Label(content_frame, 
                           text="Modern Peak Tracking System", 
                           font=('Arial', 16, 'bold'))
    title_label.pack(pady=10)
    
    version_label = ttk.Label(content_frame, 
                             text="Version 2.0 - Production Ready", 
                             font=('Arial', 12))
    version_label.pack(pady=5)
    
    description_text = """This system replaces custom mathematical implementations with established, production-ready solutions:

• YOLOv8/YOLOv9 for object detection
• DeepSORT for multi-object tracking  
• UMAP for dimensionality reduction
• HDBSCAN for trajectory clustering
• Scikit-learn for feature extraction

Key improvements:
✓ Better detection accuracy
✓ Robust tracking performance
✓ Memory-efficient video processing
✓ Modern machine learning methods
✓ Scalable architecture
✓ Production-ready components

Fallback systems ensure operation even without GPU acceleration or when modern components are unavailable."""
    
    description_label = ttk.Label(content_frame, 
                                 text=description_text, 
                                 justify=tk.LEFT, 
                                 wraplength=450)
    description_label.pack(pady=10)
    
    # Close button
    close_button = ttk.Button(content_frame, 
                             text="Close", 
                             command=about_window.destroy)
    close_button.pack(pady=10)


def test_system_components():
    """Test system components and report status"""
    print("Testing Modern Peak Tracking System Components...")
    print("=" * 60)
    
    # Test imports
    components_status = {}
    
    # Test YOLO
    try:
        from ultralytics import YOLO
        # Try to load a model
        model = YOLO('yolov8n.pt')  # This will download if not present
        components_status['YOLO'] = "✓ Available"
    except Exception as e:
        components_status['YOLO'] = f"✗ Failed: {str(e)}"
    
    # Test DeepSORT
    try:
        from deep_sort_realtime import DeepSort
        tracker = DeepSort(max_age=10, n_init=3)
        components_status['DeepSORT'] = "✓ Available"
    except Exception as e:
        components_status['DeepSORT'] = f"✗ Failed: {str(e)}"
    
    # Test scikit-learn components
    try:
        from sklearn.manifold import UMAP
        from sklearn.cluster import HDBSCAN
        from sklearn.preprocessing import StandardScaler
        reducer = UMAP(n_components=2, random_state=42)
        clusterer = HDBSCAN(min_cluster_size=3)
        scaler = StandardScaler()
        components_status['Scikit-learn'] = "✓ Available"
    except Exception as e:
        components_status['Scikit-learn'] = f"✗ Failed: {str(e)}"
    
    # Test OpenCV
    try:
        import cv2
        cap = cv2.VideoCapture()  # Test creation
        components_status['OpenCV'] = "✓ Available"
    except Exception as e:
        components_status['OpenCV'] = f"✗ Failed: {str(e)}"
    
    # Test other dependencies
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        from PIL import Image
        components_status['Visualization'] = "✓ Available"
    except Exception as e:
        components_status['Visualization'] = f"✗ Failed: {str(e)}"
    
    # Print results
    for component, status in components_status.items():
        print(f"{component:15}: {status}")
    
    print("\n" + "=" * 60)
    
    # Count successful components
    successful = sum(1 for status in components_status.values() if status.startswith("✓"))
    total = len(components_status)
    
    print(f"System Status: {successful}/{total} components available")
    
    if successful == total:
        print("🎉 All components ready! System fully operational.")
    elif successful >= 3:
        print("⚠️  System partially ready. Some advanced features may use fallbacks.")
    else:
        print("❌ System not ready. Please install required packages.")
    
    print("\nTo install missing packages:")
    print("pip install ultralytics deep-sort-realtime scikit-learn opencv-python matplotlib pillow numpy umap-learn hdbscan")
    
    return successful == total


# Main execution
if __name__ == "__main__":
    import sys
    
    # Set up logging for testing
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        # Run component tests
        test_system_components()
    else:
        # Run the application
        try:
            root, app = create_standalone_app()
            
            # Handle window closing
            def on_closing():
                if app and hasattr(app, 'video_player') and app.video_player:
                    app.video_player.release()
                root.destroy()
            
            root.protocol("WM_DELETE_WINDOW", on_closing)
            
            # Start the application
            print("Starting Modern Peak Tracking System...")
            print("Close the window or press Ctrl+C to exit.")
            
            root.mainloop()
            
        except KeyboardInterrupt:
            print("\nApplication interrupted by user")
        except Exception as e:
            print(f"Application error: {e}")
            logger.error(f"Application error: {e}")
        finally:
            print("Application closed.")
