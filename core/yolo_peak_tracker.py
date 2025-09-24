"""
YOLOv11 + SAHI NMR Peak Detection and Tracking Module
Pure AI-based detection with intelligent tracking across frames
"""

import numpy as np
import cv2
import torch
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import json
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
import uuid
from collections import defaultdict
import matplotlib.pyplot as plt

@dataclass
class AIPeak:
    """AI-detected peak with unique identifier"""
    id: str
    centroid: Tuple[float, float]
    bbox: Dict[str, float]  # x1, y1, x2, y2
    confidence: float
    intensity: float
    frame_id: int
    timestamp: float = 0.0
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())[:8]  # Generate unique 8-char ID

class YOLOv11PeakTracker:
    def __init__(self, model_path: str = None, device: str = 'auto'):
        """
        Initialize YOLOv11 + SAHI peak tracker
        
        Args:
            model_path: Path to trained YOLOv11 model (if None, uses default)
            device: Device to run inference on ('auto', 'cpu', 'cuda')
        """
        self.device = self._get_device(device)
        self.model = self._load_model(model_path)
        self.detection_model = self._setup_sahi_model()
        
        # Tracking state
        self.peaks_history = {}  # frame_id -> List[AIPeak]
        self.peak_trajectories = defaultdict(list)  # peak_id -> List[AIPeak]
        self.active_peaks = {}  # peak_id -> last_seen_frame
        self.confidence_threshold = 0.3
        self.iou_threshold = 0.5
        self.max_disappeared_frames = 3
        
    def _get_device(self, device: str) -> str:
        """Determine the best device for inference"""
        if device == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return device
    
    def _load_model(self, model_path: str = None) -> YOLO:
        """Load YOLOv11 model"""
        if model_path and Path(model_path).exists():
            return YOLO(model_path)
        else:
            # Use YOLOv11 nano for fast inference (you can change to 's', 'm', 'l', 'x')
            return YOLO('yolo11n.pt')
    
    def _setup_sahi_model(self) -> AutoDetectionModel:
        """Setup SAHI detection model"""
        return AutoDetectionModel.from_pretrained(
            model_type='yolov8',  # SAHI uses yolov8 interface for v11
            model_path=self.model.ckpt_path,
            confidence_threshold=self.confidence_threshold,
            device=self.device,
        )
    
    def preprocess_nmr_frame(self, nmr_data: np.ndarray) -> np.ndarray:
        """
        Convert NMR data to image format suitable for YOLO
        
        Args:
            nmr_data: 2D NMR spectrum data
            
        Returns:
            RGB image array
        """
        # Normalize data to 0-255
        normalized = np.abs(nmr_data)
        normalized = (normalized - normalized.min()) / (normalized.max() - normalized.min() + 1e-8)
        normalized = (normalized * 255).astype(np.uint8)
        
        # Convert to 3-channel RGB
        rgb_image = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
        
        return rgb_image
    
    def detect_peaks_frame(self, nmr_data: np.ndarray, frame_id: int) -> List[AIPeak]:
        """
        Detect peaks in a single NMR frame using YOLOv11 + SAHI
        
        Args:
            nmr_data: 2D NMR spectrum data
            frame_id: Frame identifier
            
        Returns:
            List of detected peaks
        """
        # Preprocess NMR data to image
        image = self.preprocess_nmr_frame(nmr_data)
        
        # Perform sliced prediction with SAHI
        result = get_sliced_prediction(
            image,
            self.detection_model,
            slice_height=512,
            slice_width=512,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
            postprocess_type="NMS",
            postprocess_match_metric="IOS",
            postprocess_match_threshold=0.5,
            postprocess_class_agnostic=True,
        )
        
        # Convert detections to AIPeak objects
        peaks = []
        for detection in result.object_prediction_list:
            bbox = detection.bbox.to_coco_bbox()  # [x, y, width, height]
            
            # Calculate centroid
            centroid = (
                bbox[0] + bbox[2] / 2,  # x center
                bbox[1] + bbox[3] / 2   # y center
            )
            
            # Calculate intensity from original NMR data
            x1, y1 = int(bbox[0]), int(bbox[1])
            x2, y2 = int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])
            
            # Ensure bounds are within image
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(nmr_data.shape[1], x2), min(nmr_data.shape[0], y2)
            
            intensity = float(np.mean(np.abs(nmr_data[y1:y2, x1:x2])))
            
            peak = AIPeak(
                id="",  # Will be assigned during tracking
                centroid=centroid,
                bbox={
                    'x1': float(bbox[0]),
                    'y1': float(bbox[1]),
                    'x2': float(bbox[0] + bbox[2]),
                    'y2': float(bbox[1] + bbox[3])
                },
                confidence=float(detection.score.value),
                intensity=intensity,
                frame_id=frame_id
            )
            peaks.append(peak)
        
        return peaks
    
    def track_peaks_ai(self, detected_peaks: List[AIPeak], frame_id: int) -> List[AIPeak]:
        """
        AI-based peak tracking using detection confidence and spatial coherence
        
        Args:
            detected_peaks: Newly detected peaks
            frame_id: Current frame ID
            
        Returns:
            List of tracked peaks with assigned IDs
        """
        if frame_id == 0 or not self.peaks_history:
            # First frame - assign new IDs to all peaks
            for peak in detected_peaks:
                peak.id = str(uuid.uuid4())[:8]
                self.active_peaks[peak.id] = frame_id
                self.peak_trajectories[peak.id].append(peak)
            return detected_peaks
        
        # Get previous frame peaks
        prev_frame_id = max(self.peaks_history.keys())
        previous_peaks = self.peaks_history[prev_frame_id]
        
        # AI-based matching using detection features
        tracked_peaks = []
        used_prev_peaks = set()
        
        for curr_peak in detected_peaks:
            best_match = None
            best_score = 0
            
            for prev_peak in previous_peaks:
                if prev_peak.id in used_prev_peaks:
                    continue
                
                # Calculate AI matching score
                match_score = self._calculate_ai_match_score(prev_peak, curr_peak)
                
                if match_score > best_score and match_score > 0.5:  # Threshold for matching
                    best_score = match_score
                    best_match = prev_peak
            
            if best_match:
                # Assign existing ID
                curr_peak.id = best_match.id
                used_prev_peaks.add(best_match.id)
                self.active_peaks[curr_peak.id] = frame_id
                self.peak_trajectories[curr_peak.id].append(curr_peak)
            else:
                # New peak - assign new ID
                curr_peak.id = str(uuid.uuid4())[:8]
                self.active_peaks[curr_peak.id] = frame_id
                self.peak_trajectories[curr_peak.id].append(curr_peak)
            
            tracked_peaks.append(curr_peak)
        
        # Handle disappeared peaks
        self._handle_disappeared_peaks(frame_id)
        
        return tracked_peaks
    
    def _calculate_ai_match_score(self, prev_peak: AIPeak, curr_peak: AIPeak) -> float:
        """
        Calculate AI-based matching score between peaks using multiple features
        
        Args:
            prev_peak: Peak from previous frame
            curr_peak: Peak from current frame
            
        Returns:
            Matching score between 0 and 1
        """
        # Spatial proximity score (normalized by image dimensions)
        spatial_dist = np.sqrt(
            (prev_peak.centroid[0] - curr_peak.centroid[0])**2 +
            (prev_peak.centroid[1] - curr_peak.centroid[1])**2
        )
        max_spatial_dist = 50  # pixels - adjust based on your data
        spatial_score = max(0, 1 - spatial_dist / max_spatial_dist)
        
        # Confidence similarity score
        conf_diff = abs(prev_peak.confidence - curr_peak.confidence)
        conf_score = max(0, 1 - conf_diff)
        
        # Intensity similarity score
        intensity_diff = abs(prev_peak.intensity - curr_peak.intensity)
        max_intensity = max(prev_peak.intensity, curr_peak.intensity)
        intensity_score = max(0, 1 - intensity_diff / (max_intensity + 1e-6))
        
        # Size similarity score (bbox area)
        prev_area = (prev_peak.bbox['x2'] - prev_peak.bbox['x1']) * (prev_peak.bbox['y2'] - prev_peak.bbox['y1'])
        curr_area = (curr_peak.bbox['x2'] - curr_peak.bbox['x1']) * (curr_peak.bbox['y2'] - curr_peak.bbox['y1'])
        area_ratio = min(prev_area, curr_area) / (max(prev_area, curr_area) + 1e-6)
        size_score = area_ratio
        
        # Weighted combination
        weights = [0.4, 0.3, 0.2, 0.1]  # spatial, confidence, intensity, size
        scores = [spatial_score, conf_score, intensity_score, size_score]
        
        return sum(w * s for w, s in zip(weights, scores))
    
    def _handle_disappeared_peaks(self, frame_id: int):
        """Handle peaks that have disappeared from detection"""
        disappeared_peaks = []
        
        for peak_id, last_seen in list(self.active_peaks.items()):
            frames_missing = frame_id - last_seen
            
            if frames_missing > self.max_disappeared_frames:
                # Peak has been missing too long - mark as disappeared
                disappeared_peaks.append(peak_id)
                del self.active_peaks[peak_id]
        
        return disappeared_peaks
    
    def process_nmr_sequence(self, nmr_frames: List[np.ndarray]) -> Dict:
        """
        Process a complete sequence of NMR frames
        
        Args:
            nmr_frames: List of 2D NMR spectrum arrays
            
        Returns:
            Complete tracking results
        """
        results = {
            'frames': {},
            'trajectories': {},
            'statistics': {}
        }
        
        for frame_id, nmr_data in enumerate(nmr_frames):
            print(f"Processing frame {frame_id + 1}/{len(nmr_frames)}")
            
            # Detect peaks
            detected_peaks = self.detect_peaks_frame(nmr_data, frame_id)
            
            # Track peaks
            tracked_peaks = self.track_peaks_ai(detected_peaks, frame_id)
            
            # Store results
            self.peaks_history[frame_id] = tracked_peaks
            results['frames'][frame_id] = [asdict(peak) for peak in tracked_peaks]
        
        # Generate trajectories and statistics
        results['trajectories'] = self._generate_trajectory_data()
        results['statistics'] = self._calculate_tracking_statistics()
        
        return results
    
    def _generate_trajectory_data(self) -> Dict:
        """Generate trajectory data for all tracked peaks"""
        trajectories = {}
        
        for peak_id, trajectory in self.peak_trajectories.items():
            if len(trajectory) > 0:  # Only include peaks that were tracked
                trajectories[peak_id] = {
                    'length': len(trajectory),
                    'start_frame': trajectory[0].frame_id,
                    'end_frame': trajectory[-1].frame_id,
                    'positions': [(p.centroid[0], p.centroid[1]) for p in trajectory],
                    'confidences': [p.confidence for p in trajectory],
                    'intensities': [p.intensity for p in trajectory],
                    'frames': [p.frame_id for p in trajectory]
                }
        
        return trajectories
    
    def _calculate_tracking_statistics(self) -> Dict:
        """Calculate comprehensive tracking statistics"""
        stats = {
            'total_unique_peaks': len(self.peak_trajectories),
            'active_peaks': len(self.active_peaks),
            'avg_trajectory_length': 0,
            'avg_confidence': 0,
            'avg_intensity': 0,
            'frame_count': len(self.peaks_history)
        }
        
        if self.peak_trajectories:
            trajectory_lengths = [len(traj) for traj in self.peak_trajectories.values()]
            stats['avg_trajectory_length'] = np.mean(trajectory_lengths)
            
            all_peaks = [peak for traj in self.peak_trajectories.values() for peak in traj]
            if all_peaks:
                stats['avg_confidence'] = np.mean([p.confidence for p in all_peaks])
                stats['avg_intensity'] = np.mean([p.intensity for p in all_peaks])
        
        return stats
    
    def export_results(self, results: Dict, output_dir: str):
        """
        Export tracking results in multiple formats
        
        Args:
            results: Results dictionary from process_nmr_sequence
            output_dir: Output directory path
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Export frame-by-frame results as JSON
        with open(output_path / 'frame_detections.json', 'w') as f:
            json.dump(results['frames'], f, indent=2)
        
        # Export trajectories as JSON
        with open(output_path / 'trajectories.json', 'w') as f:
            json.dump(results['trajectories'], f, indent=2)
        
        # Export statistics as JSON
        with open(output_path / 'statistics.json', 'w') as f:
            json.dump(results['statistics'], f, indent=2)
        
        # Export as CSV for easy analysis
        self._export_csv_format(results, output_path)
        
        print(f"Results exported to {output_path}")
    
    def _export_csv_format(self, results: Dict, output_path: Path):
        """Export results in CSV format"""
        # Detailed peak data
        peak_data = []
        for frame_id, peaks in results['frames'].items():
            for peak in peaks:
                peak_data.append({
                    'frame_id': frame_id,
                    'peak_id': peak['id'],
                    'centroid_x': peak['centroid'][0],
                    'centroid_y': peak['centroid'][1],
                    'confidence': peak['confidence'],
                    'intensity': peak['intensity'],
                    'bbox_x1': peak['bbox']['x1'],
                    'bbox_y1': peak['bbox']['y1'],
                    'bbox_x2': peak['bbox']['x2'],
                    'bbox_y2': peak['bbox']['y2']
                })
        
        df_peaks = pd.DataFrame(peak_data)
        df_peaks.to_csv(output_path / 'peak_detections.csv', index=False)
        
        # Trajectory summary
        traj_data = []
        for peak_id, traj_info in results['trajectories'].items():
            traj_data.append({
                'peak_id': peak_id,
                'trajectory_length': traj_info['length'],
                'start_frame': traj_info['start_frame'],
                'end_frame': traj_info['end_frame'],
                'avg_confidence': np.mean(traj_info['confidences']),
                'avg_intensity': np.mean(traj_info['intensities']),
                'total_displacement': self._calculate_total_displacement(traj_info['positions'])
            })
        
        df_trajectories = pd.DataFrame(traj_data)
        df_trajectories.to_csv(output_path / 'trajectory_summary.csv', index=False)
    
    def _calculate_total_displacement(self, positions: List[Tuple[float, float]]) -> float:
        """Calculate total displacement of a trajectory"""
        if len(positions) < 2:
            return 0.0
        
        total_dist = 0
        for i in range(1, len(positions)):
            dist = np.sqrt(
                (positions[i][0] - positions[i-1][0])**2 +
                (positions[i][1] - positions[i-1][1])**2
            )
            total_dist += dist
        
        return total_dist
    
    def visualize_tracking_results(self, results: Dict, output_path: str = None):
        """
        Create visualizations of tracking results
        
        Args:
            results: Results dictionary from process_nmr_sequence
            output_path: Optional path to save visualizations
        """
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('YOLOv11 + SAHI NMR Peak Tracking Results', fontsize=16)
        
        # Plot 1: Peak count per frame
        frame_ids = list(results['frames'].keys())
        peak_counts = [len(results['frames'][fid]) for fid in frame_ids]
        
        axes[0, 0].plot(frame_ids, peak_counts, 'b-o', markersize=4)
        axes[0, 0].set_xlabel('Frame ID')
        axes[0, 0].set_ylabel('Number of Peaks')
        axes[0, 0].set_title('Peak Count per Frame')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Trajectory lengths histogram
        traj_lengths = [info['length'] for info in results['trajectories'].values()]
        if traj_lengths:
            axes[0, 1].hist(traj_lengths, bins=20, alpha=0.7, color='green')
            axes[0, 1].set_xlabel('Trajectory Length')
            axes[0, 1].set_ylabel('Count')
            axes[0, 1].set_title('Distribution of Trajectory Lengths')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Confidence distribution
        all_confidences = []
        for frame_peaks in results['frames'].values():
            all_confidences.extend([peak['confidence'] for peak in frame_peaks])
        
        if all_confidences:
            axes[1, 0].hist(all_confidences, bins=20, alpha=0.7, color='orange')
            axes[1, 0].set_xlabel('Detection Confidence')
            axes[1, 0].set_ylabel('Count')
            axes[1, 0].set_title('Detection Confidence Distribution')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Sample trajectories
        sample_trajs = list(results['trajectories'].items())[:10]  # Show first 10 trajectories
        colors = plt.cm.tab10(np.linspace(0, 1, len(sample_trajs)))
        
        for i, (traj_id, traj_info) in enumerate(sample_trajs):
            positions = traj_info['positions']
            if len(positions) > 1:
                x_coords = [pos[0] for pos in positions]
                y_coords = [pos[1] for pos in positions]
                axes[1, 1].plot(x_coords, y_coords, 'o-', color=colors[i], 
                               label=f'Peak {traj_id[:4]}', markersize=3, linewidth=1)
        
        axes[1, 1].set_xlabel('X Position (pixels)')
        axes[1, 1].set_ylabel('Y Position (pixels)')
        axes[1, 1].set_title('Sample Peak Trajectories')
        axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {output_path}")
        
        return fig
    
    def get_peak_disappearance_analysis(self) -> Dict:
        """
        Analyze peak disappearance patterns for tracking quality assessment
        """
        disappearance_analysis = {
            'peaks_by_lifespan': defaultdict(int),
            'confidence_decay_patterns': [],
            'intensity_fade_patterns': []
        }
        
        for peak_id, trajectory in self.peak_trajectories.items():
            lifespan = len(trajectory)
            disappearance_analysis['peaks_by_lifespan'][lifespan] += 1
            
            if lifespan > 1:
                # Analyze confidence decay
                confidences = [p.confidence for p in trajectory]
                conf_trend = np.polyfit(range(len(confidences)), confidences, 1)[0]  # Slope
                disappearance_analysis['confidence_decay_patterns'].append(conf_trend)
                
                # Analyze intensity fade
                intensities = [p.intensity for p in trajectory]
                intensity_trend = np.polyfit(range(len(intensities)), intensities, 1)[0]
                disappearance_analysis['intensity_fade_patterns'].append(intensity_trend)
        
        # Convert defaultdict to regular dict for JSON serialization
        disappearance_analysis['peaks_by_lifespan'] = dict(disappearance_analysis['peaks_by_lifespan'])
        
        return disappearance_analysis