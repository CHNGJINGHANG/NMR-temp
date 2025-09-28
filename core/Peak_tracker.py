"""
Modern Trajectory Tracking System
Uses established production-ready solutions instead of custom implementations
"""

import numpy as np
import cv2
from ultralytics import YOLO
from deep_sort_realtime import DeepSort
from sklearn.manifold import UMAP
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import StandardScaler
import threading
import queue
from collections import deque
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any
import logging
import json
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TrajectoryPoint:
    """Single point in a trajectory with metadata"""
    x: float
    y: float
    frame_id: int
    confidence: float = 1.0
    timestamp: float = 0.0

@dataclass
class Trajectory:
    """Complete trajectory with metadata"""
    track_id: int
    points: List[TrajectoryPoint]
    creation_frame: int
    last_frame: int
    is_complete: bool = False
    
    def __len__(self):
        return len(self.points)
    
    def get_positions(self) -> np.ndarray:
        """Get position array (N, 2)"""
        return np.array([[p.x, p.y] for p in self.points])

class ModernDetector:
    """YOLO-based object detection"""
    
    def __init__(self, model_size='n', confidence_threshold=0.3):
        """
        Initialize detector with pre-trained YOLO model
        
        Args:
            model_size: 'n', 's', 'm', 'l', 'x' for nano, small, medium, large, extra-large
            confidence_threshold: Detection confidence threshold
        """
        try:
            self.model = YOLO(f'yolov8{model_size}.pt')
            self.confidence_threshold = confidence_threshold
            logger.info(f"YOLO detector initialized with model: yolov8{model_size}.pt")
        except Exception as e:
            logger.warning(f"Failed to load YOLO model: {e}")
            # Fallback to gradient-based detection
            self.model = None
            self.use_fallback = True
            self._init_fallback_detector()
    
    def _init_fallback_detector(self):
        """Initialize fallback gradient-based detector"""
        self.threshold = 40.0
        self.min_area = 3.0
        self.max_area = 500.0
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        logger.info("Using fallback gradient-based detector")
    
    def detect(self, frame):
        """Detect objects in frame"""
        if self.model is not None:
            return self._yolo_detect(frame)
        else:
            return self._gradient_detect(frame)
    
    def _yolo_detect(self, frame):
        """YOLO-based detection"""
        try:
            results = self.model(frame, conf=self.confidence_threshold, verbose=False)
            detections = []
            
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        
                        detections.append({
                            'bbox': [int(x1), int(y1), int(x2-x1), int(y2-y1)],
                            'confidence': float(conf),
                            'centroid': [float((x1+x2)/2), float((y1+y2)/2)]
                        })
            return detections
        except Exception as e:
            logger.error(f"YOLO detection failed: {e}")
            return []
    
    def _gradient_detect(self, frame):
        """Fallback gradient-based detection"""
        try:
            # Convert to grayscale if needed
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame.copy()
            
            # Gradient-based detection
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            grad_mag = cv2.magnitude(grad_x, grad_y)
            grad_mag = np.uint8(np.clip(grad_mag, 0, 255))
            
            # Threshold and morphology
            _, thresh = cv2.threshold(grad_mag, self.threshold, 255, cv2.THRESH_BINARY)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, self.kernel)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, self.kernel)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            detections = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                
                if self.min_area <= area <= self.max_area:
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cx = M["m10"] / M["m00"]
                        cy = M["m01"] / M["m00"]
                        
                        x, y, w, h = cv2.boundingRect(cnt)
                        
                        detections.append({
                            'centroid': [float(cx), float(cy)],
                            'bbox': [int(x), int(y), int(w), int(h)],
                            'confidence': 1.0
                        })
            
            return detections
        except Exception as e:
            logger.error(f"Gradient detection failed: {e}")
            return []
    
    def update_params(self, **kwargs):
        """Update detector parameters"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

class ModernTracker:
    """DeepSORT-based object tracking"""
    
    def __init__(self, max_age=50, n_init=3, max_cosine_distance=0.3, max_iou_distance=0.7):
        """
        Initialize tracker with DeepSORT
        
        Args:
            max_age: Maximum frames to keep a track without detection
            n_init: Number of consecutive detections before track is confirmed
            max_cosine_distance: Maximum cosine distance for feature matching
            max_iou_distance: Maximum IoU distance for matching
        """
        try:
            self.tracker = DeepSort(
                max_age=max_age,
                n_init=n_init,
                max_cosine_distance=max_cosine_distance,
                max_iou_distance=max_iou_distance,
                embedder="mobilenet",  # Feature extractor
                half=True  # FP16 for speed
            )
            self.trajectories = {}
            self.frame_count = 0
            logger.info("DeepSORT tracker initialized")
        except Exception as e:
            logger.error(f"Failed to initialize DeepSORT: {e}")
            # Fallback to simple tracking
            self.tracker = None
            self._init_simple_tracker()
    
    def _init_simple_tracker(self):
        """Initialize simple centroid-based tracker as fallback"""
        self.tracks = {}
        self.next_id = 1
        self.max_distance = 25.0
        self.max_disappeared = 8
        self.trajectories = {}
        logger.info("Using simple centroid tracker as fallback")
    
    def update(self, detections, frame):
        """Update tracker with new detections"""
        self.frame_count += 1
        
        if self.tracker is not None:
            return self._deepsort_update(detections, frame)
        else:
            return self._simple_update(detections)
    
    def _deepsort_update(self, detections, frame):
        """DeepSORT-based update"""
        try:
            # Format detections for DeepSORT
            det_list = []
            for det in detections:
                bbox = det['bbox']
                conf = det['confidence']
                det_list.append(([bbox[0], bbox[1], bbox[2], bbox[3]], conf, 'object'))
            
            # Update tracker
            tracks = self.tracker.update_tracks(det_list, frame=frame)
            
            # Collect trajectories
            active_tracks = []
            for track in tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                
                track_id = track.track_id
                bbox = track.to_tlbr()  # [x1, y1, x2, y2]
                centroid = [(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2]
                
                # Store trajectory points
                if track_id not in self.trajectories:
                    self.trajectories[track_id] = []
                
                self.trajectories[track_id].append({
                    'frame': self.frame_count,
                    'position': centroid,
                    'bbox': bbox.tolist()
                })
                
                active_tracks.append({
                    'id': track_id,
                    'bbox': bbox.tolist(),
                    'centroid': centroid,
                    'trajectory_length': len(self.trajectories[track_id])
                })
            
            return active_tracks
        except Exception as e:
            logger.error(f"DeepSORT update failed: {e}")
            return []
    
    def _simple_update(self, detections):
        """Simple centroid-based tracking update"""
        try:
            from scipy.optimize import linear_sum_assignment
            
            # Get existing track centroids
            if not self.tracks:
                # Initialize tracks
                for i, det in enumerate(detections):
                    self.tracks[self.next_id] = {
                        'centroid': det['centroid'],
                        'bbox': det['bbox'],
                        'disappeared': 0,
                        'creation_frame': self.frame_count
                    }
                    
                    if self.next_id not in self.trajectories:
                        self.trajectories[self.next_id] = []
                    
                    self.trajectories[self.next_id].append({
                        'frame': self.frame_count,
                        'position': det['centroid'],
                        'bbox': det['bbox']
                    })
                    
                    self.next_id += 1
            else:
                # Assignment using Hungarian algorithm
                if detections:
                    track_ids = list(self.tracks.keys())
                    cost_matrix = np.zeros((len(track_ids), len(detections)))
                    
                    for i, tid in enumerate(track_ids):
                        for j, det in enumerate(detections):
                            dist = np.linalg.norm(np.array(self.tracks[tid]['centroid']) - 
                                                np.array(det['centroid']))
                            cost_matrix[i, j] = dist if dist <= self.max_distance else 1e6
                    
                    # Hungarian assignment
                    row_indices, col_indices = linear_sum_assignment(cost_matrix)
                    
                    # Update matched tracks
                    matched_tracks = set()
                    matched_detections = set()
                    
                    for row, col in zip(row_indices, col_indices):
                        if cost_matrix[row, col] < 1e6:
                            tid = track_ids[row]
                            det = detections[col]
                            
                            self.tracks[tid]['centroid'] = det['centroid']
                            self.tracks[tid]['bbox'] = det['bbox']
                            self.tracks[tid]['disappeared'] = 0
                            
                            self.trajectories[tid].append({
                                'frame': self.frame_count,
                                'position': det['centroid'],
                                'bbox': det['bbox']
                            })
                            
                            matched_tracks.add(tid)
                            matched_detections.add(col)
                    
                    # Handle unmatched tracks
                    for tid in track_ids:
                        if tid not in matched_tracks:
                            self.tracks[tid]['disappeared'] += 1
                    
                    # Remove disappeared tracks
                    to_remove = []
                    for tid, track in self.tracks.items():
                        if track['disappeared'] > self.max_disappeared:
                            to_remove.append(tid)
                    
                    for tid in to_remove:
                        del self.tracks[tid]
                    
                    # Create new tracks for unmatched detections
                    for j, det in enumerate(detections):
                        if j not in matched_detections:
                            self.tracks[self.next_id] = {
                                'centroid': det['centroid'],
                                'bbox': det['bbox'],
                                'disappeared': 0,
                                'creation_frame': self.frame_count
                            }
                            
                            if self.next_id not in self.trajectories:
                                self.trajectories[self.next_id] = []
                            
                            self.trajectories[self.next_id].append({
                                'frame': self.frame_count,
                                'position': det['centroid'],
                                'bbox': det['bbox']
                            })
                            
                            self.next_id += 1
                
                else:
                    # No detections, increment disappeared counter
                    for track in self.tracks.values():
                        track['disappeared'] += 1
            
            # Return active tracks
            active_tracks = []
            for tid, track in self.tracks.items():
                if track['disappeared'] <= self.max_disappeared:
                    active_tracks.append({
                        'id': tid,
                        'bbox': track['bbox'],
                        'centroid': track['centroid'],
                        'trajectory_length': len(self.trajectories.get(tid, []))
                    })
            
            return active_tracks
            
        except Exception as e:
            logger.error(f"Simple tracking failed: {e}")
            return []

class ModernTrajectoryAnalyzer:
    """Modern trajectory analysis using scikit-learn"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.reducer = UMAP(n_components=10, random_state=42)
        self.clusterer = HDBSCAN(min_cluster_size=3)
        
    def extract_features(self, trajectory):
        """Extract comprehensive features from trajectory"""
        if isinstance(trajectory, list):
            positions = np.array([p['position'] for p in trajectory])
        else:
            positions = trajectory
            
        if len(positions) < 5:
            return None
            
        features = []
        
        # Basic geometric features
        total_distance = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
        displacement = np.linalg.norm(positions[-1] - positions[0])
        features.extend([total_distance, displacement, total_distance/max(displacement, 1e-6)])
        
        # Statistical features
        features.extend([
            np.std(positions[:, 0]),  # x variance
            np.std(positions[:, 1]),  # y variance
            np.mean(positions[:, 0]), # x mean
            np.mean(positions[:, 1])  # y mean
        ])
        
        # Velocity features
        velocities = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        features.extend([
            np.mean(velocities),
            np.std(velocities),
            np.max(velocities)
        ])
        
        # Direction changes
        if len(positions) > 2:
            angles = []
            for i in range(1, len(positions)-1):
                v1 = positions[i] - positions[i-1]
                v2 = positions[i+1] - positions[i]
                angle = np.arccos(np.clip(np.dot(v1, v2) / 
                                 (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1))
                angles.append(angle)
            
            features.extend([
                np.mean(angles),
                np.std(angles),
                np.sum(angles)  # Total curvature
            ])
        else:
            features.extend([0, 0, 0])
            
        return np.array(features)
    
    def analyze_trajectories(self, trajectories_dict):
        """Analyze and cluster trajectories"""
        # Extract features
        feature_list = []
        valid_ids = []
        
        for track_id, trajectory in trajectories_dict.items():
            features = self.extract_features(trajectory)
            if features is not None:
                feature_list.append(features)
                valid_ids.append(track_id)
        
        if len(feature_list) < 2:
            return {'status': 'insufficient_data'}
        
        feature_matrix = np.array(feature_list)
        
        # Normalize features
        try:
            feature_matrix_scaled = self.scaler.fit_transform(feature_matrix)
            
            # Dimensionality reduction with UMAP
            feature_matrix_reduced = self.reducer.fit_transform(feature_matrix_scaled)
            
            # Clustering with HDBSCAN
            cluster_labels = self.clusterer.fit_predict(feature_matrix_reduced)
            
            # Compile results
            results = {
                'status': 'success',
                'n_trajectories': len(valid_ids),
                'n_clusters': len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0),
                'n_noise': list(cluster_labels).count(-1),
                'cluster_labels': cluster_labels.tolist(),
                'trajectory_ids': valid_ids,
                'feature_matrix_shape': feature_matrix.shape,
                'reduced_features': feature_matrix_reduced.tolist()
            }
            
            return results
        except Exception as e:
            logger.error(f"Trajectory analysis failed: {e}")
            return {'status': 'analysis_failed', 'error': str(e)}

class EfficientVideoProcessor:
    """Memory-efficient video processing with buffering"""
    
    def __init__(self, video_path, buffer_size=30):
        self.video_path = video_path
        self.buffer_size = buffer_size
        self.frame_queue = queue.Queue(maxsize=buffer_size)
        self.cap = cv2.VideoCapture(video_path)
        
        # Video properties
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        self.current_frame = 0
        self.is_playing = False
        
    def start_reading(self):
        """Start background thread for reading frames"""
        def reader():
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                try:
                    self.frame_queue.put(frame, timeout=1)
                except queue.Full:
                    # Skip frame if buffer full
                    continue
        
        self.reader_thread = threading.Thread(target=reader, daemon=True)
        self.reader_thread.start()
    
    def get_next_frame(self):
        """Get next frame from buffer"""
        try:
            frame = self.frame_queue.get(timeout=1)
            self.current_frame += 1
            return frame
        except queue.Empty:
            return None
    
    def seek_frame(self, frame_number):
        """Seek to specific frame"""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        if ret:
            self.current_frame = frame_number
            return frame
        return None

class ModernTrajectorySystem:
    """Main modern trajectory tracking system"""
    
    def __init__(self, video_path=None):
        self.detector = ModernDetector()
        self.tracker = ModernTracker()
        self.analyzer = ModernTrajectoryAnalyzer()
        
        self.video_processor = None
        if video_path:
            self.video_processor = EfficientVideoProcessor(video_path)
        
        self.frame_count = 0
        self.processing_results = []
        
    def load_video(self, video_path):
        """Load video for processing"""
        try:
            self.video_processor = EfficientVideoProcessor(video_path)
            logger.info(f"Video loaded: {video_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load video: {e}")
            return False
        
    def process_frame(self, frame):
        """Process a single frame"""
        try:
            # Detect objects
            detections = self.detector.detect(frame)
            
            # Update tracks
            active_tracks = self.tracker.update(detections, frame)
            
            self.frame_count += 1
            
            # Return results for real-time processing
            result = {
                'frame': frame,
                'frame_number': self.frame_count,
                'detections': detections,
                'tracks': active_tracks
            }
            
            return result
        except Exception as e:
            logger.error(f"Frame processing failed: {e}")
            return {
                'frame': frame,
                'frame_number': self.frame_count,
                'detections': [],
                'tracks': []
            }
    
    def process_video(self, max_frames=None):
        """Process video and extract trajectories"""
        if not self.video_processor:
            raise ValueError("No video loaded")
        
        self.video_processor.start_reading()
        
        while True:
            frame = self.video_processor.get_next_frame()
            if frame is None:
                break
            
            result = self.process_frame(frame)
            self.processing_results.append(result)
            
            # Yield results for real-time processing
            yield result
            
            if max_frames and self.frame_count >= max_frames:
                break
    
    def analyze_collected_trajectories(self):
        """Analyze all collected trajectories"""
        return self.analyzer.analyze_trajectories(self.tracker.trajectories)
    
    def get_trajectories(self):
        """Get current trajectories"""
        return self.tracker.trajectories
    
    def reset_session(self):
        """Reset tracking session"""
        self.tracker = ModernTracker()  # Reinitialize tracker
        self.frame_count = 0
        self.processing_results = []
        logger.info("Session reset")
    
    def export_results(self, output_path):
        """Export tracking results"""
        try:
            output_dir = Path(output_path)
            output_dir.mkdir(exist_ok=True)
            
            # Export trajectories
            trajectories_data = {}
            for track_id, trajectory in self.tracker.trajectories.items():
                trajectories_data[str(track_id)] = trajectory
            
            with open(output_dir / 'trajectories.json', 'w') as f:
                json.dump(trajectories_data, f, indent=2, default=str)
            
            # Export analysis results
            analysis_results = self.analyze_collected_trajectories()
            with open(output_dir / 'analysis_results.json', 'w') as f:
                json.dump(analysis_results, f, indent=2, default=str)
            
            logger.info(f"Results exported to {output_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return False
    
    def update_detector_params(self, **kwargs):
        """Update detector parameters"""
        self.detector.update_params(**kwargs)

# Example usage
if __name__ == "__main__":
    # Initialize system
    system = ModernTrajectorySystem()
    
    # Load video
    if system.load_video("example_video.mp4"):
        # Process frames
        for result in system.process_video(max_frames=100):
            print(f"Frame {result['frame_number']}: {len(result['tracks'])} tracks")
        
        # Analyze trajectories
        analysis = system.analyze_collected_trajectories()
        print(f"Analysis: {analysis['status']}")
        
        # Export results
        system.export_results("./output")
    
    logger.info("Modern trajectory tracking system ready")
