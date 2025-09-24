# core/video_generator.py
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import ffmpeg
import os
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import logging
import subprocess


def diagnose_ffmpeg_environment():
    """Comprehensive FFmpeg diagnostics"""
    import sys
    import importlib.util
    
    print("=" * 60)
    print("FFMPEG ENVIRONMENT DIAGNOSTICS")
    print("=" * 60)
    
    # Check ffmpeg-python package
    try:
        import ffmpeg
        print("✓ ffmpeg-python package found")
        if hasattr(ffmpeg, 'input') and hasattr(ffmpeg, 'output') and hasattr(ffmpeg, 'run'):
            print("✓ All required ffmpeg functions available")
        else:
            print("✗ Missing required ffmpeg functions")
    except ImportError:
        print("✗ ffmpeg-python package NOT found - pip install ffmpeg-python")
    
    # Check system FFmpeg binary
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            version_line = result.stdout.split('\n')[0]
            print(f"✓ System FFmpeg found: {version_line}")
        else:
            print("✗ FFmpeg command failed")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("✗ FFmpeg binary not found in system PATH")
    
    print("=" * 60)
    return True


def create_video_with_opencv(frame_files, output_path, fps=24):
    """Fallback method using OpenCV"""
    try:
        import cv2
        print("Using OpenCV fallback for video creation...")
        
        if not frame_files:
            raise ValueError("No frame files provided")
        
        first_frame = cv2.imread(frame_files[0])
        if first_frame is None:
            raise ValueError(f"Could not read frame: {frame_files[0]}")
            
        height, width, layers = first_frame.shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for i, frame_path in enumerate(frame_files):
            frame = cv2.imread(frame_path)
            if frame is not None:
                video_writer.write(frame)
        
        video_writer.release()
        cv2.destroyAllWindows()
        print(f"✓ Video created using OpenCV: {output_path}")
        return True
        
    except ImportError:
        print("✗ OpenCV not available. Install with: pip install opencv-python")
        return False
    except Exception as e:
        print(f"✗ OpenCV video creation failed: {e}")
        return False


def create_video_with_subprocess(frame_dir, output_path, fps=24):
    """Direct subprocess call to system ffmpeg"""
    try:
        input_pattern = os.path.join(frame_dir, 'frame_%04d.png')
        
        cmd = [
            'ffmpeg', '-y', '-framerate', str(fps),
            '-i', input_pattern, '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p', '-crf', '18', output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print(f"✓ Video created using direct FFmpeg: {output_path}")
            return True
        else:
            print(f"✗ FFmpeg command failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"✗ Direct FFmpeg call failed: {e}")
        return False


class SpectrumVideoGenerator:
    """Generates scientifically accurate grayscale videos from 2D NMR spectra"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.temp_dir = None
        self.frame_files = []
        
    def __enter__(self):
        self.temp_dir = tempfile.mkdtemp(prefix='nmr_video_')
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
    
    def cleanup(self):
        """Clean up temporary files"""
        if self.frame_files:
            for frame_file in self.frame_files:
                try:
                    os.unlink(frame_file)
                except FileNotFoundError:
                    pass
            self.frame_files.clear()
        
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                os.rmdir(self.temp_dir)
            except OSError:
                pass
    
    def load_spectra_from_npy(self, file_paths: List[str]) -> Tuple[List[np.ndarray], List[Dict]]:
        """Load multiple spectra from .npy files"""
        spectra = []
        metadata_list = []
        
        for path in file_paths:
            try:
                data = np.load(path, allow_pickle=True)
                
                if isinstance(data, np.ndarray):
                    if data.dtype == object:
                        spectrum = data.item()
                        if isinstance(spectrum, dict):
                            spectra.append(spectrum.get('data', spectrum.get('spectrum', data)))
                            metadata_list.append(spectrum.get('metadata', {}))
                        else:
                            spectra.append(spectrum)
                            metadata_list.append({})
                    else:
                        spectra.append(data)
                        metadata_list.append({})
                else:
                    spectra.append(data)
                    metadata_list.append({})
                    
            except Exception as e:
                self.logger.error(f"Failed to load spectrum from {path}: {e}")
                continue
        
        return spectra, metadata_list

    def load_2d_spectra_from_npy(self, file_paths: List[str]) -> Tuple[List[np.ndarray], List[str]]:
        """Load multiple 2D spectra from .npy files for heatmap video"""
        spectra = []
        filenames = []
        
        for path in file_paths:
            try:
                data = np.load(path, allow_pickle=True)
                
                if isinstance(data, np.ndarray):
                    if data.dtype == object:
                        spectrum = data.item()
                        if isinstance(spectrum, dict):
                            spectrum_data = spectrum.get('data', spectrum.get('spectrum', data))
                        else:
                            spectrum_data = spectrum
                    else:
                        spectrum_data = data
                else:
                    spectrum_data = data
                
                if spectrum_data.ndim == 2:
                    spectra.append(spectrum_data)
                    filenames.append(os.path.basename(path))
                else:
                    self.logger.warning(f"Skipping {path}: not a 2D spectrum (shape: {spectrum_data.shape})")
                    
            except Exception as e:
                self.logger.error(f"Failed to load 2D spectrum from {path}: {e}")
                continue
        
        return spectra, filenames

    def create_2d_heatmap_video(self, spectra_2d: List[np.ndarray], 
                            output_path: str,
                            fps: int = 24,
                            titles: Optional[List[str]] = None,
                            colormap: str = 'gray',
                            save_frames_folder: Optional[str] = None) -> bool:
        """
        Create scientifically accurate grayscale video from 2D NMR spectra
        No normalization, exact pixel mapping, no axes/colorbar
        """
        try:
            for i, spectrum in enumerate(spectra_2d):
                if spectrum.dtype == complex:
                    plot_data = np.abs(spectrum)
                else:
                    plot_data = spectrum.copy()
                
                # Get exact dimensions
                height, width = plot_data.shape
                
                # Create figure with exact pixel size, no margins
                fig = plt.figure(figsize=(width/100, height/100), dpi=100)
                ax = fig.add_axes([0, 0, 1, 1])  # Full figure, no margins
                
                # Display as grayscale image with no normalization
                ax.imshow(plot_data, 
                         cmap='gray',
                         aspect='equal',
                         interpolation='nearest',
                         origin='lower')
                
                # Remove all axes, ticks, labels
                ax.set_xticks([])
                ax.set_yticks([])
                ax.axis('off')
                
                # Save frame with exact pixel dimensions
                frame_filename = f'frame_{i:04d}.png'
                
                if save_frames_folder:
                    frame_path = os.path.join(save_frames_folder, frame_filename)
                else:
                    frame_path = os.path.join(self.temp_dir, frame_filename)
                
                plt.savefig(frame_path, 
                           dpi=100,
                           bbox_inches='tight',
                           pad_inches=0,
                           facecolor='black',
                           edgecolor='none')
                plt.close(fig)
                
                self.frame_files.append(frame_path)
                
                if (i + 1) % 10 == 0:
                    self.logger.info(f"Created frame {i+1}/{len(spectra_2d)} - Size: {width}x{height}")
            
            return self._create_video_from_frames(output_path, fps)
            
        except Exception as e:
            self.logger.error(f"Failed to create 2D heatmap video: {e}")
            return False

    def _create_video_from_frames(self, output_path: str, fps: int) -> bool:
        """Robust video creation with multiple fallback methods"""
        if not self.frame_files:
            raise ValueError("No frame files found")
        
        frame_dir = os.path.dirname(self.frame_files[0])
        
        # Method 1: Try ffmpeg-python
        try:
            import ffmpeg
            if hasattr(ffmpeg, 'input') and hasattr(ffmpeg, 'output') and hasattr(ffmpeg, 'run'):
                print("Attempting video creation with ffmpeg-python...")
                
                input_pattern = os.path.join(frame_dir, 'frame_%04d.png')
                stream = ffmpeg.input(input_pattern, framerate=fps)
                stream = ffmpeg.output(stream, output_path,
                                    vcodec='libx264',
                                    pix_fmt='yuv420p',
                                    crf=18,
                                    preset='medium')
                ffmpeg.run(stream, overwrite_output=True, quiet=True)
                
                self.logger.info(f"Video created with ffmpeg-python: {output_path}")
                return True
                
        except Exception as e:
            print(f"ffmpeg-python failed: {e}")
        
        # Method 2: Try direct subprocess call
        try:
            if create_video_with_subprocess(frame_dir, output_path, fps):
                return True
        except Exception as e:
            print(f"Direct ffmpeg failed: {e}")
        
        # Method 3: Try OpenCV fallback
        try:
            if create_video_with_opencv(self.frame_files, output_path, fps):
                return True
        except Exception as e:
            print(f"OpenCV fallback failed: {e}")
        
        self.logger.error("All video creation methods failed")
        return False


if __name__ == "__main__":
    # Simple test
    print("Video generator module loaded successfully")