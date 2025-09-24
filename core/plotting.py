"""
Plotting functionality for the Enhanced Bruker NMR Data Reader
"""

import numpy as np
import matplotlib.pyplot as plt
from .data_reader import BrukerDataReader


class PlottingManager:
    def __init__(self):
        self.data_reader = BrukerDataReader()

    def plot_spectrum(self, matrix, dic, title="NMR Spectrum"):
        """Plot a spectrum in a new matplotlib window"""
        # Create new figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if matrix.ndim == 1:
            self._plot_1d_spectrum(ax, matrix, dic, title)
        elif matrix.ndim == 2:
            self._plot_2d_spectrum(ax, matrix, dic, title)
        else:
            ax.text(0.5, 0.5, f"Cannot plot {matrix.ndim}D data", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"{title} - {matrix.ndim}D Data")
        
        plt.tight_layout()
        plt.show()

    def _plot_1d_spectrum(self, ax, matrix, dic, title):
        """Plot 1D spectrum"""
        x_axis = self.data_reader.get_axis_ranges(dic, 1)
        if x_axis is not None and len(x_axis) == len(matrix.real):
            ax.plot(x_axis, matrix.real)
            ax.set_xlabel("Chemical Shift (ppm)")
            ax.invert_xaxis()
        else:
            ax.plot(matrix.real)
            ax.set_xlabel("Data Points")
        
        ax.set_ylabel("Intensity")
        ax.set_title(f"1D Spectrum: {title}")
        ax.grid(True, alpha=0.3)

    def _plot_2d_spectrum(self, ax, matrix, dic, title):
        """Plot 2D spectrum"""
        axes_data = self.data_reader.get_axis_ranges(dic, 2)
        
        if axes_data is not None:
            f1_axis, f2_axis = axes_data
            if f1_axis is not None and f2_axis is not None:
                extent = [f2_axis[-1], f2_axis[0], f1_axis[-1], f1_axis[0]]
                im = ax.imshow(matrix.real, aspect='auto', extent=extent, 
                              cmap='viridis', origin='lower')
                ax.set_xlabel("F2 (ppm)")
                ax.set_ylabel("F1 (ppm)")
                plt.colorbar(im, ax=ax)
            else:
                im = ax.imshow(matrix.real, aspect='auto', cmap='viridis', origin='lower')
                ax.set_xlabel("F2 (points)")
                ax.set_ylabel("F1 (points)")
                plt.colorbar(im, ax=ax)
        else:
            im = ax.imshow(matrix.real, aspect='auto', cmap='viridis', origin='lower')
            ax.set_xlabel("F2 (points)")
            ax.set_ylabel("F1 (points)")
            plt.colorbar(im, ax=ax)
        
        ax.set_title(f"2D Spectrum: {title}")

    def plot_multiple_1d(self, matrices_list, labels=None, title="Multiple 1D Spectra"):
        """Plot multiple 1D spectra in the same window"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        if labels is None:
            labels = [f"Spectrum {i+1}" for i in range(len(matrices_list))]
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(matrices_list)))
        
        for i, (matrix, dic) in enumerate(matrices_list):
            if matrix.ndim != 1:
                continue
                
            x_axis = self.data_reader.get_axis_ranges(dic, 1)
            if x_axis is not None and len(x_axis) == len(matrix.real):
                ax.plot(x_axis, matrix.real, label=labels[i], color=colors[i])
            else:
                ax.plot(matrix.real, label=labels[i], color=colors[i])
        
        ax.set_xlabel("Chemical Shift (ppm)")
        ax.set_ylabel("Intensity")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()
        
        plt.tight_layout()
        plt.show()

    def plot_comparison_2d(self, matrix1, dic1, matrix2, dic2, 
                          title1="Spectrum 1", title2="Spectrum 2"):
        """Plot two 2D spectra side by side for comparison"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot first spectrum
        self._plot_2d_spectrum(ax1, matrix1, dic1, title1)
        
        # Plot second spectrum
        self._plot_2d_spectrum(ax2, matrix2, dic2, title2)
        
        plt.tight_layout()
        plt.show()

    def plot_difference_2d(self, matrix1, matrix2, dic1, title="Difference Spectrum"):
        """Plot the difference between two 2D spectra"""
        if matrix1.shape != matrix2.shape:
            print("Error: Matrices must have the same shape for difference plot")
            return
        
        difference = matrix1 - matrix2
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        axes_data = self.data_reader.get_axis_ranges(dic1, 2)
        
        if axes_data is not None:
            f1_axis, f2_axis = axes_data
            if f1_axis is not None and f2_axis is not None:
                extent = [f2_axis[-1], f2_axis[0], f1_axis[-1], f1_axis[0]]
                im = ax.imshow(difference.real, aspect='auto', extent=extent, 
                              cmap='RdBu_r', origin='lower')
                ax.set_xlabel("F2 (ppm)")
                ax.set_ylabel("F1 (ppm)")
            else:
                im = ax.imshow(difference.real, aspect='auto', cmap='RdBu_r', origin='lower')
                ax.set_xlabel("F2 (points)")
                ax.set_ylabel("F1 (points)")
        else:
            im = ax.imshow(difference.real, aspect='auto', cmap='RdBu_r', origin='lower')
            ax.set_xlabel("F2 (points)")
            ax.set_ylabel("F1 (points)")
        
        plt.colorbar(im, ax=ax)
        ax.set_title(f"Difference: {title}")
        
        plt.tight_layout()
        plt.show()

    def save_plot(self, matrix, dic, filename, title="NMR Spectrum", dpi=300):
        """Save a spectrum plot to file"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if matrix.ndim == 1:
            self._plot_1d_spectrum(ax, matrix, dic, title)
        elif matrix.ndim == 2:
            self._plot_2d_spectrum(ax, matrix, dic, title)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        
        return True

    def create_summary_plot(self, matrices_dict, max_plots=6):
        """Create a summary plot showing multiple spectra"""
        # Get first few spectra
        plot_data = []
        labels = []
        count = 0
        
        for expno, procnos in matrices_dict.items():
            if count >= max_plots:
                break
            for procno, data_info in procnos.items():
                if count >= max_plots:
                    break
                
                matrix = data_info['data']
                dic = data_info['dic']
                
                if matrix.ndim == 1:
                    plot_data.append((matrix, dic))
                    labels.append(f"Exp {expno}/Proc {procno}")
                    count += 1
        
        if plot_data:
            self.plot_multiple_1d(plot_data, labels, "Summary of 1D Spectra")
        else:
            print("No 1D spectra found for summary plot")

    def plot_npy_file(self, npy_path, title=None):
        """Plot spectrum stored as a .npy file (no Bruker metadata)"""
        try:
            matrix = np.load(npy_path)
        except Exception as e:
            print(f"Error loading {npy_path}: {e}")
            return

        if title is None:
            title = f"Processed Spectrum: {npy_path}"

        fig, ax = plt.subplots(figsize=(10, 8))

        if matrix.ndim == 1:
            ax.plot(matrix.real)
            ax.set_xlabel("Data Points")
            ax.set_ylabel("Intensity")
            ax.set_title(title)
            ax.grid(True, alpha=0.3)

        elif matrix.ndim == 2:
            im = ax.imshow(matrix.real, aspect="auto", cmap="viridis", origin="lower")
            ax.set_xlabel("F2 (points)")
            ax.set_ylabel("F1 (points)")
            ax.set_title(title)
            plt.colorbar(im, ax=ax)

        else:
            ax.text(0.5, 0.5, f"Cannot plot {matrix.ndim}D data",
                   ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{title} - {matrix.ndim}D Data")

        plt.tight_layout()
        plt.show()
