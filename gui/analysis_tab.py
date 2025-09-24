"""
Analysis and visualization tab for the Enhanced Bruker NMR Data Reader
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import os
import tkinter.simpledialog

class AnalysisTab:
    def __init__(self, parent, main_app):
        self.parent = parent
        self.main_app = main_app
        
        # Create the main frame (but don't pack it)
        self.frame = ttk.Frame(parent)
        self.create_widgets()
        self.loaded_npy_files = {}


    def create_widgets(self):
        """Create all widgets for the analysis tab"""
        self.create_summary_section()
        self.create_plot_section()

    def create_summary_section(self):
        """Create data summary section"""
        summary_frame = ttk.LabelFrame(self.frame, text="Data Summary")
        summary_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.summary_text = tk.Text(summary_frame, height=2, wrap=tk.WORD)
        summary_scroll = ttk.Scrollbar(summary_frame, orient=tk.VERTICAL, command=self.summary_text.yview)
        self.summary_text.configure(yscrollcommand=summary_scroll.set)
        
        self.summary_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        summary_scroll.pack(side=tk.RIGHT, fill=tk.Y)

    def create_plot_section(self):
        """Create plot visualization section"""
        plot_frame = ttk.LabelFrame(self.frame, text="Spectrum Visualization")
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Plot controls
        control_frame = ttk.Frame(plot_frame)
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(control_frame, text="Select spectrum:").pack(side=tk.LEFT)
        self.spectrum_selector = ttk.Combobox(control_frame, state="readonly")
        self.spectrum_selector.pack(side=tk.LEFT, padx=5)
        self.spectrum_selector.bind("<<ComboboxSelected>>", self.on_spectrum_change)

        ttk.Button(control_frame, text="Plot in New Window", command=self.plot_in_new_window).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Load .npy File", command=self.load_npy_file).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Save as .npy", command=self.save_as_npy).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="View NPY Raw", command=self.view_npy_raw).pack(side=tk.LEFT, padx=5)

        # Preview canvas frame
        canvas_frame = ttk.LabelFrame(plot_frame, text="Preview")
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create matplotlib canvas for preview
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure
        
        self.preview_fig = Figure(figsize=(10, 6), dpi=80)
        self.preview_canvas = FigureCanvasTkAgg(self.preview_fig, canvas_frame)
        self.preview_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Intensity filter section (only for 2D spectra)
        filter_frame = ttk.LabelFrame(plot_frame, text="2D Intensity Filter")
        filter_frame.pack(fill=tk.X, padx=5, pady=5)

        # Min intensity controls
        ttk.Label(filter_frame, text="Min Intensity:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.min_intensity = tk.DoubleVar(value=0.0)
        self.min_scale = tk.Scale(filter_frame, from_=0, to=100, orient=tk.HORIZONTAL, 
                                variable=self.min_intensity, command=self.update_intensity_filter,
                                resolution=0.1, length=150)
        self.min_scale.grid(row=0, column=1, padx=5, pady=2)

        self.min_entry = tk.Entry(filter_frame, textvariable=self.min_intensity, width=10)
        self.min_entry.grid(row=0, column=2, padx=5, pady=2)
        self.min_entry.bind('<Return>', self.on_entry_change)
        self.min_scale.bind('<MouseWheel>', lambda e: self.on_mouse_wheel(e, self.min_intensity))

        # Max intensity controls
        ttk.Label(filter_frame, text="Max Intensity:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.max_intensity = tk.DoubleVar(value=100.0)
        self.max_scale = tk.Scale(filter_frame, from_=0, to=100, orient=tk.HORIZONTAL,
                                variable=self.max_intensity, command=self.update_intensity_filter,
                                resolution=0.1, length=150)
        self.max_scale.grid(row=1, column=1, padx=5, pady=2)

        self.max_entry = tk.Entry(filter_frame, textvariable=self.max_intensity, width=10)
        self.max_entry.grid(row=1, column=2, padx=5, pady=2)
        self.max_entry.bind('<Return>', self.on_entry_change)
        self.max_scale.bind('<MouseWheel>', lambda e: self.on_mouse_wheel(e, self.max_intensity))

        # Reset filter button
        ttk.Button(filter_frame, text="Reset Filter", command=self.reset_intensity_filter).grid(row=0, column=3, rowspan=2, padx=10)
        
        
        # Initialize filter state
        self.current_spectrum = None
        self.original_spectrum = None

    def update_summary(self, matrices_dict):
        """Update the summary text"""
        self.summary_text.delete(1.0, tk.END)
        
        total_spectra = sum(len(procnos) for procnos in matrices_dict.values())
        self.summary_text.insert(tk.END, f"Total loaded spectra: {total_spectra}\n\n")
        
        for expno, procnos in matrices_dict.items():
            self.summary_text.insert(tk.END, f"Experiment {expno}:\n")
            for procno, data_info in procnos.items():
                matrix = data_info['data']
                self.summary_text.insert(tk.END, f"  Procno {procno}: {matrix.shape}, {matrix.dtype}\n")
            self.summary_text.insert(tk.END, "\n")
        # Add .npy files to summary
        if self.loaded_npy_files:
            self.summary_text.insert(tk.END, "Loaded .npy files:\n")
            for npy_name, file_info in self.loaded_npy_files.items():
                matrix = file_info['data']
                self.summary_text.insert(tk.END, f"  {npy_name}: {matrix.shape}, {matrix.dtype}\n")
            self.summary_text.insert(tk.END, "\n")

    def update_spectrum_selector(self, matrices_dict):
        """Update the spectrum selector dropdown and listbox"""
        spectra_list = []
        
        # Add Bruker spectra
        for expno, procnos in matrices_dict.items():
            for procno in procnos:
                spectra_list.append(f"{expno}/{procno}")
        
        # Add loaded .npy files
        for npy_name in self.loaded_npy_files:
            spectra_list.append(f"npy:{npy_name}")
        
        # Update dropdown
        self.spectrum_selector['values'] = spectra_list
        if spectra_list:
            self.spectrum_selector.set(spectra_list[0])

    def plot_in_new_window(self):
        """Open selected spectrum in a new matplotlib window"""
        selection = self.spectrum_selector.get()
        if not selection:
            return

        # Handle .npy files
        if selection.startswith("npy:"):
            npy_name = selection[4:]  # Remove "npy:" prefix
            if npy_name in self.loaded_npy_files:
                file_info = self.loaded_npy_files[npy_name]
                try:
                    self.main_app.plotting_manager.plot_npy_file(
                        file_info['path'], 
                        title=f"NPY: {npy_name}"
                    )
                except Exception as e:
                    print(f"Error plotting npy file: {e}")
            return

        # Handle Bruker spectra (existing code)
        expno, procno = selection.split('/')
        if expno in self.main_app.matrices_dict and procno in self.main_app.matrices_dict[expno]:
            data_info = self.main_app.matrices_dict[expno][procno]

            # Case 1: processed .npy spectrum
            if data_info['path'].endswith('.npy') and os.path.exists(data_info['path']):
                try:
                    self.main_app.plotting_manager.plot_npy_file(data_info['path'])
                except Exception as e:
                    print(f"Error plotting npy: {e}")
                    # fallback to stored data
                    matrix = data_info['data']
                    self.main_app.plotting_manager.plot_spectrum(matrix, None, f"Exp {expno}, Proc {procno}")

            # Case 2: Bruker spectrum
            else:
                matrix = data_info['data']
                dic = data_info['dic']
                self.main_app.plotting_manager.plot_spectrum(matrix, dic, f"Exp {expno}, Proc {procno}")

    def load_npy_file(self):
        """Load a .npy file and add it to the spectrum selector"""
        file_path = filedialog.askopenfilename(
            title="Select .npy file",
            filetypes=[("NumPy files", "*.npy"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
        
        try:
            # Load the .npy file
            matrix = np.load(file_path)
            
            # Create a unique identifier for this file
            filename = os.path.basename(file_path)
            base_name = os.path.splitext(filename)[0]
            
            # Ensure unique naming if file already loaded
            counter = 1
            unique_name = base_name
            while unique_name in self.loaded_npy_files:
                unique_name = f"{base_name}_{counter}"
                counter += 1
            
            # Store the loaded data
            self.loaded_npy_files[unique_name] = {
                'data': matrix,
                'path': file_path,
                'dic': None  # .npy files don't have Bruker metadata
            }
            
            # Update the spectrum selector
            self.update_spectrum_selector(self.main_app.matrices_dict)
            
            # Select the newly loaded file
            current_values = list(self.spectrum_selector['values'])
            for value in current_values:
                if value.startswith(f"npy:{unique_name}"):
                    self.spectrum_selector.set(value)
                    break
            
            messagebox.showinfo("Success", f"Loaded .npy file: {filename}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load .npy file:\n{str(e)}")

    def save_as_npy(self):
        """Save selected spectrum as .npy file with custom name"""
        selection = self.spectrum_selector.get()
        if not selection:
            messagebox.showwarning("Warning", "No spectrum selected")
            return
        
        # Get the data
        matrix = None
        if selection.startswith("npy:"):
            npy_name = selection[4:]
            if npy_name in self.loaded_npy_files:
                matrix = self.loaded_npy_files[npy_name]['data']
        else:
            expno, procno = selection.split('/')
            if expno in self.main_app.matrices_dict and procno in self.main_app.matrices_dict[expno]:
                matrix = self.main_app.matrices_dict[expno][procno]['data']
        
        if matrix is None:
            messagebox.showerror("Error", "Could not retrieve spectrum data")
            return
        
        # Custom naming dialog
        custom_name = tk.simpledialog.askstring(
            "Save as .npy", 
            f"Enter name for spectrum ({selection}):",
            initialvalue=selection.replace('/', '_').replace(':', '_')
        )
        
        if not custom_name:
            return
        
        # Ensure .npy extension
        if not custom_name.endswith('.npy'):
            custom_name += '.npy'
        
        # Save file dialog
        file_path = filedialog.asksaveasfilename(
            title="Save spectrum as .npy",
            defaultextension=".npy",
            initialvalue=custom_name,
            filetypes=[("NumPy files", "*.npy"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                np.save(file_path, matrix)
                messagebox.showinfo("Success", f"Spectrum saved as: {os.path.basename(file_path)}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save spectrum:\n{str(e)}")

    def _get_spectrum_data(self, selection):
        """Helper method to get spectrum data from selection string"""
        if selection.startswith("npy:"):
            npy_name = selection[4:]
            if npy_name in self.loaded_npy_files:
                file_info = self.loaded_npy_files[npy_name]
                return file_info['data'], file_info['dic'], f"NPY: {npy_name}"
        else:
            expno, procno = selection.split('/')
            if expno in self.main_app.matrices_dict and procno in self.main_app.matrices_dict[expno]:
                data_info = self.main_app.matrices_dict[expno][procno]
                return data_info['data'], data_info['dic'], f"Exp {expno}/Proc {procno}"
        
        return None, None, None
    
    def on_spectrum_change(self, event=None):
        """Handle spectrum selection change"""
        selection = self.spectrum_selector.get()
        if not selection:
            return
        
        matrix, dic, label = self._get_spectrum_data(selection)
        if matrix is not None:
            self.current_spectrum = matrix
            self.original_spectrum = matrix.copy()
            
            if matrix.ndim == 2:
                # Update slider ranges based on data
                data_min = float(np.min(np.abs(matrix)))
                data_max = float(np.max(np.abs(matrix)))
                
                self.min_scale.config(from_=data_min, to=data_max)
                self.max_scale.config(from_=data_min, to=data_max)
                
                self.min_intensity.set(data_min)
                self.max_intensity.set(data_max)
                
            self.update_preview()

    def update_intensity_filter(self, *args):
        """Update intensity filter and preview"""
        if self.original_spectrum is not None and self.original_spectrum.ndim == 2:
            min_val = self.min_intensity.get()
            max_val = self.max_intensity.get()
            
            # Apply filter
            filtered = self.original_spectrum.copy()
            abs_data = np.abs(filtered)
            filtered[abs_data < min_val] = 0
            filtered[abs_data > max_val] = 0
            
            self.current_spectrum = filtered
            self.update_preview()

    def reset_intensity_filter(self):
        """Reset intensity filter to show full range"""
        if self.original_spectrum is not None and self.original_spectrum.ndim == 2:
            data_min = float(np.min(np.abs(self.original_spectrum)))
            data_max = float(np.max(np.abs(self.original_spectrum)))
            
            self.min_intensity.set(data_min)
            self.max_intensity.set(data_max)
            self.current_spectrum = self.original_spectrum.copy()
            self.update_preview()

    def on_entry_change(self, event=None):
        """Handle manual entry changes"""
        self.update_intensity_filter()

    def on_mouse_wheel(self, event, var):
        """Handle mouse wheel scrolling with logarithmic sensitivity based on current range"""
        current_val = var.get()
        
        # Get the current min and max intensity values (not scale limits)
        current_min = self.min_intensity.get()
        current_max = self.max_intensity.get()
        
        # Calculate the actual working range between min and max intensities
        working_range = current_max - (current_min+1)
        
        if working_range > 0:
            # Use log10 of the working range to determine increment sensitivity
            log_factor = np.log10(working_range + 1)
            increment = working_range * (log_factor / 100)  # Adjust divisor for desired sensitivity
        else:
            increment = 0.01  # Fallback for edge cases
        
        # Get scale limits for clamping
        if var == self.min_intensity:
            scale_min, scale_max = self.min_scale.cget('from'), self.min_scale.cget('to')
        else:
            scale_min, scale_max = self.max_scale.cget('from'), self.max_scale.cget('to')
        
        # Reverse scroll direction
        scroll_direction = -1 if event.delta < 0 else 1
        new_val = current_val + (increment * scroll_direction)
        
        # Clamp to valid range
        new_val = max(scale_min, min(scale_max, new_val))
        
        var.set(round(new_val, 6))  # More precision for small values
        self.update_intensity_filter()

    def update_preview(self):
        """Update the preview plot with colorbar"""
        if self.current_spectrum is None:
            return
        
        self.preview_fig.clear()
        ax = self.preview_fig.add_subplot(111)
        
        if self.current_spectrum.ndim == 1:
            ax.plot(self.current_spectrum.real)
            ax.set_title("1D Spectrum Preview")
        elif self.current_spectrum.ndim == 2:
            # Use imshow for better colorbar support
            abs_data = np.abs(self.current_spectrum)
            im = ax.imshow(abs_data, aspect='auto', origin='lower', cmap='viridis')
            
            # Add colorbar
            cbar = self.preview_fig.colorbar(im, ax=ax)
            cbar.set_label('Intensity')
            
            ax.set_title("2D Spectrum Preview (Filtered)")
            ax.set_xlabel("F2 (points)")
            ax.set_ylabel("F1 (points)")
        
        self.preview_fig.tight_layout()
        self.preview_canvas.draw()

    def view_npy_raw(self):
        """Open NPY file and display raw data in popup"""
        file_path = filedialog.askopenfilename(
            title="Select .npy file to view",
            filetypes=[("NumPy files", "*.npy"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
            
        try:
            # Load NPY file in original format
            data = np.load(file_path, allow_pickle=True)
            
            # Create popup window
            popup = tk.Toplevel(self.parent)
            popup.title(f"NPY Raw Data: {os.path.basename(file_path)}")
            popup.geometry("800x600")
            
            # Info frame
            info_frame = ttk.Frame(popup)
            info_frame.pack(fill=tk.X, padx=10, pady=5)
            
            tk.Label(info_frame, text=f"File: {os.path.basename(file_path)}", font=('Arial', 10, 'bold')).pack(anchor=tk.W)
            tk.Label(info_frame, text=f"Shape: {data.shape}").pack(anchor=tk.W)
            tk.Label(info_frame, text=f"Data Type: {data.dtype}").pack(anchor=tk.W)
            tk.Label(info_frame, text=f"Size: {data.size} elements").pack(anchor=tk.W)
            if data.dtype.kind in 'fc':  # complex or float
                tk.Label(info_frame, text=f"Range: {data.min():.3e} to {data.max():.3e}").pack(anchor=tk.W)
            
            # Data display
            text_frame = ttk.Frame(popup)
            text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
            
            text_widget = tk.Text(text_frame, wrap=tk.NONE, font=('Consolas', 9))
            h_scroll = ttk.Scrollbar(text_frame, orient=tk.HORIZONTAL, command=text_widget.xview)
            v_scroll = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=text_widget.yview)
            text_widget.configure(xscrollcommand=h_scroll.set, yscrollcommand=v_scroll.set)
            
            # Display full data with better formatting
            with np.printoptions(threshold=np.inf, linewidth=120, precision=0, suppress=True):
                text_widget.insert(tk.END, str(data))
            
            # Pack scrollbars and text
            text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
            h_scroll.pack(side=tk.BOTTOM, fill=tk.X)
            
            # Make read-only
            text_widget.config(state=tk.DISABLED)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load NPY file:\n{str(e)}")
