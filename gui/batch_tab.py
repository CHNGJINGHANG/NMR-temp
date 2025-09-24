"""
Enhanced Streamlined Batch operations tab for the Enhanced Bruker NMR Data Reader
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import os
from datetime import datetime


class BatchTab:
    def __init__(self, parent, main_app):
        self.parent = parent
        self.main_app = main_app
        self.frame = ttk.Frame(parent)
        self.setup_ui()

    def setup_ui(self):
        """Create and organize all UI elements"""
        self._create_operations_panel()
        self._create_macro_panel()
        self._create_results_panel()

    def _create_operations_panel(self):
        """Create enhanced batch operations panel with scalar support"""
        ops_frame = ttk.LabelFrame(self.frame, text="Batch Operations")
        ops_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Basic operations
        basic_frame = ttk.Frame(ops_frame)
        basic_frame.pack(fill=tk.X, pady=5)
        
        operations = [
            ("Add Spectra", self._add_spectra),
            ("Subtract Spectra", self._subtract_spectra), 
            ("Multiply Spectra", self._multiply_spectra),
            ("Average Spectra", self._average_spectra),
            ("Calculate Gradient", self._gradient_spectrum),
            ("Dot Product", self._dot_product),
            ("Cross Product", self._cross_product),
            ("Convolution", self._convolution),
        ]
        
        for text, command in operations:
            ttk.Button(basic_frame, text=text, command=command).pack(
                side=tk.LEFT, padx=5)
        
        # Scalar operations frame
        scalar_frame = ttk.LabelFrame(ops_frame, text="Scalar Operations")
        scalar_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(scalar_frame, text="Factor:").pack(side=tk.LEFT, padx=5)
        self.scalar_entry = ttk.Entry(scalar_frame, width=10)
        self.scalar_entry.pack(side=tk.LEFT, padx=5)
        self.scalar_entry.insert(0, "1.0")
        
        ttk.Button(scalar_frame, text="Multiply by Factor", 
                  command=self._scalar_multiply).pack(side=tk.LEFT, padx=5)
        ttk.Button(scalar_frame, text="Divide by Factor", 
                  command=self._scalar_divide).pack(side=tk.LEFT, padx=5)

    def _create_macro_panel(self):
        """Create enhanced macro operations panel"""
        macro_frame = ttk.LabelFrame(self.frame, text="Macro Operations")
        macro_frame.pack(fill=tk.X, padx=10, pady=5)
        
        macro_buttons = ttk.Frame(macro_frame)
        macro_buttons.pack(fill=tk.X, pady=5)
        
        ttk.Button(macro_buttons, text="Generate Frames", 
                  command=self._generate_frames).pack(side=tk.LEFT, padx=5)
        ttk.Button(macro_buttons, text="Export to NPY", 
                  command=self._export_npy).pack(side=tk.LEFT, padx=5)
        ttk.Button(macro_buttons, text="Script Editor", 
                  command=self._open_script_editor).pack(side=tk.LEFT, padx=5)
        ttk.Button(macro_buttons, text="Batch Process Folder", 
                  command=self._batch_process_folder).pack(side=tk.LEFT, padx=5)

    def _create_results_panel(self):
        """Create enhanced results display panel"""
        results_frame = ttk.LabelFrame(self.frame, text="Operation Results")
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Results text area with scrollbar
        text_frame = ttk.Frame(results_frame)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.results_text = tk.Text(text_frame, wrap=tk.WORD, height=12, font=('Courier', 9))
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, 
                                 command=self.results_text.yview)
        
        self.results_text.configure(yscrollcommand=scrollbar.set)
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Control buttons
        button_frame = ttk.Frame(results_frame)
        button_frame.pack(pady=5)
        
        ttk.Button(button_frame, text="Clear Results", 
                  command=self._clear_results).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Save Log", 
                  command=self._save_log).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Validate Data", 
                  command=self._validate_all_data).pack(side=tk.LEFT, padx=5)

    def _clear_results(self):
        """Clear the results display"""
        self.results_text.delete(1.0, tk.END)

    def _log_result(self, message, level="INFO"):
        """Log a result message with timestamp and level"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {level}: {message}\n"
        self.results_text.insert(tk.END, formatted_message)
        self.results_text.see(tk.END)
        
        # Color coding for different levels
        if level == "ERROR":
            # Find the last line and color it red
            last_line = self.results_text.index("end-1c linestart")
            self.results_text.tag_add("error", last_line, "end-1c")
            self.results_text.tag_configure("error", foreground="red")
        elif level == "SUCCESS":
            last_line = self.results_text.index("end-1c linestart")
            self.results_text.tag_add("success", last_line, "end-1c")
            self.results_text.tag_configure("success", foreground="green")
        elif level == "WARNING":
            last_line = self.results_text.index("end-1c linestart")
            self.results_text.tag_add("warning", last_line, "end-1c")
            self.results_text.tag_configure("warning", foreground="orange")

    def _save_log(self):
        """Save the current log to a file"""
        log_content = self.results_text.get(1.0, tk.END)
        if not log_content.strip():
            messagebox.showinfo("Empty Log", "No log content to save.")
            return
            
        filename = filedialog.asksaveasfilename(
            title="Save Operation Log",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        
        if filename:
            try:
                with open(filename, 'w') as f:
                    f.write(log_content)
                self._log_result(f"Log saved to {os.path.basename(filename)}", "SUCCESS")
            except Exception as e:
                self._log_result(f"Failed to save log: {str(e)}", "ERROR")

    def _validate_all_data(self):
        """Validate all loaded data and report status"""
        if not self.main_app.matrices_dict:
            self._log_result("No data loaded", "WARNING")
            return
        
        total_spectra = 0
        valid_2d = 0
        errors = []
        
        for expno, procnos in self.main_app.matrices_dict.items():
            if expno in ['batch_results', 'frame_results']:
                continue
            for procno, data_info in procnos.items():
                total_spectra += 1
                try:
                    data = data_info['data']
                    if data.ndim == 2:
                        valid_2d += 1
                    # Additional validation checks
                    if data.size == 0:
                        errors.append(f"{expno}/{procno} - Empty data")
                    if np.any(np.isnan(data)):
                        errors.append(f"{expno}/{procno} - Contains NaN values")
                except Exception as e:
                    errors.append(f"{expno}/{procno} - {str(e)}")
        
        self._log_result(f"Validation complete: {total_spectra} spectra, {valid_2d} are 2D", "INFO")
        if errors:
            self._log_result(f"Found {len(errors)} issues:", "WARNING")
            for error in errors[:5]:  # Show first 5 errors
                self._log_result(f"  {error}", "WARNING")
            if len(errors) > 5:
                self._log_result(f"  ... and {len(errors)-5} more", "WARNING")
        else:
            self._log_result("All data validation passed", "SUCCESS")

    def _validate_data_loaded(self):
        """Check if data is loaded with enhanced feedback"""
        if not self.main_app.matrices_dict:
            self._log_result("No spectra loaded. Please load data first.", "ERROR")
            messagebox.showwarning("No Data", "No spectra loaded. Please load data first.")
            return False
        return True

    def _select_spectra(self, min_count=1, max_count=None, operation="operation"):
        """Enhanced spectrum selection dialog with validation preview"""
        if not self._validate_data_loaded():
            return []
        
        # Get available 2D spectra
        available = []
        for expno, procnos in self.main_app.matrices_dict.items():
            if expno in ['batch_results', 'frame_results']:
                continue
            for procno, data_info in procnos.items():
                if data_info['data'].ndim == 2:
                    available.append(f"{expno}/{procno}")
        
        if len(available) < min_count:
            message = f"Need at least {min_count} 2D spectra, found {len(available)}"
            self._log_result(message, "ERROR")
            messagebox.showwarning("Insufficient Data", message)
            return []
        
        return self._show_enhanced_selection_dialog(available, min_count, max_count, operation)

    def _show_enhanced_selection_dialog(self, items, min_count, max_count, operation):
        """Enhanced spectrum selection dialog with preview and validation"""
        dialog = tk.Toplevel(self.main_app.root)
        dialog.title(f"Select Spectra for {operation.title()}")
        dialog.geometry("500x450")
        dialog.transient(self.main_app.root)
        dialog.grab_set()
        
        # Instructions
        if max_count == 1:
            instruction = f"Select exactly 1 spectrum:"
        elif max_count:
            instruction = f"Select {min_count}-{max_count} spectra:"
        else:
            instruction = f"Select at least {min_count} spectra:"
        
        ttk.Label(dialog, text=instruction, font=('TkDefaultFont', 10, 'bold')).pack(pady=10)
        
        # Listbox with scrollbar
        list_frame = ttk.Frame(dialog)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        listbox = tk.Listbox(list_frame, selectmode=tk.MULTIPLE if max_count != 1 else tk.SINGLE)
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=listbox.yview)
        listbox.configure(yscrollcommand=scrollbar.set)
        
        for item in items:
            listbox.insert(tk.END, item)
        
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Preview frame
        preview_frame = ttk.LabelFrame(dialog, text="Selection Preview")
        preview_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.preview_label = ttk.Label(preview_frame, text="No selection")
        self.preview_label.pack(pady=5)
        
        def update_preview():
            indices = listbox.curselection()
            if not indices:
                self.preview_label.config(text="No selection")
                return
            
            selected = [items[i] for i in indices]
            if len(selected) <= 3:
                preview_text = f"Selected: {', '.join(selected)}"
            else:
                preview_text = f"Selected: {selected[0]}, {selected[1]}, ... (+{len(selected)-2} more)"
            
            # Validate compatibility
            stats = self.main_app.batch_ops.get_batch_statistics(
                self.main_app.matrices_dict, selected)
            
            if not stats.get('compatible', True):
                preview_text += " [INCOMPATIBLE SHAPES]"
                self.preview_label.config(text=preview_text, foreground="red")
            else:
                total_size = stats.get('size_mb', 0)
                preview_text += f" [Size: {total_size:.1f} MB]"
                self.preview_label.config(text=preview_text, foreground="green")
        
        listbox.bind('<<ListboxSelect>>', lambda e: update_preview())
        
        # Selection controls
        if max_count != 1:
            controls = ttk.Frame(dialog)
            controls.pack(pady=5)
            ttk.Button(controls, text="Select All", 
                      command=lambda: [listbox.select_set(0, tk.END), update_preview()]).pack(side=tk.LEFT, padx=5)
            ttk.Button(controls, text="Clear All", 
                      command=lambda: [listbox.select_clear(0, tk.END), update_preview()]).pack(side=tk.LEFT, padx=5)
        
        # Buttons
        selected_items = []
        
        def confirm():
            nonlocal selected_items
            indices = listbox.curselection()
            if len(indices) < min_count:
                messagebox.showwarning("Invalid Selection", 
                                     f"Please select at least {min_count} spectra.")
                return
            if max_count and len(indices) > max_count:
                messagebox.showwarning("Invalid Selection", 
                                     f"Please select at most {max_count} spectra.")
                return
            
            selected_items = [items[i] for i in indices]
            
            # Final validation
            valid, message = self.main_app.batch_ops.validate_spectra_compatibility(
                self.main_app.matrices_dict, selected_items)
            if not valid:
                if messagebox.askyesno("Compatibility Warning", 
                                     f"Spectra may not be compatible: {message}\n\nProceed anyway?"):
                    dialog.destroy()
                else:
                    return
            else:
                dialog.destroy()
        
        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=10)
        ttk.Button(button_frame, text="OK", command=confirm).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT, padx=5)
        
        dialog.wait_window()
        return selected_items

    def _save_result_with_validation(self, data, operation_name, selected_spectra=None):
        """Save operation result with validation and enhanced feedback"""
        save_dir = filedialog.askdirectory(title=f"Save {operation_name} Result")
        if not save_dir:
            self._log_result(f"Save cancelled for {operation_name}", "WARNING")
            return None
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if selected_spectra:
            expno = selected_spectra[0].split('/')[0]
            procnos = '_'.join([spec.split('/')[1] for spec in selected_spectra])
            filename = f"{expno}_{operation_name}_{procnos}_{timestamp}.npy"
        else:
            filename = f"{operation_name}_{timestamp}.npy"
        
        filepath = os.path.join(save_dir, filename)
        
        try:
            # Save the data
            np.save(filepath, data)
            
            # Validate the save
            valid, message = self.main_app.batch_ops.validate_file_save(filepath, data)
            
            if valid:
                self._log_result(f"{operation_name} saved successfully: {os.path.basename(filepath)}", "SUCCESS")
                self._log_result(f"  {message}", "INFO")
                
                # Show save success dialog with option to preview
                result = messagebox.askyesnocancel("Save Complete", 
                    f"File saved successfully!\n\nLocation: {filepath}\n\nWould you like to preview the result?")
                
                if result:  # Yes - show preview
                    self._show_result_preview(data, f"{operation_name} Result")
                    
                return filepath
            else:
                self._log_result(f"Save validation failed for {operation_name}: {message}", "ERROR")
                messagebox.showerror("Save Error", f"File validation failed: {message}")
                return None
                
        except Exception as e:
            error_msg = f"Failed to save {operation_name}: {str(e)}"
            self._log_result(error_msg, "ERROR")
            messagebox.showerror("Save Error", error_msg)
            return None

    def _show_result_preview(self, data, title):
        """Show enhanced result preview with statistics"""
        try:
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            
            preview_window = tk.Toplevel(self.main_app.root)
            preview_window.title(f"Preview: {title}")
            preview_window.geometry("700x600")
            preview_window.transient(self.main_app.root)
            
            # Statistics frame
            stats_frame = ttk.LabelFrame(preview_window, text="Data Statistics")
            stats_frame.pack(fill=tk.X, padx=10, pady=5)
            
            stats_text = f"Shape: {data.shape} | "
            stats_text += f"Data type: {data.dtype} | "
            stats_text += f"Size: {data.nbytes/(1024*1024):.2f} MB\n"
            if np.iscomplexobj(data):
                stats_text += f"Real range: [{np.real(data).min():.3e}, {np.real(data).max():.3e}] | "
                stats_text += f"Imag range: [{np.imag(data).min():.3e}, {np.imag(data).max():.3e}]"
            else:
                stats_text += f"Range: [{data.min():.3e}, {data.max():.3e}]"
            
            ttk.Label(stats_frame, text=stats_text, font=('Courier', 9)).pack(pady=5)
            
            # Plot frame
            fig, ax = plt.subplots(figsize=(8, 6))
            
            if data.ndim == 2:
                plot_data = np.abs(data) if np.iscomplexobj(data) else data
                im = ax.imshow(plot_data, aspect='auto', cmap='viridis')
                ax.set_title(f"{title} (Magnitude)" if np.iscomplexobj(data) else title)
                ax.set_xlabel('F2 (ppm)')
                ax.set_ylabel('F1 (ppm)')
                plt.colorbar(im, ax=ax)
            else:
                ax.text(0.5, 0.5, f"Preview not available\nShape: {data.shape}\nType: {data.dtype}", 
                    ha='center', va='center', transform=ax.transAxes)
                ax.set_title(title)
            
            canvas = FigureCanvasTkAgg(fig, preview_window)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
            
            # Close button
            ttk.Button(preview_window, text="Close", 
                    command=preview_window.destroy).pack(pady=10)
            
        except ImportError:
            # Matplotlib not available
            messagebox.showinfo("Preview", 
                f"Preview not available (matplotlib required)\n\nData shape: {data.shape}\nData type: {data.dtype}")
        except Exception as e:
            self._log_result(f"Preview error: {str(e)}", "ERROR")

    def _store_batch_result(self, data, key, operation_name):
        """Store result in matrices_dict for further use with logging"""
        if 'batch_results' not in self.main_app.matrices_dict:
            self.main_app.matrices_dict['batch_results'] = {}
        
        self.main_app.matrices_dict['batch_results'][key] = {
            'data': data,
            'dic': {},
            'path': f'batch_{operation_name}'
        }
        
        # Update UI
        self.main_app.update_summary()
        self.main_app.update_spectrum_selector()
        
        self._log_result(f"Result stored as {key} in batch_results", "INFO")

    # Enhanced Batch Operations
    def _add_spectra(self):
        """Add selected spectra with enhanced validation and feedback"""
        selected = self._select_spectra(min_count=2, operation="addition")
        if not selected:
            return
        
        self._log_result(f"Starting addition of {len(selected)} spectra", "INFO")
        
        result, count, message = self.main_app.batch_ops.add_spectra(
            self.main_app.matrices_dict, selected)
        
        if result is not None:
            self._store_batch_result(result, '2d_sum', 'add')
            saved_path = self._save_result_with_validation(result, "2D_sum", selected)
            if saved_path:
                self._log_result(f"Addition complete: {message}", "SUCCESS")
            
            # Ask about scalar operation
            if messagebox.askyesno("Scalar Operation", 
                "Would you like to apply a scalar operation (multiply/divide) to the result?"):
                self._apply_scalar_to_result(result, "Addition Result")
        else:
            self._log_result(f"Addition failed: {message}", "ERROR")

    def _subtract_spectra(self):
        """Subtract selected spectra with enhanced validation and feedback"""
        selected = self._select_spectra(min_count=2, operation="subtraction")
        if not selected:
            return
        
        self._log_result(f"Starting subtraction: {selected[0]} - others", "INFO")
        
        result, count, first_name, message = self.main_app.batch_ops.subtract_spectra(
            self.main_app.matrices_dict, selected)
        
        if result is not None:
            self._store_batch_result(result, '2d_diff', 'subtract')
            saved_path = self._save_result_with_validation(result, "2D_diff", selected)
            if saved_path:
                self._log_result(f"Subtraction complete: {message}", "SUCCESS")
                
            # Ask about scalar operation
            if messagebox.askyesno("Scalar Operation", 
                "Would you like to apply a scalar operation (multiply/divide) to the result?"):
                self._apply_scalar_to_result(result, "Subtraction Result")
        else:
            self._log_result(f"Subtraction failed: {message}", "ERROR")

    def _multiply_spectra(self):
        """Multiply selected spectra with enhanced validation and feedback"""
        selected = self._select_spectra(min_count=2, operation="multiplication")
        if not selected:
            return
        
        self._log_result(f"Starting multiplication of {len(selected)} spectra", "INFO")
        
        result, count, message = self.main_app.batch_ops.multiply_spectra(
            self.main_app.matrices_dict, selected)
        
        if result is not None:
            self._store_batch_result(result, '2d_mult', 'multiply')
            saved_path = self._save_result_with_validation(result, "2D_mult", selected)
            if saved_path:
                self._log_result(f"Multiplication complete: {message}", "SUCCESS")
                
            # Ask about scalar operation
            if messagebox.askyesno("Scalar Operation", 
                "Would you like to apply a scalar operation (multiply/divide) to the result?"):
                self._apply_scalar_to_result(result, "Multiplication Result")
        else:
            self._log_result(f"Multiplication failed: {message}", "ERROR")

    def _average_spectra(self):
        """Calculate average of selected spectra"""
        selected = self._select_spectra(min_count=2, operation="averaging")
        if not selected:
            return
        
        self._log_result(f"Starting averaging of {len(selected)} spectra", "INFO")
        
        result, count, message = self.main_app.batch_ops.average_spectra(
            self.main_app.matrices_dict, selected)
        
        if result is not None:
            self._store_batch_result(result, '2d_avg', 'average')
            saved_path = self._save_result_with_validation(result, "2D_average", selected)
            if saved_path:
                self._log_result(f"Averaging complete: {message}", "SUCCESS")
        else:
            self._log_result(f"Averaging failed: {message}", "ERROR")

    def _scalar_multiply(self):
        """Apply scalar multiplication to selected spectrum"""
        selected = self._select_spectra(min_count=1, max_count=1, operation="scalar multiplication")
        if not selected:
            return
        
        try:
            factor = float(self.scalar_entry.get())
        except ValueError:
            self._log_result("Invalid scalar factor entered", "ERROR")
            messagebox.showerror("Invalid Input", "Please enter a valid number for the scalar factor.")
            return
        
        expno, procno = selected[0].split('/')
        data = self.main_app.matrices_dict[expno][procno]['data']
        
        result, message = self.main_app.batch_ops.scalar_multiply_spectrum(data, factor)
        
        if result is not None:
            self._store_batch_result(result, '2d_scalar_mult', 'scalar_multiply')
            saved_path = self._save_result_with_validation(result, f"2D_mult_{factor}", selected)
            if saved_path:
                self._log_result(f"Scalar multiplication: {selected[0]} * {factor}", "SUCCESS")
        else:
            self._log_result(f"Scalar multiplication failed: {message}", "ERROR")

    def _scalar_divide(self):
        """Apply scalar division to selected spectrum"""
        selected = self._select_spectra(min_count=1, max_count=1, operation="scalar division")
        if not selected:
            return
        
        try:
            divisor = float(self.scalar_entry.get())
        except ValueError:
            self._log_result("Invalid scalar divisor entered", "ERROR")
            messagebox.showerror("Invalid Input", "Please enter a valid number for the scalar divisor.")
            return
        
        expno, procno = selected[0].split('/')
        data = self.main_app.matrices_dict[expno][procno]['data']
        
        result, message = self.main_app.batch_ops.scalar_divide_spectrum(data, divisor)
        
        if result is not None:
            self._store_batch_result(result, '2d_scalar_div', 'scalar_divide')
            saved_path = self._save_result_with_validation(result, f"2D_div_{divisor}", selected)
            if saved_path:
                self._log_result(f"Scalar division: {selected[0]} / {divisor}", "SUCCESS")
        else:
            self._log_result(f"Scalar division failed: {message}", "ERROR")

    def _apply_scalar_to_result(self, data, operation_name):
        """Apply scalar operation to an existing result"""
        scalar_dialog = tk.Toplevel(self.main_app.root)
        scalar_dialog.title("Apply Scalar Operation")
        scalar_dialog.geometry("300x200")
        scalar_dialog.transient(self.main_app.root)
        scalar_dialog.grab_set()
        
        ttk.Label(scalar_dialog, text=f"Apply scalar operation to:\n{operation_name}").pack(pady=10)
        
        # Factor entry
        factor_frame = ttk.Frame(scalar_dialog)
        factor_frame.pack(pady=10)
        
        ttk.Label(factor_frame, text="Factor:").pack(side=tk.LEFT, padx=5)
        factor_entry = ttk.Entry(factor_frame, width=10)
        factor_entry.pack(side=tk.LEFT, padx=5)
        factor_entry.insert(0, "1.0")
        factor_entry.focus()
        
        # Operation buttons
        button_frame = ttk.Frame(scalar_dialog)
        button_frame.pack(pady=10)
        
        def apply_multiply():
            try:
                factor = float(factor_entry.get())
                result, message = self.main_app.batch_ops.scalar_multiply_spectrum(data, factor)
                if result is not None:
                    self._store_batch_result(result, '2d_scaled', 'scaled')
                    self._save_result_with_validation(result, f"Scaled_mult_{factor}")
                    self._log_result(f"Applied scalar multiply by {factor}", "SUCCESS")
                else:
                    self._log_result(f"Scalar multiply failed: {message}", "ERROR")
                scalar_dialog.destroy()
            except ValueError:
                messagebox.showerror("Invalid Input", "Please enter a valid number.")
        
        def apply_divide():
            try:
                divisor = float(factor_entry.get())
                result, message = self.main_app.batch_ops.scalar_divide_spectrum(data, divisor)
                if result is not None:
                    self._store_batch_result(result, '2d_scaled', 'scaled')
                    self._save_result_with_validation(result, f"Scaled_div_{divisor}")
                    self._log_result(f"Applied scalar divide by {divisor}", "SUCCESS")
                else:
                    self._log_result(f"Scalar divide failed: {message}", "ERROR")
                scalar_dialog.destroy()
            except ValueError:
                messagebox.showerror("Invalid Input", "Please enter a valid number.")
        
        ttk.Button(button_frame, text="Multiply", command=apply_multiply).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Divide", command=apply_divide).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=scalar_dialog.destroy).pack(side=tk.LEFT, padx=5)

    def _gradient_spectrum(self):
        """Calculate gradient of a spectrum with enhanced feedback"""
        selected = self._select_spectra(min_count=1, max_count=1, operation="gradient")
        if not selected:
            return
        
        self._log_result(f"Calculating gradient for {selected[0]}", "INFO")
        
        grad_x, grad_y, message = self.main_app.batch_ops.nabla_spectrum(
            self.main_app.matrices_dict, selected[0])
        
        if grad_x is not None and grad_y is not None:
            self._store_batch_result(grad_x, '2d_grad_x', 'gradient_x')
            self._store_batch_result(grad_y, '2d_grad_y', 'gradient_y')
            
            self._save_result_with_validation(grad_x, "2D_grad_x", selected)
            self._save_result_with_validation(grad_y, "2D_grad_y", selected)
            self._log_result(f"Gradient calculation complete: {message}", "SUCCESS")
        else:
            self._log_result(f"Gradient calculation failed: {message}", "ERROR")

    def _dot_product(self):
        """Calculate dot product with enhanced feedback"""
        selected = self._select_spectra(min_count=2, max_count=2, operation="dot product")
        if not selected:
            return
        
        self._log_result(f"Calculating dot product: {selected[0]} • {selected[1]}", "INFO")
        
        result, count, message = self.main_app.batch_ops.dot_product_spectra(
            self.main_app.matrices_dict, selected)
        
        if result is not None:
            dot_message = f"Dot product: {selected[0]} • {selected[1]} = {result:.6e}"
            self._log_result(dot_message, "SUCCESS")
            self._log_result(message, "INFO")
        else:
            self._log_result(f"Dot product failed: {message}", "ERROR")

    def _cross_product(self):
        """Calculate cross product with enhanced feedback"""
        selected = self._select_spectra(min_count=2, max_count=2, operation="cross product")
        if not selected:
            return
        
        self._log_result(f"Calculating cross product: {selected[0]} × {selected[1]}", "INFO")
        
        result, count, message = self.main_app.batch_ops.cross_product_spectra(
            self.main_app.matrices_dict, selected)
        
        if result is not None:
            self._store_batch_result(result, '2d_cross', 'cross')
            saved_path = self._save_result_with_validation(result, "2D_cross", selected)
            if saved_path:
                self._log_result(f"Cross product complete: {message}", "SUCCESS")
        else:
            self._log_result(f"Cross product failed: {message}", "ERROR")

    def _convolution(self):
        """Convolve spectrum with external kernel using file dialog"""
        selected = self._select_spectra(min_count=1, max_count=1, operation="convolution")
        if not selected:
            return
        
        kernel_path = filedialog.askopenfilename(
            title="Select Convolution Kernel File",
            filetypes=[("NumPy files", "*.npy"), ("All files", "*.*")])
        
        if not kernel_path:
            self._log_result("Kernel file selection cancelled", "WARNING")
            return
        
        self._log_result(f"Loading kernel from: {os.path.basename(kernel_path)}", "INFO")
        
        try:
            kernel_data = np.load(kernel_path)
            self._log_result(f"Kernel loaded: shape {kernel_data.shape}, type {kernel_data.dtype}", "INFO")
            
            self._log_result(f"Starting convolution: {selected[0]} with kernel", "INFO")
            
            result, message = self.main_app.batch_ops.convolve_spectrum_with_kernel(
                self.main_app.matrices_dict, selected[0], kernel_data)
            
            if result is not None:
                self._store_batch_result(result, '2d_conv', 'convolve')
                saved_path = self._save_result_with_validation(result, "2D_conv", selected)
                if saved_path:
                    self._log_result(f"Convolution complete: {message}", "SUCCESS")
            else:
                self._log_result(f"Convolution failed: {message}", "ERROR")
                
        except Exception as e:
            error_msg = f"Kernel loading failed: {str(e)}"
            self._log_result(error_msg, "ERROR")
            messagebox.showerror("Error", error_msg)

    def _batch_process_folder(self):
        """Process entire folder with enhanced file dialogs"""
        input_folder = filedialog.askdirectory(title="Select Input Folder with EXPNO Directories")
        if not input_folder:
            return
        
        output_folder = filedialog.askdirectory(title="Select Output Folder for Results")
        if not output_folder:
            return
        
        # Operation selection dialog
        op_dialog = tk.Toplevel(self.main_app.root)
        op_dialog.title("Select Batch Operation")
        op_dialog.geometry("400x300")
        op_dialog.transient(self.main_app.root)
        op_dialog.grab_set()
        
        ttk.Label(op_dialog, text="Select operation to apply:").pack(pady=10)
        
        operation_var = tk.StringVar(value="add")
        operations = [("Add PROCNO 231 + 232", "add"),
                     ("Multiply PROCNO 231 * 232", "multiply"),
                     ("Convolve with kernel", "convolve")]
        
        for text, value in operations:
            ttk.Radiobutton(op_dialog, text=text, variable=operation_var, value=value).pack(anchor=tk.W, padx=20, pady=5)
        
        # Kernel selection for convolution
        kernel_frame = ttk.LabelFrame(op_dialog, text="Kernel File (for convolution)")
        kernel_frame.pack(fill=tk.X, padx=10, pady=10)
        
        kernel_path_var = tk.StringVar()
        ttk.Label(kernel_frame, textvariable=kernel_path_var).pack(pady=5)
        
        def select_kernel():
            path = filedialog.askopenfilename(title="Select Kernel File", 
                                            filetypes=[("NumPy files", "*.npy")])
            if path:
                kernel_path_var.set(os.path.basename(path))
        
        ttk.Button(kernel_frame, text="Browse Kernel", command=select_kernel).pack(pady=5)
        
        result_info = {"cancelled": True}
        
        def start_processing():
            operation = operation_var.get()
            kernel_path = kernel_path_var.get()
            
            if operation == "convolve" and not kernel_path:
                messagebox.showwarning("Missing Kernel", "Please select a kernel file for convolution.")
                return
            
            result_info["cancelled"] = False
            result_info["operation"] = operation
            result_info["kernel_path"] = kernel_path if kernel_path else None
            op_dialog.destroy()
        
        button_frame = ttk.Frame(op_dialog)
        button_frame.pack(pady=20)
        ttk.Button(button_frame, text="Start Processing", command=start_processing).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=op_dialog.destroy).pack(side=tk.LEFT, padx=5)
        
        op_dialog.wait_window()
        
        if result_info["cancelled"]:
            return
        
        # Process the folder
        try:
            self._log_result(f"Starting batch processing: {input_folder}", "INFO")
            self._log_result(f"Operation: {result_info['operation']}", "INFO")
            
            results, operation_folder = self.main_app.batch_ops.process_macro_operation(
                input_folder, output_folder, ['231', '232'], 
                result_info["operation"], result_info.get("kernel_path"))
            
            self._log_result(f"Batch processing complete: {results['processed']} processed", "SUCCESS")
            if results['errors']:
                self._log_result(f"Errors encountered: {len(results['errors'])}", "WARNING")
                for error in results['errors'][:3]:
                    self._log_result(f"  {error}", "WARNING")
                    
        except Exception as e:
            error_msg = f"Batch processing failed: {str(e)}"
            self._log_result(error_msg, "ERROR")
            messagebox.showerror("Error", error_msg)

    def _generate_frames(self):
        """Generate combined frames with enhanced validation and feedback"""
        if not self._validate_data_loaded():
            return
        
        # Get available experiments
        expnos = [expno for expno in self.main_app.matrices_dict.keys() 
                 if expno not in ['batch_results', 'frame_results']]
        
        if not expnos:
            self._log_result("No experiments available for frame generation", "ERROR")
            messagebox.showwarning("No Data", "No experiments available.")
            return
        
        save_location = filedialog.askdirectory(title="Select Output Folder for Generated Frames")
        if not save_location:
            self._log_result("Frame generation cancelled - no output folder", "WARNING")
            return
        
        self._log_result(f"Starting frame generation for {len(expnos)} experiments", "INFO")
        
        try:
            results, main_folder = self.main_app.batch_ops.generate_combined_frames(
                self.main_app.matrices_dict, expnos, save_location)
            
            self._log_result(f"Frame generation complete: {results['processed']} processed", "SUCCESS")
            self._log_result(f"Output folder: {os.path.basename(main_folder)}", "INFO")
            
            if results['saved_files']:
                self._log_result(f"Generated files: {', '.join(results['saved_files'][:5])}", "INFO")
                if len(results['saved_files']) > 5:
                    self._log_result(f"... and {len(results['saved_files'])-5} more", "INFO")
            
            if results['errors']:
                self._log_result(f"Errors: {len(results['errors'])}", "WARNING")
                for error in results['errors'][:3]:
                    self._log_result(f"  {error}", "WARNING")
            
            self.main_app.update_summary()
            self.main_app.update_spectrum_selector()
            
            # Show completion dialog
            messagebox.showinfo("Frame Generation Complete", 
                f"Generated {results['processed']} combined frames\nSaved to: {main_folder}")
            
        except Exception as e:
            error_msg = f"Frame generation failed: {str(e)}"
            self._log_result(error_msg, "ERROR")
            messagebox.showerror("Error", error_msg)
            
    def _export_npy(self):
        """Export selected spectra to .npy files with enhanced feedback"""
        if not self._validate_data_loaded():
            return
        
        # Get all available spectra
        available = []
        for expno, procnos in self.main_app.matrices_dict.items():
            if expno in ['batch_results', 'frame_results']:
                continue
            for procno in procnos.keys():
                available.append(f"{expno}/{procno}")
        
        if not available:
            self._log_result("No spectra available for export", "ERROR")
            messagebox.showwarning("No Data", "No spectra available for export.")
            return
        
        selected = self._show_enhanced_selection_dialog(available, 1, None, "export")
        if not selected:
            return
        
        save_location = filedialog.askdirectory(title="Select Export Folder")
        if not save_location:
            self._log_result("Export cancelled - no output folder", "WARNING")
            return
        
        self._log_result(f"Starting export of {len(selected)} spectra", "INFO")
        
        try:
            results = self.main_app.batch_ops.export_to_npy(
                self.main_app.matrices_dict, selected, save_location)
            
            self._log_result(f"Export complete: {results['exported']} files exported", "SUCCESS")
            self._log_result(f"Export location: {save_location}", "INFO")
            
            if results['saved_files']:
                self._log_result(f"Exported files: {', '.join(results['saved_files'][:5])}", "INFO")
                if len(results['saved_files']) > 5:
                    self._log_result(f"... and {len(results['saved_files'])-5} more", "INFO")
            
            if results['errors']:
                self._log_result(f"Export errors: {len(results['errors'])}", "WARNING")
                for error in results['errors'][:3]:
                    self._log_result(f"  {error}", "WARNING")
                    
        except Exception as e:
            error_msg = f"Export failed: {str(e)}"
            self._log_result(error_msg, "ERROR")
            messagebox.showerror("Error", error_msg)

    def _open_script_editor(self):
        """Open the enhanced macro script editor"""
        ScriptEditor(self.main_app, self)


class ScriptEditor:
    """Enhanced macro script editor with validation and examples"""
    
    def __init__(self, main_app, batch_tab):
        self.main_app = main_app
        self.batch_tab = batch_tab
        self.create_editor()
    
    def create_editor(self):
        """Create the enhanced script editor window"""
        self.window = tk.Toplevel(self.main_app.root)
        self.window.title("Enhanced Macro Script Editor")
        self.window.geometry("800x600")
        
        # Create notebook for tabs
        notebook = ttk.Notebook(self.window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Script editing tab
        self.script_frame = ttk.Frame(notebook)
        notebook.add(self.script_frame, text="Script Editor")
        self._create_script_tab()
        
        # Examples tab
        self.examples_frame = ttk.Frame(notebook)
        notebook.add(self.examples_frame, text="Examples")
        self._create_examples_tab()
        
        # Validation tab
        self.validation_frame = ttk.Frame(notebook)
        notebook.add(self.validation_frame, text="Validation")
        self._create_validation_tab()
    
    def _create_script_tab(self):
        """Create the main script editing tab"""
        # Toolbar
        toolbar = ttk.Frame(self.script_frame)
        toolbar.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(toolbar, text="Load Script", command=self._load_script).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Save Script", command=self._save_script).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Validate Script", command=self._validate_script).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Execute Script", command=self._execute_script).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Clear", command=self._clear_script).pack(side=tk.LEFT, padx=2)
        
        # Script text area with line numbers
        text_frame = ttk.Frame(self.script_frame)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Line numbers
        self.line_numbers = tk.Text(text_frame, width=4, padx=3, takefocus=0,
                                   border=0, state='disabled', wrap='none',
                                   font=('Courier', 10), background='lightgray')
        self.line_numbers.pack(side=tk.LEFT, fill=tk.Y)
        
        # Script text
        self.script_text = tk.Text(text_frame, wrap=tk.NONE, font=('Courier', 10),
                                 undo=True, maxundo=20)
        scrollbar_v = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.script_text.yview)
        scrollbar_h = ttk.Scrollbar(self.window, orient=tk.HORIZONTAL, command=self.script_text.xview)
        
        self.script_text.configure(yscrollcommand=scrollbar_v.set, xscrollcommand=scrollbar_h.set)
        self.script_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar_v.pack(side=tk.RIGHT, fill=tk.Y)
        scrollbar_h.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Bind events for line numbers
        self.script_text.bind('<KeyRelease>', self._update_line_numbers)
        self.script_text.bind('<ButtonRelease>', self._update_line_numbers)
        
        # Load enhanced template
        self._load_enhanced_template()
    
    def _create_examples_tab(self):
        """Create examples tab with common script patterns"""
        examples_text = tk.Text(self.examples_frame, wrap=tk.WORD, font=('Courier', 9))
        examples_scroll = ttk.Scrollbar(self.examples_frame, orient=tk.VERTICAL, command=examples_text.yview)
        examples_text.configure(yscrollcommand=examples_scroll.set)
        
        examples_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        examples_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        examples_content = '''# ENHANCED NMR MACRO SCRIPT EXAMPLES

# Example 1: Basic Addition with Averaging
SET INPUT_PATH "/path/to/input"
SET OUTPUT_PATH "/path/to/output"

LOAD spectrum1.npy AS spec1
LOAD spectrum2.npy AS spec2
ADD spec1 spec2 AS sum
DIVIDE sum 2 AS average
SAVE average TO "$OUTPUT_PATH/averaged_result.npy"

# Example 2: Complex Processing with Scaling
LOOP EXPNO IN $INPUT_PATH
    LOAD $EXPNO/231 AS real_part
    LOAD $EXPNO/232 AS imag_part
    
    ADD real_part imag_part AS combined
    MULTIPLY combined 0.5 AS scaled
    
    SAVE scaled TO "$OUTPUT_PATH/$EXPNO_processed.npy"
    PRINT "Processed experiment $EXPNO"
END LOOP

# Example 3: Sequential Operations
LOAD data1.npy AS d1
LOAD data2.npy AS d2
LOAD data3.npy AS d3

ADD d1 d2 AS intermediate
MULTIPLY intermediate d3 AS product
DIVIDE product 3 AS normalized

SAVE normalized TO "final_result.npy"
PRINT "Processing complete"

# Example 4: Conditional-style Processing
SET SCALE_FACTOR 2.5

LOAD input_spectrum.npy AS input
MULTIPLY input $SCALE_FACTOR AS scaled
SUBTRACT scaled input AS difference
SAVE difference TO "scaled_difference.npy"

# Available Commands:
# SET variable value - Set a variable
# LOAD path AS alias - Load spectrum data
# ADD spec1 spec2 [spec3...] AS result - Add spectra
# SUBTRACT spec1 spec2 AS result - Subtract (spec1 - spec2)
# MULTIPLY operand1 operand2 AS result - Multiply (spectrum*spectrum or spectrum*scalar)
# DIVIDE spectrum divisor AS result - Divide spectrum by scalar
# SAVE alias TO "path" - Save spectrum to file
# LOOP EXPNO IN path ... END LOOP - Loop over experiments
# PRINT "message" - Print message to log
'''
        examples_text.insert(1.0, examples_content)
        examples_text.configure(state='disabled')
    
    def _create_validation_tab(self):
        """Create validation tab with syntax checking"""
        validation_frame = ttk.Frame(self.validation_frame)
        validation_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        ttk.Label(validation_frame, text="Script Validation Results", 
                 font=('TkDefaultFont', 12, 'bold')).pack(pady=5)
        
        self.validation_text = tk.Text(validation_frame, wrap=tk.WORD, height=20, font=('Courier', 9))
        validation_scroll = ttk.Scrollbar(validation_frame, orient=tk.VERTICAL, 
                                        command=self.validation_text.yview)
        self.validation_text.configure(yscrollcommand=validation_scroll.set)
        
        self.validation_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        validation_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        ttk.Button(self.validation_frame, text="Run Validation", 
                  command=self._validate_script).pack(pady=10)
    
    def _update_line_numbers(self, event=None):
        """Update line numbers display"""
        line_count = int(self.script_text.index('end-1c').split('.')[0])
        line_numbers_string = "\n".join(str(i) for i in range(1, line_count + 1))
        
        self.line_numbers.config(state='normal')
        self.line_numbers.delete(1.0, tk.END)
        self.line_numbers.insert(1.0, line_numbers_string)
        self.line_numbers.config(state='disabled')
    
    def _load_enhanced_template(self):
        """Load enhanced script template with scalar operations"""
        template = '''# Enhanced NMR Macro Script Template
# Available commands: SET, LOAD, ADD, SUBTRACT, MULTIPLY, DIVIDE, SAVE, LOOP, PRINT

SET INPUT_PATH "/path/to/input"
SET OUTPUT_PATH "/path/to/output"

# Example: Process and average spectra
LOOP EXPNO IN $INPUT_PATH
    LOAD $EXPNO/231 AS spec1
    LOAD $EXPNO/232 AS spec2
    
    ADD spec1 spec2 AS sum
    DIVIDE sum 2 AS average
    
    MULTIPLY average 1.5 AS scaled
    
    SAVE scaled TO "$OUTPUT_PATH/$EXPNO_processed.npy"
    PRINT "Processed and averaged experiment $EXPNO"
END LOOP

# Standalone operations example
LOAD data1.npy AS d1
LOAD data2.npy AS d2

MULTIPLY d1 d2 AS product
DIVIDE product 10 AS normalized

SAVE normalized TO "final_result.npy"
'''
        self.script_text.insert(tk.END, template)
        self._update_line_numbers()
    
    def _validate_script(self):
        """Enhanced script validation with detailed feedback"""
        script_content = self.script_text.get(1.0, tk.END)
        lines = [line.strip() for line in script_content.split('\n') if line.strip() and not line.strip().startswith('#')]
        
        validation_results = []
        errors = []
        warnings = []
        variables = set()
        aliases = set()
        
        validation_results.append("=== SCRIPT VALIDATION RESULTS ===\n")
        validation_results.append(f"Total lines to process: {len(lines)}\n\n")
        
        # Validate syntax
        for i, line in enumerate(lines, 1):
            try:
                if line.startswith('SET '):
                    parts = line[4:].split(' ', 1)
                    if len(parts) != 2:
                        errors.append(f"Line {i}: Invalid SET syntax - {line}")
                    else:
                        variables.add(parts[0])
                        
                elif line.startswith('LOAD '):
                    if ' AS ' not in line:
                        errors.append(f"Line {i}: LOAD missing AS clause - {line}")
                    else:
                        alias = line.split(' AS ')[1].strip()
                        aliases.add(alias)
                        
                elif line.startswith('SAVE '):
                    if ' TO ' not in line:
                        errors.append(f"Line {i}: SAVE missing TO clause - {line}")
                    else:
                        alias = line[5:].split(' TO ')[0].strip()
                        if alias not in aliases:
                            warnings.append(f"Line {i}: Saving undefined alias '{alias}'")
                            
                elif any(op in line for op in ['ADD ', 'SUBTRACT ', 'MULTIPLY ', 'DIVIDE ']):
                    if ' AS ' not in line:
                        errors.append(f"Line {i}: Operation missing AS clause - {line}")
                    else:
                        # Check operands exist
                        parts = line.split(' AS ')[0]
                        if line.startswith('MULTIPLY ') or line.startswith('DIVIDE '):
                            operands = parts.split()[1:]  # Skip command
                            if len(operands) == 2:
                                try:
                                    float(operands[1])  # Check if scalar
                                except ValueError:
                                    if operands[1] not in aliases:
                                        warnings.append(f"Line {i}: Undefined operand '{operands[1]}'")
                                if operands[0] not in aliases:
                                    warnings.append(f"Line {i}: Undefined operand '{operands[0]}'")
                        
                        result_alias = line.split(' AS ')[1].strip()
                        aliases.add(result_alias)
                        
                elif line.startswith('LOOP '):
                    if 'END LOOP' not in '\n'.join(lines[i:]):
                        errors.append(f"Line {i}: LOOP without matching END LOOP")
                        
            except Exception as e:
                errors.append(f"Line {i}: Parse error - {str(e)}")
        
        # Report results
        if not errors and not warnings:
            validation_results.append("✓ Script validation PASSED\n")
            validation_results.append("No syntax errors or warnings found.\n\n")
        else:
            if errors:
                validation_results.append(f"✗ Found {len(errors)} ERRORS:\n")
                for error in errors:
                    validation_results.append(f"  ERROR: {error}\n")
                validation_results.append("\n")
            
            if warnings:
                validation_results.append(f"⚠ Found {len(warnings)} WARNINGS:\n")
                for warning in warnings:
                    validation_results.append(f"  WARNING: {warning}\n")
                validation_results.append("\n")
        
        validation_results.append(f"Variables defined: {', '.join(variables) if variables else 'None'}\n")
        validation_results.append(f"Aliases used: {', '.join(aliases) if aliases else 'None'}\n")
        
        # Display results
        self.validation_text.delete(1.0, tk.END)
        self.validation_text.insert(1.0, ''.join(validation_results))
        
        return len(errors) == 0
    
    def _load_script(self):
        """Load script from file with validation"""
        filename = filedialog.askopenfilename(
            title="Load Macro Script",
            filetypes=[("Text files", "*.txt"), ("Macro files", "*.mac"), ("All files", "*.*")])
        
        if filename:
            try:
                with open(filename, 'r') as f:
                    content = f.read()
                self.script_text.delete(1.0, tk.END)
                self.script_text.insert(1.0, content)
                self._update_line_numbers()
                
                # Auto-validate loaded script
                if self._validate_script():
                    self.batch_tab._log_result(f"Script loaded successfully: {os.path.basename(filename)}", "SUCCESS")
                else:
                    self.batch_tab._log_result(f"Script loaded with validation issues: {os.path.basename(filename)}", "WARNING")
                    
            except Exception as e:
                error_msg = f"Failed to load script: {str(e)}"
                self.batch_tab._log_result(error_msg, "ERROR")
                messagebox.showerror("Error", error_msg)
    
    def _save_script(self):
        """Save script to file with validation"""
        # Validate before saving
        if not self._validate_script():
            if not messagebox.askyesno("Validation Issues", 
                "Script has validation issues. Save anyway?"):
                return
        
        filename = filedialog.asksaveasfilename(
            title="Save Macro Script",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("Macro files", "*.mac"), ("All files", "*.*")])
        
        if filename:
            try:
                content = self.script_text.get(1.0, tk.END)
                with open(filename, 'w') as f:
                    f.write(content)
                self.batch_tab._log_result(f"Script saved successfully: {os.path.basename(filename)}", "SUCCESS")
                messagebox.showinfo("Success", "Script saved successfully")
            except Exception as e:
                error_msg = f"Failed to save script: {str(e)}"
                self.batch_tab._log_result(error_msg, "ERROR")
                messagebox.showerror("Error", error_msg)
    
    def _execute_script(self):
        """Execute the current script with enhanced error handling"""
        # Validate before execution
        if not self._validate_script():
            if not messagebox.askyesno("Validation Issues", 
                "Script has validation issues. Execute anyway?"):
                return
        
        script_content = self.script_text.get(1.0, tk.END)
        
        if not script_content.strip():
            self.batch_tab._log_result("Cannot execute empty script", "ERROR")
            return
        
        self.batch_tab._log_result("Starting script execution...", "INFO")
        
        try:
            results = self.main_app.batch_ops.execute_macro_script(script_content)
            
            # Log detailed results to batch tab
            self.batch_tab._log_result(f"Script execution complete: {results.get('processed', 0)} operations processed", "SUCCESS")
            
            if results.get('messages'):
                for message in results['messages']:
                    self.batch_tab._log_result(f"Script: {message}", "INFO")
            
            if results.get('errors'):
                self.batch_tab._log_result(f"Script errors: {len(results['errors'])}", "WARNING")
                for error in results['errors'][:5]:  # Show first 5 errors
                    self.batch_tab._log_result(f"Script error: {error}", "ERROR")
                if len(results['errors']) > 5:
                    self.batch_tab._log_result(f"... and {len(results['errors'])-5} more errors", "WARNING")
            
            if results.get('skipped'):
                self.batch_tab._log_result(f"Skipped operations: {len(results['skipped'])}", "WARNING")
            
            # Show completion dialog
            success_count = results.get('processed', 0)
            error_count = len(results.get('errors', []))
            
            if error_count == 0:
                messagebox.showinfo("Script Complete", f"Script executed successfully!\n{success_count} operations completed.")
            else:
                messagebox.showwarning("Script Complete with Errors", 
                    f"Script completed with issues:\n{success_count} operations successful\n{error_count} errors encountered")
            
        except Exception as e:
            error_msg = f"Script execution failed: {str(e)}"
            self.batch_tab._log_result(error_msg, "ERROR")
            messagebox.showerror("Execution Error", error_msg)
    
    def _clear_script(self):
        """Clear the script editor with confirmation"""
        if messagebox.askyesno("Clear Script", "Clear the current script? This cannot be undone."):
            self.script_text.delete(1.0, tk.END)
            self._update_line_numbers()
            self.validation_text.delete(1.0, tk.END)