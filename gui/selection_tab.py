"""
Data selection tab for the Enhanced Bruker NMR Data Reader
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
from utils.helpers import get_dimension

class SelectionTab:
    def __init__(self, parent, main_app):
        self.parent = parent
        self.main_app = main_app
        
        # Create the main frame (but don't pack it)
        self.frame = ttk.Frame(parent)
        self.create_widgets()

    def create_widgets(self):
        """Create all widgets for the selection tab"""
        self.create_folder_selection()
        self.create_filter_options()
        self.create_main_selection_area()
        self.create_action_buttons()

    def create_folder_selection(self):
        """Create folder selection widgets"""
        folder_frame = ttk.Frame(self.frame)
        folder_frame.pack(fill=tk.X, padx=10, pady=5)
        
        select_btn = ttk.Button(folder_frame, text="Select Bruker Experiment Folder",
                                command=self.select_experiment_folder)
        select_btn.pack(side=tk.LEFT)
        
        self.folder_label = ttk.Label(folder_frame, text="No folder selected")
        self.folder_label.pack(side=tk.LEFT, padx=(10, 0))

    def create_filter_options(self):
        """Create filter option widgets"""
        filter_frame = ttk.Frame(self.frame)
        filter_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(filter_frame, text="Filter by dimension:").pack(side=tk.LEFT)
        self.dim_filter = ttk.Combobox(filter_frame, values=["All", "1D", "2D", "3D"], state="readonly")
        self.dim_filter.set("All")
        self.dim_filter.pack(side=tk.LEFT, padx=(5, 10))
        self.dim_filter.bind("<<ComboboxSelected>>", self.apply_filter)

        ttk.Button(filter_frame, text="Refresh", command=self.refresh_list).pack(side=tk.LEFT, padx=5)

    def create_main_selection_area(self):
        """Create the main selection area with listboxes"""
        main_frame = ttk.Frame(self.frame)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10)

        # Left side - Available data
        self.create_available_list(main_frame)
        
        # Middle buttons
        self.create_control_buttons(main_frame)
        
        # Right side - Selected data
        self.create_selected_list(main_frame)

    def create_available_list(self, parent):
        """Create available experiments listbox"""
        left_frame = ttk.Frame(parent)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        ttk.Label(left_frame, text="Available experiment folders").pack()
        
        listbox_frame = ttk.Frame(left_frame)
        listbox_frame.pack(fill=tk.BOTH, expand=True)
        
        self.listbox = tk.Listbox(listbox_frame, selectmode=tk.MULTIPLE)
        scrollbar1 = ttk.Scrollbar(listbox_frame, orient=tk.VERTICAL, command=self.listbox.yview)
        self.listbox.configure(yscrollcommand=scrollbar1.set)
        
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar1.pack(side=tk.RIGHT, fill=tk.Y)

    def create_control_buttons(self, parent):
        """Create control buttons between listboxes"""
        button_frame = ttk.Frame(parent)
        button_frame.pack(side=tk.LEFT, padx=10, pady=50)
        
        ttk.Button(button_frame, text="Add Selected →", command=self.add_selected).pack(pady=5)
        ttk.Button(button_frame, text="← Remove", command=self.remove_selected).pack(pady=5)
        ttk.Button(button_frame, text="Add All →", command=self.add_all).pack(pady=5)
        ttk.Button(button_frame, text="Clear All", command=self.clear_all).pack(pady=5)

    def create_selected_list(self, parent):
        """Create selected experiments listbox"""
        right_frame = ttk.Frame(parent)
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        ttk.Label(right_frame, text="Selected experiments").pack()
        
        selected_frame = ttk.Frame(right_frame)
        selected_frame.pack(fill=tk.BOTH, expand=True)
        
        self.selected_listbox = tk.Listbox(selected_frame)
        scrollbar2 = ttk.Scrollbar(selected_frame, orient=tk.VERTICAL, command=self.selected_listbox.yview)
        self.selected_listbox.configure(yscrollcommand=scrollbar2.set)
        
        self.selected_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar2.pack(side=tk.RIGHT, fill=tk.Y)

    def create_action_buttons(self):
        """Create action buttons and progress bar"""
        action_frame = ttk.Frame(self.frame)
        action_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(action_frame, text="Read Selected Data", command=self.read_selected).pack(side=tk.LEFT)
        ttk.Button(action_frame, text="Export Metadata", command=self.export_metadata).pack(side=tk.LEFT, padx=5)
        
        # Progress bar
        self.progress = ttk.Progressbar(action_frame, mode='determinate')
        self.progress.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(10, 0))

    def select_experiment_folder(self):
        """Handle folder selection"""
        folder = filedialog.askdirectory(title="Select Bruker Experiment Folder")
        if not folder:
            return

        self.main_app.current_folder = folder
        self.folder_label.config(text=f"Selected: {os.path.basename(folder)}")
        self.refresh_list()

    def refresh_list(self):
        """Refresh the available experiments list"""
        if not self.main_app.current_folder:
            return
            
        self.listbox.delete(0, tk.END)
        self.main_app.experiment_paths.clear()

        # Use data reader to find experiments
        experiments = self.main_app.data_reader.find_experiments(self.main_app.current_folder)
        self.main_app.experiment_paths = experiments
        self.apply_filter()

    def apply_filter(self, event=None):
        """Apply dimension filter to the list"""
        filter_dim = self.dim_filter.get()
        self.listbox.delete(0, tk.END)
        
        for exp_path, dimensions, display_name in self.main_app.experiment_paths:
            if filter_dim == "All":
                self.listbox.insert(tk.END, display_name)
            else:
                filter_num = int(filter_dim[0])  # Extract number from "1D", "2D", etc.
                if filter_num in dimensions:
                    self.listbox.insert(tk.END, display_name)

    def add_selected(self):
        """Add selected experiments to the selected list"""
        selected_indices = self.listbox.curselection()
        current_items = [self.listbox.get(i) for i in range(self.listbox.size())]
        
        for i in selected_indices:
            display_name = current_items[i]
            # Find the actual path
            for exp_path, dimensions, disp in self.main_app.experiment_paths:
                if disp == display_name:
                    # Check if path is already selected
                    if exp_path not in self.main_app.selected_paths:
                        self.main_app.selected_paths.append(exp_path)
                        self.selected_listbox.insert(tk.END, display_name)
                    break

    def remove_selected(self):
        """Remove selected experiments from the selected list"""
        selected_indices = list(self.selected_listbox.curselection())
        for i in reversed(selected_indices):
            self.selected_listbox.delete(i)
            if i < len(self.main_app.selected_paths):
                self.main_app.selected_paths.pop(i)

    def add_all(self):
        """Add all experiments to the selected list"""
        for exp_path, dimensions, display_name in self.main_app.experiment_paths:
            if exp_path not in self.main_app.selected_paths:
                self.main_app.selected_paths.append(exp_path)
                self.selected_listbox.insert(tk.END, display_name)

    def clear_all(self):
        """Clear all selected experiments"""
        self.selected_listbox.delete(0, tk.END)
        self.main_app.selected_paths.clear()

    def read_selected(self):
        """Read the selected experiments"""
        if not self.main_app.selected_paths:
            self.main_app.show_message("No Selection", "Please select experiments to read.", "warning")
            return

        # Use data reader to load experiments
        success = self.main_app.data_reader.read_experiments(
            self.main_app.selected_paths,
            self.main_app.matrices_dict,
            self.progress,
            self.main_app.root
        )
        
        if success:
            self.main_app.update_summary()
            self.main_app.update_spectrum_selector()
            
            total_spectra = sum(len(procnos) for procnos in self.main_app.matrices_dict.values())
            total_exps = len(self.main_app.selected_paths)
            self.main_app.show_message("Complete", 
                                     f"Successfully loaded {total_spectra} spectra from {total_exps} experiments")

    def export_metadata(self):
        """Export metadata of loaded spectra"""
        if not self.main_app.matrices_dict:
            self.main_app.show_message("No Data", "No spectra loaded.", "warning")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if not filename:
            return
        
        success = self.main_app.data_reader.export_metadata(self.main_app.matrices_dict, filename)
        if success:
            self.main_app.show_message("Complete", f"Metadata exported to {filename}")
        else:
            self.main_app.show_message("Error", "Failed to export metadata", "error")