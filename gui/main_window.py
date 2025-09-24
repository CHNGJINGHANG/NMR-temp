"""
Main window and controller for the Enhanced Bruker NMR Data Reader
"""

import tkinter as tk
from tkinter import ttk, messagebox
import os
from core.data_reader import BrukerDataReader
from core.batch_ops import BatchOperations
from core.plotting import PlottingManager
from .selection_tab import SelectionTab
from .analysis_tab import AnalysisTab
from .batch_tab import BatchTab
from .video import VideoTab
from .cluster_tab import ClusterTab
from gui.peak_tracking_tab import PeakTrackingTab


class EnhancedBrukerReader:
    def __init__(self, root):
        self.root = root
        self.root.title("Enhanced Bruker NMR Data Reader")
        self.root.geometry("1200x800")

        # Initialize data components
        self.data_reader = BrukerDataReader()
        self.batch_ops = BatchOperations()
        self.plotting_manager = PlottingManager()
        
        # Data storage
        self.experiment_paths = []
        self.selected_paths = []
        self.matrices_dict = {}
        self.current_folder = ""
        
        self.create_widgets()

    def create_widgets(self):
        """Create the main interface with menu, toolbar, and workspace"""
        # Create main container first
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # Create toolbar on the left
        self.create_toolbar(main_container)
        
        # Create central workspace
        self.workspace = ttk.Frame(main_container)
        self.workspace.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # Create status bar
        self.create_status_bar()
        
        # Initialize tabs FIRST (but don't pack them yet)
        self.initialize_tab_objects()
        
        # NOW create menu bar (after tabs exist)
        self.create_menu_bar()
        
        # Show default tab
        self.current_tool = None
        self.show_tool('selection')

    def create_menu_bar(self):
        """Create the menu bar"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open Experiment Folder...", command=self.selection_tab.select_experiment_folder)
        file_menu.add_separator()
        file_menu.add_command(label="Export Metadata...", command=self.selection_tab.export_metadata)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Data Selection", command=lambda: self.show_tool('selection'))
        tools_menu.add_command(label="Analysis & Visualization", command=lambda: self.show_tool('analysis'))
        tools_menu.add_command(label="Batch Operations", command=lambda: self.show_tool('batch'))
        tools_menu.add_command(label="Video Export", command=lambda: self.show_tool('video'))
        tools_menu.add_command(label="Cluster Tasks", command=lambda: self.show_tool('cluster'))
        tools_menu.add_command(label="Peak Tracking", command=lambda: self.show_tool('peak_tracking'))
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)

    def create_toolbar(self, parent):
        """Create the vertical toolbar with enhanced styling"""
        # Create toolbar frame with background
        toolbar = ttk.Frame(parent, width=130, relief=tk.RAISED, borderwidth=1)
        toolbar.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
        toolbar.pack_propagate(False)
        
        # Add toolbar title
        title_label = ttk.Label(toolbar, text="Tools", font=('Arial', 10, 'bold'))
        title_label.pack(pady=(5, 10))
        
        # Toolbar buttons
        self.toolbar_buttons = {}
        
        tools = [
            ('selection', 'Data\nSelection', '📂'),
            ('analysis', 'Analysis &\nVisualization', '📊'),
            ('batch', 'Batch\nOperations', '⚙️'),
            ('video', 'Video\nExport', '🎬'),
            ('cluster', 'Cluster\nTasks', '🖥️'),
            ('peak_tracking', 'Peak\nTracking', '🎯')
        ]
        
        for tool_id, label, icon in tools:
            btn_frame = ttk.Frame(toolbar)
            btn_frame.pack(fill=tk.X, pady=2)
            
            btn = ttk.Button(
                btn_frame, 
                text=f"{icon}\n{label}",
                command=lambda t=tool_id: self.show_tool(t),
                width=15
            )
            btn.pack(fill=tk.X)
            self.toolbar_buttons[tool_id] = btn

    def create_status_bar(self):
        """Create the status bar"""
        self.status_bar = ttk.Frame(self.root)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.status_text = tk.StringVar(value="Ready")
        status_label = ttk.Label(self.status_bar, textvariable=self.status_text, relief=tk.SUNKEN)
        status_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2, pady=1)
        
    def show_message(self, title, message, msg_type="info"):
        """Show a message dialog and update status bar"""
        # Update status bar
        self.status_text.set(f"{title}: {message}")
        
        # Show dialog
        if msg_type == "info":
            messagebox.showinfo(title, message)
        elif msg_type == "warning":
            messagebox.showwarning(title, message)
        elif msg_type == "error":
            messagebox.showerror(title, message)

    def initialize_tab_objects(self):
        """Initialize tab objects without packing them"""
        self.selection_tab = SelectionTab(self.workspace, self)
        self.analysis_tab = AnalysisTab(self.workspace, self)
        self.batch_tab = BatchTab(self.workspace, self)
        self.video_tab = VideoTab(self.workspace, self)
        self.cluster_tab = ClusterTab(self.workspace, self)
        self.peak_tracking_tab = PeakTrackingTab(self.workspace, self)
        
        # Don't pack any of the frames yet - they'll be shown/hidden by show_tool()

    def show_tool(self, tool_name):
        """Show the selected tool in the workspace"""
        if self.current_tool == tool_name:
            return
        
        # Hide current tool
        if self.current_tool:
            self.get_tab_frame(self.current_tool).pack_forget()
        
        # Show new tool
        tool_frame = self.get_tab_frame(tool_name)
        tool_frame.pack(fill=tk.BOTH, expand=True)
        
        # Update button states
        for btn_id, btn in self.toolbar_buttons.items():
            if btn_id == tool_name:
                btn.state(['pressed'])
            else:
                btn.state(['!pressed'])
        
        self.current_tool = tool_name
        self.status_text.set(f"Tool: {tool_name.replace('_', ' ').title()}")

    def get_tab_frame(self, tool_name):
        """Get the frame for a specific tool"""
        tab_mapping = {
            'selection': self.selection_tab.frame,
            'analysis': self.analysis_tab.frame,
            'batch': self.batch_tab.frame,
            'video': self.video_tab.frame,
            'cluster': self.cluster_tab.frame,
            'peak_tracking': self.peak_tracking_tab.frame
        }
        return tab_mapping[tool_name]

    def show_about(self):
        """Show about dialog"""
        messagebox.showinfo("About", "Enhanced Bruker NMR Data Reader v1.0.0\nDeveloped by JH")

    def update_summary(self):
        """Update the analysis summary"""
        self.analysis_tab.update_summary(self.matrices_dict)

    def update_spectrum_selector(self):
        """Update the spectrum selector dropdown"""
        self.analysis_tab.update_spectrum_selector(self.matrices_dict)