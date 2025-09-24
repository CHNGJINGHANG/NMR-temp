"""
Cluster management tab with integrated SFTP file management
"""

from core.ssh_utils import SSHConnection
import paramiko
import stat
import json
import time
from core.cluster_core import ClusterCore
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox, scrolledtext
import tkinter.simpledialog
import threading
from pathlib import Path
import os
import stat

class ClusterTab:
    def __init__(self, parent, main_app):
        self.parent = parent
        self.main_app = main_app
        self.core = ClusterCore()
        
        # Create the main frame
        self.frame = ttk.Frame(parent)
        self.create_widgets()



    def create_widgets(self):
        """Create all widgets for the cluster tab"""
        self.create_connection_section()
        self.create_file_transfer_section()
        self.create_job_submission_section()
        self.create_monitoring_section()

    def create_connection_section(self):
        """Create SSH connection section"""
        conn_frame = ttk.LabelFrame(self.frame, text="SSH Connection")
        conn_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Connection presets
        preset_frame = ttk.Frame(conn_frame)
        preset_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(preset_frame, text="Preset:").pack(side=tk.LEFT)
        self.preset_combo = ttk.Combobox(preset_frame, values=list(self.core.config['connections'].keys()), state="readonly")
        self.preset_combo.pack(side=tk.LEFT, padx=5)
        self.preset_combo.bind("<<ComboboxSelected>>", self.load_preset)
        
        # Connection details
        details_frame = ttk.Frame(conn_frame)
        details_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(details_frame, text="Host:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.host_var = tk.StringVar()
        ttk.Entry(details_frame, textvariable=self.host_var, width=25).grid(row=0, column=1, padx=5, pady=2)
        
        ttk.Label(details_frame, text="Username:").grid(row=0, column=2, sticky=tk.W, padx=5, pady=2)
        self.username_var = tk.StringVar()
        ttk.Entry(details_frame, textvariable=self.username_var, width=15).grid(row=0, column=3, padx=5, pady=2)
        
        ttk.Label(details_frame, text="Port:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.port_var = tk.IntVar(value=22)
        tk.Spinbox(details_frame, textvariable=self.port_var, from_=1, to=65535, width=10).grid(row=1, column=1, padx=5, pady=2)
        
        # Working directory
        ttk.Label(details_frame, text="Work Dir:").grid(row=1, column=2, sticky=tk.W, padx=5, pady=2)
        self.workdir_var = tk.StringVar(value=self.core.config.get('remote_workdir', '~/nmr_processing'))
        ttk.Entry(details_frame, textvariable=self.workdir_var, width=15).grid(row=1, column=3, padx=5, pady=2)
        
        # Connect/Disconnect button
        self.connect_btn = ttk.Button(details_frame, text="Connect", command=self.connect_ssh)
        self.connect_btn.grid(row=2, column=1, padx=5, pady=5)
        
        # Status
        self.status_var = tk.StringVar(value="Not connected")
        ttk.Label(conn_frame, textvariable=self.status_var).pack(padx=5, pady=2)
        
        # Load first preset if available
        if self.preset_combo['values']:
            self.preset_combo.current(0)
            self.load_preset()

    def download_file_or_folder(self):
        """Download file or folder based on selection"""
        if not self.ensure_connected():
            return
        
        download_type = self.download_type.get()
        folder_name = self.download_combo.get()
        
        if not folder_name:
            messagebox.showerror("Error", "Please select a download path")
            return
            
        remote_dir = self.core.config['download_paths'][folder_name]
        self.open_file_browser(remote_dir, download_type)
        
    def create_file_transfer_section(self):
        """Create comprehensive file transfer section"""
        transfer_frame = ttk.LabelFrame(self.frame, text="File Transfer")
        transfer_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Upload section
        upload_frame = ttk.LabelFrame(transfer_frame, text="Upload")
        upload_frame.pack(fill=tk.X, padx=5, pady=2)
        
        # Upload path management
        upload_path_frame = ttk.Frame(upload_frame)
        upload_path_frame.pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Label(upload_path_frame, text="Destination:").pack(side=tk.LEFT)
        self.upload_combo = ttk.Combobox(upload_path_frame, state="readonly", width=25)
        self.upload_combo.pack(side=tk.LEFT, padx=5)
        ttk.Button(upload_path_frame, text="Manage", command=lambda: self.manage_paths("upload")).pack(side=tk.LEFT, padx=5)
        
        # Upload type and buttons
        upload_btn_frame = ttk.Frame(upload_frame)
        upload_btn_frame.pack(fill=tk.X, padx=5, pady=2)
        
        self.upload_type = tk.StringVar(value="file")
        ttk.Radiobutton(upload_btn_frame, text="File", variable=self.upload_type, value="file").pack(side=tk.LEFT)
        ttk.Radiobutton(upload_btn_frame, text="Folder", variable=self.upload_type, value="folder").pack(side=tk.LEFT, padx=10)
        ttk.Button(upload_btn_frame, text="Select & Upload", command=self.upload_file_or_folder).pack(side=tk.RIGHT, padx=5)
        
        # Download section
        download_frame = ttk.LabelFrame(transfer_frame, text="Download")
        download_frame.pack(fill=tk.X, padx=5, pady=2)
        
        # Download path management
        download_path_frame = ttk.Frame(download_frame)
        download_path_frame.pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Label(download_path_frame, text="Source:").pack(side=tk.LEFT)
        self.download_combo = ttk.Combobox(download_path_frame, state="readonly", width=25)
        self.download_combo.pack(side=tk.LEFT, padx=5)
        ttk.Button(download_path_frame, text="Manage", command=lambda: self.manage_paths("download")).pack(side=tk.LEFT, padx=5)
        
        # Download type and buttons
        download_btn_frame = ttk.Frame(download_frame)
        download_btn_frame.pack(fill=tk.X, padx=5, pady=2)
        
        self.download_type = tk.StringVar(value="file")
        ttk.Radiobutton(download_btn_frame, text="File", variable=self.download_type, value="file").pack(side=tk.LEFT)
        ttk.Radiobutton(download_btn_frame, text="Folder", variable=self.download_type, value="folder").pack(side=tk.LEFT, padx=10)
        ttk.Button(download_btn_frame, text="Browse & Download", command=self.download_file_or_folder).pack(side=tk.RIGHT, padx=5)
        
        # Progress bar
        self.progress = ttk.Progressbar(transfer_frame, mode='indeterminate')
        self.progress.pack(fill=tk.X, padx=5, pady=5)
        
        # Update dropdowns
        self.update_dropdowns()

    def connect_ssh(self):
        """Connect to SSH server"""
        host = self.host_var.get().strip()
        username = self.username_var.get().strip()
        port = self.port_var.get()
        workdir = self.workdir_var.get().strip()
        
        if not host or not username:
            messagebox.showwarning("Warning", "Please enter host and username")
            return
        
        # Get password
        try:
            password = tk.simpledialog.askstring(
                "SSH Password", 
                f"Enter password for {username}@{host}:", 
                show='*'
            )
        except Exception as e:
            messagebox.showerror("Error", f"Password dialog error: {str(e)}")
            return
        
        if password is None:
            return
        
        if not password.strip():
            messagebox.showwarning("Warning", "Password cannot be empty")
            return
        
        def _connect():
            try:
                self.status_var.set("Connecting...")
                
                success = self.core.connect(host, username, password, port, workdir)
                
                if success:
                    self.status_var.set(f"Connected to {host}")
                    messagebox.showinfo("Success", f"Connected to {host}!")
                    # Set up shell output callback
                    self.core.set_shell_output_callback(self.handle_shell_output)
                else:
                    self.status_var.set("Connection failed")
                    messagebox.showerror("Error", "Failed to connect to SSH server")
                    
            except Exception as e:
                self.status_var.set("Connection error")
                messagebox.showerror("Error", f"SSH connection error: {str(e)}")
        
        threading.Thread(target=_connect, daemon=True).start()

    def create_job_submission_section(self):
        """Create interactive terminal section"""
        job_frame = ttk.LabelFrame(self.frame, text="Interactive Terminal")
        job_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Command input
        cmd_frame = ttk.Frame(job_frame)
        cmd_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(cmd_frame, text="$ ").pack(side=tk.LEFT)
        self.command_var = tk.StringVar()
        self.command_entry = ttk.Entry(cmd_frame, textvariable=self.command_var)
        self.command_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        self.command_entry.bind('<Return>', self.send_command)
        self.command_entry.bind('<Up>', self.command_history_up)
        self.command_entry.bind('<Down>', self.command_history_down)
        
        # Command history
        self.command_history = []
        self.history_index = -1

    def create_monitoring_section(self):
        """Create output monitoring section"""
        monitor_frame = ttk.LabelFrame(self.frame, text="Command Output")
        monitor_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Output text area
        self.output_text = scrolledtext.ScrolledText(monitor_frame, height=10, wrap=tk.WORD)
        self.output_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def load_preset(self, event=None):
        """Load connection preset"""
        preset_name = self.preset_combo.get()
        if preset_name in self.core.config['connections']:
            preset = self.core.config['connections'][preset_name]
            self.host_var.set(preset['host'])
            self.username_var.set(preset['username'])
            self.port_var.set(preset['port'])

    def ensure_connected(self):
        """Check if SSH connection is active"""
        if not self.core.is_connected():
            messagebox.showerror("Error", "Please connect first")
            return False
        return True

    def manage_paths(self, path_type):
        """Open path management window"""
        manage_window = tk.Toplevel(self.main_app.root)
        manage_window.title(f"Manage {path_type.capitalize()} Paths")
        manage_window.geometry("500x400")
        manage_window.grab_set()
        
        # Current paths list
        ttk.Label(manage_window, text=f"Current {path_type} paths:").pack(pady=5)
        
        # Listbox with scrollbar
        list_frame = ttk.Frame(manage_window)
        list_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side="right", fill="y")
        
        paths_listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set)
        paths_listbox.pack(fill="both", expand=True)
        scrollbar.config(command=paths_listbox.yview)
        
        # Populate listbox
        def refresh_listbox():
            paths_listbox.delete(0, tk.END)
            paths_dict = self.core.config[f'{path_type}_paths']
            for name, path in paths_dict.items():
                paths_listbox.insert(tk.END, f"{name}: {path}")
        
        refresh_listbox()
        
        # Button frame
        button_frame = ttk.Frame(manage_window)
        button_frame.pack(fill="x", padx=10, pady=5)
        
        # Add path button
        def add_path():
            dialog = PathDialog(manage_window, "Add New Path")
            if dialog.result:
                name, path = dialog.result
                success, message = self.core.add_path(path_type, name, path)
                if not success:
                    messagebox.showerror("Error", message)
                refresh_listbox()
                self.update_dropdowns()

        # Remove path button
        def remove_path():
            selection = paths_listbox.curselection()
            if not selection:
                messagebox.showwarning("Warning", "Please select a path to remove")
                return
            
            selected_text = paths_listbox.get(selection[0])
            path_name = selected_text.split(":")[0].strip()
            
            if messagebox.askyesno("Confirm", f"Remove path '{path_name}'?"):
                success, message = self.core.remove_path(path_type, path_name)
                if success:
                    refresh_listbox()
                    self.update_dropdowns()
                else:
                    messagebox.showerror("Error", message)

        ttk.Button(button_frame, text="Add Path", command=add_path).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Remove Path", command=remove_path).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Close", command=manage_window.destroy).pack(side="right", padx=5)

    def update_dropdowns(self):
        """Update dropdown values"""
        upload_values = list(self.core.config.get('upload_paths', {}).keys())
        download_values = list(self.core.config.get('download_paths', {}).keys())
        
        self.upload_combo['values'] = upload_values
        self.download_combo['values'] = download_values
        
        if upload_values:
            self.upload_combo.current(0)
        if download_values:
            self.download_combo.current(0)

    def ensure_remote_dir(self, remote_path):
        """Recursively create remote directory if it doesn't exist"""
        remote_path_sftp = remote_path.replace("\\", "/")
        try:
            self.sftp.stat(remote_path_sftp)
        except FileNotFoundError:
            parent_dir = str(Path(remote_path_sftp).parent)
            if parent_dir != remote_path_sftp:
                self.ensure_remote_dir(parent_dir.replace("/", "\\"))
            try:
                self.sftp.mkdir(remote_path_sftp)
            except Exception as e:
                print(f"Warning: Could not create directory {remote_path}: {e}")

    def upload_file_or_folder(self):
        """Upload file or folder based on selection"""
        if not self.ensure_connected():
            return
        
        upload_type = self.upload_type.get()
        folder_name = self.upload_combo.get()
        
        if not folder_name:
            messagebox.showerror("Error", "Please select an upload path")
            return
            
        remote_dir = self.core.config['upload_paths'][folder_name]
        
        if upload_type == "file":
            self.upload_file(remote_dir)
        else:
            self.upload_folder(remote_dir)

    def upload_file(self, remote_dir):
        """Upload a single file"""
        local_file = filedialog.askopenfilename(title="Select file to upload")
        if not local_file:
            return
        
        def _upload():
            try:
                self.progress.start()
                self.status_var.set("Uploading file...")
                
                success, message = self.core.upload_file(local_file, remote_dir)
                
                if success:
                    self.status_var.set("Upload completed")
                    messagebox.showinfo("Success", message)
                else:
                    self.status_var.set("Upload failed")
                    messagebox.showerror("Error", message)
                    
            except Exception as e:
                messagebox.showerror("Error", f"Upload failed: {e}")
                self.status_var.set("Upload failed")
            finally:
                self.progress.stop()
        
        threading.Thread(target=_upload, daemon=True).start()

    def upload_folder(self, remote_dir):
        """Upload an entire folder"""
        local_folder = filedialog.askdirectory(title="Select folder to upload")
        if not local_folder:
            return
        
        def _upload():
            try:
                self.progress.start()
                self.status_var.set("Uploading folder...")
                
                success, message = self.core.upload_folder(local_folder, remote_dir)
                
                if success:
                    self.status_var.set("Folder upload completed")
                    messagebox.showinfo("Success", message)
                else:
                    self.status_var.set("Upload failed")
                    messagebox.showerror("Error", message)
                    
            except Exception as e:
                messagebox.showerror("Error", f"Folder upload failed: {e}")
                self.status_var.set("Upload failed")
            finally:
                self.progress.stop()
        
        threading.Thread(target=_upload, daemon=True).start()

    def open_file_browser(self, start_path, operation_type):
        """Open the navigable file browser"""
        RemoteFileBrowser(self, start_path, operation_type)

    def send_command(self, event=None):
        """Send command to interactive shell"""
        if not self.ensure_connected():
            return
        
        command = self.command_var.get().strip()
        if not command:
            return
        
        # Add to history
        if command != "" and (not self.command_history or self.command_history[-1] != command):
            self.command_history.append(command)
        self.history_index = len(self.command_history)
        
        # Display command in output
        self.output_text.insert(tk.END, f"$ {command}\n")
        self.output_text.see(tk.END)
        
        # Send to shell
        self.core.send_shell_command(command)
        
        # Clear input
        self.command_var.set("")

    def command_history_up(self, event):
        """Navigate command history up"""
        if self.command_history and self.history_index > 0:
            self.history_index -= 1
            self.command_var.set(self.command_history[self.history_index])

    def command_history_down(self, event):
        """Navigate command history down"""
        if self.command_history and self.history_index < len(self.command_history) - 1:
            self.history_index += 1
            self.command_var.set(self.command_history[self.history_index])
        elif self.history_index >= len(self.command_history) - 1:
            self.history_index = len(self.command_history)
            self.command_var.set("")

    def handle_shell_output(self, data):
        """Handle output from shell (called from core)"""
        def _update_output():
            self.output_text.insert(tk.END, data)
            self.output_text.see(tk.END)
        
        # Schedule GUI update from main thread
        self.main_app.root.after(0, _update_output)

class RemoteFileBrowser:
    def __init__(self, cluster_tab, start_path, operation_type):
        self.cluster_tab = cluster_tab
        self.core = cluster_tab.core
        self.current_path = self.core.to_win_path(start_path)
        self.operation_type = operation_type
        self.path_history = []
        
        self.setup_browser()
        self.refresh_view()
    
    def setup_browser(self):
        self.window = tk.Toplevel(self.cluster_tab.main_app.root)
        self.window.title(f"Remote File Browser - {self.operation_type.capitalize()} Mode")
        self.window.geometry("600x500")
        self.window.grab_set()
        
        # Navigation frame
        nav_frame = ttk.Frame(self.window)
        nav_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Button(nav_frame, text="Back", command=self.go_back).pack(side="left", padx=5)
        ttk.Button(nav_frame, text="Up", command=self.go_up).pack(side="left", padx=5)
        ttk.Button(nav_frame, text="Refresh", command=self.refresh_view).pack(side="left", padx=5)
        
        # Current path display
        self.path_var = tk.StringVar(value=self.current_path)
        path_entry = ttk.Entry(nav_frame, textvariable=self.path_var, width=40)
        path_entry.pack(side="left", padx=10, fill="x", expand=True)
        path_entry.bind('<Return>', self.navigate_to_path)
        
        ttk.Button(nav_frame, text="Go", command=self.navigate_to_path).pack(side="right", padx=5)
        
        # File list frame
        list_frame = ttk.Frame(self.window)
        list_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Create treeview
        columns = ('Name', 'Type', 'Size')
        self.tree = ttk.Treeview(list_frame, columns=columns, show='tree headings')
        
        self.tree.heading('#0', text='')
        self.tree.column('#0', width=30, minwidth=30)
        
        for col in columns:
            self.tree.heading(col, text=col)
            if col == 'Name':
                self.tree.column(col, width=300, minwidth=100)
            else:
                self.tree.column(col, width=100, minwidth=80)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        self.tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Double click to navigate
        self.tree.bind('<Double-1>', self.on_double_click)
        
        # Action buttons
        action_frame = ttk.Frame(self.window)
        action_frame.pack(fill="x", padx=10, pady=5)
        
        if self.operation_type == "file":
            ttk.Button(action_frame, text="Download File", command=self.download_selected_file).pack(side="left", padx=5)
        else:
            ttk.Button(action_frame, text="Download Folder", command=self.download_selected_folder).pack(side="left", padx=5)
        
        ttk.Button(action_frame, text="Close", command=self.window.destroy).pack(side="right", padx=5)
        
        # Status
        self.browser_status = tk.StringVar(value="Ready")
        ttk.Label(self.window, textvariable=self.browser_status).pack(pady=5)
    
    def refresh_view(self):
        """Refresh the current directory view"""
        def _refresh():
            try:
                self.browser_status.set("Loading...")
                self.path_var.set(self.current_path)
                
                for item in self.tree.get_children():
                    self.tree.delete(item)
                
                current_path_sftp = self.current_path.replace("\\", "/").rstrip("/")
                items = self.core.list_remote_directory(self.current_path)
                if items is None:
                    messagebox.showerror("Error", "Failed to list directory")
                    self.browser_status.set("Error loading directory")
                    return
                
                items.sort(key=lambda x: (not stat.S_ISDIR(x.st_mode), x.filename.lower()))
                
                for item in items:
                    is_dir = stat.S_ISDIR(item.st_mode)
                    icon = "📁" if is_dir else "📄"
                    item_type = "Folder" if is_dir else "File"
                    size = "-" if is_dir else f"{item.st_size} B"
                    
                    self.tree.insert('', 'end', text=icon, values=(item.filename, item_type, size))
                
                self.browser_status.set(f"Ready - {len(items)} items")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to list directory: {e}")
                self.browser_status.set("Error loading directory")
        
        threading.Thread(target=_refresh, daemon=True).start()
    
    def on_double_click(self, event):
        """Handle double click on items"""
        selection = self.tree.selection()
        if not selection:
            return
        
        item = self.tree.item(selection[0])
        filename = item['values'][0]
        item_type = item['values'][1]
        
        if item_type == "Folder":
            self.navigate_to_folder(filename)
        elif self.operation_type == "file":
            self.download_file(filename)
    
    def navigate_to_folder(self, folder_name):
        """Navigate to a subfolder"""
        self.path_history.append(self.current_path)
        new_path = self.core.to_win_path(f"{self.current_path}{folder_name}")
        self.current_path = new_path
        self.refresh_view()
    
    def navigate_to_path(self, event=None):
        """Navigate to the path in the entry field"""
        new_path = self.core.to_win_path(self.path_var.get().strip())
        if new_path and new_path != self.current_path:
            try:
                test_path = new_path.replace("\\", "/").rstrip("/")
                self.cluster_tab.sftp.listdir(test_path)
                self.path_history.append(self.current_path)
                self.current_path = new_path
                self.refresh_view()
            except Exception as e:
                messagebox.showerror("Error", f"Cannot access path: {e}")
                self.path_var.set(self.current_path)
    
    def go_back(self):
        """Go back to previous directory"""
        if self.path_history:
            self.current_path = self.path_history.pop()
            self.refresh_view()
    
    def go_up(self):
        """Go up one directory level"""
        current_forward = self.current_path.replace("\\", "/").rstrip("/")
        parent_path = str(Path(current_forward).parent)
        parent_path_win = self.core.to_win_path(parent_path)
        
        if parent_path_win != self.current_path:
            self.path_history.append(self.current_path)
            self.current_path = parent_path_win
            self.refresh_view()
    
    def download_selected_file(self):
        """Download the selected file"""
        selection = self.tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a file to download")
            return
        
        item = self.tree.item(selection[0])
        filename = item['values'][0]
        item_type = item['values'][1]
        
        if item_type != "File":
            messagebox.showwarning("Warning", "Please select a file, not a folder")
            return
        
        self.download_file(filename)
    
    def download_selected_folder(self):
        """Download the selected folder"""
        selection = self.tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a folder to download")
            return
        
        item = self.tree.item(selection[0])
        filename = item['values'][0]
        item_type = item['values'][1]
        
        if item_type != "Folder":
            messagebox.showwarning("Warning", "Please select a folder, not a file")
            return
        
        remote_folder_path = self.core.to_win_path(f"{self.current_path}{filename}")
        
        local_parent = filedialog.askdirectory(title="Select destination folder")
        if local_parent:
            self.download_folder_to_local(remote_folder_path, local_parent)
    
    def download_file(self, filename):
        """Download a specific file"""
        remote_file_path = self.cluster_tab.to_win_path(f"{self.current_path}{filename}")
        
        local_file = filedialog.asksaveasfilename(
            title="Save file as",
            initialvalue=filename
        )
        
        if local_file:
            def _download():
                try:
                    self.cluster_tab.progress.start()
                    self.cluster_tab.status_var.set("Downloading...")
                    
                    remote_path_sftp = remote_file_path.replace("\\", "/").rstrip("/")
                    self.cluster_tab.sftp.get(remote_path_sftp, local_file)
                    
                    self.cluster_tab.status_var.set("Download completed")
                    messagebox.showinfo("Success", f"File downloaded to {local_file}")
                    self.window.destroy()
                    
                except Exception as e:
                    messagebox.showerror("Error", f"Download failed: {e}")
                    self.cluster_tab.status_var.set("Download failed")
                finally:
                    self.cluster_tab.progress.stop()
            
            threading.Thread(target=_download, daemon=True).start()
    
    def download_folder_to_local(self, remote_folder, local_parent):
        """Download an entire folder"""
        folder_name = Path(remote_folder.replace("\\", "/")).name
        local_folder = self.cluster_tab.to_win_path(os.path.join(local_parent, folder_name))
        
        def _download():
            try:
                self.cluster_tab.progress.start()
                self.cluster_tab.status_var.set("Downloading folder...")
                self.download_folder_recursive(remote_folder, local_folder)
                
                self.cluster_tab.status_var.set("Folder download completed")
                messagebox.showinfo("Success", f"Folder downloaded to {local_folder}")
                self.window.destroy()
                
            except Exception as e:
                messagebox.showerror("Error", f"Folder download failed: {e}")
                self.cluster_tab.status_var.set("Download failed")
            finally:
                self.cluster_tab.progress.stop()
        
        threading.Thread(target=_download, daemon=True).start()
    
    def download_folder_recursive(self, remote_folder, local_folder):
        """Recursively download a folder and its contents"""
        # Ensure local folder exists
        os.makedirs(local_folder.rstrip("\\"), exist_ok=True)
        
        remote_folder_sftp = remote_folder.replace("\\", "/").rstrip("/")
        try:
            items = self.cluster_tab.sftp.listdir_attr(remote_folder_sftp)
        except Exception as e:
            raise Exception(f"Failed to list remote directory {remote_folder}: {e}")
        
        local_folder_clean = local_folder.rstrip("\\")
        for item in items:
            remote_path = self.cluster_tab.to_win_path(f"{remote_folder}{item.filename}")
            local_path = self.cluster_tab.to_win_path(os.path.join(local_folder_clean, item.filename))
            
            if stat.S_ISDIR(item.st_mode):
                # Recursively download subdirectory
                self.download_folder_recursive(remote_path, local_path)
            else:
                # Download file
                remote_path_sftp = remote_path.replace("\\", "/").rstrip("/")
                self.cluster_tab.sftp.get(remote_path_sftp, local_path.rstrip("\\"))



class PathDialog:
    def __init__(self, parent, title, name="", path=""):
        self.result = None
        
        self.dialog = tk.Toplevel(parent)
        self.dialog.title(title)
        self.dialog.geometry("400x150")
        self.dialog.grab_set()
        self.dialog.transient(parent)
        
        # Center the dialog on parent
        self.dialog.geometry("+%d+%d" % (parent.winfo_rootx() + 50, parent.winfo_rooty() + 50))
        
        # Name entry
        ttk.Label(self.dialog, text="Name:").grid(row=0, column=0, sticky="w", padx=10, pady=5)
        self.name_var = tk.StringVar(value=name)
        name_entry = ttk.Entry(self.dialog, textvariable=self.name_var, width=40)
        name_entry.grid(row=0, column=1, padx=10, pady=5)
        name_entry.focus()
        
        # Path entry
        ttk.Label(self.dialog, text="Path:").grid(row=1, column=0, sticky="w", padx=10, pady=5)
        self.path_var = tk.StringVar(value=path)
        ttk.Entry(self.dialog, textvariable=self.path_var, width=40).grid(row=1, column=1, padx=10, pady=5)
        
        # Buttons
        button_frame = ttk.Frame(self.dialog)
        button_frame.grid(row=2, column=0, columnspan=2, pady=10)
        
        ttk.Button(button_frame, text="OK", command=self.ok_clicked).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Cancel", command=self.cancel_clicked).pack(side="left", padx=5)
        
        # Bind Enter key to OK
        self.dialog.bind('<Return>', lambda e: self.ok_clicked())
        self.dialog.bind('<Escape>', lambda e: self.cancel_clicked())
        
        # Wait for dialog to close
        self.dialog.wait_window()
    
    def ok_clicked(self):
        name = self.name_var.get().strip()
        path = self.path_var.get().strip()
        
        if not name or not path:
            messagebox.showwarning("Warning", "Please enter both name and path")
            return
        
        self.result = (name, path)
        self.dialog.destroy()
    
    def cancel_clicked(self):
        self.result = None
        self.dialog.destroy()