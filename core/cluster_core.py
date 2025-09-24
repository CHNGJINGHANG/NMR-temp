"""
Core cluster management functionality without GUI dependencies
"""

import threading
import os
import stat
import json
import shutil
from pathlib import Path
from core.ssh_utils import SSHConnection
import paramiko
import time

class ClusterCore:
    def __init__(self, config_file="cluster_config.json"):
        self.ssh_connection = None
        self.ssh = None
        self.sftp = None
        
        # Set default config file if none provided
        if not config_file:
            config_file = os.path.join(os.path.expanduser("~"), ".cluster_config.json")
        
        self.config_file = config_file
        self.config = None
        self.shell_channel = None
        self._shell_output_callback = None
        self._shell_running = False
        
        self.load_config()

    def to_posix_path(self, path):
        """Convert path to POSIX style for remote systems"""
        return str(path).replace("\\", "/").rstrip("/")

    def to_local_path(self, path):
        """Convert path to local OS style"""
        if os.name == 'nt':  # Windows
            path = str(path).replace("/", "\\")
            if not path.endswith("\\"):
                path += "\\"
            return path
        else:  # Unix-like systems
            return str(path).replace("\\", "/").rstrip("/") + "/"

    def load_config(self):
        """Load SSH connection presets and file transfer paths"""
        default_config = {
            "connections": {
                "JH_NSCC": {
                    "host": "aspire2antu.nscc.sg", 
                    "username": "chng0145", 
                    "port": 22,
                },
                "NMR_700": {
                    "host": "", 
                    "username": "", 
                    "port": 22,
                },
                "NMR_600": {
                    "host": "", 
                    "username": "", 
                    "port": 22,
                },
                "Others": {
                    "host": "", 
                    "username": "", 
                    "port": 22,
                },
            },
            "remote_workdir": "/home",
            "upload_paths": {},
            "download_paths": {},
            "config_version": "1.0"
        }
        
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                    
                # Merge with defaults to ensure all keys exist
                self.config = self._merge_configs(default_config, loaded_config)
                
                # Validate and normalize paths
                self._validate_config()
            else:
                self.config = default_config
                self.save_config()  # Create default config file
                
        except json.JSONDecodeError as e:
            print(f"Config JSON error: {e}")
            self.config = default_config
        except Exception as e:
            print(f"Config load error: {e}")
            self.config = default_config

    def _merge_configs(self, default, loaded):
        """Recursively merge loaded config with defaults"""
        merged = default.copy()
        for key, value in loaded.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._merge_configs(merged[key], value)
            else:
                merged[key] = value
        return merged

    def _validate_config(self):
        """Validate and normalize config values"""
        # Validate connections
        for name, conn in self.config["connections"].items():
            if not isinstance(conn.get("port"), int):
                conn["port"] = 22
            if not conn.get("remote_workdir"):
                conn["remote_workdir"] = "/home"
        
        # Normalize paths based on local OS
        for path_type in ["upload_paths", "download_paths"]:
            normalized_paths = {}
            for name, path in self.config[path_type].items():
                normalized_paths[name] = self.to_local_path(path)
            self.config[path_type] = normalized_paths

    def save_config(self):
        """Save connection presets with backup"""
        try:
            # Create backup if config exists
            if os.path.exists(self.config_file):
                backup_file = f"{self.config_file}.backup"
                shutil.copy2(self.config_file, backup_file)
            
            # Ensure directory exists
            config_dir = os.path.dirname(self.config_file)
            if config_dir:
                os.makedirs(config_dir, exist_ok=True)
            
            # Write config with proper encoding
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            
            return True
            
        except Exception as e:
            print(f"Failed to save config: {e}")
            return False

    # Connection Management
    def connect(self, host, username, password, port=22, workdir=None):
        """Connect to SSH server and initialize shell"""
        try:
            self.ssh_connection = SSHConnection()
            success = self.ssh_connection.connect(
                host=host,
                username=username, 
                password=password,
                port=port
            )
            
            if success:
                self.ssh = self.ssh_connection.ssh_client
                self.sftp = self.ssh_connection.ssh_client.open_sftp()
                
                # Initialize interactive shell
                self.shell_channel = self.ssh.invoke_shell()
                self.shell_channel.settimeout(0.1)  # Non-blocking reads
                
                # Set working directory if provided
                if workdir:
                    self.send_shell_command(f"cd {workdir}")
                
                # Start shell output reader thread
                self._shell_running = True
                threading.Thread(target=self._read_shell_output, daemon=True).start()
                
                return True
            return False
            
        except Exception as e:
            print(f"SSH connection error: {str(e)}")
            return False

    def connect_with_preset(self, preset_name, password):
        """Connect using a saved preset"""
        connection = self.get_connection(preset_name)
        if not connection:
            return False, "Connection preset not found"
        
        success = self.connect(
            host=connection["host"],
            username=connection["username"], 
            password=password,
            port=connection["port"],
            workdir=connection["remote_workdir"]
        )
        
        return success, "Connected successfully" if success else "Connection failed"

    def disconnect(self):
        """Disconnect from SSH server"""
        self._shell_running = False
        if self.shell_channel:
            self.shell_channel.close()
        if self.sftp:
            self.sftp.close()
        if self.ssh_connection:
            self.ssh_connection.disconnect()
        
        self.ssh = None
        self.sftp = None
        self.shell_channel = None

    def is_connected(self):
        """Check if SSH connection is active"""
        if not self.sftp or not self.ssh:
            return False
        try:
            self.sftp.listdir('.')
            return True
        except:
            return False

    # Connection Presets Management
    def add_connection(self, name, host, username, port=22, remote_workdir="/home"):
        """Add or update a connection preset"""
        self.config["connections"][name] = {
            "host": host,
            "username": username, 
            "port": int(port),
            "remote_workdir": remote_workdir
        }
        return self.save_config()

    def remove_connection(self, name):
        """Remove a connection preset"""
        if name in self.config["connections"]:
            del self.config["connections"][name]
            return self.save_config()
        return False

    def get_connection(self, name):
        """Get connection details by name"""
        return self.config["connections"].get(name, None)

    # Shell Management
    def set_shell_output_callback(self, callback):
        """Set callback function for shell output"""
        self._shell_output_callback = callback

    def send_shell_command(self, command):
        """Send command to interactive shell"""
        if not self.shell_channel:
            return False
        try:
            self.shell_channel.send(command + '\n')
            return True
        except Exception as e:
            print(f"Failed to send command: {e}")
            return False

    def _read_shell_output(self):
        """Continuously read shell output and call callback"""
        import re
        # Pattern to remove ANSI escape sequences (color codes)
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        
        while self._shell_running and self.shell_channel:
            try:
                if self.shell_channel.recv_ready():
                    data = self.shell_channel.recv(1024).decode('utf-8', errors='ignore')
                    if self._shell_output_callback and data:
                        # Strip ANSI escape sequences before sending to callback
                        clean_data = ansi_escape.sub('', data)
                        self._shell_output_callback(clean_data)
                time.sleep(0.01)  # Small delay to prevent excessive CPU usage
            except Exception as e:
                if self._shell_running:  # Only log if we're supposed to be running
                    print(f"Shell output read error: {e}")
                break
    def get_current_remote_dir(self):
        """Get current remote working directory"""
        if not self.shell_channel:
            return None
        
        try:
            # Send pwd command and capture output
            self.shell_channel.send('pwd\n')
            time.sleep(0.5)  # Wait for command to execute
            
            output = ""
            while self.shell_channel.recv_ready():
                data = self.shell_channel.recv(1024).decode('utf-8', errors='ignore')
                output += data
            
            # Extract path from output
            lines = output.strip().split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('/') and not line.startswith('/usr') and 'pwd' not in line:
                    return line
            
            return None
        except Exception as e:
            print(f"Failed to get current directory: {e}")
            return None

    # Remote Directory Operations
    def list_remote_directory(self, path):
        """List files in remote directory"""
        if not self.sftp:
            return None
        try:
            path_posix = self.to_posix_path(path)
            return self.sftp.listdir_attr(path_posix)
        except Exception as e:
            print(f"Failed to list directory {path}: {e}")
            return None

    def ensure_remote_dir(self, remote_path):
        """Recursively create remote directory if it doesn't exist"""
        remote_path_posix = self.to_posix_path(remote_path)
        try:
            self.sftp.stat(remote_path_posix)
        except FileNotFoundError:
            parent_dir = str(Path(remote_path_posix).parent)
            if parent_dir != remote_path_posix and parent_dir != '.':
                self.ensure_remote_dir(parent_dir)
            try:
                self.sftp.mkdir(remote_path_posix)
            except Exception as e:
                print(f"Warning: Could not create directory {remote_path_posix}: {e}")

    # File Transfer Operations
    def upload_file(self, local_file, remote_dir, progress_callback=None):
        """Upload a single file"""
        if not self.sftp:
            return False, "Not connected"
        
        try:
            filename = Path(local_file).name
            remote_dir_posix = self.to_posix_path(remote_dir)
            if not remote_dir_posix.endswith('/'):
                remote_dir_posix += '/'
            remote_file_posix = remote_dir_posix + filename
            
            # Ensure remote directory exists
            self.ensure_remote_dir(remote_dir_posix)
            self.sftp.put(local_file, remote_file_posix)
            return True, f"File uploaded to {remote_dir_posix}"
        except Exception as e:
            return False, f"Upload failed: {e}"

    def upload_folder(self, local_folder, remote_dir, progress_callback=None):
        """Upload an entire folder"""
        if not self.sftp:
            return False, "Not connected"
        
        try:
            folder_name = Path(local_folder).name
            remote_dir_posix = self.to_posix_path(remote_dir)
            if not remote_dir_posix.endswith('/'):
                remote_dir_posix += '/'
            remote_folder_posix = remote_dir_posix + folder_name
            
            self.upload_folder_recursive(local_folder, remote_folder_posix)
            return True, f"Folder uploaded to {remote_folder_posix}"
        except Exception as e:
            return False, f"Folder upload failed: {e}"

    def upload_folder_recursive(self, local_folder, remote_folder):
        """Recursively upload a folder and its contents"""
        remote_folder_posix = self.to_posix_path(remote_folder)
        self.ensure_remote_dir(remote_folder_posix)
        
        for item in os.listdir(local_folder):
            local_path = os.path.join(local_folder, item)
            remote_path_posix = remote_folder_posix + '/' + item
            
            if os.path.isfile(local_path):
                self.sftp.put(local_path, remote_path_posix)
            elif os.path.isdir(local_path):
                self.upload_folder_recursive(local_path, remote_path_posix)

    def download_file(self, remote_file, local_file, progress_callback=None):
        """Download a single file"""
        if not self.sftp:
            return False, "Not connected"
        
        try:
            remote_path_posix = self.to_posix_path(remote_file)
            self.sftp.get(remote_path_posix, local_file)
            return True, f"File downloaded to {local_file}"
        except Exception as e:
            return False, f"Download failed: {e}"

    def download_folder(self, remote_folder, local_parent, progress_callback=None):
        """Download an entire folder"""
        if not self.sftp:
            return False, "Not connected"
        
        try:
            remote_folder_posix = self.to_posix_path(remote_folder)
            folder_name = Path(remote_folder_posix).name
            local_folder = self.to_local_path(os.path.join(local_parent, folder_name))
            self.download_folder_recursive(remote_folder_posix, local_folder)
            return True, f"Folder downloaded to {local_folder}"
        except Exception as e:
            return False, f"Folder download failed: {e}"

    def download_folder_recursive(self, remote_folder, local_folder):
        """Recursively download a folder and its contents"""
        # Ensure local folder exists
        local_folder_clean = local_folder.rstrip("\\/")
        os.makedirs(local_folder_clean, exist_ok=True)
        
        remote_folder_posix = self.to_posix_path(remote_folder)
        try:
            items = self.sftp.listdir_attr(remote_folder_posix)
        except Exception as e:
            raise Exception(f"Failed to list remote directory {remote_folder}: {e}")
        
        for item in items:
            remote_path_posix = remote_folder_posix + '/' + item.filename
            local_path = os.path.join(local_folder_clean, item.filename)
            
            if stat.S_ISDIR(item.st_mode):
                # Recursively download subdirectory
                self.download_folder_recursive(remote_path_posix, local_path)
            else:
                # Download file
                self.sftp.get(remote_path_posix, local_path)

    # Path Management
    def add_path(self, path_type, name, path):
        """Add a new upload/download path"""
        if path_type not in ['upload', 'download']:
            return False, "Invalid path type"
        
        key = f'{path_type}_paths'
        local_path = self.to_local_path(path)
        self.config[key][name] = local_path
        success = self.save_config()
        return success, "Path added successfully" if success else "Failed to save config"
    
    def remove_path(self, path_type, name):
        """Remove an upload/download path"""
        if path_type not in ['upload', 'download']:
            return False, "Invalid path type"
        
        key = f'{path_type}_paths'
        if name not in self.config[key]:
            return False, "Path not found"
        
        del self.config[key][name]
        success = self.save_config()
        return success, "Path removed successfully" if success else "Failed to save config"

    def get_paths(self, path_type):
        """Get all paths of specified type"""
        if path_type not in ['upload', 'download']:
            return {}
        
        key = f'{path_type}_paths'
        return self.config.get(key, {})

    # Utility Methods
    def get_config(self):
        """Get current configuration"""
        return self.config.copy()

    def reload_config(self):
        """Reload configuration from file"""
        self.load_config()

    def reset_config(self):
        """Reset configuration to defaults"""
        if os.path.exists(self.config_file):
            backup_file = f"{self.config_file}.backup"
            shutil.copy2(self.config_file, backup_file)
        
        self.config = None
        self.load_config()  # This will create default config
        return True