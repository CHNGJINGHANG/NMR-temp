"""
General SSH utilities for cluster integration
"""
import paramiko
import os
import logging
from typing import Optional, Tuple, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)

class SSHConnection:
    """General SSH connection manager for any cluster/server"""
    
    def __init__(self, host: str = None, username: str = None, port: int = 22):
        self.ssh_client = None
        self.sftp_client = None
        self.host = host
        self.username = username
        self.port = port
        self.remote_workdir = None
        self.connected = False
        
    def connect(self, host: str = None, username: str = None, password: str = None, 
                key_path: str = None, port: int = 22) -> bool:
        """
        Establish SSH connection
        """
        # Use provided parameters or fall back to instance variables
        self.host = host or self.host
        self.username = username or self.username
        self.port = port
        
        if not self.host or not self.username:
            raise ValueError("Host and username must be provided")
        
        try:
            self.ssh_client = paramiko.SSHClient()
            self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            # Try key-based auth first if key provided
            if key_path and os.path.exists(key_path):
                self.ssh_client.connect(
                    hostname=self.host,
                    username=self.username,
                    port=self.port,
                    key_filename=key_path,
                    timeout=30
                )
            elif password:
                self.ssh_client.connect(
                    hostname=self.host,
                    username=self.username,
                    port=self.port,
                    password=password,
                    timeout=30
                )
            else:
                raise ValueError("No authentication method provided (password or key)")
            
            # Test connection with a simple command
            exit_code, _, _ = self.execute_command("echo 'Connection test'")
            if exit_code != 0:
                raise RuntimeError("Connection test failed")
            
            # Open SFTP channel
            self.sftp_client = self.ssh_client.open_sftp()
            self.connected = True
            
            # Set default remote work directory to home
            self.remote_workdir = self.get_home_directory()
            
            logger.info(f"Successfully connected to {self.host}")
            return True
            
        except Exception as e:
            logger.error(f"SSH connection failed: {e}")
            self.disconnect()
            return False
    
    def disconnect(self):
        """Clean up connections"""
        try:
            if self.sftp_client:
                self.sftp_client.close()
                self.sftp_client = None
            if self.ssh_client:
                self.ssh_client.close()
                self.ssh_client = None
            self.connected = False
            logger.info("SSH connection closed")
        except Exception as e:
            logger.error(f"Error during disconnect: {e}")
    
    def get_home_directory(self) -> str:
        """Get remote home directory"""
        if not self.connected:
            raise RuntimeError("Not connected")
        
        try:
            exit_code, stdout, _ = self.execute_command("pwd")
            if exit_code == 0:
                return stdout.strip()
            return "/home/" + self.username  # fallback
        except:
            return "/home/" + self.username  # fallback
    
    def set_work_directory(self, remote_path: str):
        """Set the remote working directory"""
        self.remote_workdir = remote_path
        self.ensure_remote_dir(remote_path)
    
    def ensure_remote_dir(self, remote_path: str):
        """Create remote directory if it doesn't exist"""
        if not self.sftp_client:
            raise RuntimeError("No SFTP connection")
        
        try:
            self.sftp_client.stat(remote_path)
        except FileNotFoundError:
            # Try to create directory recursively
            self.execute_command(f"mkdir -p {remote_path}")
    
    def upload_file(self, local_path: str, remote_path: str = None) -> str:
        """Upload file to remote server"""
        if not self.sftp_client:
            raise RuntimeError("No SFTP connection")
        
        if not remote_path:
            filename = os.path.basename(local_path)
            remote_path = f"{self.remote_workdir}/{filename}"
        
        try:
            # Ensure remote directory exists
            remote_dir = os.path.dirname(remote_path)
            self.ensure_remote_dir(remote_dir)
            
            self.sftp_client.put(local_path, remote_path)
            logger.info(f"Uploaded {local_path} -> {remote_path}")
            return remote_path
        except Exception as e:
            logger.error(f"Upload failed: {e}")
            raise
    
    def download_file(self, remote_path: str, local_path: str) -> bool:
        """Download file from remote server"""
        if not self.sftp_client:
            raise RuntimeError("No SFTP connection")
        
        try:
            # Ensure local directory exists
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            self.sftp_client.get(remote_path, local_path)
            logger.info(f"Downloaded {remote_path} -> {local_path}")
            return True
        except Exception as e:
            logger.error(f"Download failed: {e}")
            return False
    
    def execute_command(self, command: str, timeout: int = 300) -> Tuple[int, str, str]:
        """Execute command on remote server"""
        if not self.ssh_client:
            raise RuntimeError("No SSH connection")
        
        try:
            stdin, stdout, stderr = self.ssh_client.exec_command(command, timeout=timeout)
            exit_code = stdout.channel.recv_exit_status()
            stdout_text = stdout.read().decode('utf-8')
            stderr_text = stderr.read().decode('utf-8')
            
            logger.debug(f"Executed: {command}")
            logger.debug(f"Exit code: {exit_code}")
            
            return exit_code, stdout_text, stderr_text
        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            raise
    
    def list_directory(self, remote_path: str = None) -> list:
        """List files in remote directory"""
        if not self.sftp_client:
            raise RuntimeError("No SFTP connection")
        
        path = remote_path or self.remote_workdir
        try:
            return self.sftp_client.listdir(path)
        except Exception as e:
            logger.error(f"Directory listing failed: {e}")
            return []
    
    def file_exists(self, remote_path: str) -> bool:
        """Check if remote file exists"""
        if not self.sftp_client:
            return False
        
        try:
            self.sftp_client.stat(remote_path)
            return True
        except FileNotFoundError:
            return False
        except Exception:
            return False
    
    def get_connection_info(self) -> Dict[str, Any]:
        """Get connection information"""
        return {
            'host': self.host,
            'username': self.username,
            'port': self.port,
            'connected': self.connected,
            'remote_workdir': self.remote_workdir
        }
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()


# Helper function for common cluster types
def create_nscc_connection() -> SSHConnection:
    """Create NSCC-configured connection (user still needs to call connect())"""
    return SSHConnection(
        host="nscc.sg",  # Default NSCC hostname
        username=None,   # User must provide
        port=22
    )