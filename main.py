#!/usr/bin/env python3
"""
Enhanced Bruker NMR Data Reader
Main entry point for the application
"""

import tkinter as tk
from gui.main_window import EnhancedBrukerReader

def main():
    """Main entry point"""
    root = tk.Tk()
    app = EnhancedBrukerReader(root)
    root.mainloop()

if __name__ == "__main__":
    main()