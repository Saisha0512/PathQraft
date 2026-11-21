import subprocess
import sys
from tkinter import messagebox


def check_and_install_osmnx():
    required_packages = ['osmnx', 'matplotlib', 'networkx', 'geopandas', 'descartes']
    missing_packages = []
    for pkg in required_packages:
        try:
            __import__(pkg)
        except ImportError:
            missing_packages.append(pkg)

    if missing_packages:
        print(f"Missing packages: {', '.join(missing_packages)}. Attempting to install...")
        try:
            # Use subprocess to run pip in a new process
            subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing_packages)
            print("Required packages installed successfully.")
            messagebox.showinfo("Installation Complete", "Required dependencies have been installed. You might need to restart the application for changes to full effect.")
        except Exception as e:
            print(f"Error installing packages: {e}")
            messagebox.showerror("Installation Error", f"Failed to install required packages. Please try running 'pip install {' '.join(required_packages)}' manually in your terminal.\nError: {e}")
    else:
        print("All required packages are already installed.")