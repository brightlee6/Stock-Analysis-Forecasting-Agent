"""
Installation script for the Stock Analysis Agent.
This script installs the package in development mode.
"""
import subprocess
import sys

def install_package():
    """Install the package in development mode."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", "."])
        print("Package installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error installing package: {e}")
        sys.exit(1)

if __name__ == "__main__":
    install_package() 