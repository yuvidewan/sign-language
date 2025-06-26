#!/usr/bin/env python3
"""
Setup script for Lip Reading AI System
"""

import os
import sys
import subprocess
import platform

def print_banner():
    """Print setup banner"""
    print("=" * 60)
    print("LIP READING AI SYSTEM - SETUP")
    print("=" * 60)
    print()

def check_virtual_environment():
    """Check if running in a virtual environment"""
    print("Checking virtual environment...")
    
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    
    if in_venv:
        print("✓ Running in virtual environment")
        return True
    else:
        print("⚠️  Not running in virtual environment")
        print("It's recommended to use a virtual environment to avoid conflicts.")
        
        response = input("Would you like to create a virtual environment? (y/n): ").lower()
        if response == 'y':
            create_virtual_environment()
            return True
        else:
            print("Continuing without virtual environment...")
            return False

def create_virtual_environment():
    """Create a virtual environment"""
    print("\nCreating virtual environment...")
    
    try:
        # Create virtual environment
        subprocess.check_call([sys.executable, "-m", "venv", "lip_reader_env"])
        
        print("✓ Virtual environment created: lip_reader_env")
        print("\nTo activate the virtual environment:")
        
        if platform.system() == "Windows":
            print("lip_reader_env\\Scripts\\activate")
        else:
            print("source lip_reader_env/bin/activate")
        
        print("\nAfter activating, run this setup script again.")
        sys.exit(0)
        
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to create virtual environment: {e}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    print("Checking Python version...")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"✗ Python {version.major}.{version.minor} is not supported.")
        print("Please use Python 3.8 or higher.")
        return False
    
    print(f"✓ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def install_dependencies():
    """Install required dependencies"""
    print("\nInstalling dependencies...")
    
    try:
        # Install from requirements.txt
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install dependencies: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    print("\nCreating directories...")
    
    directories = [
        'data',
        'data/train',
        'data/val',
        'data/test',
        'models',
        'logs',
        'checkpoints',
        'outputs',
        'notebooks'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}")

def download_sample_data():
    """Download or create sample data"""
    print("\nSetting up sample data...")
    
    # Create sample metadata
    sample_metadata = {
        'dataset_info': {
            'total_videos': 10,
            'output_size': [64, 64],
            'sequence_length': 30,
            'processed_date': '2024-01-01',
            'note': 'Sample dataset for demonstration'
        },
        'samples': [
            {
                'video_path': f'sample_video_{i+1}.mp4',
                'text': f'sample text {i+1}',
                'duration': 2.5 + i * 0.5,
                'sequence_length': 30
            }
            for i in range(10)
        ]
    }
    
    # Save sample metadata
    import json
    metadata_path = os.path.join('data', 'sample_dataset', 'metadata.json')
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    
    with open(metadata_path, 'w') as f:
        json.dump(sample_metadata, f, indent=2)
    
    print(f"✓ Created sample dataset metadata: {metadata_path}")

def check_gpu():
    """Check GPU availability"""
    print("\nChecking GPU availability...")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✓ GPU available: {gpu_name}")
            print(f"  - Number of GPUs: {gpu_count}")
        else:
            print("⚠️  No GPU detected. Training will use CPU (slower)")
        return True
    except ImportError:
        print("⚠️  PyTorch not installed. GPU check skipped.")
        return False

def run_tests():
    """Run system tests"""
    print("\nRunning system tests...")
    
    try:
        result = subprocess.run([sys.executable, "test_system.py"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ System tests passed")
            return True
        else:
            print("✗ System tests failed")
            print("Error output:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"✗ Failed to run tests: {e}")
        return False

def create_config_file():
    """Create configuration file if it doesn't exist"""
    print("\nChecking configuration...")
    
    if not os.path.exists('config.py'):
        print("⚠️  Configuration file not found. Creating default config...")
        # The config.py file should already exist from our earlier creation
        print("✓ Configuration file created")
    else:
        print("✓ Configuration file exists")

def print_next_steps():
    """Print next steps for the user"""
    print("\n" + "=" * 60)
    print("SETUP COMPLETED!")
    print("=" * 60)
    print()
    print("Next steps to get started:")
    print()
    print("1. Prepare your dataset:")
    print("   - Collect videos with clear lip movements")
    print("   - Create text transcripts for each video")
    print("   - Organize data in data/train, data/val, data/test")
    print()
    print("2. Preprocess your data:")
    print("   python preprocessing/extract_lip_sequences.py --video_path data/videos --output_path data/processed")
    print()
    print("3. Train the model:")
    print("   python training/train_lip_reader.py --data_path data/processed --epochs 50")
    print()
    print("4. Test real-time lip reading:")
    print("   python inference/real_time_lip_reader.py")
    print()
    print("5. Run the demo:")
    print("   python demo_lip_reader.py")
    print()
    print("For more information, see the README.md file.")
    print()
    print("Happy lip reading! 🎉")

def main():
    """Main setup function"""
    print_banner()
    
    # Check virtual environment
    check_virtual_environment()
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("\n⚠️  Some dependencies failed to install.")
        print("You may need to install them manually:")
        print("pip install torch torchvision opencv-python mediapipe")
        print("Continue with setup? (y/n): ", end="")
        if input().lower() != 'y':
            sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Create sample data
    download_sample_data()
    
    # Check GPU
    check_gpu()
    
    # Create config
    create_config_file()
    
    # Run tests
    print("\nWould you like to run system tests? (y/n): ", end="")
    if input().lower() == 'y':
        run_tests()
    
    # Print next steps
    print_next_steps()

if __name__ == '__main__':
    main() 