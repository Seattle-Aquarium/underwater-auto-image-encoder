#!/usr/bin/env python3
"""
Underwater Image Enhancer GUI Application
Main entry point for the desktop application
"""

__version__ = "0.7.1"

import sys
import os
from pathlib import Path
import logging
import traceback
from datetime import datetime

# Add source directory to path (parent of gui directory)
sys.path.insert(0, str(Path(__file__).parent.parent))

def setup_logging(version=__version__):
    """Setup file logging for crash diagnosis"""
    # Determine log directory based on platform
    if sys.platform == 'win32':
        log_dir = Path(os.environ.get('APPDATA', '.')) / 'UnderwaterEnhancer' / 'logs'
    elif sys.platform == 'darwin':
        log_dir = Path.home() / 'Library' / 'Logs' / 'UnderwaterEnhancer'
    else:
        log_dir = Path.home() / '.local' / 'share' / 'UnderwaterEnhancer' / 'logs'

    # Create log directory if it doesn't exist
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create log filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'underwater_enhancer_{timestamp}.log'

    # Configure root logger
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

    # Log startup information
    logger = logging.getLogger(__name__)
    logger.info(f"Underwater Enhancer v{version} starting")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Platform: {sys.platform}")
    logger.info(f"Log file: {log_file}")

    # Store version for use in exception handler
    app_version = version

    # Set up global exception handler to catch unhandled exceptions
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

        # Write crash report
        crash_file = log_dir / f'crash_{timestamp}.txt'
        with open(crash_file, 'w') as f:
            f.write(f"Underwater Enhancer Crash Report\n")
            f.write(f"Version: {app_version}\n")
            f.write(f"Time: {datetime.now().isoformat()}\n")
            f.write(f"Python: {sys.version}\n")
            f.write(f"Platform: {sys.platform}\n\n")
            f.write("Exception:\n")
            traceback.print_exception(exc_type, exc_value, exc_traceback, file=f)

        logger.info(f"Crash report saved to: {crash_file}")

    sys.excepthook = handle_exception

    return log_file

def smoke_test():
    """Run smoke tests for CI validation"""
    print(f"Underwater Enhancer v{__version__}")
    print("Running smoke tests...")
    
    
    errors = []
    
    # Test critical imports
    try:
        import torch
        print("  [OK] PyTorch imported")
    except ImportError as e:
        errors.append(f"PyTorch import failed: {e}")
        print(f"  [FAIL] PyTorch: {e}")
    
    try:
        import cv2
        print("  [OK] OpenCV imported")
    except (ImportError, Exception) as e:
        # OpenCV has known issues with PyInstaller on macOS, warn but don't fail
        if "recursion is detected" in str(e):
            print(f"  [WARN] OpenCV: Known PyInstaller issue - {e}")
        else:
            errors.append(f"OpenCV import failed: {e}")
            print(f"  [FAIL] OpenCV: {e}")
    
    try:
        import customtkinter
        print("  [OK] CustomTkinter imported")
    except ImportError as e:
        errors.append(f"CustomTkinter import failed: {e}")
        print(f"  [FAIL] CustomTkinter: {e}")
    
    try:
        from src.models.unet_autoencoder import UNetAutoencoder
        print("  [OK] Model imported")
    except ImportError as e:
        errors.append(f"Model import failed: {e}")
        print(f"  [FAIL] Model: {e}")
    
    try:
        from src.converters.gpr_converter import GPRConverter
        print("  [OK] GPR converter imported")
    except ImportError as e:
        errors.append(f"GPR converter import failed: {e}")
        print(f"  [FAIL] GPR converter: {e}")
    
    try:
        from src.gui.main_window import UnderwaterEnhancerApp
        print("  [OK] GUI imported")
    except ImportError as e:
        errors.append(f"GUI import failed: {e}")
        print(f"  [FAIL] GUI: {e}")
    
    if errors:
        print("\nSmoke test FAILED with errors:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)
    else:
        print("\nSmoke test PASSED - all modules imported successfully!")
        sys.exit(0)

def main():
    """Main application entry point"""
    # Check for smoke test mode
    if os.environ.get('SMOKE_TEST') == '1' or '--smoke-test' in sys.argv:
        smoke_test()
        return

    # Set up logging first
    log_file = setup_logging(__version__)
    logger = logging.getLogger(__name__)

    try:
        # Log system information for debugging GPU issues
        import torch
        logger.info(f"Underwater Enhancer v{__version__}")
        logger.info(f"PyTorch Version: {torch.__version__}")
        logger.info(f"CUDA Available: {torch.cuda.is_available()}")
        logger.info(f"MPS Available: {torch.backends.mps.is_available()}")

        if torch.cuda.is_available():
            logger.info(f"CUDA Version: {torch.version.cuda}")
            logger.info(f"GPU Device: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Count: {torch.cuda.device_count()}")
        elif torch.backends.mps.is_available():
            logger.info("Running in MPS mode (Apple Silicon GPU)")
        else:
            logger.info("Running in CPU mode")

        # Print log location for user
        print(f"Log file: {log_file}")
        print("-" * 40)

        from src.gui.main_window import UnderwaterEnhancerApp
        logger.info("Creating main application window")
        app = UnderwaterEnhancerApp()
        logger.info("Starting main event loop")
        app.mainloop()
        logger.info("Application exited normally")

    except Exception as e:
        logger.exception("Fatal error in main()")
        raise

if __name__ == "__main__":
    main()