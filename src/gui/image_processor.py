"""
Image processor that handles GPR, TIFF, and JPEG inputs
"""
from pathlib import Path
import tempfile
import logging
from typing import Optional, Callable
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.converters.gpr_converter import GPRConverter
from inference.inference import Inferencer

logger = logging.getLogger(__name__)

class ImageProcessor:
    """Handles image processing with GPR support"""
    
    def __init__(self, model_path: str, config_path: str = None):
        """
        Initialize the image processor
        
        Args:
            model_path: Path to the ML model checkpoint
            config_path: Optional path to config file
        """
        self.model_path = model_path
        self.config_path = config_path
        self.inferencer = None
        self.gpr_converter = GPRConverter()
        
        # Check if GPR support is available
        self.gpr_support = GPRConverter.is_available()
        if not self.gpr_support:
            logger.warning("GPR support is not available - gpr_tools binary not found")
        
    def load_model(self):
        """Load the ML model - matching inference.py exactly"""
        if not self.inferencer:
            # Create inferencer exactly like inference.py
            self.inferencer = Inferencer(self.model_path, self.config_path)

            # Override config for full-size processing exactly like inference.py --full-size
            config_override = {'inference': {'resize_inference': False}}
            self.inferencer.config.update(config_override)
            self.inferencer.setup_transforms()  # Refresh transforms with new config

            # Log device information clearly
            import torch
            if torch.cuda.is_available():
                device_type = "GPU (CUDA)"
            elif torch.backends.mps.is_available():
                device_type = "GPU (MPS)"
            else:
                device_type = "CPU"
            logger.info(f"Model loaded from {self.model_path} - Using {device_type} - Full resolution processing enabled")
    
    def process_image(self, input_path: Path, output_path: Path,
                     output_format: str = 'TIFF', jpeg_options: Optional[dict] = None,
                     progress_callback=None) -> Path:
        """
        Process a single image, handling GPR conversion if needed

        Args:
            input_path: Path to input image
            output_path: Path for output image
            output_format: Output format ('TIFF' or 'JPEG')
            jpeg_options: Optional dict of PIL JPEG save params (quality, subsampling, etc.)
            progress_callback: Optional callback for progress updates

        Returns:
            Path to processed image
        """
        # Ensure model is loaded
        self.load_model()
        
        # Check if input is GPR
        if self.gpr_converter.is_gpr_file(input_path):
            if not self.gpr_support:
                raise RuntimeError(
                    "GPR file detected but GPR support is not available. "
                    "The gpr_tools binary is missing from the bundled application."
                )
            # Convert to TIFF first (let converter create temp file)
            tiff_path = self.gpr_converter.convert(input_path, output_path=None)
            
            try:
                # Process the TIFF using inferencer exactly like inference.py
                # The inferencer.process_image method will handle tiling for large images
                output_img = self.inferencer.process_image(tiff_path, output_path, progress_callback=progress_callback, save_options=jpeg_options)
            finally:
                # Clean up temp TIFF
                if tiff_path and tiff_path.exists():
                    try:
                        tiff_path.unlink()
                    except:
                        pass
            
            # Convert output format if needed
            if output_format.upper() == 'JPEG' and output_path.suffix.lower() != '.jpg':
                from PIL import Image
                img = Image.open(output_path)
                jpeg_path = output_path.with_suffix('.jpg')
                img.save(jpeg_path, 'JPEG', **(jpeg_options or {'quality': 95}))
                if output_path != jpeg_path:
                    output_path.unlink()
                return jpeg_path
            
            return output_path
        else:
            # Direct processing for TIFF/JPEG - using inferencer exactly like inference.py
            # The inferencer.process_image method will handle tiling for large images
            output_img = self.inferencer.process_image(input_path, output_path, progress_callback=progress_callback, save_options=jpeg_options)

            # Convert output format if needed
            if output_format.upper() == 'JPEG' and not output_path.suffix.lower() in ['.jpg', '.jpeg']:
                from PIL import Image
                img = Image.open(output_path)
                jpeg_path = output_path.with_suffix('.jpg')
                img.save(jpeg_path, 'JPEG', **(jpeg_options or {'quality': 95}))
                if output_path != jpeg_path:
                    output_path.unlink()
                return jpeg_path
            
            return output_path
    
    def process_batch(self, input_files: list[Path], output_dir: Path,
                     output_format: str = 'TIFF',
                     jpeg_options: Optional[dict] = None,
                     progress_callback: Optional[Callable] = None,
                     cancel_check: Optional[Callable] = None) -> list[tuple[Path, Path, bool, str]]:
        """
        Process multiple images in batch

        Args:
            input_files: List of input image paths
            output_dir: Directory for output images
            output_format: Output format ('TIFF' or 'JPEG')
            jpeg_options: Optional dict of PIL JPEG save params (quality, subsampling, etc.)
            progress_callback: Optional callback(current, total, filename, status)
            cancel_check: Optional callback that returns True to cancel
        
        Returns:
            List of tuples (input_path, output_path, success, error_message)
        """
        # Ensure model is loaded
        self.load_model()
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = []
        
        for i, input_path in enumerate(input_files):
            # Check for cancellation
            if cancel_check and cancel_check():
                logger.info("Batch processing cancelled by user")
                break
            
            # Update progress
            if progress_callback:
                progress_callback(i, len(input_files), input_path.name, "Processing")
            
            # Determine output filename
            ext = '.tiff' if output_format.upper() == 'TIFF' else '.jpg'
            output_path = output_dir / f"{input_path.stem}_enhanced{ext}"
            
            try:
                # Create a callback to pass tile updates to the main progress callback
                def tile_progress(message):
                    if progress_callback:
                        # Append tile info to the current file status
                        progress_callback(i, len(input_files), input_path.name, f"Processing - {message}")
                
                # Process the image with tile progress callback
                actual_output = self.process_image(input_path, output_path, output_format, jpeg_options=jpeg_options, progress_callback=tile_progress)
                results.append((input_path, actual_output, True, "Success"))
                
                if progress_callback:
                    progress_callback(i + 1, len(input_files), input_path.name, "Complete")
                    
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Failed to process {input_path.name}: {error_msg}")
                results.append((input_path, None, False, error_msg))
                
                if progress_callback:
                    progress_callback(i + 1, len(input_files), input_path.name, f"Failed: {error_msg}")
        
        return results
    
    def get_device_info(self) -> tuple[str, bool]:
        """
        Get information about the device being used for processing

        Returns:
            Tuple of (device_name, is_gpu)
        """
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "CUDA GPU"
            return (device_name, True)
        elif torch.backends.mps.is_available():
            return ("Apple Silicon (MPS)", True)
        else:
            return ("CPU", False)

    @staticmethod
    def get_supported_formats() -> list[str]:
        """Get list of supported input formats"""
        return ['.gpr', '.tiff', '.tif', '.jpg', '.jpeg', '.png']

    @staticmethod
    def filter_supported_files(files: list[Path]) -> list[Path]:
        """Filter list of files to only supported formats"""
        supported = ImageProcessor.get_supported_formats()
        return [f for f in files if f.suffix.lower() in supported]