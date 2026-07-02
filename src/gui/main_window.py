"""
Main GUI window for Underwater Image Enhancer using CustomTkinter
"""
import customtkinter as ctk
from tkinter import filedialog, messagebox
import threading
from pathlib import Path
import logging
from datetime import datetime, timedelta
import sys
from typing import Optional
import torch

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import version from main app module
try:
    from app import __version__
except ImportError:
    __version__ = "dev"

from src.gui.image_processor import ImageProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set CustomTkinter appearance
ctk.set_appearance_mode("system")  # Modes: "system", "dark", "light"
ctk.set_default_color_theme("blue")  # Themes: "blue", "green", "dark-blue"

class UnderwaterEnhancerApp(ctk.CTk):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        
        # Window configuration
        self.title(f"🌊 Underwater Image Enhancer v{__version__}")
        self.geometry("980x680")  # Reduced height for laptop screens
        self.minsize(950, 650)    # Reduced minimum height
        
        # Processing state
        self.processing = False
        self.cancel_processing = False
        self.processor: Optional[ImageProcessor] = None
        self.input_files = []
        
        # Create GUI elements
        self.create_widgets()
        
        # Force update to calculate sizes
        self.update_idletasks()
        
        # Center window on screen
        self.center_window()
        
    def center_window(self):
        """Center the window on screen"""
        self.update_idletasks()
        # Use the configured size
        width = 980
        height = 680
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry(f"{width}x{height}+{x}+{y}")
    
    def create_widgets(self):
        """Create all GUI widgets"""
        
        # Main container with padding
        main_container = ctk.CTkFrame(self)
        main_container.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Title
        title_label = ctk.CTkLabel(
            main_container,
            text="Underwater Image Enhancer",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        title_label.pack(pady=(0, 8))
        
        # Model Selection Frame
        model_frame = ctk.CTkFrame(main_container)
        model_frame.pack(fill="x", pady=(0, 8))
        
        ctk.CTkLabel(model_frame, text="Model Selection", font=ctk.CTkFont(size=16, weight="bold")).pack(anchor="w", padx=10, pady=(5, 3))
        
        model_path_frame = ctk.CTkFrame(model_frame)
        model_path_frame.pack(fill="x", padx=10, pady=(0, 5))
        
        self.model_path_var = ctk.StringVar(value="")
        self.model_entry = ctk.CTkEntry(
            model_path_frame, 
            textvariable=self.model_path_var,
            placeholder_text="Select model file (.pth)",
            width=500
        )
        self.model_entry.pack(side="left", padx=(10, 10), pady=5)
        
        self.model_browse_btn = ctk.CTkButton(
            model_path_frame, 
            text="Browse...",
            command=self.browse_model,
            width=100
        )
        self.model_browse_btn.pack(side="left", padx=(0, 10), pady=5)
        
        # Input/Output Frame
        io_frame = ctk.CTkFrame(main_container)
        io_frame.pack(fill="x", pady=(0, 8))
        
        ctk.CTkLabel(io_frame, text="Input/Output Folders", font=ctk.CTkFont(size=16, weight="bold")).pack(anchor="w", padx=10, pady=(5, 3))
        
        # Input folder
        input_frame = ctk.CTkFrame(io_frame)
        input_frame.pack(fill="x", padx=10, pady=(0, 5))
        
        ctk.CTkLabel(input_frame, text="Input:", width=60).pack(side="left", padx=(10, 5))
        
        self.input_path_var = ctk.StringVar(value="")
        self.input_entry = ctk.CTkEntry(
            input_frame,
            textvariable=self.input_path_var,
            placeholder_text="Select input folder",
            width=430
        )
        self.input_entry.pack(side="left", padx=5)
        
        self.input_browse_btn = ctk.CTkButton(
            input_frame,
            text="Browse...",
            command=self.browse_input,
            width=100
        )
        self.input_browse_btn.pack(side="left", padx=(5, 10))
        
        # Output folder
        output_frame = ctk.CTkFrame(io_frame)
        output_frame.pack(fill="x", padx=10, pady=(0, 5))
        
        ctk.CTkLabel(output_frame, text="Output:", width=60).pack(side="left", padx=(10, 5))
        
        self.output_path_var = ctk.StringVar(value="")
        self.output_entry = ctk.CTkEntry(
            output_frame,
            textvariable=self.output_path_var,
            placeholder_text="Select output folder",
            width=430
        )
        self.output_entry.pack(side="left", padx=5)
        
        self.output_browse_btn = ctk.CTkButton(
            output_frame,
            text="Browse...",
            command=self.browse_output,
            width=100
        )
        self.output_browse_btn.pack(side="left", padx=(5, 10))
        
        # File info and format selection
        info_frame = ctk.CTkFrame(main_container)
        info_frame.pack(fill="x", pady=(0, 8))
        
        self.files_label = ctk.CTkLabel(info_frame, text="Files Found: 0 images")
        self.files_label.pack(side="left", padx=15)
        
        ctk.CTkLabel(info_frame, text="Output Format:").pack(side="left", padx=(50, 10))
        
        self.format_var = ctk.StringVar(value="TIFF")
        self.tiff_radio = ctk.CTkRadioButton(info_frame, text="TIFF", variable=self.format_var,
                                             value="TIFF", command=self.on_format_change)
        self.tiff_radio.pack(side="left", padx=5)

        self.jpeg_radio = ctk.CTkRadioButton(info_frame, text="JPEG", variable=self.format_var,
                                             value="JPEG", command=self.on_format_change)
        self.jpeg_radio.pack(side="left", padx=5)

        # JPEG export options (enabled only when JPEG format is selected)
        jpeg_frame = ctk.CTkFrame(main_container)
        jpeg_frame.pack(fill="x", pady=(0, 8))

        ctk.CTkLabel(jpeg_frame, text="JPEG Quality:").pack(side="left", padx=(15, 5))
        self.jpeg_quality_var = ctk.IntVar(value=95)
        self.jpeg_quality_slider = ctk.CTkSlider(
            jpeg_frame, from_=1, to=100, number_of_steps=99,
            variable=self.jpeg_quality_var, width=180,
            command=lambda v: self.jpeg_quality_value_label.configure(text=str(int(float(v))))
        )
        self.jpeg_quality_slider.pack(side="left", padx=5)
        self.jpeg_quality_value_label = ctk.CTkLabel(jpeg_frame, text="95", width=30)
        self.jpeg_quality_value_label.pack(side="left", padx=(0, 15))

        ctk.CTkLabel(jpeg_frame, text="Subsampling:").pack(side="left", padx=(0, 5))
        self.jpeg_subsampling_var = ctk.StringVar(value="Auto")
        self.jpeg_subsampling_menu = ctk.CTkOptionMenu(
            jpeg_frame, variable=self.jpeg_subsampling_var,
            values=["Auto", "4:4:4", "4:2:2", "4:2:0"], width=90
        )
        self.jpeg_subsampling_menu.pack(side="left", padx=(0, 15))

        self.jpeg_optimize_var = ctk.BooleanVar(value=False)
        self.jpeg_optimize_check = ctk.CTkCheckBox(jpeg_frame, text="Optimize", variable=self.jpeg_optimize_var)
        self.jpeg_optimize_check.pack(side="left", padx=5)

        self.jpeg_progressive_var = ctk.BooleanVar(value=False)
        self.jpeg_progressive_check = ctk.CTkCheckBox(jpeg_frame, text="Progressive", variable=self.jpeg_progressive_var)
        self.jpeg_progressive_check.pack(side="left", padx=5)

        # Set initial enabled state of JPEG controls based on default format (TIFF)
        self.on_format_change()

        # Progress Frame
        progress_frame = ctk.CTkFrame(main_container)
        progress_frame.pack(fill="x", pady=(0, 8))
        
        ctk.CTkLabel(progress_frame, text="Progress", font=ctk.CTkFont(size=16, weight="bold")).pack(anchor="w", padx=10, pady=(5, 3))
        
        self.progress_text = ctk.CTkLabel(progress_frame, text="Ready to process")
        self.progress_text.pack(anchor="w", padx=20, pady=(0, 5))
        
        self.progress_bar = ctk.CTkProgressBar(progress_frame, width=840)
        self.progress_bar.pack(padx=20, pady=(0, 5))
        self.progress_bar.set(0)
        
        self.time_label = ctk.CTkLabel(progress_frame, text="")
        self.time_label.pack(anchor="w", padx=20, pady=(0, 5))
        
        # Log Frame
        log_frame = ctk.CTkFrame(main_container)
        log_frame.pack(fill="both", expand=True, pady=(0, 8))
        
        log_header = ctk.CTkFrame(log_frame)
        log_header.pack(fill="x", padx=10, pady=(5, 3))
        
        ctk.CTkLabel(log_header, text="Log", font=ctk.CTkFont(size=16, weight="bold")).pack(side="left")
        
        self.clear_log_btn = ctk.CTkButton(
            log_header,
            text="Clear",
            command=self.clear_log,
            width=60,
            height=25
        )
        self.clear_log_btn.pack(side="right")
        
        self.log_text = ctk.CTkTextbox(log_frame, height=120, width=840)
        self.log_text.pack(padx=10, pady=(0, 5))
        
        # Control Buttons Frame
        control_frame = ctk.CTkFrame(main_container)
        control_frame.pack(fill="x", pady=(3, 0))
        
        self.process_btn = ctk.CTkButton(
            control_frame,
            text="▶ Start Processing",
            command=self.start_processing,
            width=150,
            height=35,
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.process_btn.pack(side="left", padx=(20, 10))
        
        self.cancel_btn = ctk.CTkButton(
            control_frame,
            text="■ Cancel",
            command=self.cancel_process,
            width=100,
            height=35,
            state="disabled"
        )
        self.cancel_btn.pack(side="left", padx=5)

        # GPU/CPU indicator in the middle
        has_gpu = torch.cuda.is_available() or torch.backends.mps.is_available()
        device_type = "GPU" if has_gpu else "CPU"
        device_color = "green" if has_gpu else "orange"

        # Get additional device info
        if torch.cuda.is_available():
            try:
                gpu_name = torch.cuda.get_device_name(0)
                device_tooltip = f"{gpu_name}"
            except:
                device_tooltip = "Hardware Accelerated"
        elif torch.backends.mps.is_available():
            device_tooltip = "Apple Silicon (MPS)"
        else:
            device_tooltip = "CPU Processing"

        # Create a frame for the device indicator
        device_frame = ctk.CTkFrame(control_frame, fg_color="transparent")
        device_frame.pack(side="left", padx=(50, 0), expand=True)

        # Device label with colored background
        self.device_label = ctk.CTkLabel(
            device_frame,
            text=f" {device_type} Mode ",
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color=device_color,
            corner_radius=6
        )
        self.device_label.pack(side="left")

        # Add device info
        self.device_info_label = ctk.CTkLabel(
            device_frame,
            text=f" • {device_tooltip}",
            font=ctk.CTkFont(size=11),
            text_color="gray"
        )
        self.device_info_label.pack(side="left", padx=(5, 0))

        # Theme toggle
        self.theme_btn = ctk.CTkButton(
            control_frame,
            text="🌙 Dark Mode",
            command=self.toggle_theme,
            width=120,
            height=35
        )
        self.theme_btn.pack(side="right", padx=(10, 20))
        
    def browse_model(self):
        """Browse for model file"""
        filename = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[
                ("Model files", "*.pth"),
                ("All files", "*.*")
            ]
        )
        if filename:
            self.model_path_var.set(filename)
            self.log(f"Model selected: {Path(filename).name}")
    
    def browse_input(self):
        """Browse for input folder"""
        folder = filedialog.askdirectory(title="Select Input Folder")
        if folder:
            self.input_path_var.set(folder)
            self.scan_input_folder()
    
    def browse_output(self):
        """Browse for output folder"""
        folder = filedialog.askdirectory(title="Select Output Folder")
        if folder:
            self.output_path_var.set(folder)
            self.log(f"Output folder selected: {folder}")
    
    def scan_input_folder(self):
        """Scan input folder for supported images"""
        input_path = Path(self.input_path_var.get())
        if not input_path.exists():
            return
        
        # Get all files
        all_files = list(input_path.iterdir())
        
        # Filter supported formats
        self.input_files = ImageProcessor.filter_supported_files(all_files)
        
        # Count by type
        gpr_count = len([f for f in self.input_files if f.suffix.lower() == '.gpr'])
        other_count = len(self.input_files) - gpr_count
        
        # Update display
        total = len(self.input_files)
        if total > 0:
            file_info = f"Files Found: {total} images"
            if gpr_count > 0:
                file_info += f" ({gpr_count} GPR)"
            self.files_label.configure(text=file_info)
            self.log(f"Found {total} supported images in {input_path.name}")
        else:
            self.files_label.configure(text="Files Found: 0 images")
            self.log(f"No supported images found in {input_path.name}")
    
    def validate_inputs(self) -> bool:
        """Validate all inputs before processing"""
        
        # Check model path
        if not self.model_path_var.get():
            messagebox.showerror("Error", "Please select a model file")
            return False
        
        model_path = Path(self.model_path_var.get())
        if not model_path.exists():
            messagebox.showerror("Error", "Model file does not exist")
            return False
        
        # Check input folder
        if not self.input_path_var.get():
            messagebox.showerror("Error", "Please select an input folder")
            return False
        
        if not self.input_files:
            messagebox.showerror("Error", "No supported images found in input folder")
            return False
        
        # Check output folder
        if not self.output_path_var.get():
            messagebox.showerror("Error", "Please select an output folder")
            return False
        
        return True
    
    def start_processing(self):
        """Start processing images"""
        if not self.validate_inputs():
            return
        
        # Disable controls
        self.processing = True
        self.cancel_processing = False
        self.process_start_time = datetime.now()
        self.set_controls_enabled(False)
        self.cancel_btn.configure(state="normal")
        
        # Start processing in background thread
        thread = threading.Thread(target=self.process_images_thread)
        thread.daemon = True
        thread.start()
    
    def process_images_thread(self):
        """Process images in background thread"""
        try:
            # Initialize processor
            model_path = self.model_path_var.get()
            # Config is optional - Inferencer creates default from checkpoint if not provided
            self.processor = ImageProcessor(model_path, config_path=None)
            
            # Log GPR support status
            if self.processor.gpr_support:
                import platform
                check = "[OK]" if platform.system() == "Windows" else "✓"
                self.log(f"GPR support: Available {check}")
            else:
                self.log("GPR support: Not available (gpr_tools binary missing)")
            
            self.log("Loading model...")
            self.processor.load_model()

            # Get and log device information
            device_name, is_gpu = self.processor.get_device_info()
            if is_gpu:
                self.log(f"✓ Model loaded successfully - Using GPU: {device_name}")
            else:
                self.log(f"✓ Model loaded successfully - Using CPU (GPU not available)")
            self.log("Full resolution processing enabled (with tiling for large images)")
            
            # Setup paths
            output_dir = Path(self.output_path_var.get())
            output_format = self.format_var.get()
            jpeg_options = self.get_jpeg_options() if output_format.upper() == "JPEG" else None
            if jpeg_options:
                self.log(f"JPEG export options: {jpeg_options}")

            # Track timing
            start_time = datetime.now()
            
            # Process batch
            self.log(f"Starting batch processing of {len(self.input_files)} images...")
            
            # Check for GPR files and GPR support
            gpr_files = [f for f in self.input_files if f.suffix.lower() == '.gpr']
            if gpr_files:
                # Check if GPR support is available
                if not self.processor.gpr_support:
                    error_msg = (
                        f"Cannot process {len(gpr_files)} GPR file(s).\n\n"
                        "GPR support is not available - the gpr_tools binary is missing.\n"
                        "Please rebuild the application with GPR support enabled."
                    )
                    self.log(f"Error: {error_msg}")
                    messagebox.showerror("GPR Support Missing", error_msg)
                    return
                
                self.log(f"Note: Processing {len(gpr_files)} GPR files at 4606×4030 resolution")
                self.log("This may take several minutes per image...")
            
            results = self.processor.process_batch(
                self.input_files,
                output_dir,
                output_format,
                jpeg_options=jpeg_options,
                progress_callback=self.update_progress,
                cancel_check=lambda: self.cancel_processing
            )
            
            # Summary
            successful = sum(1 for _, _, success, _ in results if success)
            failed = len(results) - successful
            
            elapsed = datetime.now() - start_time
            
            if self.cancel_processing:
                self.log(f"Processing cancelled. Completed {successful} of {len(self.input_files)} images")
            else:
                self.log(f"Batch processing complete!")
                self.log(f"Successfully processed: {successful} images")
                if failed > 0:
                    self.log(f"Failed: {failed} images")
                self.log(f"Total time: {str(elapsed).split('.')[0]}")
            
            # Show completion message
            if not self.cancel_processing:
                self.after(0, lambda: messagebox.showinfo(
                    "Processing Complete",
                    f"Successfully processed {successful} of {len(self.input_files)} images\n"
                    f"Time elapsed: {str(elapsed).split('.')[0]}"
                ))
            
        except Exception as e:
            error_msg = f"Processing error: {str(e)}"
            logger.error(error_msg)
            self.log(error_msg)
            self.after(0, lambda: messagebox.showerror("Error", error_msg))
        
        finally:
            # Re-enable controls
            self.processing = False
            self.after(0, lambda: self.set_controls_enabled(True))
            self.after(0, lambda: self.cancel_btn.configure(state="disabled"))
            self.after(0, lambda: self.progress_bar.set(0))
            self.after(0, lambda: self.progress_text.configure(text="Ready to process"))
            self.after(0, lambda: self.time_label.configure(text=""))
    
    def update_progress(self, current: int, total: int, filename: str, status: str):
        """Update progress display"""
        progress = current / total if total > 0 else 0
        
        # Calculate time estimates
        if current > 0:
            elapsed = datetime.now() - self.process_start_time
            per_image = elapsed / current
            remaining = per_image * (total - current)
            time_text = f"Time Elapsed: {str(elapsed).split('.')[0]} | Est. Remaining: {str(remaining).split('.')[0]}"
        else:
            time_text = ""
        
        # Update UI in main thread
        self.after(0, lambda: self.progress_bar.set(progress))
        self.after(0, lambda: self.progress_text.configure(
            text=f"Processing: {filename} ({current}/{total}) - {status}"
        ))
        self.after(0, lambda: self.time_label.configure(text=time_text))
        
        # Log status
        if "Complete" in status:
            self.log(f"✓ {filename} processed")
        elif "Failed" in status:
            self.log(f"✗ {filename} failed: {status.replace('Failed: ', '')}")
        elif "tile" in status.lower():
            # Log tile processing updates
            self.log(f"  • {filename}: {status.replace('Processing - ', '')}")
    
    def cancel_process(self):
        """Cancel ongoing processing"""
        self.cancel_processing = True
        self.log("Cancelling processing...")
        self.cancel_btn.configure(state="disabled")
    
    @property
    def _jpeg_controls(self):
        """JPEG option widgets, toggled together based on format/processing state"""
        return (self.jpeg_quality_slider, self.jpeg_subsampling_menu,
                self.jpeg_optimize_check, self.jpeg_progressive_check)

    def on_format_change(self):
        """Enable the JPEG option controls only when JPEG output is selected"""
        state = "normal" if self.format_var.get().upper() == "JPEG" else "disabled"
        for widget in self._jpeg_controls:
            widget.configure(state=state)

    def get_jpeg_options(self) -> dict:
        """Collect PIL JPEG save options from the UI controls"""
        options = {'quality': int(self.jpeg_quality_var.get())}
        subsampling_map = {"4:4:4": 0, "4:2:2": 1, "4:2:0": 2}
        subsampling = self.jpeg_subsampling_var.get()
        if subsampling in subsampling_map:  # "Auto" omits the param (PIL default)
            options['subsampling'] = subsampling_map[subsampling]
        if self.jpeg_optimize_var.get():
            options['optimize'] = True
        if self.jpeg_progressive_var.get():
            options['progressive'] = True
        return options

    def set_controls_enabled(self, enabled: bool):
        """Enable/disable controls during processing"""
        state = "normal" if enabled else "disabled"
        self.model_browse_btn.configure(state=state)
        self.input_browse_btn.configure(state=state)
        self.output_browse_btn.configure(state=state)
        self.process_btn.configure(state=state)
        self.tiff_radio.configure(state=state)
        self.jpeg_radio.configure(state=state)
        if enabled:
            self.on_format_change()  # restore JPEG controls to match selected format
        else:
            for widget in self._jpeg_controls:
                widget.configure(state="disabled")
    
    def log(self, message: str):
        """Add message to log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        self.after(0, lambda: self.log_text.insert("end", log_entry))
        self.after(0, lambda: self.log_text.see("end"))
        logger.info(message)
    
    def clear_log(self):
        """Clear the log text"""
        self.log_text.delete("1.0", "end")
    
    def toggle_theme(self):
        """Toggle between dark and light theme"""
        current = ctk.get_appearance_mode()
        if current == "Dark":
            ctk.set_appearance_mode("light")
            self.theme_btn.configure(text="🌙 Dark Mode")
        else:
            ctk.set_appearance_mode("dark")
            self.theme_btn.configure(text="☀️ Light Mode")