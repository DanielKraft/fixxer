#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FIXXER Engine
High-level workflow orchestration for photo organization and processing.

CHAOS PATCHED // LOGIC INJECTED
"""

from __future__ import annotations

import os
import json
import time
import threading
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Tuple, List, Dict, Any, Callable
from collections import defaultdict, Counter
from io import BytesIO
import re
import subprocess
import sys
import math

# Import from new modules
from .config import (
    SUPPORTED_EXTENSIONS,
    RAW_SUPPORT,
    GROUP_KEYWORDS,
    BEST_PICK_PREFIX,
    PREP_FOLDER_NAME,
    TIER_A_FOLDER,
    TIER_B_FOLDER,
    TIER_C_FOLDER,
    SESSION_DATE,
    SESSION_TIMESTAMP,
    MAX_WORKERS,
    DEFAULT_MODEL_NAME,
    OLLAMA_URL,
    load_app_config,
    save_app_config
)

from .security import (
    calculate_sha256,
    verify_file_move_with_hash,
    read_existing_sidecar,
    write_sidecar_file
)

from .vision import (
    convert_raw_to_jpeg,
    encode_image,
    get_image_bytes_for_analysis,
    check_ollama_connection,
    get_ai_description,
    get_ai_name_with_cache,
    critique_single_image
)

# --- Optional Libs (Required for Real Engine) ---

try:
    import imagehash
    from PIL import Image, ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    V5_LIBS_AVAILABLE = True
except ImportError:
    V5_LIBS_AVAILABLE = False
    imagehash = None  # Type stub for type hints

try:
    import cv2
    import numpy as np
    V6_CULL_LIBS_AVAILABLE = True
except ImportError:
    V6_CULL_LIBS_AVAILABLE = False

try:
    import exifread
    V6_4_EXIF_LIBS_AVAILABLE = True
except ImportError:
    V6_4_EXIF_LIBS_AVAILABLE = False

# ==============================================================================
# STATS TRACKER
# ==============================================================================

class StatsTracker:
    """
    Real-time statistics tracker for workflow progress.
    Designed for thread-safe callback communication between engine and TUI.
    
    Usage:
        tracker = StatsTracker(callback=my_callback_function)
        tracker.start_timer()
        tracker.update('bursts', 42)
        tracker.stop_timer()
    """
    
    def __init__(self, callback: Optional[Callable[[str, Any], None]] = None):
        """
        Args:
            callback: Function to call when stats update (receives key, value)
        """
        self.callback = callback
        self._stats = {
            'bursts': 0,
            'tier_a': 0,
            'tier_b': 0,
            'tier_c': 0,
            'heroes': 0,
            'archived': 0,
            'time': '--'
        }
        self._start_time = None
    
    def update(self, key: str, value: Any) -> None:
        """
        Update a stat and trigger callback.
        
        Args:
            key: Stat identifier (e.g., 'bursts', 'tier_a', 'archived')
            value: New value for the stat
        """
        self._stats[key] = value
        if self.callback:
            self.callback(key, value)
    
    def start_timer(self) -> None:
        """Start the workflow timer."""
        self._start_time = datetime.now()
        self.update('time', 'Running...')
    
    def stop_timer(self) -> None:
        """Stop the timer and calculate human-readable duration."""
        if self._start_time:
            duration = datetime.now() - self._start_time
            total_seconds = int(duration.total_seconds())
            minutes = total_seconds // 60
            seconds = total_seconds % 60
            
            # Format: "2m 34s" (human-readable for quick glances)
            if minutes > 0:
                time_str = f"{minutes}m {seconds}s"
            else:
                time_str = f"{seconds}s"
            
            self.update('time', time_str)
    
    def reset(self) -> None:
        """Reset all stats to default values."""
        for key in self._stats.keys():
            if key == 'time':
                self._stats[key] = '--'
            else:
                self._stats[key] = 0
        self._start_time = None


# ==============================================================================
# IV. CORE UTILITIES
# ==============================================================================

def no_op_logger(message: str) -> None:
    """A dummy logger that does nothing, for when no callback is provided."""
    pass

def check_rawpy(log_callback: Callable[[str], None] = no_op_logger):
    """Check if rawpy is available and update RAW support"""
    from . import config
    try:
        import rawpy
        config.RAW_SUPPORT = True
        # Add ALL common RAW formats that rawpy supports
        # rawpy uses libraw which supports 100+ RAW formats
        raw_extensions = {'.rw2', '.arw', '.cr2', '.cr3', '.nef', '.dng', '.raf', '.orf', '.pef', '.srw',
                         '.3fr', '.ari', '.bay', '.crw', '.cs1', '.dc2', '.dcr', '.drf', '.eip', '.erf',
                         '.fff', '.iiq', '.k25', '.kdc', '.mdc', '.mef', '.mos', '.mrw', '.nrw', '.obm',
                         '.ptx', '.pxn', '.r3d', '.raw', '.rwl', '.rw1', '.rwz', '.sr2', '.srf', '.sti', '.x3f'}
        config.SUPPORTED_EXTENSIONS.update(raw_extensions)
        log_callback(f"✓ [green]rawpy found.[/green] RAW support enabled.")
        log_callback(f"  Common formats: RW2, CR2, CR3, NEF, ARW, DNG, RAF, ORF, PEF, SRW + 40 more")
    except ImportError:
        config.RAW_SUPPORT = False
        log_callback("✗ [yellow]rawpy not found.[/yellow] RAW support disabled.")
        log_callback("  Install with: pip install rawpy")
    except Exception as e:
        config.RAW_SUPPORT = False
        log_callback(f"✗ [red]rawpy check failed:[/red] {e}")

def get_available_models(log_callback: Callable[[str], None] = no_op_logger) -> Optional[List[str]]:
    """Get list of available Ollama models."""
    try:
        log_callback("   [grey]Checking Ollama connection...[/grey]")
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, check=True)
        lines = result.stdout.strip().split('\n')[1:]
        models = [line.split()[0] for line in lines if line.strip()]
        log_callback(f"   [green]✓ Ollama connected.[/green] Found {len(models)} models.")
        return models
    except subprocess.CalledProcessError as e:
        log_callback(f"   [red]✗ Ollama command failed:[/red] {e.stdout}")
        return None
    except FileNotFoundError:
        log_callback("   [red]✗ Ollama not found.[/red] Please ensure 'ollama' is in your PATH.")
        return None
    except Exception as e:
        log_callback(f"   [red]✗ Ollama connection error:[/red] {e}")
        return None

def get_unique_filename(base_name: str, extension: str, destination: Path) -> Path:
    """Generate unique filename if file already exists"""
    filename = destination / f"{base_name}{extension}"
    if not filename.exists():
        return filename
    counter = 1
    while True:
        filename = destination / f"{base_name}-{counter:02d}{extension}"
        if not filename.exists():
            return filename
        counter += 1

def get_unique_filename_simulated(base_name: str, extension: str, destination: Path, simulated_paths: set) -> Path:
    """
    Generate unique filename for preview mode using simulated_paths set.

    This is critical for accurate collision detection in dry-run mode where
    files don't actually exist yet but we need to simulate naming conflicts.

    Args:
        base_name: Base filename without extension
        extension: File extension (including dot)
        destination: Destination directory path
        simulated_paths: Set of Path objects representing files that will exist

    Returns:
        Path object with collision-free filename
    """
    filename = destination / f"{base_name}{extension}"
    if filename not in simulated_paths:
        return filename
    counter = 1
    while True:
        filename = destination / f"{base_name}-{counter:02d}{extension}"
        if filename not in simulated_paths:
            return filename
        counter += 1

def format_duration(duration: timedelta) -> str:
    """Converts timedelta to readable string like '1d 4h 15m'"""
    total_seconds = int(duration.total_seconds())
    days, remainder = divmod(total_seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, _ = divmod(remainder, 60)
    parts = []
    if days > 0: parts.append(f"{days}d")
    if hours > 0: parts.append(f"{hours}h")
    if minutes > 0 or (days == 0 and hours == 0): parts.append(f"{minutes}m")
    return " ".join(parts) if parts else "0m"

def generate_bar_chart(data: dict, bar_width: int = 25, bar_char: str = "■") -> List[str]:
    """Generates ASCII bar chart lines from a dictionary"""
    output_lines = []
    if not data: return output_lines
    max_val = max(data.values()); max_val = 1 if max_val == 0 else max_val
    max_key_len = max(len(key) for key in data.keys())
    for key, val in data.items():
        bar_len = int(math.ceil((val / max_val) * bar_width))
        bar = bar_char * bar_len
        line = f"     {key.ljust(max_key_len)}: [bold]{str(val).ljust(4)}[/bold] {bar}"
        output_lines.append(line)
    return output_lines

def clean_filename(description: str) -> str:
    """Convert AI description to clean filename"""
    clean = description.strip('"\'.,!?')
    clean = re.sub(r'[^\w\s-]', '', clean)
    clean = re.sub(r'[-\s]+', '-', clean)
    clean = clean.lower()[:60]
    return clean.strip('-')

def categorize_description(description: str) -> str:
    """Determine category based on keywords in description"""
    description_lower = description.lower()
    category_scores = {}
    for category, keywords in GROUP_KEYWORDS.items():
        score = sum(1 for keyword in keywords if keyword in description_lower)
        if score > 0:
            category_scores[category] = score
    if category_scores:
        return max(category_scores, key=category_scores.get)
    return "Miscellaneous"

def write_rename_log(log_path: Path, original_name: str, new_name: str, destination: Path):
    """(V9.3) Append an AI rename operation to the log file."""
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"{timestamp} | {original_name} -> {new_name} | {destination}\n"
        with open(log_path, 'a') as f:
            f.write(log_entry)
    except Exception:
        pass # Silent fail

def initialize_rename_log(log_path: Path):
    """(V9.3) Initialize the rename log file with a header."""
    try:
        header = f"# FIXXER AI Rename Log - {SESSION_TIMESTAMP}\n"
        header += f"# Format: timestamp | original_name -> new_name | destination\n"
        header += "=" * 80 + "\n"
        with open(log_path, 'w') as f:
            f.write(header)
    except Exception:
        pass

# ==============================================================================
# V. AI & ANALYSIS MODULES (The "Brains")
# ==============================================================================

def is_already_ai_named(filename: str) -> bool:
    """(V9.2) Check if a PICK file already has an AI-generated name."""
    if not re.search(r'_PICK\.\w+$', filename, re.IGNORECASE):
        return False
    if filename.startswith('_PICK_'):
        return False
    return True

def get_image_hash(image_path: Path, log_callback: Callable[[str], None] = no_op_logger) -> Tuple[Path, Optional[Any]]:
    """Calculates perceptual hash (visual fingerprint) of an image."""
    if not V5_LIBS_AVAILABLE:
        log_callback("[red]Missing 'imagehash' library. Burst grouping will fail.[/red]")
        return image_path, None
    
    # NEW: Use your existing rawpy helper instead of dcraw subprocess!
    # This handles ARW, CR2, NEF, etc. purely in memory.
    try:
        raw_formats = {'.rw2', '.cr2', '.cr3', '.nef', '.arw', '.dng', '.raf', '.orf', '.pef', '.srw'}
        if image_path.suffix.lower() in raw_formats:
            jpeg_bytes = convert_raw_to_jpeg(image_path, log_callback)
            if jpeg_bytes:
                img = Image.open(BytesIO(jpeg_bytes))
                return image_path, imagehash.phash(img)
            else:
                return image_path, None
           
        # Standard images (JPG, PNG, etc.)
        with Image.open(image_path) as img:
            return image_path, imagehash.phash(img)
            
    except Exception as e:
        log_callback(f"     [yellow]Skipping hash for {image_path.name}: {e}[/yellow]")
        return image_path, None

def analyze_image_quality(image_bytes: bytes) -> Dict[str, float]:
    """Analyzes image bytes for sharpness and exposure."""
    if not V6_CULL_LIBS_AVAILABLE:
        return {'sharpness': 0.0, 'blacks_pct': 0.0, 'whites_pct': 0.0}
        
    scores = {'sharpness': 0.0, 'blacks_pct': 0.0, 'whites_pct': 0.0}
    try:
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None: return scores

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        scores['sharpness'] = float(laplacian_var)

        total_pixels = gray.size
        crushed_blacks = np.sum(gray < 10)
        scores['blacks_pct'] = float(crushed_blacks / total_pixels)
        blown_whites = np.sum(gray > 245)
        scores['whites_pct'] = float(blown_whites / total_pixels)
        return scores
    except Exception:
        return scores

def analyze_single_exif(image_path: Path) -> Optional[Dict]:
    """Thread-pool worker: Opens image and extracts key EXIF data."""
    if not V6_4_EXIF_LIBS_AVAILABLE:
        return None
        
    try:
        with open(image_path, 'rb') as f:
            tags = exifread.process_file(f, details=False, stop_tag='EXIF DateTimeOriginal')
            if not tags or 'EXIF DateTimeOriginal' not in tags: return None
            timestamp_str = str(tags['EXIF DateTimeOriginal'])
            dt_obj = datetime.strptime(timestamp_str, '%Y:%m:%d %H:%M:%S')
            camera = str(tags.get('Image Model', 'Unknown')).strip()
            focal_len = str(tags.get('EXIF FocalLength', 'Unknown')).split(' ')[0]
            aperture_str = "Unknown"
            aperture_tag = tags.get('EXIF FNumber')
            
            if aperture_tag:
                val = aperture_tag.values[0]
                if hasattr(val, 'num') and hasattr(val, 'den'):
                    aperture_val = 0.0 if val.den == 0 else float(val.num) / float(val.den)
                    aperture_str = f"f/{aperture_val:.1f}"
                else:
                    aperture_str = f"f/{val:.1f}"

            if not camera: camera = "Unknown"
            if not focal_len: focal_len = "Unknown"
            if aperture_str == "f/0.0": aperture_str = "Unknown"

            return {
                'timestamp': dt_obj,
                'camera': camera,
                'focal_length': f"{focal_len} mm",
                'aperture': aperture_str
            }
    except Exception:
        return None

def process_single_image(
    image_path: Path,
    destination_base: Path,
    model_name: str,
    rename_log_path: Optional[Path] = None,
    log_callback: Callable[[str], None] = no_op_logger,
    preview_mode: bool = False,
    ai_cache: Optional[Dict[str, Dict]] = None,
    cache_lock: Optional[threading.Lock] = None,
    simulated_paths: Optional[set] = None
) -> Tuple[Path, bool, str, str]:
    """
    (V9.3) Process one image: get AI name/tags, rename, move to temp location.

    New in dry-run feature:
        preview_mode: If True, simulate operations without moving files
        ai_cache: Optional cache dict for AI results
        cache_lock: Optional threading.Lock for thread-safe cache access
        simulated_paths: Set of paths for collision detection in preview mode
    """
    try:
        if is_already_ai_named(image_path.name):
            extension = image_path.suffix.lower()
            base_name = image_path.stem
            clean_base = base_name[:-5] if base_name.endswith('_PICK') else base_name

            if preview_mode and simulated_paths is not None:
                new_path = get_unique_filename_simulated(clean_base, extension, destination_base, simulated_paths)
                simulated_paths.add(new_path)
                log_callback(f"   [cyan]WOULD MOVE:[/cyan] {image_path.name} → {new_path.name}")
            else:
                new_path = get_unique_filename(clean_base, extension, destination_base)
                # FIXXER v1.0: Hash-verified move
                verify_file_move_with_hash(image_path, new_path, log_callback, generate_sidecar=True)

            if rename_log_path:
                write_rename_log(rename_log_path, image_path.name, new_path.name, destination_base)
            description_for_categorization = clean_base.replace('-', ' ')
            return image_path, True, new_path.name, description_for_categorization

        # Use cache-aware AI naming
        ai_filename, ai_tags = get_ai_name_with_cache(
            image_path, model_name, ai_cache, cache_lock, log_callback
        )
        if not ai_filename or not ai_tags:
            return image_path, False, "Failed to get valid AI JSON response", ""

        description_for_categorization = " ".join(ai_tags)
        clean_name = Path(ai_filename).stem
        extension = image_path.suffix.lower()

        if preview_mode and simulated_paths is not None:
            new_path = get_unique_filename_simulated(clean_name, extension, destination_base, simulated_paths)
            simulated_paths.add(new_path)
            log_callback(f"   [cyan]WOULD MOVE:[/cyan] {image_path.name} → {new_path.name}")
        else:
            new_path = get_unique_filename(clean_name, extension, destination_base)
            # FIXXER v1.0: Hash-verified move
            verify_file_move_with_hash(image_path, new_path, log_callback, generate_sidecar=True)

        if rename_log_path:
            write_rename_log(rename_log_path, image_path.name, new_path.name, destination_base)

        return image_path, True, new_path.name, description_for_categorization
    except Exception as e:
        return image_path, False, str(e), ""

def organize_into_folders(
    processed_files: List[Dict], 
    files_source: Path, 
    destination_base: Path,
    log_callback: Callable[[str], None] = no_op_logger
):
    """Group files into folders based on their descriptions."""
    log_callback("\n[bold]🗂️  Organizing into smart folders...[/bold]")
    
    categories = defaultdict(list)
    for file_info in processed_files:
        filename = file_info['new_name']
        description = file_info['description']
        category = categorize_description(description)
        categories[category].append({
            'filename': filename,
            'description': description
        })
    
    for category, files in categories.items():
        folder_name = category
        folder_path = destination_base / folder_name
        folder_path.mkdir(exist_ok=True)
        log_callback(f"   [green]✓[/green] Creating {folder_name}/ ({len(files)} files)")

        moved_count = 0
        for file_info in files:
            src = files_source / file_info['filename']
            dst = folder_path / file_info['filename']
            if src.exists():
                # FIXXER v1.0: Hash-verified move
                log_callback(f"     Moving {src.name} → {folder_name}/")
                verify_file_move_with_hash(src, dst, log_callback, generate_sidecar=True)
                moved_count += 1
            else:
                log_callback(f"     [red]✗ File not found at expected location:[/red] {src}")
                log_callback(f"       [dim]Looking for: {file_info['filename']}[/dim]")
                log_callback(f"       [dim]In directory: {files_source}[/dim]")

        if moved_count > 0:
            log_callback(f"     [green]✓ Moved {moved_count}/{len(files)} files to {folder_name}/[/green]")

    log_callback(f"\n   [bold]✓ Organized into {len(categories)} folders.[/bold]")

# AI session naming removed - adds too much time for minimal value
# Users can rename folders themselves after workflow completes

# ==============================================================================
# VIII. FEATURE WORKFLOWS (The "Tools")
# ==============================================================================

# --- Simple Sort Workflow (Legacy Mode) ---

def simple_sort_workflow(
    log_callback: Callable[[str], None] = no_op_logger,
    app_config: Optional[Dict[str, Any]] = None,
    stop_event: Optional[threading.Event] = None,
    preview_mode: bool = False,
    ai_cache: Optional[Dict[str, Dict]] = None,
    cache_lock: Optional[threading.Lock] = None
) -> Dict[str, Any]:
    """
    Simple workflow: AI name + organize by keyword into folders.
    No burst detection, no culling - just straightforward naming and sorting.
    Perfect for home users who want a simple "point and organize" experience.

    This is the "legacy mode" from the original CLI photosort.py that just:
    1. AI names all images
    2. Groups them into folders by keyword
    That's it. Powerful for the home user who just needs to organize photos.

    New in dry-run feature:
        preview_mode: If True, simulate operations without moving files
        ai_cache: Optional cache dict for AI results
        cache_lock: Optional threading.Lock for thread-safe cache access
    """
    start_time = datetime.now()

    if app_config is None:
        app_config = load_app_config()

    source_str = app_config.get('last_source_path')
    dest_str = app_config.get('last_destination_path')
    chosen_model = app_config.get('default_model', DEFAULT_MODEL_NAME)

    if not source_str or not dest_str:
        log_callback("[bold red]✗ FATAL: Source or Destination not set in config.[/bold red]")
        return {}

    directory = Path(source_str)
    chosen_destination = Path(dest_str)

    if not directory.is_dir():
        log_callback(f"[bold red]✗ FATAL: Source directory not found:[/bold red] {directory}")
        return {}

    if not preview_mode:
        try:
            chosen_destination.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            log_callback(f"[bold red]✗ FATAL: Could not create destination:[/bold red] {e}")
            return {}

    # Create temp staging area for renamed files
    temp_staging = chosen_destination / "_staging"
    if not preview_mode:
        temp_staging.mkdir(exist_ok=True)

    log_callback(f"\n[bold cyan]📁 Simple Sort: AI Naming + Keyword Folders[/bold cyan]")
    if preview_mode:
        log_callback(f"   [yellow]MODE: DRY RUN (No files will be moved)[/yellow]")
    log_callback(f"   Source:      {directory}")
    log_callback(f"   Destination: {chosen_destination}")
    log_callback(f"   Model:       {chosen_model}")

    # Get all image files (case-insensitive)
    image_files = []
    for ext in SUPPORTED_EXTENSIONS:
        # Try both lowercase and uppercase
        found_lower = list(directory.glob(f"*{ext}"))
        found_upper = list(directory.glob(f"*{ext.upper()}"))
        image_files.extend(found_lower)
        image_files.extend(found_upper)

    if not image_files:
        log_callback("[yellow]No image files found in source directory.[/yellow]")
        return {}

    log_callback(f"\n   Found {len(image_files)} images to process")
    
    # Process images: AI naming
    log_callback("\n[bold]🤖 AI Naming Images...[/bold]")
    processed_files = []
    success_count = 0

    # Track simulated paths to prevent collisions in preview
    simulated_paths = set() if preview_mode else None

    for idx, img in enumerate(image_files, 1):
        if stop_event and stop_event.is_set():
            log_callback("\n[yellow]🛑 Workflow stopped by user.[/yellow]")
            return {}

        log_callback(f"   [{idx}/{len(image_files)}] {img.name}")
        original_path, success, new_name, description = process_single_image(
            img, temp_staging, chosen_model,
            log_callback=log_callback,
            preview_mode=preview_mode,
            ai_cache=ai_cache,
            cache_lock=cache_lock,
            simulated_paths=simulated_paths
        )
        
        if success:
            processed_files.append({
                'original': original_path.name,
                'new_name': new_name,
                'description': description
            })
            success_count += 1
        else:
            log_callback(f"     [red]Failed: {new_name}[/red]")
    
    log_callback(f"\n   [green]✓[/green] Successfully named {success_count}/{len(image_files)} images")
    
    if not processed_files:
        log_callback("[yellow]No files were successfully processed.[/yellow]")
        try:
            temp_staging.rmdir()
        except:
            pass
        return {}
    
    # Organize into keyword folders
    log_callback("\n[bold]📂 Organizing into Keyword Folders...[/bold]")
    if preview_mode:
        # In preview mode, simulate organization based on descriptions
        categories = defaultdict(list)
        for file_info in processed_files:
            cat = categorize_description(file_info['description'])
            categories[cat].append(file_info['new_name'])

        for cat, files in categories.items():
            log_callback(f"   [dim]Preview: Would move {len(files)} files to[/dim] [cyan]{cat}/[/cyan]")
    else:
        organize_into_folders(processed_files, temp_staging, chosen_destination, log_callback)

        # Clean up staging directory
        try:
            temp_staging.rmdir()
        except:
            pass
    
    duration = datetime.now() - start_time
    log_callback(f"\n[bold green]✓ Simple Sort Complete![/bold green]")
    log_callback(f"   Duration: {format_duration(duration)}")
    log_callback(f"   {success_count} images organized into folders")
    
    return {
        'total_images': len(image_files),
        'success_count': success_count,
        'duration': str(duration)
    }


# --- Auto Workflow ---

def auto_workflow(
    log_callback: Callable[[str], None] = no_op_logger,
    app_config: Optional[Dict[str, Any]] = None,
    tracker: Optional[StatsTracker] = None,
    stop_event: Optional[threading.Event] = None,
    preview_mode: bool = False,
    ai_cache: Optional[Dict[str, Dict]] = None,
    cache_lock: Optional[threading.Lock] = None
) -> Dict[str, Any]:
    """
    (V9.3) Complete automated workflow: Stack → Cull → AI-Name → Archive.

    New in dry-run feature:
        preview_mode: If True, simulate operations without moving files
        ai_cache: Optional cache dict for AI results (model-aware)
        cache_lock: Optional threading.Lock for thread-safe cache access
    """

    # --- PREVIEW MODE BANNER ---
    if preview_mode:
        log_callback("\n[bold yellow]═══ DRY RUN MODE ═══[/bold yellow]")
        log_callback("[dim]No files will be moved. Preview only.[/dim]\n")

    # --- 1. CONFIGURATION ---
    if app_config is None:
        app_config = load_app_config()

    # The TUI now handles source/dest selection. Get them from config.
    source_str = app_config.get('last_source_path')
    dest_str = app_config.get('last_destination_path')

    if not source_str or not dest_str:
        log_callback("[bold red]✗ FATAL: Source or Destination not set in config.[/bold red]")
        return {}

    directory = Path(source_str)
    chosen_destination = Path(dest_str)

    if not directory.is_dir():
        log_callback(f"[bold red]✗ FATAL: Source directory not found:[/bold red] {directory}")
        return {}

    if not preview_mode:
        try:
            chosen_destination.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            log_callback(f"[bold red]✗ FATAL: Could not create destination:[/bold red] {e}")
            return {}

    chosen_model = app_config['default_model']
    log_callback(f"   Source:      {directory}")
    log_callback(f"   Destination: {chosen_destination}")
    log_callback(f"   Model:       {chosen_model}")
    if preview_mode:
        log_callback(f"   [yellow]Mode:        DRY RUN (preview only)[/yellow]")
    
    # Check for critical libs
    if not V5_LIBS_AVAILABLE or not V6_CULL_LIBS_AVAILABLE or not V6_4_EXIF_LIBS_AVAILABLE:
        log_callback("[bold red]✗ FATAL: Missing required libraries.[/bold red]")
        log_callback("   Please run: pip install imagehash opencv-python numpy exifread")
        return {}
    
    # Session tracking removed - was mock/stub code

    # Start HUD timer
    if tracker:
        tracker.start_timer()
    
    # --- 2. STATS PREVIEW ---
    log_callback("\n[bold]Step 2/5: Analyzing session (read-only)...[/bold]")
    try:
        if stop_event and stop_event.is_set(): return {}
        show_exif_insights(log_callback, app_config, simulated=True, directory_override=directory, stop_event=stop_event)
    except Exception as e:
        log_callback(f"     [yellow]Could not run EXIF analysis: {e}[/yellow]")

    # --- 3. GROUP BURSTS ---
    if stop_event and stop_event.is_set(): return {}
    log_callback("\n[bold]Step 3/5: Stacking burst shots (with AI naming)...[/bold]")
    if preview_mode:
        log_callback("[dim yellow]PREVIEW MODE: No files will be moved[/dim yellow]")
    # v1.1: Auto workflow ALWAYS does AI naming, regardless of burst_auto_name config
    auto_config = app_config.copy()
    auto_config['burst_auto_name'] = True

    # [FIX] Capture returned picks (dry-run feature)
    burst_picks = group_bursts_in_directory(log_callback, auto_config, directory_override=directory, tracker=tracker, stop_event=stop_event, preview_mode=preview_mode, ai_cache=ai_cache, cache_lock=cache_lock)

    # --- 4. CULL SINGLES ---
    if stop_event and stop_event.is_set(): return {}
    log_callback("\n[bold]Step 4/5: Culling single shots...[/bold]")
    if preview_mode:
        log_callback("[dim yellow]PREVIEW MODE: No files will be moved[/dim yellow]")

    # [FIX] Capture returned Tier A files (dry-run feature)
    tier_a_files = cull_images_in_directory(log_callback, app_config, directory_override=directory, tracker=tracker, stop_event=stop_event, preview_mode=preview_mode)
    tier_a_dir = directory / TIER_A_FOLDER
    
    # --- 5. FIND & ARCHIVE HEROES ---
    log_callback("\n[bold]Step 5/5: Finding and archiving 'hero' files...[/bold]")

    hero_files = []

    if preview_mode:
        # [CRITICAL FIX] In Dry Run, folders don't exist. Use the returned lists!
        # Combine lists and remove duplicates (if any file appeared in both)
        if burst_picks:
            hero_files.extend(burst_picks)
        if tier_a_files:
            # Only add Tier A files that aren't already in burst picks
            picks_set = set(burst_picks) if burst_picks else set()
            for f in tier_a_files:
                if f not in picks_set:
                    hero_files.append(f)
        log_callback(f"   [dim]Preview: Found {len(hero_files)} hero files from returned lists[/dim]")
    else:
        # [REAL RUN] Use existing filesystem scanning logic
        if tier_a_dir.is_dir():
            for f in tier_a_dir.iterdir():
                if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS:
                    hero_files.append(f)

        burst_parent = directory / "_Bursts"
        burst_folders = []
        if burst_parent.exists() and burst_parent.is_dir():
            burst_folders = [f for f in burst_parent.iterdir() if f.is_dir()]

        for burst_folder in burst_folders:
            if burst_folder.is_dir():
                for f in burst_folder.iterdir():
                    if f.is_file() and (f.name.startswith(BEST_PICK_PREFIX) or is_already_ai_named(f.name)):
                        hero_files.append(f)

    if not hero_files:
        log_callback(f"\n   No '{TIER_A_FOLDER}' or '_PICK_' files found. Nothing to archive.")
        log_callback("[bold green]✓ Auto workflow complete (no heroes found).[/bold green]")
        return {}

    already_named = [f for f in hero_files if is_already_ai_named(f.name)]
    needs_naming = [f for f in hero_files if not is_already_ai_named(f.name)]
    log_callback(f"   Found {len(hero_files)} 'hero' files total:")
    if already_named:
        log_callback(f"     • {len(already_named)} already AI-named (from burst stacking)")
    log_callback(f"     • {len(needs_naming)} to process")
    
    # Update HUD: Heroes count
    if tracker:
        tracker.update('heroes', len(hero_files))
    
    results = {"success": [], "failed": []}

    # [LOGIC PATCH] Only create the physical rename log if this is a REAL run
    if preview_mode:
        rename_log_path = None
    else:
        rename_log_path = chosen_destination / f"_ai_rename_log_{SESSION_TIMESTAMP}.txt"
        initialize_rename_log(rename_log_path)

    # Preview mode: track simulated paths for collision detection
    simulated_paths = set() if preview_mode else None

    log_callback(f"\n   [grey]Archiving {len(hero_files)} files...[/grey]")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_file = {
            executor.submit(
                process_single_image,
                img_path,
                chosen_destination,
                chosen_model,
                rename_log_path,
                log_callback,
                preview_mode,
                ai_cache,
                cache_lock,
                simulated_paths
            ): img_path
            for img_path in hero_files
        }

        for i, future in enumerate(as_completed(future_to_file)):
            if stop_event and stop_event.is_set():
                log_callback("\n[yellow]🛑 Workflow stopped by user.[/yellow]")
                executor.shutdown(wait=False, cancel_futures=True)
                return {}

            log_callback(f"   [grey]Processing item {i+1}/{len(hero_files)}...[/grey]")
            original, success, message, description = future.result()
            if success:
                results["success"].append({
                    "original": original.name,
                    "new_name": message,
                    "description": description
                })
            else:
                results["failed"].append((original.name, message))
                log_callback(f"   [red]✗ {original.name}: {message}[/red]")
    
    log_callback(f"\n[green]✓ Successfully archived: {len(results['success'])}[/green]")
    log_callback(f"[red]✗ Failed to archive: {len(results['failed'])}[/red]")
    
    # Update HUD: Archived count
    if tracker:
        tracker.update('archived', len(results['success']))
    
    summary = {
        "archived": len(results['success']),
        "failed": len(results['failed']),
    }

    if results["success"]:
        categories = {}
        for item in results["success"]:
            cat = categorize_description(item["description"])
            categories[cat] = categories.get(cat, 0) + 1

        # In preview mode, skip file organization (files not actually moved)
        if preview_mode:
            log_callback(f"\n[dim]Preview: Would organize into {len(categories)} categories:[/dim]")
            for cat, count in categories.items():
                log_callback(f"   [dim]• {cat}: {count} files[/dim]")
            summary["categories"] = len(categories)
            summary["preview_categories"] = categories
        else:
            # Use simple date-based folder naming
            dated_folder = f"{SESSION_DATE}_Session"
            final_destination = chosen_destination / dated_folder
            if not preview_mode:
                final_destination.mkdir(parents=True, exist_ok=True)

            organize_into_folders(results["success"], chosen_destination, final_destination, log_callback)
            summary["final_destination"] = str(final_destination.name)
            summary["categories"] = len(categories)

    # Stop HUD timer
    if tracker:
        tracker.stop_timer()

    # Different completion messages for preview vs real mode
    if preview_mode:
        log_callback("\n[bold green]✓ Dry Run Complete[/bold green]")
        log_callback(f"   [dim]Preview generated {len(results['success'])} AI names[/dim]")
        if ai_cache:
            log_callback(f"   [cyan]⚡ Cached {len(ai_cache)} AI results for instant execution[/cyan]")
    else:
        log_callback("\n[bold green]🚀 AUTO WORKFLOW COMPLETE[/bold green]")
        log_callback(f"   Your 'hero' photos are now in: {chosen_destination}")
        if rename_log_path:
            log_callback(f"   Rename log saved: {rename_log_path.name}")

    return summary

# --- Burst Workflow ---

def group_bursts_in_directory(
    log_callback: Callable[[str], None] = no_op_logger,
    app_config: Optional[Dict[str, Any]] = None,
    simulated: bool = False,
    directory_override: Optional[Path] = None,
    tracker: Optional[StatsTracker] = None,
    stop_event: Optional[threading.Event] = None,
    preview_mode: bool = False,
    ai_cache: Optional[Dict[str, Dict]] = None,
    cache_lock: Optional[threading.Lock] = None
) -> List[Path]:
    """
    (V1.2) Finds and stacks burst groups, optionally AI-naming the best pick.

    Returns:
        List of 'Pick' files (best images from each burst group)

    By default (burst_auto_name=false), uses fast numeric naming.
    Set burst_auto_name=true in config to enable AI naming (slower).
    Auto workflow always uses AI naming regardless of this setting.

    New in dry-run feature:
        preview_mode: If True, simulate operations without moving files
    """
    
    if app_config is None: app_config = load_app_config()
    
    # Use override dir (from auto) or config dir
    if directory_override:
        directory = directory_override
    elif app_config.get('last_source_path'):
        directory = Path(app_config['last_source_path'])
    else:
        log_callback("[red]✗ No source directory specified.[/red]")
        return
        
    if not directory.is_dir():
        log_callback(f"[red]✗ Source directory not found: {directory}[/red]")
        return

    if not V5_LIBS_AVAILABLE or not V6_CULL_LIBS_AVAILABLE:
        log_callback("[bold red]✗ FATAL: Missing required libraries.[/bold red]")
        log_callback("   Please run: pip install imagehash opencv-python numpy")
        return

    log_callback(f"[grey]Scanning for bursts in: {directory.name}[/grey]")
    burst_threshold = app_config['burst_threshold']
    
    image_files = [
        f for f in directory.iterdir() 
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    if len(image_files) < 2:
        log_callback("   Not enough images to compare.")
        return

    all_hashes = {}
    log_callback(f"   [grey]Calculating {len(image_files)} visual fingerprints...[/grey]")
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_path = {executor.submit(get_image_hash, path, log_callback): path for path in image_files}
        for i, future in enumerate(as_completed(future_to_path)):
            if stop_event and stop_event.is_set():
                log_callback("\n[yellow]🛑 Workflow stopped by user.[/yellow]")
                executor.shutdown(wait=False, cancel_futures=True)
                return
                
            log_callback(f"   [grey]Hashing image {i+1}/{len(image_files)}...[/grey]",)
            path, img_hash = future.result()
            if img_hash:
                all_hashes[path] = img_hash

    log_callback("   [grey]Comparing fingerprints to find burst groups...[/grey]")
    visited_paths = set()
    all_burst_groups = []
    sorted_paths = sorted(all_hashes.keys(), key=lambda p: p.name)
    
    for path in sorted_paths:
        if path in visited_paths: continue
        current_group = [path]
        visited_paths.add(path)
        for other_path in sorted_paths:
            if other_path in visited_paths: continue
            hash1 = all_hashes.get(path)
            hash2 = all_hashes.get(other_path)
            if hash1 and hash2:
                distance = hash1 - hash2
                if distance <= burst_threshold:
                    current_group.append(other_path)
                    visited_paths.add(other_path)
        if len(current_group) > 1:
            all_burst_groups.append(current_group)

    if not all_burst_groups:
        log_callback("   No burst groups found. All images are unique!")
        return []
        
    log_callback(f"   [green]✓ Found {len(all_burst_groups)} burst groups.[/green] Analyzing for best pick...")

    # Update HUD: Bursts count
    if tracker:
        tracker.update('bursts', len(all_burst_groups))

    # Track burst picks for return value (dry-run feature)
    burst_picks: List[Path] = []

    best_picks: Dict[int, Tuple[Path, float]] = {}
    for i, group in enumerate(all_burst_groups):
        best_sharpness = -1.0
        best_file = None
        for file_path in group:
            image_bytes = get_image_bytes_for_analysis(file_path, log_callback)
            if image_bytes:
                scores = analyze_image_quality(image_bytes)
                sharpness = scores.get('sharpness', 0.0)
                if sharpness > best_sharpness:
                    best_sharpness = sharpness
                    best_file = file_path
        if best_file:
            best_picks[i] = (best_file, best_sharpness)

    use_parent_folder = app_config.get('burst_parent_folder', True)
    bursts_parent = directory / "_Bursts" if use_parent_folder else directory
    if use_parent_folder:
        log_callback(f"   Organizing burst groups into: {bursts_parent.name}/")
        if not preview_mode:
            bursts_parent.mkdir(exist_ok=True)
    
    # v1.1: Check if AI naming is enabled for burst workflow
    burst_auto_name = app_config.get('burst_auto_name', False)

    if burst_auto_name:
        # [LOGIC PATCH] Only create the physical rename log if this is a REAL run
        if preview_mode:
            rename_log_path = None
        else:
            rename_log_path = directory / f"_ai_rename_log_{SESSION_TIMESTAMP}.txt"
            initialize_rename_log(rename_log_path)
        ai_model = app_config.get('default_model', DEFAULT_MODEL_NAME)
        log_callback("   [grey]AI naming enabled for bursts...[/grey]")

    for i, group in enumerate(all_burst_groups):
        winner_data = best_picks.get(i)
        sample_image = winner_data[0] if winner_data else group[0]

        # Add the winner to our return list (for dry-run feature)
        burst_picks.append(sample_image)

        # Only run AI naming if enabled
        if burst_auto_name:
            log_callback(f"   [grey]Burst {i+1}/{len(all_burst_groups)}: Naming...[/grey]")

            # Use cache-aware AI naming (for dry-run feature)
            ai_filename, ai_tags = get_ai_name_with_cache(
                sample_image, ai_model, ai_cache, cache_lock, log_callback
            )

            if ai_filename and ai_tags:
                base_name = Path(ai_filename).stem  # Extract base name without extension
                folder_name = f"{base_name}_burst"
                log_callback(f"     [green]✓ AI named:[/green] {base_name}")
            else:
                base_name = f"burst-{i+1:03d}"
                folder_name = base_name
                log_callback(f"     [yellow]⚠️ AI naming failed, using:[/yellow] {base_name}")
        else:
            # Fast mode: Just use numeric naming
            base_name = f"burst-{i+1:03d}"
            folder_name = base_name
            log_callback(f"   [grey]Burst {i+1}/{len(all_burst_groups)}: {folder_name} ({len(group)} files)[/grey]")
        
        folder_path = bursts_parent / folder_name
        if folder_path.exists():
            counter = 2
            original_name = folder_name
            while folder_path.exists():
                folder_name = f"{original_name}-{counter}"
                folder_path = bursts_parent / folder_name
                counter += 1
        
        log_callback(f"     [grey]📁 Moving {len(group)} files to {folder_path.relative_to(directory)}/...[/grey]")
        if not preview_mode:
            folder_path.mkdir(parents=True, exist_ok=True)
        
        alternate_counter = 1
        for file_path in group:
            extension = file_path.suffix
            if winner_data and file_path == winner_data[0]:
                new_name = f"{base_name}_PICK{extension}"
            else:
                new_name = f"{base_name}_{alternate_counter:03d}{extension}"
                alternate_counter += 1
            
            new_file_path = folder_path / new_name
            try:
                if preview_mode:
                    log_callback(f"     [cyan]WOULD MOVE:[/cyan] {file_path.name} → {folder_path.name}/{new_name}")
                else:
                    # FIXXER v1.0: Hash-verified move
                    verify_file_move_with_hash(file_path, new_file_path, log_callback, generate_sidecar=True)
                if burst_auto_name:
                    write_rename_log(rename_log_path, file_path.name, new_name, folder_path)
            except Exception as e:
                log_callback(f"     [red]FAILED to move {file_path.name}: {e}[/red]")

    if burst_auto_name and rename_log_path:
        log_callback(f"   Rename log saved: {rename_log_path.name}")

    return burst_picks  # Return the winners for dry-run feature

# --- Cull Workflow ---

def cull_images_in_directory(
    log_callback: Callable[[str], None] = no_op_logger,
    app_config: Optional[Dict[str, Any]] = None,
    simulated: bool = False,
    directory_override: Optional[Path] = None,
    tracker: Optional[StatsTracker] = None,
    stop_event: Optional[threading.Event] = None,
    preview_mode: bool = False
) -> List[Path]:
    """
    (V9.4) Finds and groups images by technical quality using Tier A/B/C naming.

    Returns:
        List of 'Tier A' files (best quality images)

    New in dry-run feature:
        preview_mode: If True, simulate operations without moving files
    """
    
    if app_config is None: app_config = load_app_config()
    
    if directory_override:
        directory = directory_override
    elif app_config.get('last_source_path'):
        directory = Path(app_config['last_source_path'])
    else:
        log_callback("[red]✗ No source directory specified.[/red]")
        return
        
    if not directory.is_dir():
        log_callback(f"[red]✗ Source directory not found: {directory}[/red]")
        return

    if not V6_CULL_LIBS_AVAILABLE:
        log_callback("[bold red]✗ FATAL: Missing required libraries.[/bold red]")
        log_callback("   Please run: pip install opencv-python numpy")
        return
    
    log_callback(f"[grey]Analyzing technical quality in: {directory.name}[/grey]")

    image_files = [
        f for f in directory.iterdir() 
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    if not image_files:
        log_callback("     No supported images to analyze.")
        return

    all_scores = {}
    log_callback(f"   [grey]Analyzing sharpness/exposure for {len(image_files)} images...[/grey]")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_path = {
            executor.submit(process_image_for_culling, path, log_callback): path 
            for path in image_files
        }
        for i, future in enumerate(as_completed(future_to_path)):
            if stop_event and stop_event.is_set():
                log_callback("\n[yellow]🛑 Workflow stopped by user.[/yellow]")
                executor.shutdown(wait=False, cancel_futures=True)
                return

            log_callback(f"   [grey]Analyzing image {i+1}/{len(image_files)}...[/grey]")
            path, scores = future.result()
            if scores:
                all_scores[path] = scores
    
    log_callback("   [grey]Triaging images into quality tiers...[/grey]")
    th = app_config['cull_thresholds']
    tiers = {"Tier_A": [], "Tier_B": [], "Tier_C": []}

    for path, scores in all_scores.items():
        sharp = scores['sharpness']
        blacks = scores['blacks_pct']
        whites = scores['whites_pct']
        is_exposure_bad = (blacks > th['exposure_dud_pct']) or (whites > th['exposure_dud_pct'])
        is_exposure_good = (blacks < th['exposure_good_pct']) and (whites < th['exposure_good_pct'])
        is_sharp_bad = sharp < th['sharpness_dud']
        is_sharp_good = sharp > th['sharpness_good']

        tier = "Tier_B"
        if is_sharp_bad or is_exposure_bad:
            tier = "Tier_C"
        elif is_sharp_good and is_exposure_good:
            tier = "Tier_A"
        tiers[tier].append(path)

    log_callback(f"   [green]Found {len(tiers['Tier_A'])} Tier A[/green], [yellow]{len(tiers['Tier_B'])} Tier B[/yellow], [red]{len(tiers['Tier_C'])} Tier C[/red].")
    
    # Update HUD: Tier A/B/C counts
    if tracker:
        tracker.update('tier_a', len(tiers['Tier_A']))
        tracker.update('tier_b', len(tiers['Tier_B']))
        tracker.update('tier_c', len(tiers['Tier_C']))
    
    folder_map = {
        "Tier_A": directory / TIER_A_FOLDER,
        "Tier_B": directory / TIER_B_FOLDER,
        "Tier_C": directory / TIER_C_FOLDER
    }
    
    for tier, paths in tiers.items():
        if not paths: continue
        folder_path = folder_map[tier]
        log_callback(f"   [grey]Moving {len(paths)} files to {folder_path.name}/...[/grey]")
        if not preview_mode:
            folder_path.mkdir(exist_ok=True)
            
        for file_path in paths:
            new_file_path = folder_path / file_path.name
            try:
                if preview_mode:
                    log_callback(f"     [cyan]WOULD MOVE:[/cyan] {file_path.name} → {folder_path.name}/{file_path.name}")
                else:
                    # FIXXER v1.0: Hash-verified move
                    verify_file_move_with_hash(file_path, new_file_path, log_callback, generate_sidecar=True)
            except Exception as e:
                log_callback(f"     [red]FAILED to move {file_path.name}: {e}[/red]")

    log_callback("   Culling complete!")

    return tiers["Tier_A"]  # Return Tier A files for dry-run feature

def process_image_for_culling(image_path: Path, log_callback: Callable[[str], None] = no_op_logger) -> Tuple[Path, Optional[Dict[str, float]]]:
    """Thread-pool worker: Gets bytes and runs analysis engine"""
    image_bytes = get_image_bytes_for_analysis(image_path, log_callback)
    if not image_bytes:
        return image_path, None
    scores = analyze_image_quality(image_bytes)
    return image_path, scores

# --- Stats Workflow ---

def show_exif_insights(
    log_callback: Callable[[str], None] = no_op_logger,
    app_config: Optional[Dict[str, Any]] = None,
    simulated: bool = False,
    directory_override: Optional[Path] = None,
    stop_event: Optional[threading.Event] = None
) -> None:
    """(V6.4) Scans images, aggregates EXIF data, prints summary"""
    
    if app_config is None: app_config = load_app_config()
    
    if directory_override:
        directory = directory_override
    elif app_config.get('last_source_path'):
        directory = Path(app_config['last_source_path'])
    else:
        log_callback("[red]✗ No source directory specified.[/red]")
        return
        
    if not directory.is_dir():
        log_callback(f"[red]✗ Source directory not found: {directory}[/red]")
        return
        
    if not V6_4_EXIF_LIBS_AVAILABLE:
        log_callback("[bold red]✗ FATAL: Missing required library.[/bold red]")
        log_callback("   Please run: pip install exifread")
        return
        
    log_callback(f"[grey]Scanning EXIF data in: {directory.name}[/grey]")
    
    image_files = [
        f for f in directory.iterdir() 
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    if not image_files:
        log_callback("     No supported images to analyze.")
        return

    all_stats = []
    log_callback(f"   [grey]Reading EXIF data from {len(image_files)} files...[/grey]")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_path = {
            executor.submit(analyze_single_exif, path): path 
            for path in image_files
        }
        for i, future in enumerate(as_completed(future_to_path)):
            if stop_event and stop_event.is_set():
                log_callback("\n[yellow]🛑 Workflow stopped by user.[/yellow]")
                executor.shutdown(wait=False, cancel_futures=True)
                return

            if not simulated:
                log_callback(f"   [grey]Scanning image {i+1}/{len(image_files)}...[/grey]")
            result_dict = future.result()
            if result_dict:
                all_stats.append(result_dict)

    if not all_stats:
        log_callback(f"   [yellow]No EXIF data found in {len(image_files)} scanned images.[/yellow]")
        return

    log_callback("   [grey]Aggregating statistics...[/grey]")
    
    timestamps = sorted([s['timestamp'] for s in all_stats])
    start_time = timestamps[0]
    end_time = timestamps[-1]
    duration_str = format_duration(end_time - start_time)

    LIGHTING_TABLE = {
        (0, 4):   "Night", (5, 7):   "Golden Hour (AM)", (8, 10):  "Morning",
        (11, 13): "Midday", (14, 16): "Afternoon", (17, 18): "Golden Hour (PM)",
        (19, 21): "Dusk", (22, 23): "Night",
    }
    lighting_buckets = defaultdict(int)
    camera_counter = Counter()
    focal_len_counter = Counter()
    aperture_counter = Counter()

    for stats in all_stats:
        hour = stats['timestamp'].hour
        for (start, end), name in LIGHTING_TABLE.items():
            if start <= hour <= end:
                lighting_buckets[name] += 1
                break
        camera_counter[stats['camera']] += 1
        focal_len_counter[stats['focal_length']] += 1
        aperture_counter[stats['aperture']] += 1

    # --- Display Results to Log ---
    log_callback("\n[bold]📖 Session Story:[/bold]")
    log_callback(f"   Started:     {start_time.strftime('%a, %b %d at %I:%M %p')}")
    log_callback(f"   Ended:       {end_time.strftime('%a, %b %d at %I:%M %p')}")
    log_callback(f"   Duration:    [bold]{duration_str}[/bold]")
    log_callback(f"   Total Shots: [bold]{len(image_files)}[/bold] ({len(all_stats)} with EXIF)")

    log_callback("\n[bold]☀️ Lighting Conditions:[/bold]")
    bar_lines = generate_bar_chart(lighting_buckets, bar_width=30)
    for line in bar_lines:
        log_callback(line)
    
    log_callback("\n[bold]🎨 Creative Habits (Top 3):[/bold]")
    log_callback("    [cyan]Cameras:[/cyan]")
    for cam, count in camera_counter.most_common(3):
        log_callback(f"      {cam}: [bold]{count} shots[/bold]")
    log_callback("    [cyan]Focal Lengths:[/cyan]")
    for focal, count in focal_len_counter.most_common(3):
        log_callback(f"      {focal}: [bold]{count} shots[/bold]")
    log_callback("    [cyan]Apertures:[/cyan]")
    for ap, count in aperture_counter.most_common(3):
        log_callback(f"      {ap}: [bold]{count} shots[/bold]")
