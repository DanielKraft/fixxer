#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FIXXER Configuration
Application settings, defaults, and configuration file management.
"""

from __future__ import annotations

import os
import configparser
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# ==============================================================================
# CONFIGURATION CONSTANTS
# ==============================================================================

# --- Ollama / AI Settings ---
OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_API_PROVIDER = "ollama"
DEFAULT_MODEL_NAME = "qwen2.5vl:3b"
DEFAULT_CRITIQUE_MODEL = "qwen2.5vl:3b"

# --- Path Settings ---
DEFAULT_DESTINATION_BASE = Path.home() / "Pictures" / "FIXXER_Output"
CONFIG_FILE_PATH = Path.home() / ".fixxer.conf"

# --- File Processing Settings ---
SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png'}
RAW_SUPPORT = False  # Updated dynamically by check_rawpy()

# --- Workflow Settings ---
MAX_WORKERS = 5
INGEST_TIMEOUT = 120
CRITIQUE_TIMEOUT = 120

# --- Algorithm Defaults ---
DEFAULT_CULL_ALGORITHM = 'legacy'
DEFAULT_BURST_ALGORITHM = 'legacy'

DEFAULT_CULL_THRESHOLDS = {
    'sharpness_good': 40.0,
    'sharpness_dud': 15.0,
    'exposure_dud_pct': 0.20,
    'exposure_good_pct': 0.05
}
DEFAULT_BURST_THRESHOLD = 8

# --- Session Metadata ---
SESSION_DATE = datetime.now().strftime("%Y-%m-%d")
SESSION_TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H%M")

# --- Folder Names ---
BEST_PICK_PREFIX = "_PICK_"
PREP_FOLDER_NAME = "_ReadyForLightroom"
TIER_A_FOLDER = "_Tier_A"
TIER_B_FOLDER = "_Tier_B"
TIER_C_FOLDER = "_Tier_C"

# --- AI Classification Keywords ---
GROUP_KEYWORDS = {
    "Architecture": ["building", "architecture"],
    "Street-Scenes": ["street", "road", "city"],
    "People": ["people", "person", "man", "woman"],
    "Nature": ["tree", "forest", "mountain", "lake"],
    "Transportation": ["car", "bus", "train", "vehicle"],
    "Signs-Text": ["sign", "text", "billboard"],
    "Food-Dining": ["food", "restaurant", "cafe"],
    "Animals": ["dog", "cat", "bird", "animal"],
    "Interior": ["interior", "room", "inside"],
}


# ==============================================================================
# CONFIGURATION FILE MANAGEMENT
# ==============================================================================

def load_app_config() -> Dict[str, Any]:
    """
    Load settings from ~/.fixxer.conf with fallback defaults.

    Returns:
        Dictionary containing all application settings
    """
    parser = configparser.ConfigParser()
    config_loaded = False

    if CONFIG_FILE_PATH.exists():
        try:
            parser.read(CONFIG_FILE_PATH)
            config_loaded = True
        except configparser.Error:
            pass  # Will use fallbacks

    config = {}

    config['config_file_found'] = config_loaded
    config['config_file_path'] = str(CONFIG_FILE_PATH)

    config['default_destination'] = Path(parser.get(
        'ingest', 'default_destination',
        fallback=str(DEFAULT_DESTINATION_BASE)
    )).expanduser()

    config['default_model'] = parser.get(
        'ingest', 'default_model',
        fallback=DEFAULT_MODEL_NAME
    )

    config['cull_thresholds'] = {
        'sharpness_good': parser.getfloat('cull', 'sharpness_good', fallback=DEFAULT_CULL_THRESHOLDS['sharpness_good']),
        'sharpness_dud': parser.getfloat('cull', 'sharpness_dud', fallback=DEFAULT_CULL_THRESHOLDS['sharpness_dud']),
        'exposure_dud_pct': parser.getfloat('cull', 'exposure_dud_pct', fallback=DEFAULT_CULL_THRESHOLDS['exposure_dud_pct']),
        'exposure_good_pct': parser.getfloat('cull', 'exposure_good_pct', fallback=DEFAULT_CULL_THRESHOLDS['exposure_good_pct']),
    }

    config['cull_algorithm'] = parser.get(
        'cull', 'cull_algorithm',
        fallback=DEFAULT_CULL_ALGORITHM
    )

    config['burst_threshold'] = parser.getint(
        'burst', 'similarity_threshold',
        fallback=DEFAULT_BURST_THRESHOLD
    )

    config['burst_algorithm'] = parser.get(
        'burst', 'burst_algorithm',
        fallback=DEFAULT_BURST_ALGORITHM
    )

    config['burst_auto_name'] = parser.getboolean(
        'burst', 'burst_auto_name',
        fallback=False
    )

    config['critique_model'] = parser.get(
        'critique', 'default_model',
        fallback=DEFAULT_CRITIQUE_MODEL
    )

    config['last_source_path'] = parser.get(
        'behavior', 'last_source_path', fallback=None
    )

    config['last_destination_path'] = parser.get(
        'behavior', 'last_destination_path', fallback=None
    )

    config['burst_parent_folder'] = parser.getboolean(
        'folders', 'burst_parent_folder', fallback=True
    )

    config['pro_mode'] = parser.getboolean(
        'behavior', 'pro_mode', fallback=False
    )

    # --- API / Provider Settings ---
    # Priority: Env Vars > Config File > Defaults
    config['api_provider'] = os.environ.get(
        'FIXXER_API_PROVIDER',
        parser.get('api', 'provider', fallback=DEFAULT_API_PROVIDER)
    )
    
    config['api_endpoint'] = os.environ.get(
        'FIXXER_API_ENDPOINT',
        parser.get('api', 'endpoint', fallback=OLLAMA_URL)
    )

    return config


def save_app_config(config: Dict[str, Any]) -> bool:
    """
    Save settings back to ~/.fixxer.conf.

    Only saves paths and model settings that are commonly changed during TUI use.

    Args:
        config: Dictionary with settings to save

    Returns:
        True on success, False on error
    """
    parser = configparser.ConfigParser()

    # Load existing config first to preserve other settings
    if CONFIG_FILE_PATH.exists():
        try:
            parser.read(CONFIG_FILE_PATH)
        except configparser.Error:
            pass

    # Ensure sections exist
    if not parser.has_section('behavior'):
        parser.add_section('behavior')
    if not parser.has_section('ingest'):
        parser.add_section('ingest')

    # Save the key settings
    if 'last_source_path' in config and config['last_source_path']:
        parser.set('behavior', 'last_source_path', str(config['last_source_path']))

    if 'last_destination_path' in config and config['last_destination_path']:
        parser.set('behavior', 'last_destination_path', str(config['last_destination_path']))

    if 'default_model' in config and config['default_model']:
        parser.set('ingest', 'default_model', str(config['default_model']))

    if 'pro_mode' in config:
        parser.set('behavior', 'pro_mode', 'true' if config['pro_mode'] else 'false')

    try:
        with open(CONFIG_FILE_PATH, 'w') as f:
            parser.write(f)
        return True
    except Exception:
        return False
