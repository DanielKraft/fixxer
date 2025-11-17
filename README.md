# 📷 PhotoSort v12.1

**AI-Powered Photography Workflow Automation**

PhotoSort is a professional-grade tool that automates the tedious parts of photography post-processing using local AI inference. Built for photographers who shoot hundreds of photos per session and need smart, reliable automation.

> Created by Nick (∞vision crew)  
> Engineered with Claude (Anthropic) + Gemini (Google)

---

## 🆕 What's New in v12.1

- **Full TUI Interface** - Terminal User Interface with mouse support, file browser, and real-time logs
- **Engine Detection** - BRISQUE and CLIP availability checks at startup
- **Animated Progress** - Block spinner with rotating motivational phrases
- **Persistent Config** - Settings auto-save to `~/.photosort.conf`
- **RAW Support** - Enhanced dcraw integration for RW2, ARW, CR2, NEF, DNG, and more
- **AI Session Naming** - Vision-based folder naming from actual image content

---

## 📦 What's Included

### Core Files
- `photosort_tui_styled.py` - **NEW** Terminal User Interface (v12.1)
- `photosort_visioncrew.css` - **NEW** Theme stylesheet
- `photosort_engine.py` - Backend processing engine (v10.7)
- `photosort.py` - Original CLI version (v9.3)
- `phrases.py` - Rotating progress messages

### Documentation
- `README.md` - This file
- `README_TUI.md` - TUI-specific documentation
- `requirements.txt` - Python dependencies

### Configuration
- `.photosort.conf` - User configuration (auto-generated)

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Clone the repo
git clone https://github.com/yourusername/photosort.git
cd photosort

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # macOS/Linux

# Install Python packages
pip install -r requirements.txt
```

### 2. Install System Tools

**macOS (Homebrew):**
```bash
brew install exiftool dcraw
```

**Ubuntu/Debian:**
```bash
sudo apt-get install exiftool dcraw
```

**Fedora:**
```bash
sudo dnf install perl-Image-ExifTool dcraw
```

### 3. Install Ollama (for AI features)

```bash
# Download from https://ollama.com/download
# Then pull the recommended model:
ollama pull qwen2.5vl:3b
```

### 4. Run PhotoSort

**TUI (Recommended):**
```bash
python photosort_tui_styled.py
```

**CLI (Original):**
```bash
python photosort.py
```

---

## 🎮 TUI Interface

The new Terminal User Interface provides:

- **Visual file browser** - Navigate with mouse or keyboard
- **Real-time log panel** - Watch operations as they happen
- **Status dashboard** - Current settings and workflow state
- **Button bar** - Quick access to all workflows

```
┌─────────────────────────────────────────────────────────────┐
│ VISION CREW                                                 │
├─────────────────────────────────────────────────────────────┤
│ Source Browser          │ Status: Ready                     │
│ [File Tree]             │ [■ □ □] Processing...             │
│                         │ [Log Output]                      │
├─────────────────────────────────────────────────────────────┤
│ [Auto] [Bursts] [Cull] [Stats] [Critique] [Source] [Dest]  │
└─────────────────────────────────────────────────────────────┘
```

See `README_TUI.md` for detailed TUI documentation.

---

## ⚙️ Features

### Auto Workflow
Complete pipeline: Burst Detection → Quality Culling → AI Renaming → Smart Organization

### Burst Detection
Groups similar consecutive shots using:
- **CLIP Engine** - Semantic similarity (recommended)
- **pHash Fallback** - Perceptual hashing

### Quality Culling
Sorts images into quality tiers:
- **Tier A** - Hero shots (sharp, well-exposed)
- **Tier B** - Usable images
- **Tier C** - Review needed

Uses:
- **BRISQUE** - No-reference image quality assessment
- **Laplacian Variance** - Sharpness detection (fallback)

### AI-Powered Features
- **Smart Renaming** - Descriptive filenames from image content
- **Session Naming** - Evocative folder names based on visual themes
- **Image Critique** - Professional feedback on composition and technique

---

## 📁 Directory Structure

After running Auto workflow:

```
~/Pictures/
└── 2025-11-16_Meridian/          # AI-named session folder
    ├── Architecture/              # Smart category folders
    │   ├── urban-skyline-sunset.RW2
    │   └── modern-glass-facade.RW2
    ├── Nature/
    │   └── autumn-leaves-reflection.RW2
    └── Street-Scenes/
        └── busy-intersection-night.RW2
```

---

## 🔧 Configuration

Settings are stored in `~/.photosort.conf`:

```ini
[behavior]
last_source_path = /path/to/source
last_destination_path = /path/to/destination

[ingest]
default_model = qwen2.5vl:3b

[cull]
sharpness_good = 25.0
sharpness_dud = 45.0

[burst]
similarity_threshold = 12
burst_algorithm = clip
```

The TUI automatically saves changes to source, destination, and model settings.

---

## 🎯 RAW Format Support

PhotoSort supports all major RAW formats via dcraw:

- **Panasonic** - .RW2
- **Sony** - .ARW
- **Canon** - .CR2, .CR3
- **Nikon** - .NEF
- **Adobe** - .DNG
- **Fujifilm** - .RAF
- **Olympus** - .ORF
- **Pentax** - .PEF
- **Samsung** - .SRW

Plus standard formats: JPG, PNG, TIFF, BMP, WebP

---

## 📊 Version History

### v12.1 - TUI Edition (2025-11-16)
- Full Terminal User Interface with mouse support
- BRISQUE and CLIP engine detection at startup
- Animated block spinner during workflows
- Config auto-saves when changing settings
- Fixed mouse event leakage in Textual

### v10.7 - Engine Improvements (2025-11-16)
- Fixed session naming (uses destination paths)
- Pre-tests image encoding before selecting samples
- Added save_app_config() for persistence
- RAW conversion tries embedded thumbnail first

### v9.3 - Original CLI
- Complete automation pipeline
- Local AI inference with Ollama
- CLIP-based burst detection
- BRISQUE quality assessment

---

## 🤝 Contributing

PhotoSort is built through collaborative AI development with Claude (Anthropic) and Gemini (Google). Issues and feature requests welcome!

---

## 📜 License

MIT License - Use responsibly. Unleash creatively. Inference locally.

---

## 🙏 Acknowledgments

- **Ollama** - Local AI model serving
- **Textual** - Beautiful TUI framework
- **sentence-transformers** - CLIP embeddings
- **dcraw** - RAW image conversion
- **ExifTool** - Metadata extraction

---

*"Less noise, more signal."* - ∞vision crew
