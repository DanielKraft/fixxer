# FIXXER ✞ PRO
### Professional-Grade Photography Workflow Automation

![Version](https://img.shields.io/badge/version-1.0-red.svg)
![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

**"CHAOS PATCHED // LOGIC INJECTED"**

---

## 🎯 What is FIXXER?

FIXXER is a professional photography workflow automation tool that combines **AI vision models**, **cryptographic integrity verification**, and **intelligent quality analysis** to streamline your post-processing pipeline.

Built for photographers who demand both **speed** and **safety** in their digital asset management.

---

## ✨ Key Features

### 🔐 **Hash-Verified File Operations**
- **SHA256 integrity checking** on every file move
- **Halt-on-mismatch** protection prevents corruption
- **JSON sidecar files** (.fixxer.json) create audit trails
- Production-tested: 120+ files, zero corruption

### 🤖 **AI-Powered Workflows**
- **Vision-based naming** using Ollama models (qwen2.5vl, llava, etc.)
- **Semantic burst detection** with CLIP embeddings
- **AI session naming** from visual analysis
- **Creative critique mode** for artistic feedback

### 📊 **Quality Analysis Pipeline**
- **BRISQUE quality scoring** for sharpness assessment
- **Exposure analysis** (crushed blacks, blown highlights)
- **CLIP-based burst grouping** (semantic similarity, not just timestamps)
- **Automated culling** into Tier A/B/C folders

### 🎨 **Two UI Modes**
- **Standard Mode**: Warez-inspired red/white/black aesthetic
- **Pro Mode (F12)**: Phantom Redline - tactical precision dashboard
- Real-time **system monitoring** (RAM, CPU sparklines)
- **Milestone HUD** for workflow progress tracking

### 📷 **RAW File Support**
- **120+ RAW formats** via rawpy/libraw
- Cross-platform: Linux, macOS, Windows
- Supports: RW2, CR2, CR3, NEF, ARW, DNG, RAF, ORF, PEF, SRW, and more
- Zero temp files - pure in-memory processing

---

## 🚀 Quick Start

### Prerequisites
- **Python 3.10+**
- **Ollama** (for AI vision features)
- Supported OS: macOS, Linux, Windows (WSL recommended)

### Installation

```bash
# Clone the repository
git clone https://github.com/BandwagonVibes/fixxer.git
cd fixxer

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install Ollama and pull a vision model
# Visit: https://ollama.ai
ollama pull qwen2.5vl:3b
```

### Launch FIXXER

```bash
# Standard Mode (Warez aesthetic)
python3 photosort_tui_styled.py

# Toggle Pro Mode in-app with F12
```

---

## 📖 Workflow Overview

### **Auto Workflow** (Recommended)
Complete end-to-end processing:

1. **Analyze Session** - EXIF statistics and insights
2. **Stack Bursts** - CLIP-based semantic grouping + AI naming
3. **Cull Singles** - Quality analysis → Tier A/B/C
4. **Archive Heroes** - Move best shots to organized folders
5. **Verify Integrity** - SHA256 hash checking throughout

### **Individual Workflows**

- **Bursts**: Group similar shots, AI-name the best pick
- **Cull**: Analyze sharpness/exposure, sort by quality
- **Stats**: EXIF insights (cameras, focal lengths, lighting conditions)
- **Critique**: Get creative feedback from AI (composition, mood, suggestions)
- **Easy Archive**: Simple AI naming + keyword folder organization (no culling)

---

## 🎛️ Configuration

Configuration is stored in `~/.photosort.conf`:

```ini
[ingest]
default_model = qwen2.5vl:3b
default_destination = ~/Pictures/Sorted

[cull]
cull_algorithm = legacy
sharpness_good = 40.0
sharpness_dud = 15.0
exposure_dud_pct = 0.20
exposure_good_pct = 0.05

[burst]
burst_algorithm = legacy
similarity_threshold = 8

[folders]
burst_parent_folder = true
ai_session_naming = true

[behavior]
pro_mode = false
last_source_path = 
last_destination_path = 
```

---

## 🔧 Technical Architecture

### **Hash Verification Pipeline**
```
Source File
    ↓ Calculate SHA256
    ↓ Move to Destination
    ↓ Recalculate SHA256
    ↓ Compare Hashes
    ├─ MATCH → Generate .fixxer.json sidecar
    └─ MISMATCH → HALT WORKFLOW (RuntimeError)
```

### **AI Vision Integration**
- **Ollama API**: Local LLM inference (no cloud, no privacy concerns)
- **JSON-structured responses**: Deterministic parsing
- **Base64 image encoding**: Direct vision model analysis
- **Fallback chains**: Graceful degradation on timeouts

### **Quality Scoring**
- **BRISQUE** (Blind/Referenceless Image Spatial Quality Evaluator)
- **OpenCV Laplacian variance** (sharpness fallback)
- **Histogram analysis** (exposure distribution)
- **CLIP embeddings** (semantic similarity for bursts)

---

## 📂 Project Structure

```
fixxer/
├── photosort_engine.py          # Core workflow logic + hash verification
├── photosort_tui_styled.py      # Textual TUI application
├── photosort_pro.css            # Pro Mode styling (Phantom Redline)
├── photosort_visioncrew.css     # Standard Mode styling (Warez)
├── phrases.py                   # Motivational progress phrases
├── requirements.txt             # Python dependencies
├── pyproject.toml               # Packaging metadata
├── README.md                    # This file
├── LICENSE                      # MIT License
└── .gitignore                   # Git exclusions
```

---

## 🔐 Sidecar File Format

Example `.fixxer.json`:

```json
{
  "fixxer_version": "1.0",
  "filename": "golden-hour-cityscape.jpg",
  "original_path": "/source/IMG_1234.jpg",
  "final_path": "/archive/2024-11-20_Urban/Architecture/golden-hour-cityscape.jpg",
  "sha256_source": "a1b2c3d4...",
  "verified": true,
  "timestamp": "2024-11-20T14:35:22.123456"
}
```

If corruption is detected:
```json
{
  "sha256_source": "a1b2c3d4...",
  "sha256_destination": "e5f6g7h8...",
  "verified": false,
  "corruption_detected": true
}
```

---

## 🎨 UI Modes Comparison

| Feature | Standard Mode | Pro Mode (F12) |
|---------|---------------|----------------|
| **Aesthetic** | Warez (red/white/black) | Phantom Redline (tactical black) |
| **Logo** | ASCII art + tagline | Clean typography |
| **System Monitor** | Cyan sparklines | Red "redline" sparklines |
| **Progress Phrases** | "Applying physics hacks..." | "Processing active... [2m 34s]" |
| **Milestone HUD** | ❌ Hidden | ✅ Real-time stats (BURSTS, TIER A/B/C, HEROES, ARCHIVED, TIME) |
| **Button Styles** | High contrast | Minimal, subtle borders |

---

## 🧪 Testing

Hash verification stress test (included):

```bash
# Test with 120+ mixed RAW/JPEG files
python3 test_hash_verification.py

# Expected output:
# ✅ 120 files processed
# ✅ 120 hashes verified
# ✅ 0 corruption events
# ✅ 120 sidecar files generated
```

---

## 🤝 Contributing

Contributions welcome! Areas of interest:

- Additional RAW format testing
- Alternative AI vision models
- Quality scoring algorithm improvements
- Cross-platform testing (Windows native)
- Performance optimizations

---

## 📜 License

MIT License - See [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Ollama** - Local LLM inference
- **rawpy/libraw** - RAW file processing
- **CLIP** (OpenAI) - Semantic burst detection
- **BRISQUE** - Image quality assessment
- **Textual** - Modern TUI framework

---

## 📧 Contact

Issues and feature requests: [GitHub Issues](https://github.com/BandwagonVibes/fixxer/issues)

---

**Built with precision. Secured with cryptography. Powered by AI.**

✞ **FIXXER PRO** - "CHAOS PATCHED // LOGIC INJECTED"
