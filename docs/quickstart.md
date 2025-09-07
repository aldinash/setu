# Quickstart Guide

Get up and running with Setu in under 10 minutes! This guide walks you through installation, setup, and your first inference.

## Prerequisites

- **CUDA-capable GPU** with Ampere, Lovelace or Hopper architecture (A100, H100 recommended)
- **Python 3.12** 
- **CUDA 12.6+** toolkit installed
- **Git** with LFS support

## Installation Options

### Option 1: VS Code Devcontainer (Recommended)

For the most consistent development experience:

1. **Install Prerequisites:**
   - [Docker](https://www.docker.com/products/docker-desktop)
   - [VS Code](https://code.visualstudio.com/)
   - [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)

2. **Clone and Setup:**
   ```bash
   git clone https://github.com/project-vajra/setu
   cd setu
   ```

3. **Open in VS Code:**
   ```bash
   code .
   ```

4. **Launch Devcontainer:**
   - VS Code will detect the devcontainer configuration
   - Click "Reopen in Container" when prompted
   - The container will automatically set up all dependencies

5. **Build Setu:**
   - Use VS Code's Terminal > Run Build Task... > "build"
   - Or run manually: `make build`

### Option 2: Manual Setup

1. **Setup Mamba/Conda:**
   ```bash
   # If you don't have mamba, install it:
   wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
   bash Miniforge3-Linux-x86_64.sh
   ```

2. **Clone Repository:**
   ```bash
   git clone https://github.com/project-vajra/setu
   cd setu
   ```

3. **Create Environment:**
   ```bash
   mamba env create -f environment-dev.yml -p ./env
   mamba activate ./env
   ```

4. **Build Setu:**
   ```bash
   make build
   ```

## Verify Installation

```python
import setu
import torch

print(f"✅ Setu {setu.__version__} installed")
```

## Your First Setu Program

TODO(Anirudha, Sadjad)

## Next Steps

**🚀 Development & Contributing:**
- **[Development Workflow](contributing/development_workflow.md)** - Building, testing, and contributing code
- **[Contributing Guide](contributing/contributing.md)** - How to contribute to Setu

**📚 Learn More:**
- **[System Design](design/index.md)** - Learn about Setu's architecture
- **[Usage Guide](usage/entrypoints.md)** - Usage examples
<!-- - **[API Reference](usage/api_reference/index.md)** - Explore the full API documentation -->


**🔧 Need Help?**
- **[Troubleshooting Guide](troubleshooting.md)** - Common issues and solutions
- **[GitHub Issues](https://github.com/project-vajra/setu/issues)** - Bug reports and feature requests
- **[GitHub Discussions](https://github.com/project-vajra/setu/discussions)** - Questions and community
- **[Community Discord](https://discord.gg/wjaSvGgsNN)** - Interact with other users and developers.

