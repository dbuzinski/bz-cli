# Developer Guide

This guide is for contributors and developers working on the `bz-cli` project.

## Project Structure

```
bz-cli/
├── src/
│   ├── bz/                 # Core framework
│   │   ├── __init__.py     # Main trainer and core classes
│   │   ├── cli.py          # Command-line interface
│   │   ├── metrics/        # Modular metrics system
│   │   └── plugins/        # Core plugin system
│   ├── bz_optuna/          # Optuna plugin package
│   ├── bz_wandb/           # WandB plugin package
│   ├── bz_tensorboard/     # TensorBoard plugin package
│   └── bz_profiler/        # Profiler plugin package
├── tests/                  # Test suite
├── examples/               # Example projects
├── docs/                   # Documentation
└── pyproject.toml          # Project configuration
```

## Development Setup

### Prerequisites

- Git
- Python 3.10 or higher
- uv

### Quick Start

```bash
# Clone and setup
git clone https://github.com/your-org/bz-cli.git
cd bz-cli

# Install dependencies
uv sync --all-extras --dev

# Verify setup
uv run pytest
```

### Development Commands

```bash
# Run tests
uv run pytest

# Linting and formatting
uv run ruff check src tests
uv run ruff check --fix src tests
uv run black src tests

# Type checking
uv run mypy src tests
```

## Plugin Development

### Creating a New Plugin

1. **Create plugin structure**

   ```bash
   uv init --plugin my-plugin
   ```

2. **Implement the plugin**

   ```python
   # src/bz_myplugin/myplugin_plugin.py
   from typing import Optional, Dict, Any
   from bz.plugins import Plugin, PluginContext
   
   
   class MyPluginPlugin(Plugin):
       """My custom plugin."""
       
       name = "myplugin"
       
       def __init__(
           self, 
           config: Optional[Dict[str, Any]] = None, 
           **kwargs
       ) -> None:
           super().__init__(name=self.name, config=config, **kwargs)
       
       def start_training_session(self, context: PluginContext) -> None:
           """Initialize plugin at training start."""
           self.logger.info("MyPlugin initialized")
   ```

3. **Register the plugin**

   ```python
   # src/bz_myplugin/__init__.py
   from .myplugin_plugin import MyPluginPlugin
   
   __all__ = ["MyPluginPlugin"]
   ```

4. **Add entry point**

   ```toml
   # pyproject.toml
   [project.entry-points."bz.plugins"]
   myplugin = "bz_myplugin.myplugin_plugin:MyPluginPlugin"
   ```

### Plugin Testing

```python
# tests/bz_myplugin/test_myplugin.py
import pytest
from unittest.mock import Mock, patch
from bz_myplugin import MyPluginPlugin
from bz.plugins import PluginContext


def test_myplugin_initialization():
    """Test plugin initialization."""
    plugin = MyPluginPlugin()
    assert plugin.name == "myplugin"


def test_myplugin_start_training_session():
    """Test plugin start_training_session hook."""
    plugin = MyPluginPlugin()
    context = Mock(spec=PluginContext)
    
    with patch.object(plugin.logger, 'info') as mock_info:
        plugin.start_training_session(context)
        mock_info.assert_called_once_with("MyPlugin initialized")
```

## Contributing

### Workflow

1. **Fork and branch**
   ```bash
   git checkout -b feature/my-feature
   ```

2. **Make changes and test**
   ```bash
   uv run pytest
   uv run black src tests
   uv run ruff check src tests
   uv run mypy src tests
   ```

3. **Commit and push**
   ```bash
   git add .
   git commit -m "feat: add my feature"
   git push origin feature/my-feature
   ```

4. **Create pull request**
   Create a pull request to the `main` branch and a maintainer will review your changes.


### Getting Help

1. Check existing issues
2. Search documentation  
3. Create issue with:
   - Python version
   - Error message
   - Steps to reproduce
