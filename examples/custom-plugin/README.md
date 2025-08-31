# Custom Plugin Example

This example demonstrates how to create custom plugins for the `bz-cli` framework using Python entry points. You'll learn how to build, package, and distribute your own plugins.

## 🎯 What This Example Shows

- **Plugin Architecture**: How to create plugins that integrate with the training loop
- **Entry Points System**: Using Python entry points for plugin discovery
- **Package Structure**: Proper organization for distributable plugins
- **Configuration Management**: How plugins handle configuration
- **Plugin Lifecycle**: Understanding when plugin methods are called

## 📁 Project Structure

```
custom-plugin/
├── README.md                    # This file
├── custom_logger_plugin.py      # Complete plugin example with package structure
├── simple_plugin_example.py     # Minimal plugin example
└── pyproject.toml.example       # Example package configuration
```

## 🚀 Quick Start

### 1. Understanding Plugin Architecture

All plugins inherit from the base `Plugin` class and can hook into various stages of the training lifecycle:

```python
from bz.plugins import Plugin, PluginContext

class MyCustomPlugin(Plugin):
    name = "my_plugin"  # Plugin name for discovery
    
    def start_training_session(self, context: PluginContext) -> None:
        """Called at the start of training"""
        pass
    
    def end_epoch(self, context: PluginContext) -> None:
        """Called at the end of each epoch"""
        pass
```

### 2. Plugin Lifecycle Hooks

Plugins can implement these methods to hook into training:

- `start_training_session()` - Training begins
- `start_epoch()` - Each epoch starts
- `end_epoch()` - Each epoch ends
- `end_training_session()` - Training completes
- `start_training_batch()` - Each batch starts
- `end_training_batch()` - Each batch ends
- And many more...

### 3. Configuration System

Plugins use a standardized configuration system:

```python
@staticmethod
def load_config(config_data: dict) -> Dict[str, Any]:
    """Load configuration from dict data."""
    return {
        "enabled": config_data.get("enabled", True),
        "log_interval": config_data.get("log_interval", 10),
        # ... other config options
    }

@staticmethod
def create(config_data: dict, training_config) -> Optional["MyPlugin"]:
    """Create plugin instance from config data."""
    config = MyPlugin.load_config(config_data)
    if not config.get("enabled", True):
        return None
    return MyPlugin(**config)
```

## 📦 Creating a Distributable Plugin

### 1. Package Structure

```
my-bz-plugin/
├── src/
│   └── my_bz_plugin/
│       ├── __init__.py
│       └── my_plugin.py
├── tests/
│   └── test_my_plugin.py
├── pyproject.toml
└── README.md
```

### 2. Package Configuration

```toml
[project]
name = "my-bz-plugin"
version = "0.1.0"
description = "My custom plugin for bz-cli"
dependencies = ["bz-cli>=0.1.0"]

[project.entry-points."bz.plugins"]
my_plugin = "my_bz_plugin.my_plugin:MyPlugin"
```

### 3. Plugin Implementation

```python
# src/my_bz_plugin/my_plugin.py
from typing import Dict, Any, Optional
from bz.plugins import Plugin, PluginContext

class MyPlugin(Plugin):
    name = "my_plugin"
    
    def __init__(self, log_interval: int = 10):
        super().__init__(name=self.name)
        self.log_interval = log_interval
    
    def end_epoch(self, context: PluginContext) -> None:
        if context.epoch % self.log_interval == 0:
            print(f"Epoch {context.epoch}: Loss = {context.training_loss_total}")
    
    @staticmethod
    def load_config(config_data: dict) -> Dict[str, Any]:
        return {
            "enabled": config_data.get("enabled", True),
            "log_interval": config_data.get("log_interval", 10),
        }
    
    @staticmethod
    def create(config_data: dict, training_config) -> Optional["MyPlugin"]:
        config = MyPlugin.load_config(config_data)
        if not config.get("enabled", True):
            return None
        return MyPlugin(log_interval=config["log_interval"])
```

## 🔧 Using Custom Plugins

### 1. Installation

```bash
# Install your plugin
pip install my-bz-plugin

# Or install in development mode
pip install -e .
```

### 2. Configuration

Add to your `bzconfig.json`:

```json
{
  "plugins": [
    "console_out",
    {
      "my_plugin": {
        "enabled": true,
        "log_interval": 5
      }
    }
  ]
}
```

### 3. Verification

```bash
# List available plugins
bz list-plugins

# Should show your plugin in the list
```

## 📋 Plugin Examples

### Simple Logger Plugin

See `simple_plugin_example.py` for a minimal plugin that logs training progress.

### Advanced Logger Plugin

See `custom_logger_plugin.py` for a complete example with:
- File logging
- Configuration management
- Error handling
- Performance metrics

## 🧪 Testing Your Plugin

### 1. Unit Tests

```python
# tests/test_my_plugin.py
import pytest
from my_bz_plugin.my_plugin import MyPlugin

def test_plugin_creation():
    plugin = MyPlugin(log_interval=5)
    assert plugin.name == "my_plugin"
    assert plugin.log_interval == 5

def test_plugin_config():
    config_data = {"enabled": True, "log_interval": 10}
    config = MyPlugin.load_config(config_data)
    assert config["enabled"] is True
    assert config["log_interval"] == 10
```

### 2. Integration Testing

```python
def test_plugin_integration():
    # Test that your plugin works with the training loop
    from bz import Trainer
    from bz.plugins import PluginContext
    
    plugin = MyPlugin()
    context = PluginContext()
    context.epoch = 5
    
    # Test that your plugin methods work correctly
    plugin.end_epoch(context)
```

## 🚀 Best Practices

### 1. Plugin Design

- **Single Responsibility**: Each plugin should do one thing well
- **Configuration**: Make plugins configurable via JSON
- **Error Handling**: Gracefully handle errors without breaking training
- **Logging**: Use the framework's logging system

### 2. Package Structure

- **Clear Naming**: Use descriptive names for your plugin
- **Documentation**: Include README and docstrings
- **Tests**: Write comprehensive tests
- **Versioning**: Follow semantic versioning

### 3. Distribution

- **PyPI**: Publish to PyPI for easy installation
- **Entry Points**: Always use entry points for discovery
- **Dependencies**: Minimize dependencies
- **Compatibility**: Test with different bz-cli versions

## 🔍 Debugging Plugins

### 1. Check Plugin Discovery

```bash
# List all available plugins
bz list-plugins

# Check if your plugin appears
```

### 2. Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### 3. Test Plugin Methods

```python
# Test your plugin methods directly
plugin = MyPlugin()
context = PluginContext()
plugin.start_training_session(context)
```

## 📚 Advanced Topics

### 1. Plugin Communication

Plugins can communicate through the training context:

```python
def end_epoch(self, context: PluginContext) -> None:
    # Store data for other plugins
    context.extra["my_plugin_data"] = {"epoch": context.epoch}
    
    # Read data from other plugins
    other_data = context.extra.get("other_plugin_data")
```

### 2. Custom Metrics

Plugins can contribute custom metrics:

```python
def end_epoch(self, context: PluginContext) -> None:
    # Add custom metrics
    context.training_metrics["my_custom_metric"] = 0.95
```

### 3. Plugin Dependencies

Plugins can depend on other plugins:

```python
def start_training_session(self, context: PluginContext) -> None:
    # Check if required plugin is available
    if "tensorboard" not in context.extra.get("active_plugins", []):
        self.logger.warning("TensorBoard plugin not found")
```

## 🎉 Next Steps

1. **Study the Examples**: Examine the provided plugin examples
2. **Create Your Own**: Build a plugin for your specific needs
3. **Share**: Contribute your plugin to the community
4. **Improve**: Iterate based on feedback and usage

## 📖 Additional Resources

- **Plugin API Documentation**: See the main bz-cli documentation
- **Entry Points Guide**: Python packaging documentation
- **Community Plugins**: Check for existing plugins to learn from
