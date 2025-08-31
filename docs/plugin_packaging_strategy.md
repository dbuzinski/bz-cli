# Plugin Packaging Strategy: Separate Packages vs. Monolithic

## Overview

This document explores the architectural decision of whether to ship complex plugins (like Optuna, WandB, TensorBoard) as separate Python packages from the base `bz` framework, or keep them all in a single monolithic package.

## Current Approach: Optional Dependencies

### How it works:
```python
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
```

### Pros:
- ✅ **Simple implementation** - Easy to add to existing codebase
- ✅ **Backward compatibility** - Works with existing installations
- ✅ **Gradual adoption** - Users can opt-in to features
- ✅ **Single codebase** - All code in one repository
- ✅ **Unified testing** - All tests run together
- ✅ **Simplified CI/CD** - One build pipeline

### Cons:
- ❌ **Dependency bloat** - All optional deps listed in main package
- ❌ **Version conflicts** - Hard to manage complex dependency trees
- ❌ **Installation complexity** - Users need to understand optional deps
- ❌ **Testing complexity** - Need to mock optional dependencies
- ❌ **Documentation overhead** - Need to document which deps are optional
- ❌ **Plugin discovery** - No clear way to discover available plugins

## Alternative Approach: Separate Packages

### Proposed Structure:
```
bz-core/                    # Core framework
├── bz/                     # Main package
│   ├── __init__.py
│   ├── cli.py
│   ├── config.py
│   ├── trainer.py
│   └── plugins/
│       ├── __init__.py
│       ├── plugin.py       # Base plugin system
│       ├── console_out.py  # Simple plugin (no deps)
│       └── profiler.py     # Simple plugin (psutil only)
├── pyproject.toml
└── README.md

bz-optuna/                  # Optuna plugin package
├── bz_optuna/
│   ├── __init__.py
│   ├── plugin.py
│   └── config.py
├── pyproject.toml
├── requirements.txt        # optuna>=4.3.0
└── README.md

bz-wandb/                   # WandB plugin package
├── bz_wandb/
│   ├── __init__.py
│   ├── plugin.py
│   └── config.py
├── pyproject.toml
├── requirements.txt        # wandb>=0.15.0
└── README.md

bz-tensorboard/             # TensorBoard plugin package
├── bz_tensorboard/
│   ├── __init__.py
│   ├── plugin.py
│   └── config.py
├── pyproject.toml
├── requirements.txt        # tensorboard>=2.19.0
└── README.md
```

## Separate Package Approach: Detailed Analysis

### Pros:

#### 1. **Clean Dependency Management**
```python
# bz-core/pyproject.toml
[project]
name = "bz-core"
dependencies = [
    "torch>=2.7.0",
    "numpy>=2.2.5",
    "tqdm>=4.67.1",
    # No optuna, wandb, tensorboard here
]

# bz-optuna/pyproject.toml
[project]
name = "bz-optuna"
dependencies = [
    "bz-core>=0.1.0",
    "optuna>=4.3.0",
]
```

#### 2. **Independent Versioning**
```bash
# Users can choose specific versions
pip install bz-core==0.1.0
pip install bz-optuna==0.2.0  # Different version
pip install bz-wandb==0.1.5   # Different version
```

#### 3. **Focused Testing**
```python
# bz-optuna/tests/test_optuna_plugin.py
# Only tests Optuna-specific functionality
# No need to mock optuna - it's a real dependency
```

#### 4. **Plugin Discovery**
```python
# bz-core/src/bz/plugins/discovery.py
def discover_installed_plugins():
    """Find all installed bz plugin packages."""
    import pkg_resources
    
    plugins = {}
    for dist in pkg_resources.working_set:
        if dist.project_name.startswith('bz-'):
            plugin_name = dist.project_name.replace('bz-', '')
            plugins[plugin_name] = dist.version
    
    return plugins
```

#### 5. **Independent Development**
- Different teams can work on different plugins
- Different release cycles for different plugins
- Plugin-specific CI/CD pipelines
- Plugin-specific documentation

#### 6. **Reduced Installation Footprint**
```bash
# Minimal installation
pip install bz-core

# Add only what you need
pip install bz-optuna    # For hyperparameter optimization
pip install bz-wandb     # For experiment tracking
```

### Cons:

#### 1. **Increased Complexity**
- Multiple repositories to manage
- Multiple build pipelines
- More complex dependency resolution
- Plugin compatibility matrix

#### 2. **Installation Complexity**
```bash
# Users need to know about separate packages
pip install bz-core bz-optuna bz-wandb bz-tensorboard

# vs. current approach
pip install bz-cli[optuna,wandb,tensorboard]
```

#### 3. **Version Compatibility**
```python
# Need to ensure plugin compatibility
# bz-optuna/pyproject.toml
[project]
dependencies = [
    "bz-core>=0.1.0,<0.2.0",  # Version constraints
]
```

#### 4. **Plugin Registration Complexity**
```python
# Need a plugin registration system
# bz-optuna/bz_optuna/__init__.py
from bz.plugins import register_plugin
from .plugin import OptunaPlugin

register_plugin("optuna", OptunaPlugin)
```

#### 5. **Testing Complexity**
- Need to test plugin integration
- Need to test version compatibility
- More complex CI/CD setup

## Hybrid Approach: Best of Both Worlds

### Structure:
```
bz-cli/                     # Main package (current)
├── bz/                     # Core framework
│   ├── plugins/
│   │   ├── __init__.py
│   │   ├── plugin.py       # Base plugin system
│   │   ├── console_out.py  # Built-in plugin
│   │   ├── profiler.py     # Built-in plugin
│   │   └── discovery.py    # Plugin discovery
│   └── ...
├── pyproject.toml          # Core dependencies only
└── README.md

bz-plugins-optuna/          # Separate plugin package
├── bz_plugins_optuna/
│   ├── __init__.py
│   ├── plugin.py
│   └── config.py
├── pyproject.toml
└── README.md
```

### Implementation:

#### 1. **Plugin Discovery System**
```python
# bz/plugins/discovery.py
import importlib
import pkg_resources
from typing import Dict, Type
from .plugin import Plugin

class PluginDiscovery:
    """Discovers and loads installed bz plugins."""
    
    def __init__(self):
        self.plugins: Dict[str, Type[Plugin]] = {}
        self._discover_plugins()
    
    def _discover_plugins(self):
        """Find all installed bz plugins."""
        # Built-in plugins
        self._load_builtin_plugins()
        
        # External plugins
        self._load_external_plugins()
    
    def _load_builtin_plugins(self):
        """Load plugins that are part of the core package."""
        from .console_out import ConsoleOutPlugin
        from .profiler import ProfilerPlugin
        
        self.plugins["console_out"] = ConsoleOutPlugin
        self.plugins["profiler"] = ProfilerPlugin
    
    def _load_external_plugins(self):
        """Load plugins from separate packages."""
        try:
            import bz_plugins_optuna
            from bz_plugins_optuna.plugin import OptunaPlugin
            self.plugins["optuna"] = OptunaPlugin
        except ImportError:
            pass
        
        try:
            import bz_plugins_wandb
            from bz_plugins_wandb.plugin import WandBPlugin
            self.plugins["wandb"] = WandBPlugin
        except ImportError:
            pass
    
    def get_plugin(self, name: str) -> Type[Plugin]:
        """Get a plugin by name."""
        return self.plugins.get(name)
    
    def list_plugins(self) -> Dict[str, str]:
        """List all available plugins with their status."""
        result = {}
        for name, plugin_class in self.plugins.items():
            if name in ["console_out", "profiler"]:
                result[name] = "built-in"
            else:
                result[name] = "external"
        return result
```

#### 2. **Plugin Package Template**
```python
# bz-plugins-optuna/bz_plugins_optuna/__init__.py
from .plugin import OptunaPlugin
from .config import OptunaConfig

__version__ = "0.1.0"
__all__ = ["OptunaPlugin", "OptunaConfig"]

# Auto-register the plugin
try:
    from bz.plugins import register_plugin
    register_plugin("optuna", OptunaPlugin)
except ImportError:
    # bz-core not installed
    pass
```

#### 3. **CLI Integration**
```python
# bz/cli.py
def list_plugins():
    """List available plugins."""
    from .plugins.discovery import PluginDiscovery
    
    discovery = PluginDiscovery()
    plugins = discovery.list_plugins()
    
    print("Available plugins:")
    for name, status in plugins.items():
        print(f"  - {name} ({status})")
    
    print("\nTo install external plugins:")
    print("  pip install bz-plugins-optuna")
    print("  pip install bz-plugins-wandb")
```

## Recommendation: Hybrid Approach

### Phase 1: Keep Current Approach (Immediate)
- Maintain current optional dependency pattern
- Add plugin discovery system
- Improve documentation for optional dependencies

### Phase 2: Introduce Separate Packages (Future)
- Create `bz-plugins-*` packages for complex plugins
- Keep simple plugins in core package
- Implement plugin discovery system
- Provide migration guide

### Benefits of Hybrid Approach:
1. **Gradual migration** - Users can adopt at their own pace
2. **Backward compatibility** - Existing code continues to work
3. **Flexibility** - Users can choose their preferred approach
4. **Clean separation** - Complex plugins get their own packages
5. **Reduced bloat** - Core package stays lean

### Migration Strategy:
```bash
# Current (Phase 1)
pip install bz-cli[optuna,wandb]

# Future (Phase 2)
pip install bz-cli                    # Core only
pip install bz-plugins-optuna         # Add what you need
pip install bz-plugins-wandb          # Add what you need
```

## Conclusion

The hybrid approach provides the best balance between simplicity and flexibility. It allows for:

- **Immediate improvements** to the current system
- **Future scalability** with separate packages
- **User choice** in how they want to manage dependencies
- **Clean architecture** with proper separation of concerns

This approach aligns with the principle of "make the common case simple, but don't make the uncommon case impossible."
