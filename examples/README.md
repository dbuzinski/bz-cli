# bz-cli Examples

This directory contains examples demonstrating how to use the `bz-cli` framework for machine learning training.

## 📁 Available Examples

### 🎯 [Fashion MNIST](./fashion-mnist/)

A complete image classification example using the Fashion MNIST dataset.

**What you'll learn:**
- Basic training setup and configuration
- Using `bzconfig.json` for hyperparameters and plugins
- Integrating TensorBoard for logging
- Creating custom PyTorch models
- Setting up data loaders and transforms

**Quick start:**
```bash
cd examples/fashion-mnist
pip install bz-cli[tensorboard]
bz train
```

### 🔌 [Custom Plugin](./custom-plugin/)

Learn how to create and distribute custom plugins for `bz-cli`.

**What you'll learn:**
- Plugin architecture and lifecycle hooks
- Using Python entry points for plugin discovery
- Package structure for distributable plugins
- Configuration management for plugins
- Testing and debugging plugins

**Quick start:**
```bash
cd examples/custom-plugin
# Study the plugin examples and README
```

## 🚀 Getting Started

### Prerequisites

1. **Install bz-cli:**
   ```bash
   pip install bz-cli
   ```

2. **Install example-specific dependencies:**
   ```bash
   # For Fashion MNIST example
   pip install bz-cli[tensorboard]
   
   # For all examples
   pip install bz-cli[all]
   ```

### Running Examples

Each example has its own directory with:
- `README.md` - Detailed instructions
- Source code files
- Configuration files
- Example outputs

Navigate to any example directory and follow the instructions in its README.

## 📚 Learning Path

### Beginner
1. Start with **Fashion MNIST** to understand basic training
2. Experiment with different hyperparameters
3. Try adding more plugins

### Intermediate
1. Study the **Custom Plugin** example
2. Create your own simple plugin
3. Understand the plugin lifecycle

### Advanced
1. Build complex plugins with multiple features
2. Publish plugins to PyPI
3. Contribute to the community

## 🔧 Common Commands

```bash
# List available plugins
bz list-plugins

# List available metrics
bz list-metrics

# Run health check
bz health

# Get help
bz --help
```

## 🐛 Troubleshooting

### Plugin Not Found
```bash
# Install the required plugin
pip install bz-cli[tensorboard]

# Check available plugins
bz list-plugins
```

### Configuration Issues
- Ensure `bzconfig.json` is in the current directory
- Check JSON syntax is valid
- Verify plugin names match available plugins

### Performance Issues
- Enable PyTorch compilation: `"compile": true`
- Use GPU if available: `"device": "auto"`
- Adjust batch size based on your hardware

## 📖 Additional Resources

- **Main Documentation**: See the project README
- **CLI Help**: `bz --help`
- **Plugin API**: Study the plugin examples
- **Configuration Reference**: Check `bzconfig.json` examples

## 🤝 Contributing

Found a bug or have an idea for a new example?

1. Check existing issues
2. Create a new issue with details
3. Submit a pull request with your example

## 📄 License

These examples are provided under the same license as the main project.
