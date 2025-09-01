# Quick Start Guide

Get your first model training with bz-cli in under 10 minutes.

## Installation

```bash
pip install bz-cli
```

## Create Your First Project

```bash
bz init my-first-model
cd my-first-model
```

This creates:
- `train.py` - Your training script
- `model.py` - Model definition
- `bzconfig.json` - Configuration file
- `README.md` - Project documentation

## Train Your Model

```bash
bz train
```

You should see progress bars, metrics being logged, and checkpoints being saved.

**That's it!** You've successfully trained your first model with bz-cli.

## What Just Happened?

bz-cli automatically handled:
- Training loops and optimization
- Metrics tracking (loss, accuracy)
- Model checkpointing
- Error handling and recovery

## Next Steps

- **Customize Your Model**: Edit `model.py` to change the network architecture
- **Add Experiment Tracking**: Install the WandB plugin to track your experiments
- **Optimize Hyperparameters**: Use the Optuna plugin to find the best model configuration

## Troubleshooting

### Installation Issues
**Problem**: `pip install bz-cli` fails
**Solution**: Try upgrading pip first:
```bash
python -m pip install --upgrade pip
pip install bz-cli
```

### PyTorch Issues
**Problem**: PyTorch not found
**Solution**: Install PyTorch:
```bash
pip install torch torchvision
```

### Permission Issues
**Problem**: Permission denied during installation
**Solution**: Use user installation:
```bash
pip install --user bz-cli
```

## Need Help?

- **Documentation**: Check the [full documentation](index.md)
- **Examples**: Browse [working examples](examples/)
- **Issues**: Report bugs on [GitHub Issues](https://github.com/dbuzinski/bz-cli/issues)

**Pro Tip**: The `bz --help` command shows all available options and commands.
