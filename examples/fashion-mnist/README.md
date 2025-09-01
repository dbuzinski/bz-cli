# Fashion MNIST Image Classification Example

This example demonstrates how to use the `bz-cli` framework to train a simple convolutional neural network on the Fashion MNIST dataset for image classification.

## 🎯 What This Example Shows

- **Basic Training Setup**: How to configure and run a training session
- **Configuration Management**: Using `bzconfig.json` for hyperparameters and plugins
- **Plugin Integration**: Using TensorBoard for logging and early stopping
- **Model Definition**: Creating a custom PyTorch model
- **Data Loading**: Setting up data loaders with proper transforms

## 📁 Project Structure

```
fashion-mnist/
├── README.md           # This file
├── train.py           # Main training script
├── model.py           # Neural network model definition
└── bzconfig.json      # Configuration file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Install core framework
pip install bz-cli

# Install TensorBoard plugin for logging
pip install bz-cli[tensorboard]
```

### 2. Run Training

```bash
# Navigate to the example directory
cd examples/fashion-mnist

# Start training
bz train
```

### 3. Monitor Training

```bash
# View TensorBoard logs (in a separate terminal)
tensorboard --logdir runs/experiment
```

## 📊 Model Architecture

The `GarmentClassifier` model is a simple CNN with:
- 2 convolutional layers with ReLU activation and max pooling
- 3 fully connected layers
- Input: 28x28 grayscale images
- Output: 10 classes (different types of clothing)

## ⚙️ Configuration

The `bzconfig.json` file configures:

### Training Parameters
- **epochs**: 10 training epochs
- **device**: "auto" (automatically detect GPU/CPU)
- **compile**: true (use PyTorch 2.0 compilation for speed)
- **checkpoint_interval**: 5 (save checkpoints every 5 epochs)

### Hyperparameters
- **learning_rate**: 0.001
- **batch_size**: 64

### Metrics
- **accuracy**: Overall classification accuracy
- **precision**: Precision for each class
- **recall**: Recall for each class

### Plugins
- **console_out**: Formatted console output during training
- **tensorboard**: Log metrics and loss curves
- **early_stopping**: Stop training if no improvement (disabled in this example)

## 🔧 Customization

### Change Hyperparameters

Edit `bzconfig.json`:
```json
{
  "hyperparameters": {
    "lr": 0.0001,        // Lower learning rate
    "batch_size": 128    // Larger batch size
  }
}
```

### Add More Plugins

```json
{
  "plugins": [
    "console_out",
    {
      "wandb": {
        "enabled": true,
        "project_name": "fashion-mnist-experiment"
      }
    }
  ]
}
```

### Modify the Model

Edit `model.py` to change the architecture:
```python
class GarmentClassifier(nn.Module):
    def __init__(self):
        super(GarmentClassifier, self).__init__()
        # Add more layers, change activation functions, etc.
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)  # More filters
        # ... rest of the model
```

## 📈 Expected Results

With the default configuration, you should see:
- Training accuracy: ~85-90% after 10 epochs
- Validation accuracy: ~85-88%
- Training time: ~2-5 minutes (depending on hardware)

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size in `bzconfig.json`
   - Set `"device": "cpu"` to use CPU instead

2. **Slow Training**
   - Enable PyTorch compilation: `"compile": true`
   - Use GPU if available: `"device": "auto"`

3. **Plugin Not Found**
   - Install the required plugin: `pip install bz-cli[tensorboard]`
   - Check plugin name in `bzconfig.json`

### Getting Help

- Check the main documentation: `bz --help`
- List available plugins: `bz list-plugins`
- Run health check: `bz health`

## 🔄 Next Steps

After running this example, try:
1. **Experiment with hyperparameters** - change learning rate, batch size
2. **Add more plugins** - try WandB, Optuna for hyperparameter optimization
3. **Modify the model** - add more layers, change activation functions
4. **Use your own dataset** - adapt the data loading code for your images

## 📚 Related Examples

- **Custom Plugin Example**: Learn how to create your own plugins
- **Advanced Configuration**: See more complex training setups
- **Hyperparameter Optimization**: Use Optuna for automated tuning
