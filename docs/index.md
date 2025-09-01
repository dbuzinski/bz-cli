# bz

**Train machine learning models with confidence. Reproducible, efficient, and extensible.**

## Why bz?

bz is a model training framework that eliminates the complexity of ML infrastructure so teams can focus on what matters most: building better models. We believe that every ML practitioner deserves powerful, reproducible training workflows without the overhead of custom infrastructure code.

bz provides a unified command-line interface that works consistently across local development and production environments. Choose bz for its ability to reduce training boilerplate, ensure reproducibility, and enable seamless integrations with other tools in your ML pipeline.

Use bz when training experiments, running hyperparameter searches, or deploying production models - it works seamlessly for both local and automated workflows.

## Quick Start

Add `bz` to your existing project:

```bash
# Install bz
pip install bz-cli

# Set up bz for your project
bz init

# Define your model
# Edit train.py to define your model, loss_fn, and optimizer

# Start training
bz train
```

That's it! Your model is now training with automatic progress tracking, checkpointing, and metric logging.

## Common Workflows

### Training and Evaluation

Train your models with confidence using bz's built-in training loop:

```bash
# Basic training
bz train

# Training with custom configuration
bz train --config my_config.json
```

### Hyperparameter Tuning

Optimize your model performance with automated hyperparameter search:

```bash
# Optuna integration
pip install bz-cli[optuna]

# Add the optuna plugin to your bzconfig.json
# Define parameters for your study

# Run training with hyperparameter optimization
bz train
```

## Custom Integrations

Extend bz with plugins for your specific needs:

### Experiment Tracking
```bash
# Weights & Biases integration
pip install bz-cli[wandb]

# Add the wandb plugin to your bzconfig.json
# Train your models with advanced reporting
bz train

# TensorBoard logging
pip install bz-cli[tensorboard]

# Add the tensorboard plugin to your bzconfig.json
# Train your models with TensorBoard logging
bz train
```

### Custom Metrics and Plugins

Extend bz to include the metrics and integrations your team needs.

```python
# Create custom metrics
from bz.metrics import Metric

class CustomMetric(Metric):
    def compute(self, predictions, targets):
        return your_calculation(predictions, targets)

# Build your own plugins
from bz.plugins import Plugin

class CustomPlugin(Plugin):
    def on_epoch_end(self, trainer):
        # Your custom logic here
        pass
```

Now update `bzconfig.json` to use your new metric and plugin:

```json
{
    ... // bzconfig.json
    "metrics": ["custom_metric"],
    "plugins": ["custom_plugin"]
}
```
