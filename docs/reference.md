# API Reference

This document provides a comprehensive reference for all the classes, functions, and modules in the `bz` CLI.

## Core Modules

### `bz` - Main Package

The main package containing the core training functionality.

#### `Trainer`

The main training class that orchestrates the training process.

```python
class Trainer:
    def __init__(self):
        self.plugins = []
        self.logger = logger
```

**Methods:**

- `add_plugin(plugin)`: Add a plugin to the trainer
- `train(model, optimizer, loss_fn, training_loader, ...)`: Start training

**Training Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | torch.nn.Module | - | PyTorch model to train |
| `optimizer` | torch.optim.Optimizer | - | Optimizer for training |
| `loss_fn` | torch.nn.Module | - | Loss function |
| `training_loader` | torch.utils.data.DataLoader | - | Training data loader |
| `validation_loader` | torch.utils.data.DataLoader | None | Validation data loader |
| `device` | torch.device | auto | Device to use for training |
| `epochs` | int | 1 | Number of training epochs |
| `compile` | bool | True | Enable model compilation |
| `checkpoint_interval` | int | 0 | Checkpoint save interval |
| `metrics` | List[Metric] | [] | List of metrics to track |
| `hyperparameters` | Dict[str, Any] | {} | Training hyperparameters |


#### `PluginContext`

Data class containing training context information passed to plugins.

```python
@dataclass
class PluginContext:
    epoch: int = 0
    training_loss_total: float = 0.0
    validation_loss_total: float = 0.0
    training_batch_count: int = 0
    validation_batch_count: int = 0
    training_metrics: Dict[str, float] = field(default_factory=dict)
    validation_metrics: Dict[str, float] = field(default_factory=dict)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)
    should_stop_training: bool = False
```

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `epoch` | int | Current epoch number |
| `training_loss_total` | float | Cumulative training loss |
| `validation_loss_total` | float | Cumulative validation loss |
| `training_batch_count` | int | Number of training batches processed |
| `validation_batch_count` | int | Number of validation batches processed |
| `training_metrics` | Dict[str, float] | Current training metrics |
| `validation_metrics` | Dict[str, float] | Current validation metrics |
| `hyperparameters` | Dict[str, Any] | Training hyperparameters |
| `extra` | Dict[str, Any] | Additional context data |
| `should_stop_training` | bool | Flag to stop training (set by plugins) |

### `bz.plugins` - Plugin System

Plugin system for extending training functionality.

#### `Plugin`

Base class for all plugins.

```python
class Plugin:
    def start_training_session(self, context):
        pass

    def load_checkpoint(self, context):
        pass

    def start_epoch(self, context):
        pass

    def start_training_loop(self, context):
        pass

    def start_training_batch(self, context):
        pass

    def end_training_batch(self, context):
        pass

    def end_training_loop(self, context):
        pass

    def start_validation_loop(self, context):
        pass

    def start_validation_batch(self, context):
        pass

    def end_validation_batch(self, context):
        pass

    def end_validation_loop(self, context):
        pass

    def save_checkpoint(self, context):
        pass

    def end_epoch(self, context):
        pass

    def end_training_session(self, context):
        pass

    def run_stage(self, stage, context):
        getattr(self, stage)(context)
```

#### Built-in Plugins

##### `ConsoleOutPlugin`

Provides progress bars and training summaries.

```python
class ConsoleOutPlugin(Plugin):
    def __init__(self, training_data_len, validation_data_len=0, update_interval=1):
        self.training_data_len = training_data_len
        self.validation_data_len = validation_data_len
        self.update_interval = update_interval
        self._training_bar = None
        self._validation_bar = None
        self.training_start_time = None
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `training_data_len` | int | - | Number of training batches |
| `validation_data_len` | int | 0 | Number of validation batches |
| `update_interval` | int | 1 | Progress update interval |

**Static Methods:**

- `init(spec) -> ConsoleOutPlugin`: Create plugin instance from training specification

##### `TensorBoardPlugin`

Logs training metrics to TensorBoard.

```python
class TensorBoardPlugin(Plugin):
    def __init__(self, training_loader_len, log_dir):
        self.training_loader_len = training_loader_len
        self.writer = SummaryWriter(log_dir)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `training_loader_len` | int | Number of training batches |
| `log_dir` | str | TensorBoard log directory |

**Static Methods:**

- `init(spec, log_dir="runs/experiment") -> TensorBoardPlugin`: Create plugin instance

### `bz.metrics` - Metrics System

Metrics system for tracking model performance.

#### `Metric`

Abstract base class for all metrics.

```python
class Metric(ABC):
    def __init__(self, name: Optional[str] = None):
        self._name = name
    
    @abstractmethod
    def reset(self) -> None:
        """Reset the metric state."""
        pass
    
    @abstractmethod
    def update(self, preds: Tensor, targets: Tensor) -> None:
        """Update metric with new predictions and targets."""
        pass
    
    @abstractmethod
    def compute(self) -> float:
        """Compute the final metric value."""
        pass
    
    @property
    def name(self) -> str:
        """Get the metric name."""
        return self._name or self.__class__.__name__
```

#### Built-in Metrics

##### `Accuracy`

Classification accuracy metric.

```python
class Accuracy(Metric):
    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        self.correct: int = 0
        self.total: int = 0
```

##### `Precision`

Classification precision metric.

```python
class Precision(Metric):
    def __init__(self, average: str = "micro", name: Optional[str] = None):
        super().__init__(name)
        self.true_positives: int = 0
        self.predicted_positives: int = 0
        self.average = average
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `average` | str | "micro" | Averaging method ('micro', 'macro', 'weighted') |
| `name` | Optional[str] | None | Custom metric name |

##### `Recall`

Classification recall metric.

```python
class Recall(Metric):
    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        self.true_positives: int = 0
        self.actual_positives: int = 0
```

##### `F1Score`

F1 score metric for classification.

```python
class F1Score(Metric):
    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        self.precision_metric = Precision()
        self.recall_metric = Recall()
```

##### `MeanSquaredError`

Mean squared error metric for regression.

```python
class MeanSquaredError(Metric):
    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        self.sum_squared_error: float = 0.0
        self.total: int = 0
```

##### `MeanAbsoluteError`

Mean absolute error metric for regression.

```python
class MeanAbsoluteError(Metric):
    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        self.sum_absolute_error: float = 0.0
        self.total: int = 0
```

##### `TopKAccuracy`

Top-K accuracy metric for classification.

```python
class TopKAccuracy(Metric):
    def __init__(self, k: int = 5, name: Optional[str] = None):
        super().__init__(name)
        self.k = k
        self.correct: int = 0
        self.total: int = 0
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `k` | int | 5 | Number of top predictions to consider |
| `name` | Optional[str] | None | Custom metric name |

#### Metric Registry

##### `METRIC_REGISTRY`

Dictionary mapping metric names to metric classes.

```python
METRIC_REGISTRY = {
    "accuracy": Accuracy,
    "precision": Precision,
    "recall": Recall,
    "f1_score": F1Score,
    "mse": MeanSquaredError,
    "mae": MeanAbsoluteError,
    "top5_accuracy": lambda: TopKAccuracy(k=5),
}
```

##### Functions

- `get_metric(metric_name: str, **kwargs) -> Metric`: Get metric by name
- `list_available_metrics() -> List[str]`: List all available metrics

### `bz.cli` - Command Line Interface

Command line interface module.

#### Functions

- `main()`: Main CLI entry point
- `create_parser() -> argparse.ArgumentParser`: Create command line argument parser
- `run_training(args)`: Run the training process
- `run_validation(args)`: Run model validation
- `run_init(args)`: Initialize project structure
- `load_plugins_from_config(config_manager, training_spec) -> list`: Load plugins from configuration
- `load_metrics_from_config(config_manager, module) -> list`: Load metrics from configuration

#### Classes

##### `TrainingSpecification`

Data class for training specification.

```python
@dataclass
class TrainingSpecification:
    def __init__(
        self,
        model,
        loss_fn,
        optimizer,
        training_loader,
        validation_loader,
        hyperparameters,
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.training_loader = training_loader
        self.validation_loader = validation_loader
        self.hyperparameters = hyperparameters
```

## Utility Functions

### Checkpoint Management

- `find_latest_checkpoint_epoch(checkpoint_dir) -> Optional[int]`: Find the latest checkpoint epoch
- `compute_training_signature(model, optimizer, loss_fn, config) -> str`: Compute training signature for checkpointing

### Configuration

- `load_config(path=None) -> Dict[str, Any]`: Load configuration file (backward compatibility)

## Type Definitions

### Common Types

```python
from typing import Dict, List, Optional, Any, Union
from typing_extensions import TypedDict
from torch import Tensor

class BzConfig(TypedDict, total=False):
    loss_fn: Optional[Any]
    optimizer: Optional[Any]
    model: Optional[Any]
    device: Optional[str]
    epochs: int
    checkpoint_interval: int
    hyperparameters: Dict[str, Any]
    metrics: Dict[str, Any]
    plugins: Dict[str, Any]
```

## Metrics System

### `bz.metrics` - Metrics Module

The metrics system provides a comprehensive set of evaluation metrics for machine learning models.

#### Base Classes

##### `Metric`

Abstract base class for all metrics.

```python
class Metric(ABC):
    def __init__(self, name: Optional[str] = None)
    def reset(self) -> None
    def update(self, preds: Tensor, targets: Tensor) -> None
    def compute(self) -> float
    @property
    def name(self) -> str
```

#### Classification Metrics

##### `Accuracy`

Classification accuracy metric.

```python
class Accuracy(Metric):
    def __init__(self, name: Optional[str] = None)
```

**Usage:**
```python
from bz.metrics import Accuracy

metric = Accuracy()
# For multi-class: preds should be logits (N, C)
# For binary: preds can be logits (N,) or probabilities (N,)
```

##### `Precision`

Classification precision metric.

```python
class Precision(Metric):
    def __init__(self, average: str = "micro", name: Optional[str] = None)
```

**Parameters:**
- `average`: Averaging method ("micro", "macro", "weighted")

##### `Recall`

Classification recall metric.

```python
class Recall(Metric):
    def __init__(self, name: Optional[str] = None)
```

##### `F1Score`

F1 score metric (harmonic mean of precision and recall).

```python
class F1Score(Metric):
    def __init__(self, name: Optional[str] = None)
```

##### `TopKAccuracy`

Top-K accuracy for multi-class classification.

```python
class TopKAccuracy(Metric):
    def __init__(self, k: int = 5, name: Optional[str] = None)
```

**Parameters:**
- `k`: Number of top predictions to consider

#### Regression Metrics

##### `MeanSquaredError`

Mean squared error for regression tasks.

```python
class MeanSquaredError(Metric):
    def __init__(self, name: Optional[str] = None)
```

##### `MeanAbsoluteError`

Mean absolute error for regression tasks.

```python
class MeanAbsoluteError(Metric):
    def __init__(self, name: Optional[str] = None)
```

#### Metric Registry

##### `get_metric(name: str, **kwargs) -> Metric`

Get a metric instance by name.

```python
from bz.metrics import get_metric

accuracy = get_metric("accuracy")
precision = get_metric("precision", average="macro")
top5 = get_metric("top5_accuracy")  # Creates TopKAccuracy with k=5
```

##### `list_available_metrics() -> List[str]`

List all available metric names.

```python
from bz.metrics import list_available_metrics

metrics = list_available_metrics()
# Returns: ['accuracy', 'precision', 'recall', 'f1_score', 'mse', 'mae', 'top5_accuracy']
```

#### Available Metrics

| Metric Name | Class | Description |
|-------------|-------|-------------|
| `accuracy` | `Accuracy` | Classification accuracy |
| `precision` | `Precision` | Classification precision |
| `recall` | `Recall` | Classification recall |
| `f1_score` | `F1Score` | F1 score |
| `mse` | `MeanSquaredError` | Mean squared error |
| `mae` | `MeanAbsoluteError` | Mean absolute error |
| `top5_accuracy` | `TopKAccuracy` | Top-5 accuracy (k=5) |

## Plugin System

### `bz.plugins` - Plugin Module

The plugin system provides extensible hooks into the training lifecycle.

#### Base Classes

##### `Plugin`

Abstract base class for all plugins.

```python
class Plugin(ABC):
    def __init__(self, name: Optional[str] = None, config: Optional[Any] = None)
    def run_stage(self, stage: str, context: PluginContext) -> None
```

**Lifecycle Methods:**
- `start_training_session(context)`: Called at training start
- `load_checkpoint(context)`: Called when loading checkpoint
- `start_epoch(context)`: Called at epoch start
- `start_training_loop(context)`: Called at training loop start
- `start_training_batch(context)`: Called at training batch start
- `end_training_batch(context)`: Called at training batch end
- `end_training_loop(context)`: Called at training loop end
- `start_validation_loop(context)`: Called at validation start
- `start_validation_batch(context)`: Called at validation batch start
- `end_validation_batch(context)`: Called at validation batch end
- `end_validation_loop(context)`: Called at validation end
- `save_checkpoint(context)`: Called when saving checkpoint
- `end_epoch(context)`: Called at epoch end
- `end_training_session(context)`: Called at training end

#### Built-in Plugins

##### `ConsoleOutPlugin`

Provides progress bars and training summaries.

```python
class ConsoleOutPlugin(Plugin):
    def __init__(self, training_data_len: int, validation_data_len: int = 0, update_interval: int = 1)
    
    @classmethod
    def init(cls, spec) -> ConsoleOutPlugin
```

##### `TensorBoardPlugin`

Logs training metrics to TensorBoard.

```python
class TensorBoardPlugin(Plugin):
    def __init__(self, training_loader_len: int, log_dir: str = "runs/experiment")
    
    @classmethod
    def init(cls, spec, log_dir: str = "runs/experiment") -> TensorBoardPlugin
```

##### `EarlyStoppingPlugin`

Automatically stops training when monitored metric stops improving.

```python
class EarlyStoppingPlugin(Plugin):
    def __init__(self, config: Optional[EarlyStoppingConfig] = None)
```

**Configuration:**
```python
@dataclass
class EarlyStoppingConfig:
    enabled: bool = True
    patience: int = 10
    min_delta: float = 0.001
    monitor: str = "validation_loss"
    mode: str = "min"  # "min" or "max"
    restore_best_weights: bool = True
    verbose: bool = True
    baseline: Optional[float] = None
    min_epochs: int = 0
    strategy: str = "patience"  # "patience", "plateau", "custom"
    plateau_factor: float = 0.1
    plateau_patience: int = 10
    plateau_threshold: float = 0.0001
    custom_conditions: Dict[str, Any] = field(default_factory=dict)
```

#### Plugin Registry

##### `PluginRegistry`

Manages plugin discovery and loading.

```python
class PluginRegistry:
    def register(self, name: str, plugin_class: Type[Plugin], config: Optional[Dict[str, Any]] = None) -> None
    def create_plugin(self, name: str, config: Optional[Dict[str, Any]] = None) -> Optional[Plugin]
    def list_plugins(self) -> List[str]
    def unregister(self, name: str) -> None
```

##### Global Functions

```python
def get_plugin_registry() -> PluginRegistry
def register_plugin(name: str, plugin_class: Type[Plugin], config: Optional[Dict[str, Any]] = None) -> None
def create_plugin(name: str, config: Optional[Dict[str, Any]] = None) -> Optional[Plugin]
def list_plugins() -> List[str]
```

## Error Handling

### Common Exceptions

- `ValueError`: Raised for invalid configuration values
- `FileNotFoundError`: Raised when configuration files are not found
- `json.JSONDecodeError`: Raised for malformed JSON configuration files

### Error Recovery

The system includes robust error handling:

- **Plugin errors**: Logged but don't stop training
- **Batch errors**: Failed batches are skipped
- **Checkpoint errors**: Logged but training continues
- **Configuration errors**: Graceful fallback to defaults

## Examples

### Creating a Custom Plugin

```python
from bz.plugins import Plugin

class MyCustomPlugin(Plugin):
    def __init__(self, config=None):
        self.config = config or {}
    
    def start_training_session(self, context):
        print("Training started!")
    
    def end_epoch(self, context):
        if context.training_batch_count > 0:
            avg_loss = context.training_loss_total / context.training_batch_count
            print(f"Epoch {context.epoch} - Loss: {avg_loss:.4f}")
```

### Creating a Custom Metric

```python
from bz.metrics import Metric
import torch

class CustomMetric(Metric):
    def __init__(self, name=None):
        super().__init__(name)
        self.sum = 0.0
        self.count = 0
    
    def reset(self):
        self.sum = 0.0
        self.count = 0
    
    def update(self, preds, targets):
        self.sum += (preds - targets).abs().sum().item()
        self.count += targets.numel()
    
    def compute(self):
        return self.sum / self.count if self.count > 0 else 0.0
```

### Using Configuration

```python
from bz.config import get_config_manager

# Get configuration manager
config_manager = get_config_manager()

# Load configuration
config = config_manager.load()

# Get specific configurations
training_config = config_manager.get_training_config()
plugin_config = config_manager.get_plugin_config("tensorboard")
metrics_config = config_manager.get_metrics_config()
hyperparameters = config_manager.get_hyperparameters()

# Check if plugin is enabled
if config_manager.is_plugin_enabled("tensorboard"):
    print("TensorBoard plugin is enabled")
```
