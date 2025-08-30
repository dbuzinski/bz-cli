# Custom Plugins

This example demonstrates how to create custom plugins to extend the functionality of the `bz` CLI.

## Plugin Overview

Plugins allow you to hook into the training lifecycle and add custom functionality such as:
- Custom logging and monitoring
- Integration with external services
- Custom visualization
- Data processing
- Model analysis

## Basic Plugin Structure

All plugins inherit from the `Plugin` base class:

```python
from bz.plugins import Plugin

class MyCustomPlugin(Plugin):
    def __init__(self, config=None):
        self.config = config or {}
    
    def start_training_session(self, context):
        """Called at the start of training."""
        pass
    
    def end_epoch(self, context):
        """Called at the end of each epoch."""
        pass
    
    def end_training_session(self, context):
        """Called at the end of training."""
        pass
```

## Example 1: Simple Logging Plugin

Create a plugin that logs training progress to a file:

```python
import json
import os
from datetime import datetime
from bz.plugins import Plugin

class LoggingPlugin(Plugin):
    def __init__(self, log_file="training_log.json"):
        self.log_file = log_file
        self.log_data = []
    
    def start_training_session(self, context):
        """Log training session start."""
        self.log_data.append({
            "timestamp": datetime.now().isoformat(),
            "event": "training_start",
            "hyperparameters": context.hyperparameters
        })
    
    def end_epoch(self, context):
        """Log epoch completion."""
        if context.training_batch_count > 0:
            training_loss = context.training_loss_total / context.training_batch_count
            training_metrics = context.training_metrics.copy()
            
            epoch_data = {
                "timestamp": datetime.now().isoformat(),
                "event": "epoch_complete",
                "epoch": context.epoch,
                "training_loss": training_loss,
                "training_metrics": training_metrics
            }
            
            if context.validation_batch_count > 0:
                validation_loss = context.validation_loss_total / context.validation_batch_count
                epoch_data.update({
                    "validation_loss": validation_loss,
                    "validation_metrics": context.validation_metrics.copy()
                })
            
            self.log_data.append(epoch_data)
    
    def end_training_session(self, context):
        """Log training session end and save to file."""
        self.log_data.append({
            "timestamp": datetime.now().isoformat(),
            "event": "training_complete",
            "total_epochs": context.epoch
        })
        
        # Save log data to file
        with open(self.log_file, 'w') as f:
            json.dump(self.log_data, f, indent=2)
        
        print(f"Training log saved to {self.log_file}")
```

## Example 2: Email Notification Plugin

Create a plugin that sends email notifications when training completes:

```python
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from bz.plugins import Plugin

class EmailNotificationPlugin(Plugin):
    def __init__(self, email_config):
        self.email_config = email_config
        self.training_start_time = None
        self.best_validation_loss = float('inf')
    
    def start_training_session(self, context):
        """Record training start time."""
        from datetime import datetime
        self.training_start_time = datetime.now()
    
    def end_epoch(self, context):
        """Track best validation loss."""
        if context.validation_batch_count > 0:
            validation_loss = context.validation_loss_total / context.validation_batch_count
            if validation_loss < self.best_validation_loss:
                self.best_validation_loss = validation_loss
    
    def end_training_session(self, context):
        """Send email notification."""
        if self.training_start_time:
            from datetime import datetime
            training_duration = datetime.now() - self.training_start_time
            
            # Create email content
            subject = f"Training Complete - {context.hyperparameters.get('model_name', 'Model')}"
            
            body = f"""
            Training session completed successfully!
            
            Summary:
            - Total epochs: {context.epoch}
            - Training duration: {training_duration}
            - Best validation loss: {self.best_validation_loss:.4f}
            - Final training loss: {context.training_loss_total / context.training_batch_count:.4f}
            
            Training metrics: {context.training_metrics}
            """
            
            self._send_email(subject, body)
    
    def _send_email(self, subject, body):
        """Send email using configured SMTP settings."""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.email_config['from_email']
            msg['To'] = self.email_config['to_email']
            msg['Subject'] = subject
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port'])
            server.starttls()
            server.login(self.email_config['username'], self.email_config['password'])
            server.send_message(msg)
            server.quit()
            
            print(f"Email notification sent to {self.email_config['to_email']}")
        except Exception as e:
            print(f"Failed to send email notification: {e}")
```

## Example 3: Model Checkpoint Plugin

Create a plugin that manages model checkpoints with custom logic:

```python
import os
import torch
from bz.plugins import Plugin

class SmartCheckpointPlugin(Plugin):
    def __init__(self, save_dir="checkpoints", max_checkpoints=5, save_best_only=True):
        self.save_dir = save_dir
        self.max_checkpoints = max_checkpoints
        self.save_best_only = save_best_only
        self.best_validation_loss = float('inf')
        self.checkpoint_files = []
        
        os.makedirs(save_dir, exist_ok=True)
    
    def end_epoch(self, context):
        """Save checkpoint based on validation performance."""
        if context.validation_batch_count > 0:
            validation_loss = context.validation_loss_total / context.validation_batch_count
            
            should_save = False
            if self.save_best_only:
                if validation_loss < self.best_validation_loss:
                    self.best_validation_loss = validation_loss
                    should_save = True
            else:
                should_save = True
            
            if should_save:
                self._save_checkpoint(context, validation_loss)
    
    def _save_checkpoint(self, context, validation_loss):
        """Save model checkpoint."""
        checkpoint_path = os.path.join(
            self.save_dir, 
            f"checkpoint_epoch_{context.epoch}_loss_{validation_loss:.4f}.pth"
        )
        
        # Save checkpoint
        torch.save({
            'epoch': context.epoch,
            'model_state_dict': context.model.state_dict(),
            'optimizer_state_dict': context.optimizer.state_dict(),
            'validation_loss': validation_loss,
            'training_metrics': context.training_metrics,
            'validation_metrics': context.validation_metrics,
            'hyperparameters': context.hyperparameters
        }, checkpoint_path)
        
        self.checkpoint_files.append(checkpoint_path)
        
        # Remove old checkpoints if exceeding max count
        if len(self.checkpoint_files) > self.max_checkpoints:
            old_checkpoint = self.checkpoint_files.pop(0)
            if os.path.exists(old_checkpoint):
                os.remove(old_checkpoint)
        
        print(f"Checkpoint saved: {checkpoint_path}")
```

## Example 4: Custom Visualization Plugin

Create a plugin that generates custom plots during training:

```python
import matplotlib.pyplot as plt
import os
from bz.plugins import Plugin

class VisualizationPlugin(Plugin):
    def __init__(self, plot_dir="plots", update_interval=5):
        self.plot_dir = plot_dir
        self.update_interval = update_interval
        self.training_losses = []
        self.validation_losses = []
        self.epochs = []
        
        os.makedirs(plot_dir, exist_ok=True)
    
    def end_epoch(self, context):
        """Update plots every few epochs."""
        if context.epoch % self.update_interval == 0:
            self._update_plots(context)
    
    def _update_plots(self, context):
        """Generate and save plots."""
        # Collect data
        if context.training_batch_count > 0:
            training_loss = context.training_loss_total / context.training_batch_count
            self.training_losses.append(training_loss)
        
        if context.validation_batch_count > 0:
            validation_loss = context.validation_loss_total / context.validation_batch_count
            self.validation_losses.append(validation_loss)
        
        self.epochs.append(context.epoch)
        
        # Create loss plot
        plt.figure(figsize=(10, 6))
        plt.plot(self.epochs, self.training_losses, label='Training Loss', marker='o')
        if self.validation_losses:
            plt.plot(self.epochs, self.validation_losses, label='Validation Loss', marker='s')
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        plot_path = os.path.join(self.plot_dir, f"loss_plot_epoch_{context.epoch}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create metrics plot
        if context.training_metrics:
            self._plot_metrics(context)
    
    def _plot_metrics(self, context):
        """Plot training metrics."""
        metrics = list(context.training_metrics.keys())
        values = list(context.training_metrics.values())
        
        plt.figure(figsize=(8, 6))
        plt.bar(metrics, values)
        plt.xlabel('Metrics')
        plt.ylabel('Value')
        plt.title(f'Training Metrics - Epoch {context.epoch}')
        plt.xticks(rotation=45)
        
        plot_path = os.path.join(self.plot_dir, f"metrics_plot_epoch_{context.epoch}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
```

## Using Custom Plugins

### Method 1: In Training Script

Add plugins directly in your training script:

```python
from bz import Trainer
from bz.plugins import ConsoleOutPlugin
from my_plugins import LoggingPlugin, EmailNotificationPlugin

# Create trainer
trainer = Trainer()

# Add built-in plugins
trainer.add_plugin(ConsoleOutPlugin.init(training_spec))

# Add custom plugins
trainer.add_plugin(LoggingPlugin("my_training_log.json"))

email_config = {
    'smtp_server': 'smtp.gmail.com',
    'smtp_port': 587,
    'username': 'your_email@gmail.com',
    'password': 'your_app_password',
    'from_email': 'your_email@gmail.com',
    'to_email': 'recipient@example.com'
}
trainer.add_plugin(EmailNotificationPlugin(email_config))

# Train
trainer.train(model, optimizer, loss_fn, training_loader, ...)
```

### Method 2: In Configuration

Enable plugins through configuration:

```json
{
  "plugins": {
    "console_out": {"enabled": true},
    "tensorboard": {"enabled": true},
    "logging_plugin": {
      "enabled": true,
      "config": {
        "log_file": "training_log.json"
      }
    },
    "email_plugin": {
      "enabled": true,
      "config": {
        "smtp_server": "smtp.gmail.com",
        "smtp_port": 587,
        "username": "your_email@gmail.com",
        "password": "your_app_password",
        "from_email": "your_email@gmail.com",
        "to_email": "recipient@example.com"
      }
    }
  }
}
```

## Plugin Lifecycle Hooks

| Hook | Description | When Called |
|------|-------------|-------------|
| `start_training_session` | Training session begins | Before training starts |
| `load_checkpoint` | Checkpoint loaded | When resuming from checkpoint |
| `start_epoch` | Epoch begins | At start of each epoch |
| `start_training_loop` | Training loop begins | Before training batches |
| `start_training_batch` | Training batch begins | Before each training batch |
| `end_training_batch` | Training batch ends | After each training batch |
| `end_training_loop` | Training loop ends | After all training batches |
| `start_validation_loop` | Validation begins | Before validation |
| `start_validation_batch` | Validation batch begins | Before each validation batch |
| `end_validation_batch` | Validation batch ends | After each validation batch |
| `end_validation_loop` | Validation ends | After all validation batches |
| `save_checkpoint` | Checkpoint saved | When checkpoint is saved |
| `end_epoch` | Epoch ends | At end of each epoch |
| `end_training_session` | Training session ends | After training completes |

## Best Practices

1. **Keep plugins focused**: Each plugin should have a single responsibility
2. **Handle errors gracefully**: Don't let plugin errors crash training
3. **Use configuration**: Make plugins configurable
4. **Document your plugins**: Include docstrings and examples
5. **Test your plugins**: Write tests for custom plugins
6. **Use context data**: Access training information from the context object
7. **Clean up resources**: Close files, connections, etc. in `end_training_session`

## Next Steps

- Check out the [API Reference](../reference.md) for detailed plugin documentation
