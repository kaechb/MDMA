
```markdown
### README for Thesis Repository: Model Training

#### Overview
This repository contains the necessary scripts and configurations to train machine learning models on particle physics datasets using PyTorch Lightning and various custom utilities. The main script, as provided, handles complex tasks such as setting up models, loading data, and running training procedures with support for logging and checkpointing.

#### Requirements
To run the training script, you need the following:
- Python 3.8 or higher
- PyTorch 1.8 or higher
- PyTorch Lightning
- Wandb for logging and monitoring
- YAML for configuration management
- NumPy, pandas, scipy for data manipulation and computation
- tqdm for progress bars

#### Installation
1. Clone the repository:
   ```bash
   git clone git@github.com:kaechb/MDMA.git
   cd MDMA
   ```
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

#### Configuration
The script uses YAML files for configuration (`hparams/default_[model_name].yaml`). Update these configuration files according to your model and dataset specifics. Parameters include model type, dataset, optimizer settings, and more.

#### Usage
To train a model, navigate to the directory containing the training script and execute:
```bash
python train_script.py [model_name]
```
Replace `[model_name]` with the specific model configuration name you want to use, which corresponds to a YAML file.

#### Key Functions
- `setup_model`: Configures the model based on the YAML settings.
- `train`: Manages the training process, including data loading, model initialization, and training loop execution.
- `setup_scaler_calo`: Configures data scaling based on dataset characteristics.

#### Logging and Monitoring
- **Wandb**: Ensure you have an active Wandb account and your environment is set up to log metrics and outputs. The script initializes Wandb logging and will sync results to your dashboard.
- **TensorBoard**: Local logging with TensorBoard is supported if configured in the YAML file.

#### Checkpoints
The script supports saving and loading from checkpoints, allowing training to be paused and resumed. Checkpoints are saved based on validation metrics, ensuring the best models are preserved.

#### Callbacks
Several PyTorch Lightning callbacks are utilized to enhance training:
- `LearningRateMonitor`: Tracks and logs the learning rate.
- `ModelCheckpoint`: Manages the saving of model states.
- `EMA`: Exponential moving average of model parameters for stable training.

#### Extending the Script
- The training script is designed to be modular. Additional models and configurations can be added by updating the `setup_model` function and the corresponding YAML files.
- For different datasets, ensure appropriate data modules are available and configured.

#### Troubleshooting
- Ensure CUDA devices are properly configured if using GPU acceleration.
- Check Wandb configuration and internet connectivity for remote logging issues.
- Validate data paths and formats to match expected inputs by the models and data loaders.


---

```

