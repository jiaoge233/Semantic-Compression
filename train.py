# my_controlnet_train.py
import argparse
import os
import sys
import resource
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
import torch
import torch.distributed as dist

def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0

# Attempt to increase the system's file descriptor limit
try:
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
    if is_main_process():
        print(f"File descriptor limit raised from {soft} to {hard}.")
except Exception as e:
    if is_main_process():
        print(f"Warning: Could not raise file descriptor limit: {e}. This might be required on some systems.")

# Add project root to Python path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if is_main_process():
    print(f"Project root added to sys.path: {PROJECT_ROOT}")

try:
    from model.my_cldm_light import ControlLDM
    from model.logger import ImageLogger
except ImportError as e:
    if is_main_process():
        print(f"Error: Could not import required modules: {e}")
        print("Please ensure that `model/my_cldm_light.py` exists and all dependencies are installed.")
    sys.exit(1)

try:
    from Dataset import MyControlNetDataset
except ImportError as e:
    if is_main_process():
        print(f"Error: Could not import MyControlNetDataset from Dataset.py: {e}")
        print("Please create `Dataset.py` and implement the `MyControlNetDataset` class for your data.")
    sys.exit(1)


def load_state_dict_from_checkpoint(ckpt_path, location='cpu'):
    """Loads pretrained model weights, handling .safetensors and .ckpt/.pth"""
    _, extension = os.path.splitext(ckpt_path)
    if extension.lower() == ".safetensors":
        import safetensors.torch
        if is_main_process(): print(f"Loading with safetensors: {ckpt_path}")
        state_dict = safetensors.torch.load_file(ckpt_path, device=location)
    else:
        if is_main_process(): print(f"Loading with torch.load: {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location=torch.device(location))
    
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    return state_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config_path',
        type=str,
        default='./configs/my_scribble_config.yaml',
        help='Path to your training configuration YAML file'
    )
    parser.add_argument(
        '--resume_path',
        type=str,
        default=None,
        help='Path to resume training from a checkpoint'
    )
    parser.add_argument(
        '--sd_model_path',
        type=str,
        default='./ControlNet-main/models/v1-5-pruned.ckpt', # Modify to your actual path
        help='Path to the base Stable Diffusion model weights (.ckpt or .safetensors)'
    )
    parser.add_argument('--gpus', nargs='+', type=int, help='List of GPU IDs to use (overrides config), e.g., --gpus 0 1')
    parser.add_argument('--max_epochs', type=int, help='Maximum training epochs (overrides config)')
    parser.add_argument('--base_lr', type=float, help='Base learning rate (overrides config)')
    parser.add_argument('--batch_size_per_gpu', type=int, help='Batch size per GPU (overrides data.params.batch_size in config)')


    args = parser.parse_args()

    # 1. Load configuration
    if not os.path.exists(args.config_path):
        print(f"Error: Config file {args.config_path} not found!")
        sys.exit(1)
    
    config = OmegaConf.load(args.config_path)
    if is_main_process():
        print(f"Loaded configuration from {args.config_path}")

    # Override config settings with command-line arguments
    if args.gpus is not None:
        config.lightning.trainer.devices = args.gpus
        if is_main_process(): print(f"Overriding GPU devices with: {config.lightning.trainer.devices}")
    if args.max_epochs is not None:
        config.lightning.trainer.max_epochs = args.max_epochs
        if is_main_process(): print(f"Overriding max_epochs with: {config.lightning.trainer.max_epochs}")
    if args.base_lr is not None:
        config.training.base_learning_rate = args.base_lr
        if is_main_process(): print(f"Overriding base_learning_rate with: {config.training.base_learning_rate}")
    if args.batch_size_per_gpu is not None:
        config.data.params.batch_size = args.batch_size_per_gpu
        if hasattr(config.data, 'validation_params') and config.data.validation_params:
             config.data.validation_params.batch_size = args.batch_size_per_gpu # also modify validation batch size
        if is_main_process(): print(f"Overriding batch_size (per GPU) with: {config.data.params.batch_size}")


    # 2. Initialize Model (ControlLDM)
    if is_main_process():
        print("Initializing ControlLDM model...")
    # TODO: [Critical] Ensure the class pointed to by `student_control_stage_config.target` (e.g., cldm.cldm.StudentControlNet)
    # is defined in your ControlNet-main/cldm/cldm.py or another importable path.
    # The original cldm.py may not contain StudentControlNet.
    if "student_control_stage_config" not in config.model.params:
        if is_main_process(): print("Warning: `student_control_stage_config` not found in config. This is required for knowledge distillation.")

    try:
        # Make sure model.target in YAML points to our modified class
        if config.model.target != 'model.my_cldm_light.ControlLDM':
            if is_main_process():
                print(f"Warning: model.target in config is '{config.model.target}', but 'model.my_cldm_light.ControlLDM' is expected for full features.")
                print("         Attempting to use the configured target, but this may not enable vision guidance.")

        model = ControlLDM(**config.model.params)
        if is_main_process():
            print("ControlLDM model initialized successfully.")

    except Exception as e:
        if is_main_process():
            print(f"Fatal error during ControlLDM initialization: {e}")
            print("Possible causes:")
            print("1. The class specified in `student_control_stage_config.target` could not be found or imported.")
            print("   Check the path in your YAML file and ensure the class exists.")
            print("2. Incorrect or missing model parameters in the YAML config.")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    # Set training-related attributes
    model.learning_rate = config.training.base_learning_rate
    model.sd_locked = config.training.get('sd_locked', True) # Get from config, default to True
    model.only_mid_control = config.model.params.get('only_mid_control', False)
    
    if is_main_process():
        print(f"Model LR: {model.learning_rate}, SD Locked: {model.sd_locked}, Mid-Control Only: {model.only_mid_control}")

    # Load Stable Diffusion pretrained weights
    if args.sd_model_path and os.path.exists(args.sd_model_path):
        if is_main_process(): print(f"Loading base SD model weights from: {args.sd_model_path}")
        sd_state_dict = load_state_dict_from_checkpoint(args.sd_model_path, location='cpu')
        filtered_sd_state_dict = {k: v for k, v in sd_state_dict.items() if not k.startswith('control_model.')}
        missing_keys, unexpected_keys = model.load_state_dict(filtered_sd_state_dict, strict=False)
        if is_main_process(): print(f"SD weights loaded.")
        if missing_keys and is_main_process():
            print(f"  Missing SD keys (sample): {missing_keys[:5]}")
        if unexpected_keys and is_main_process():
            print(f"  Unexpected SD keys (sample): {unexpected_keys[:5]}")
        del sd_state_dict, filtered_sd_state_dict
    else:
        if is_main_process(): print(f"Warning: SD model path '{args.sd_model_path}' not provided or not found. The model might start from random weights.")

    # 3. Initialize Datasets and DataLoaders
    if is_main_process(): print("Initializing datasets and dataloaders...")
    try:
        # Get common data parameters from YAML, e.g., image resolution
        # Your MyControlNetDataset __init__ should handle these parameters
        common_data_params_config = config.data.params.copy() # Copy to avoid modifying original config
        batch_size_train = common_data_params_config.pop('batch_size', 1)
        num_workers_train = common_data_params_config.pop('num_workers', 0)
        persistent_workers_train = common_data_params_config.pop('persistent_workers', False)
        json_path_train = common_data_params_config.pop('json_file_path')
        # common_data_params_config now contains other params like image_resolution

        dataset_train = MyControlNetDataset(
            json_file_path=json_path_train,
            **common_data_params_config # Pass remaining params like image_resolution
        )
        
        dataloader_train = DataLoader(
            dataset_train,
            num_workers=num_workers_train,
            batch_size=batch_size_train,
            shuffle=True,
            persistent_workers=persistent_workers_train if num_workers_train > 0 else False,
            pin_memory=True,
        )
        if is_main_process():
            print(f"Train Dataloader: {len(dataset_train)} samples, BS={batch_size_train}, Workers={num_workers_train}")

        dataloader_val = None
        if hasattr(config.data, 'validation_params') and config.data.validation_params:
            val_config = config.data.validation_params.copy()
            batch_size_val = val_config.pop('batch_size', 1)
            num_workers_val = val_config.pop('num_workers', 0)
            persistent_workers_val = val_config.pop('persistent_workers', False)
            json_path_val = val_config.pop('json_file_path')
            # val_config now contains other params for validation

            dataset_val = MyControlNetDataset(
                json_file_path=json_path_val,
                **val_config # Pass remaining params like image_resolution
            )
            dataloader_val = DataLoader(
                dataset_val,
                num_workers=num_workers_val,
                batch_size=batch_size_val,
                shuffle=False,
                persistent_workers=persistent_workers_val if num_workers_val > 0 else False,
                pin_memory=True,
            )
            if is_main_process():
                print(f"Validation Dataloader: {len(dataset_val)} samples, BS={batch_size_val}, Workers={num_workers_val}")
        else:
            if is_main_process(): print("No validation dataset configured.")

    except Exception as e:
        if is_main_process():
            print(f"Error initializing dataset or dataloader: {e}")
            print("Please check:")
            print("1. Your `MyControlNetDataset` implementation in `Dataset.py`.")
            print("2. The paths and parameters in `data.params` and `data.validation_params` in your YAML config.")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 4. Initialize Callbacks
    if is_main_process(): print("Initializing callbacks...")
    callbacks = []
    if hasattr(config.lightning, 'callbacks'):
        for cb_name, cb_conf_dict in config.lightning.callbacks.items():
            cb_conf = OmegaConf.to_container(cb_conf_dict, resolve=True) # Convert OmegaConf to a plain dict
            target_path = cb_conf.get('target')
            params = cb_conf.get('params', {})
            if not target_path:
                if is_main_process(): print(f"Warning: Callback {cb_name} is missing a target path, skipping.")
                continue
            
            try:
                module_path, class_name = target_path.rsplit('.', 1)
                module = __import__(module_path, fromlist=[class_name])
                callback_class = getattr(module, class_name)
                
                if class_name == "ModelCheckpoint":
                    dirpath = params.get('dirpath', 'checkpoints/')
                    os.makedirs(dirpath, exist_ok=True)
                    # ModelCheckpoint filename might contain slashes, ensure directory exists
                    if 'filename' in params and '/' in params['filename']:
                        full_dir = os.path.join(dirpath, os.path.dirname(params['filename']))
                        os.makedirs(full_dir, exist_ok=True)
                
                callbacks.append(callback_class(**params))
                if is_main_process(): print(f"Added callback: {target_path}")
            except Exception as e:
                if is_main_process(): print(f"Warning: Could not initialize callback {cb_name} ({target_path}): {e}")
    
    callbacks.append(pl.callbacks.LearningRateMonitor(logging_interval='step'))
    if is_main_process(): print("Added LearningRateMonitor callback.")


    # 5. Initialize PyTorch Lightning Trainer
    if is_main_process(): print("Initializing PyTorch Lightning Trainer...")
    trainer_params = OmegaConf.to_container(config.lightning.get('trainer', {}), resolve=True)
    
    # Set up Logger (if not configured in YAML)
    if 'logger' not in trainer_params or trainer_params.get('logger') is False:
        default_save_dir = trainer_params.get('default_root_dir', 'lightning_logs')
        os.makedirs(default_save_dir, exist_ok=True)
        version_name = "my_controlnet_run"
        logger = pl.loggers.TensorBoardLogger(save_dir=default_save_dir, name=None, version=version_name)
        trainer_params['logger'] = logger
        if is_main_process(): print(f"Using default TensorBoardLogger. Logs will be saved to: {logger.log_dir}")
    elif isinstance(trainer_params.get('logger'), (dict, OmegaConf)):
        logger_conf_dict = trainer_params['logger']
        logger_conf = OmegaConf.to_container(logger_conf_dict, resolve=True) if isinstance(logger_conf_dict, OmegaConf) else logger_conf_dict
        target_path = logger_conf.get('target')
        params = logger_conf.get('params', {})
        if target_path:
            try:
                module_path, class_name = target_path.rsplit('.', 1)
                module = __import__(module_path, fromlist=[class_name])
                logger_class = getattr(module, class_name)
                if 'save_dir' not in params:
                    params['save_dir'] = trainer_params.get('default_root_dir', 'lightning_logs')
                os.makedirs(params['save_dir'], exist_ok=True)
                trainer_params['logger'] = logger_class(**params)
                if is_main_process(): print(f"Using logger from config: {target_path}")
            except Exception as e:
                if is_main_process(): print(f"Warning: Could not initialize logger from config ({target_path}): {e}. Falling back to default.")
                default_save_dir = trainer_params.get('default_root_dir', 'lightning_logs')
                os.makedirs(default_save_dir, exist_ok=True)
                trainer_params['logger'] = pl.loggers.TensorBoardLogger(save_dir=default_save_dir, name="my_controlnet_fallback_run")
        else:
            if is_main_process(): print("Logger config incomplete (missing target), falling back to default.")
            default_save_dir = trainer_params.get('default_root_dir', 'lightning_logs')
            os.makedirs(default_save_dir, exist_ok=True)
            trainer_params['logger'] = pl.loggers.TensorBoardLogger(save_dir=default_save_dir, name="my_controlnet_incomplete_logger_config_run")


    try:
        trainer = pl.Trainer(callbacks=callbacks, **trainer_params)
        if is_main_process(): print("PyTorch Lightning Trainer initialized successfully.")
    except Exception as e:
        if is_main_process():
            print(f"Error initializing PyTorch Lightning Trainer: {e}")
            print("Please check the 'lightning.trainer' section in your config file.")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 6. Start Training
    if is_main_process():
        print("\n" + "="*40)
        print("Starting training...")
        print(f"  - Config: {args.config_path}")
        print(f"  - Resume from: {args.resume_path or 'Not specified'}")
        print(f"  - Log Dir: {trainer.logger.save_dir}")
        print(f"  - Model: {config.model.target}")
        print(f"  - Batch Size (per GPU): {config.data.params.batch_size}")
        print(f"  - Epochs: {trainer.max_epochs}, Steps: {trainer.max_steps}")
        print(f"  - Learning Rate: {model.learning_rate}")
        print(f"  - Devices: {trainer.num_devices} ({trainer.device_ids})")
        print(f"  - Precision: {trainer.precision}")
        print("="*40 + "\n")
    
    resume_from_checkpoint = args.resume_path
    if resume_from_checkpoint and resume_from_checkpoint.lower() == 'last':
        if is_main_process(): print("Attempting to resume from the latest 'last.ckpt'.")
    elif resume_from_checkpoint and not os.path.exists(resume_from_checkpoint):
        if is_main_process(): print(f"Warning: Checkpoint path '{resume_from_checkpoint}' does not exist. Starting from scratch.")
        resume_from_checkpoint = None
    elif resume_from_checkpoint:
         if is_main_process(): print(f"Resuming training from checkpoint: {resume_from_checkpoint}")
    else:
        if is_main_process(): print("Starting a new training run (not resuming).")
    
    try:
        trainer.fit(
            model,
            train_dataloaders=dataloader_train,
            val_dataloaders=dataloader_val,
            ckpt_path=resume_from_checkpoint 
        )
        if is_main_process(): print("Training finished.")
    except Exception as e:
        if is_main_process():
            print(f"A fatal error occurred during training: {e}")
            import traceback
            traceback.print_exc()
            print("="*40 + "\n")

if __name__ == '__main__':
    pl.seed_everything(42, workers=True)
    
    if is_main_process():
        print("\n" + "-"*20 + " Starting ControlNet Training Script " + "-"*20)
        print("Please ensure the following are correctly configured:")
        print("1. YAML config file (path provided via --config_path).")
        print("2. `Dataset.py` is implemented correctly.")
        print("3. All paths to models and data in the config are correct.")
        print("4. All dependencies are installed.")
        print("-"*(59) + "\n")

    main()