# KNIFER
An application to train and use GANs

# Usage

```python3 -i main.py --help``` for a list of arguments.

Upon launching the script in interactive mode, you're given a tm variable which is an instance of TrainingManager.
This instance is fully initialized.

You may simply use tm.simple_training_loop(n_epochs) to start training your chosen arch.
You're also given a run_manager_n_times that runs simple_training_loop multiple times, saving a checkpoint and synthetizing some images,
at the timestep of your choosing.

## Requirements

* pytorch (preferably with cuda support)
* torchvision
