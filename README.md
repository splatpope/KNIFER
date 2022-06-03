# KNIFER
An application to train and use GANs

## Usage

```python3 -i main.py --help``` for a list of arguments.

Use the *-c* or *--config_file* argument to load a config file describing your desired GAN arch. Look in the *config* folder for more details.

Upon launching the script in interactive mode, you're given a tm variable which is an instance of TrainingManager.
This instance is fully initialized.

You may simply use ```tm.simple_training_loop(n_epochs)``` to start training your chosen arch.
You're also given ```run_manager_n_times``` that runs ```simple_training_loop``` multiple times, saving a checkpoint and synthetizing some images,
at the timestep of your choosing.

If you want to use the app as a script (without interactive mode), just make your own script importing the ```TrainingManager``` class and instanciate it like in *main.py*.

## Requirements

* Python 3
    With packages:
* pytorch (preferably with cuda support)
* torchvision
* scipy
