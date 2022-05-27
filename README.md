# KNIFER
An application to train and use GANs

# Usage

Run the main.py file on the top level with python3 to use the GUI.

To use without the GUI :
* Import the TrainingManager class from architectures.manager
* Instanciate the class, with debug=True if needed
* Create a parameters dict for the architecture you wish to train. Look at the models for the parameters you will need.
* Add an "arch" field to it corresponding to any arch present in the manager's source code.
* Add an "img_size" field to it corresponding to your dataset's image definition.
* Call the "set_dataset_folder" on the manager, with the dataset's path as an argument. As per torchvision.dataset.ImageFolder,
the folder you're passing should contain folders whose names are the classes of whatever image is inside.
* Call the "set_trainer" method on the manager, with the parameters dict as an argument.
* Grab a batch from the trainer inside the manager, then provide it to the manager's "proceed" method, along with the batch's ID.

The TrainingManager class provides a simple training loop for CLI usage.

For more insight, look at how the GUI uses the manager.

## Requirements

* pytorch (preferably with cuda support)
* torchvision
