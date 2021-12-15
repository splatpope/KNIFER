import tkinter as tk
from tkinter import ttk, StringVar
from tkinter import messagebox as MSG
import tkinter.filedialog as tkfd
from architectures.manager import TrainingManager
from .widgets import ButtonL, ComboboxL, MiscParameterFrame, IntSpinboxL, Pow2SpinboxL

KNIFER_DEBUG = True

class TrainWindow(tk.Toplevel):
    def __init__(self, master=None):
        tk.Toplevel.__init__(self, master)
        self.title("GAN Training")
        self.current_batch = 0
        self.current_epoch = -1

        self.menubar = tk.Menu(self)
        self.config(menu=self.menubar)
        self.filemenu = tk.Menu(self.menubar, tearoff=0)
        self.filemenu.add_command(label="Load Dataset...", command=self.load_dset)
        self.filemenu.add_command(label="New model...", command=self.new_model)
        self.filemenu.add_command(label="Load savestate...", command=self.load_state)
        self.filemenu.add_command(label="Quick save", command=self.save_state)

        self.menubar.add_cascade(label="File", menu=self.filemenu)

        self.do_train_button = ttk.Button(master=self, text="Train", command=self.do_train, state="disabled")
        self.stop_train_button = ttk.Button(master=self, text="Stop", command=self.stop_train, state="disabled")
        self.viz_button = ttk.Button(master=self, text="Visualize", command=self.visualize, state="disabled")

        self.do_train_button.pack()
        self.stop_train_button.pack()
        self.viz_button.pack()

        self.protocol("WM_DELETE_WINDOW", self.exit)

        self.image_size = 64
        self.batch_size = 16
        self.latent_size = 100
        self.clear_manager()

        self.dotrain = False
    
    def do_train(self):
        self.dotrain = True
        self.do_train_button.configure(state="disabled")
        self.stop_train_button.configure(state="normal")
        self.train_loop()

    def stop_train(self):
        self.dotrain = False
        self.stop_train_button.configure(state="disabled")
        self.do_train_button.configure(state="normal", text="Resume")
    
    def train_loop(self, data=None):
        if self.dotrain:
            if self.current_batch == 0:
                self.current_epoch += 1
                data = self.grab_data()
            if not data:
                data = self.current_data
            self.current_batch = self.manager.proceed(data, self.current_batch)
            ## TODO : adapt waiting time to processing time so very fast learning is not stunted
            self.after(100, lambda: self.train_loop(data))
        else:
            self.current_data = data

    def clear_manager(self):
        self.manager = TrainingManager(debug=KNIFER_DEBUG)

    def load_dset(self):
        folder = tkfd.askdirectory(initialdir=".", mustexist=True, title="Pick a dataset folder")
        if not folder:
            MSG.showwarning(message="No dataset selected !")
        self.manager.set_dataset_folder(folder)
        
    def new_model(self):    
        if not self.manager.dataset_folder:
            MSG.showerror(message="Please load a dataset first !")
            return
        NewModelWindow(self)

    def setup_training(self, *args, **kwargs):
        self.current_batch = 0
        self.current_epoch = -1
        self.manager.set_trainer(*args, **kwargs)
        self.do_train_button.configure(state="normal", text="Train")
        self.viz_button.configure(state="normal")

    def load_state(self):
        state_file = tkfd.askopenfilename(initialdir="./savestates", title="Pick a savestate...")
        if not state_file:
            return
        self.current_batch = self.manager.load(state_file)
        print(self.current_batch)
        self.current_epoch = self.manager.epoch
        if self.current_batch == 0:
            self.current_epoch -= 1
        else:
            self.current_data = self.grab_data()
            ## Skip batches to resume training 
            for i in range(self.current_batch):
                next(self.current_data)
        self.do_train_button.configure(state="normal")
        self.viz_button.configure(state="normal")
    
    def save_state(self):
        self.manager.save()

    def visualize(self):
        self.manager.synthetize_viz()

    def grab_data(self):
        return iter(self.manager.trainer.data)

    def exit(self):
        self.master.destroy_child(self)

class NewModelWindow(tk.Toplevel):
    def __init__(self, master=None):
        if not isinstance(master, TrainWindow):
            raise TypeError
        tk.Toplevel.__init__(self, master)
        self.title("New Model")

        self.trainers_box = ComboboxL(master=self, label="Architecture",
            values=("DCGAN", "WGAN_GP"),
        )
        self.trainers_box.combobox.current(0)
        self.trainers_box.combobox.configure(state="readonly")
        self.img_size_box = Pow2SpinboxL(master=self, label="Image Size")
        self.batch_size_box = IntSpinboxL(master=self, label="Batch Size", init_value=16)
        self.latent_size_box = IntSpinboxL(master=self, label="Latent Space Size", init_value=100)
        self.misc_parameters_frame = MiscParameterFrame(master=self)

        self.ok_btn = ButtonL(master=self, text="OK", command=self.exit_and_new)

        self.trainers_box.pack()
        self.img_size_box.pack()
        self.batch_size_box.pack()
        self.latent_size_box.pack()
        self.misc_parameters_frame.pack()
        self.ok_btn.pack()

    def exit_and_new(self):
        params = {
            "arch": self.trainers_box.combobox.get(),
            "img_size": self.img_size_box.get_value(),
            "batch_size": self.batch_size_box.get_value(),
            "latent_size": self.batch_size_box.get_value(),
        }
        params.update(self.misc_parameters_frame.get_params())
        self.master.setup_training(params)
        self.destroy()

class TrainerView(tk.Frame):
    def __init__(self, master = None, trainer = None):
        self.trainer = trainer
