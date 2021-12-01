import tkinter as tk
from tkinter import ttk, StringVar
from tkinter import messagebox as MSG
import tkinter.filedialog as tkfd
from architectures.common import TrainingManager
from .widgets import ButtonL, ComboboxL, HyperParameterFrame, IntSpinboxL, Pow2SpinboxL

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
        self.filemenu.add_command(label="Load model...")
        self.filemenu.add_command(label="Save trained model as...")

        self.menubar.add_cascade(label="File", menu=self.filemenu)

        self.do_train_button = ttk.Button(master=self, text="Train", command=self.do_train, state="disabled")
        self.stop_train_button = ttk.Button(master=self, text="Stop", command=self.stop_train, state="disabled")

        self.do_train_button.pack()
        self.stop_train_button.pack()

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
        self.do_train_button.configure(state="normal")
    
    def train_loop(self, data=None):
        if self.dotrain:
            if self.current_batch == 0:
                self.current_epoch += 1
                data = self.grab_data()
            if not data:
                data = self.current_data
            self.current_batch = self.manager.proceed(data, self.current_batch)
            self.after(100, lambda: self.train_loop(data))
        else:
            self.current_data = data

    def clear_manager(self):
        self.manager = TrainingManager()

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
        self.manager.set_trainer(*args, **kwargs)
        self.do_train_button.configure(state="normal")
    
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
            values=("DCGAN"),
        )
        self.trainers_box.combobox.current(0)
        self.trainers_box.combobox.configure(state="readonly")
        self.img_size_box = Pow2SpinboxL(master=self, label="Image Size")
        self.batch_size_box = IntSpinboxL(master=self, label="Batch Size", init_value=16)
        self.latent_size_box = IntSpinboxL(master=self, label="Latent Space Size", init_value=100)
        self.hyp_frame = HyperParameterFrame(master=self)

        self.ok_btn = ButtonL(master=self, text="OK", command=self.exit_and_new)

        self.trainers_box.pack()
        self.img_size_box.pack()
        self.batch_size_box.pack()
        self.latent_size_box.pack()
        self.hyp_frame.pack()
        self.ok_btn.pack()

    def exit_and_new(self):
        self.master.setup_training(
            self.trainers_box.combobox.get(), 
            self.img_size_box.get_value(),
            self.batch_size_box.get_value(),
            self.latent_size_box.get_value(),
            self.hyp_frame.get_params(),
        )
        self.destroy()

class TrainerView(tk.Frame):
    def __init__(self, master = None, trainer = None):
        self.trainer = trainer
