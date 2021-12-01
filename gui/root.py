import tkinter as tk
from tkinter import ttk
from tkinter import messagebox as MSG
from .training import TrainWindow

class Application(tk.Tk):
    def __init__(self, title) -> None:
        tk.Tk.__init__(self)
        self.title(title)
        self.windows = list()
        
        self.to_trainer = ttk.Button(self, text="Trainer", command=self.start_trainer)
        self.to_trainer.pack()

    def start_trainer(self) -> None:
        if self.windows:
            return
        self.windows.append(TrainWindow(self))
        self.withdraw()

    def destroy_child(self, child) -> None:
        if not self.windows:
            return
        child.destroy()
        self.windows.pop()
        self.deiconify()
