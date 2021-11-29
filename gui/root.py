import tkinter as tk

class Application(tk.Tk):
    def __init__(self, title):
        tk.Tk.__init__(self)
        self.title(title)
        