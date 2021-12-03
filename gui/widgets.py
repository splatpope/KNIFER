from tkinter import StringVar, ttk
import tkinter.messagebox as MSG

def _is_pow2(val: int) -> bool:
    return (val & (val - 1)) and val != 0

## These are helper widgets combining a standard widgets and a label
# kinda amazing that such widgets aren't available by default
class ButtonL(ttk.Frame):
    def __init__(self, master=None, label="", text="", command=None):
        ttk.Frame.__init__(self, master)

        self.label = ttk.Label(master=self, text=label)
        self.button = ttk.Button(master=self, text=text, command=command)

        self.label.grid(
            row=0, column=0
        )
        self.button.grid(
            row=0, column=1
        )

class ComboboxL(ttk.Frame):
    def __init__(self, master=None, label="", values=None):
        ttk.Frame.__init__(self, master)

        self.label = ttk.Label(master=self, text=label)
        self.combobox = ttk.Combobox(master=self, values=values)

        self.label.grid(
            row=0, column=0,
        )
        self.combobox.grid(
            row=0, column=1,
        )

class SpinboxL(ttk.Frame):
    def __init__(self, master=None, label=""):
        ttk.Frame.__init__(self, master)
        ## Create inner widgets
        self.side_label = ttk.Label(master=self, text=label)
        self.value_box = ttk.Spinbox(master=self)
        ## Layout
        self.side_label.grid(
            row=0, column=0,
        )
        self.value_box.grid(
            row=0, column=1,
        )

        def get_value(self):
            return self.value_box.get()

class IntSpinboxL(SpinboxL):
    def __init__(self, master=None, label="", low_value=0, max_value=100, init_value=None):
        SpinboxL.__init__(self, master, label)
        self.value = StringVar(master=self, value=str(init_value) if init_value != None else "")
        self.value_box.configure(from_=low_value, to=max_value, increment=1, textvariable=self.value)

    def get_value(self):
            return int(self.value.get())

## TODO : FloatSpinboxL and more

class Pow2SpinboxL(IntSpinboxL):
    def __init__(self, master=None, label="", low_value = 0, max_value=1024, init_value=64):
        IntSpinboxL.__init__(self, master, label, low_value, max_value, init_value)
        self.max_val = max_value
        ## Reset boundaries if they are not powers of 2
        if not _is_pow2(init_value):
            init_value = 64
        self.value.set(str(init_value))
        if not _is_pow2(max_value):
            max_val = 1024
        ## Create spinbox values
        i = 1
        values = list()
        while i <= max_val:
            values.append(i)
            i *= 2
        ## Update the spinbox
        self.value_box.configure(
            from_=None, to=None, increment=None,
            values=tuple(values), textvariable=self.value, state="readonly",
        )

## TODO add different flavors for each arch
# make sure that the training window only gets the parameters it needs to not clutter the manager
class MiscParameterFrame(ttk.Frame):
    def __init__(self, master=None):
        ttk.Frame.__init__(self, master)
        self.lr_var = StringVar(master=self, value=str(2e-4))
        self.b1_var = StringVar(master=self, value=str(0.5))
        self.b2_var = StringVar(master=self, value=str(0.999))

        self.lr_label = ttk.Label(master=self, text="Learning Rate")
        self.lr_box = ttk.Spinbox(master=self,
            from_=0.0001,
            to=1.0,
            increment=0.00001,
            textvariable=self.lr_var,
        )
        self.b1_label = ttk.Label(master=self, text="Adam momentum 1")
        self.b1_box = ttk.Spinbox(master=self,
            from_=0.001,
            to=1.0,
            increment=0.0001,
            textvariable=self.b1_var,
        )
        self.b2_label = ttk.Label(master=self, text="Adam momentum 2")
        self.b2_box = ttk.Spinbox(master=self,
            from_=0.001,
            to=1.0,
            increment=0.0001,
            textvariable=self.b2_var,
        )
        self.critic_iters_box = IntSpinboxL(master=self,
            label="Critic Iterations",
            low_value=1,
            max_value=100,
            init_value=5,
        )
        self.lambda_gp_box = IntSpinboxL(master=self,
            label="Lambda (GP)",
            low_value=1,
            max_value=100,
            init_value=10,
        )

        self.lr_label.grid(
            row=0, column=0,
            sticky='we',
        )
        self.lr_box.grid(
            row=0, column=1,
            sticky='we',
        )
        self.b1_label.grid(
            row=1, column=0,
            sticky='we',
        )
        self.b1_box.grid(
            row=1, column=1,
            sticky='we',
        )
        self.b2_label.grid(
            row=2, column=0,
            sticky='we',
        )
        self.b2_box.grid(
            row=2, column=1,
            sticky='we',
        )
        self.critic_iters_box.grid(
            row=3, column=0,
            sticky='we',
            columnspan=2,
        )
        self.lambda_gp_box.grid(
            row=4, column=0,
            sticky='we',
            columnspan=2,
        )

    def get_params(self):
        return {
            "lr": float(self.lr_var.get()),
            "b1": float(self.b1_var.get()),
            "b2": float(self.b2_var.get()),
            "critic_iters": self.critic_iters_box.get_value(),
            "lambda_gp": self.lambda_gp_box.get_value(),
        }
