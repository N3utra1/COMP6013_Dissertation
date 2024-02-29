import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Load your numpy array here
spectrogram = np.random.random((100, 100))

def create_circle(canvas, x, y, r, color='grey'):
    return canvas.create_oval(x-r, y-r, x+r, y+r, fill=color)

def cycle_back():
    pass

def cycle_forward():
    pass

def dropdown_callback(*args):
    print("called with args:")
    selected_option = dropdown_var.get()
    print(selected_option)
    colors = {'ictal': 'red', 'interictal': 'green', 'preictal': 'amber'}
    for i in range(2):
        for j in range(3):
            color = colors[selected_option] if i == 1 and colors[selected_option] == j else 'grey'
            create_circle(canvas, 50 + i*100, 50 + j*50, 20, color=color)

root = tk.Tk()

fig, ax = plt.subplots()
ax.specgram(spectrogram)
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.draw()
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

button_frame = tk.Frame(root)
button_frame.pack(side=tk.TOP)

button1 = tk.Button(button_frame, text="<", command=cycle_back)
button1.pack(side=tk.LEFT)

button2 = tk.Button(button_frame, text=">", command=cycle_forward)
button2.pack(side=tk.LEFT)

dropdown_var = tk.StringVar(root)
dropdown_var.trace_add("write", dropdown_callback)
dropdown = ttk.Combobox(button_frame, textvariable=dropdown_var, state="readonly")
dropdown['values'] = ('interictal', 'preictal', 'ictal')
dropdown.pack(side=tk.TOP)
dropdown.set("interictal")

# No need to create a new Canvas widget here

# Create traffic lights
for i in range(2):
    for j in range(3):
        create_circle(canvas, 50 + i*100, 50 + j*50, 20)

root.mainloop()