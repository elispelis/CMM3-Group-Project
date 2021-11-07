import tkinter as tk
from tkinter import *

def put_in(D, time_max, h, Xmin, Xmax, Ymin, Ymax, Nx, Ny, V):
    return
root = Tk()
root.title("project's interface")
fw = 512
fh = 382
sw = root.winfo_screenwidth()
sh = root.winfo_screenheight()
x = sw / 2 - fw / 2
y = sh / 2 - fh / 2
root.geometry("%dx%d+%d+%d" % (fw, fh, x, y))
photo = tk.PhotoImage(file = "python.png")
tk.Label(root, image = photo).place(relx=0,rely=0)
l = tk.Label(root,
text='Please input the parameters required in',
bg='black',
fg='white',
font=('Arial', 12),
width=45, height=2
).place(relx=0.1,rely=0)
tk.Label(root, text="Diffusivity:").place(rely=0.11,relwidth=0.2,relheight=0.05)
D = tk.Entry(root).place(rely=0.11,relx=0.8,relwidth=0.2,relheight=0.05)
tk.Label(root,text="Timestep h:").place(rely=0.16,relwidth=0.2,relheight=0.05)
h = tk.Entry(root).place(rely=0.16,relx=0.8,relwidth=0.2,relheight=0.05)
tk.Label(root,text="maximum solution time:").place(rely=0.21,relwidth=0.2,relheight=0.05)
time_max = tk.Entry(root).place(rely=0.21,relx=0.8,relwidth=0.2,relheight=0.05)
tk.Label(root,text="Nx:").place(rely=0.26,relwidth=0.2,relheight=0.05)
Nx = tk.Entry(root).place(rely=0.26,relx=0.8,relwidth=0.2,relheight=0.05)
tk.Label(root,text="Ny:").place(rely=0.31,relwidth=0.2,relheight=0.05)
Ny = tk.Entry(root).place(rely=0.31,relx=0.8,relwidth=0.2,relheight=0.05)
tk.Label(root,text="Xmin:").place(rely=0.36,relwidth=0.2,relheight=0.05)
Xmin = tk.Entry(root).place(rely=0.36,relx=0.8,relwidth=0.2,relheight=0.05)
tk.Label(root,text="Xmax:").place(rely=0.41,relwidth=0.2,relheight=0.05)
Xmax = tk.Entry(root).place(rely=0.41,relx=0.8,relwidth=0.2,relheight=0.05)
tk.Label(root,text="Ymin:").place(rely=0.46,relwidth=0.2,relheight=0.05)
Ymin = tk.Entry(root).place(rely=0.46,relx=0.8,relwidth=0.2,relheight=0.05)
tk.Label(root,text="Ymax:").place(rely=0.51,relwidth=0.2,relheight=0.05)
Ymax = tk.Entry(root).place(rely=0.51,relx=0.8,relwidth=0.2,relheight=0.05)
tk.Label(root,text="Velocity(0 or 1):").place(rely=0.56,relwidth=0.2,relheight=0.05)
V = tk.Entry(root).place(rely=0.56,relx=0.8,relwidth=0.2,relheight=0.05)
tk.Button(root, text="quit", width=10, command=root.quit).place(rely=0.9,relx=0.8,relwidth=0.2,relheight=0.1)
tk.Button(root, text="import in", width=10, command=put_in).place(rely=0.9,relx=0,relwidth=0.2,relheight=0.1)

root.mainloop()