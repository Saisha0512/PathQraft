import tkinter as tk
from dependencies import check_and_install_osmnx

check_and_install_osmnx()

from app import App


if __name__ == '__main__':
    root = tk.Tk()
    app = App(root)
    root.mainloop()