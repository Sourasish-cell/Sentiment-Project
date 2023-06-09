from tkinter import *
from tkinter.ttk import *
 
# creates a Tk() object
master = Tk()

master.title("Religious Sentiment Detector")
 
# sets the geometry of main
# root window
master.geometry("620x400")

bg = PhotoImage(file = "Religion.png")
label = Label(
master,
image=bg
)
label.place(x=0, y=0)
 
# function to open a new window
# on a button click
def openNewWindow():
     
    # Toplevel object which will
    # be treated as a new window
    newWindow = Toplevel(master)
 
    # sets the title of the
    # Toplevel widget
    newWindow.title("Religious Sentiment Detector")
 
    # sets the geometry of toplevel
    newWindow.geometry("620x400")
 
    # A Label widget to show in toplevel
    Label(newWindow,
          text ="This is a new window").pack()
 
 
label = Label(master,
              text ="Welcome to Religious Sentiment Detector")

label.pack(pady = 10)
 
# a button widget which will open a
# new window on button click
btn = Button(master,
             text ="Let's Go",
             command = openNewWindow)
btn.pack(pady = 10)
 
# mainloop, runs infinitely
mainloop()