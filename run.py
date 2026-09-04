
from project import *

dataBase = [ ]
dataBase_b = [ ]
buildDB_b()
buildDB_D()

buildDB()

fileName = ""

def data1():
	global fileName
	fileName = e0.get()
	plateText = main5(fileName)

master = Tk(className='Plate Detector')
master.geometry('500x500')
label0 = Label(master, text='File Name')
label0.place(x=80, y=130)
e0 = Entry(master)
e0.place(x=240, y=130)

b = Button(master, text='OK', width=20, bg='brown', fg='white', command=data1)
d = Button(master, text="Quit", width=20, bg='brown', fg='white', command=master.destroy)
b.place(x=180, y=380)
d.place(x=180, y=420)
mainloop()
