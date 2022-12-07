import sys
import os
import tkinter as tk
import cv2
from PIL import Image, ImageTk, ImageOps  # 画像データ用

from Commons.vector import Vector2D
from Controller.Controller import Controller, ControllerState
from Controller.MouseEvent import MouseEvent

BASE_COLOR = "white"
BASE_COLOR_ACTIVE = "#dddddd"
ACCENT_COLOR = "#2a93ed"
ACCENT_COLOR_ACTIVE = "#0f78d3"
SUB_COLOR = "#cfcfcf"
SUB_COLOR_ACTIVE = "#979797"
ICONS = {
    "setting": None
}

def resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

def hover(event, *, background=None):
    if background:
        event.widget['bg'] = background

class CanvasFrame(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.config(background=BASE_COLOR, width=self.master.winfo_width(), height=300)
        # Canvasの作成
        self.canvas = tk.Canvas(
            self, 
            width = self.master.winfo_width(),
            height = 100,
            bg = "gray",
            borderwidth=0,
            highlightthickness=0,
            relief='ridge'
            )
        # Canvasを配置
        self.canvas.place(x=0, y=0)
        #setting_button = tk.Button(self.master, text="OK", image=tk.PhotoImage(file="setting.png"), height=50, width=50, compound=tk.LEFT)
        #self.canvas.tag_bind("setting_button", "<Button-1>", lambda e:print("Hello"))
        #self.canvas.create_window(10, 10, anchor=tk.NW, window=setting_button)

        self.videoCapture = cv2.VideoCapture(0)
        #self.update_idletasks()
        self.cap_width = self.videoCapture.get( cv2.CAP_PROP_FRAME_WIDTH )
        self.cap_height = self.videoCapture.get( cv2.CAP_PROP_FRAME_HEIGHT )
        self.delay = 20
        self.controller_state = ControllerState()
        self.controller = Controller(controllerState=self.controller_state)
        self.mouseEvent = MouseEvent(controllerState=self.controller_state, screen_size=Vector2D(x=self.master.winfo_screenwidth(), y=self.master.winfo_screenheight()))
        self.update()

    def update(self):
        success, cv_image = self.videoCapture.read()
        #print(success)
        if not success:
            self.master.after(self.delay, self.update)
            return

        cv_image, pointer, handGesture, hand_dir = self.controller.update(cv_image)
        self.mouseEvent.update(pointer, handGesture, hand_dir)

        # NumPyのndarrayからPillowのImageへ変換
        pil_image = Image.fromarray(cv_image)

        #self.update_idletasks()

        # キャンバスのサイズを取得
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        if canvas_width==1:
            self.master.after(self.delay, self.update)
            return 

        # 画像のアスペクト比（縦横比）を崩さずに指定したサイズ（キャンバスのサイズ）全体に画像をリサイズする
        pil_image = ImageOps.pad(pil_image, (canvas_width, canvas_height))

        # PIL.ImageからPhotoImageへ変換する
        self.photo_image = ImageTk.PhotoImage(image=pil_image)

        # 画像の描画
        self.canvas.create_image(
            canvas_width / 2,       # 画像表示位置(Canvasの中心)
            canvas_height / 2,                   
            image=self.photo_image  # 表示画像データ
        )
        #self.canvas.create_image(12, 12, anchor=tk.NW, image=ICONS["setting"], tag="setting_button")
        self.master.after(self.delay, self.update)

    def resize(self, *, width, height):
        self.config(width=width, height=height)
        self.canvas.config(width=width, height=height)

class ControllerIconButton(tk.Button):
    def __init__(self, master=None, *, icon=None, text="Button", width=60, height=60, command=None, expand=True):
        super().__init__(master, text=text, image=icon, height=height, width=width, compound=tk.TOP, background=BASE_COLOR, relief=tk.FLAT, borderwidth=0, command=command)
        self.bind("<Enter>", lambda event: hover(event, background=BASE_COLOR_ACTIVE))
        self.bind("<Leave>", lambda event: hover(event, background=BASE_COLOR))
        self.pack(side=tk.LEFT, expand=expand, fill=tk.X)

class ControllerTextButton(tk.Button):
    def __init__(self, master=None, *, text="Button", width=100, height=60, font=("", 12, "bold"), command=None, background=BASE_COLOR, background_hover=None, color=None):
        super().__init__(master, text=text, font=font, width=width, height=height, background=BASE_COLOR, foreground=color, activeforeground=color, relief=tk.FLAT, borderwidth=0, command=command)
        self.bind("<Enter>", lambda event: hover(event, background=background_hover))
        self.bind("<Leave>", lambda event: hover(event, background=background))
        self.pack(side=tk.RIGHT, padx = 8, pady = 10)


class ControllerFrame(tk.Frame):
    def __init__(self, master=None, *, calibration):
        super().__init__(master)
        self.pack_propagate(0)
        self.height = 60
        self.config(background=BASE_COLOR, width=self.master.winfo_width(), height=self.height)
        #print(self.master.winfo_width())
        ICONS["question"] = tk.PhotoImage(file=resource_path("./Assets/question.png"))
        ICONS["calibration"] = tk.PhotoImage(file=resource_path("./Assets/calibration.png"))
        ICONS["setting"] = tk.PhotoImage(file=resource_path("./Assets/setting.png"))
        self.help_button = ControllerIconButton(self, icon=ICONS["question"], text="Help", height=self.height)
        self.calibration_button = ControllerIconButton(self, icon=ICONS["calibration"], text="Calibration", height=self.height, command=calibration)
        self.setting_button = ControllerIconButton(self, icon=ICONS["setting"], text="Settings", height=self.height)


    def resize(self, width):
        self.config(width=width)

class CalibrationFrame(tk.Frame):
    def __init__(self, master=None, *, close_event):
        super().__init__(master)
        self.pack_propagate(0)
        self.height = 60
        self.config(background=BASE_COLOR, width=800, height=self.height)
        self.help_button = ControllerIconButton(self, icon=ICONS["question"], text=None, width=60, height=self.height, expand=False)
        self.apply_button = ControllerTextButton(self, text="Apply", width=9, height=4, color="white" ,background=ACCENT_COLOR, background_hover=ACCENT_COLOR_ACTIVE, command=close_event)
        self.cancel_button = ControllerTextButton(self, text="Cancel", width=9, height=4, color="black" ,background=SUB_COLOR, background_hover=SUB_COLOR_ACTIVE, command=close_event)


    def resize(self, width):
        self.config(width=width)

class Application(tk.Frame):
    def __init__(self, master=tk.Tk()):
        super().__init__(master)
        self.master.geometry("800x700")
        self.master.configure(background=BASE_COLOR)
        self.width = self.master.winfo_width()
        self.height = self.master.winfo_height()
        self.__width = None
        self.__height = None
        self.name = 'AirMousePointer α'
        self.master.title(self.name)
        self.master.minsize(300, 60)
        self.canvasFrame = CanvasFrame(self)
        self.controllerFrame = ControllerFrame(self, calibration=self.start_calibration)
        self.calibrationFrame = CalibrationFrame(self, close_event=self.end_calibration)
        #self.canvasFrame.grid(row=0, column=0, sticky=tk.NSEW)
        self.canvasFrame.pack(side=tk.TOP)
        #canvasFrame.grid()
        #self.controllerFrame.grid(row=1, column=0, sticky=tk.NSEW)
        self.controllerFrame.pack(side=tk.BOTTOM)
        self.pack()
        self.resize()
        self.mainloop()
    
    def start_calibration(self):
        self.master.title(f"{self.name} - Calibrating...")
        self.controllerFrame.pack_forget()
        self.calibrationFrame.pack(side=tk.BOTTOM)
        self.calibrationFrame.apply_button.config(background=ACCENT_COLOR)
        self.calibrationFrame.cancel_button.config(background=SUB_COLOR)
        self.canvasFrame.controller_state.calibrating = True

    def end_calibration(self):
        self.master.title(f"{self.name}")
        self.calibrationFrame.pack_forget()
        self.controllerFrame.pack(side=tk.BOTTOM)
        self.canvasFrame.controller_state.calibrating = False

    def resize(self):
        width = self.master.winfo_width()
        height = self.master.winfo_height()
        controller_height = self.controllerFrame.height
        
        if self.width == width and self.height == height:
            self.width = width
            self.height = height
            if self.__width and self.__height:
                self.master.geometry(f"{self.__width}x{self.__height+controller_height}")
                pass
            self.__width = None
            self.__height = None
        elif self.width == width:
            ratio = self.canvasFrame.cap_width / self.canvasFrame.cap_height
            self.__height = height-controller_height
            self.__width = int(self.__height*ratio)
            self.canvasFrame.resize(width=self.__width, height=self.__height)
            self.controllerFrame.resize(self.__width)
            self.calibrationFrame.resize(self.__width)
        else:
            ratio = self.canvasFrame.cap_height / self.canvasFrame.cap_width
            self.__width = width
            self.__height = int(width*ratio)
            self.canvasFrame.resize(width=self.__width, height=self.__height)
            self.controllerFrame.resize(self.__width)
            self.calibrationFrame.resize(self.__width)
        self.width = width
        self.height = height
        self.master.after(100, self.resize)