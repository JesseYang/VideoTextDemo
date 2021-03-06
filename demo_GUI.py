import tkinter as tk
from tkinter.filedialog import askopenfilename, askdirectory
from tkinter.messagebox import askquestion
from demo_terminal import Extractor
from PIL import Image
from PIL import ImageTk
import os
import threading
import time
import numpy as np
import cv2
import pdb
from GUI.widgets import *

from cfgs.config import cfg

STATE_UNINITIALIZED = 'Uninitialized'
STATE_INITIALIZING = 'Initializing...'
STATE_PREDICTING = 'Predicting...'
STATE_READY = 'Ready'
STATE_FILE_ERROR = 'File error!!!'
frames = 0

class Extractor_GUI():
    def __init__(self):
        self.__init_gui()
        self.__init_model()
    def __init_gui(self):
       
        self.window = tk.Tk()
        self.window.wm_title('VideoText')
        self.window.config(background = '#FFFFFF')
        self.language_name = 'english'
        # ====================================================================
        # canvas
        # ====================================================================
        #self.fm_canvas = tk.Frame(self.window,  height = 800)
        #self.fm_canvas.grid(row = 0, column = 0, padx=10, pady=2)

        self.canvas = ICanvas(self.window, width = 800, height = 800)
        self.canvas.grid(row = 0, column = 0)

        self.fm_pred = tk.Frame(self.window, width = 800, height =800)
        self.fm_pred.grid(row = 0, column = 1)
        # self.fm_pred.grid_propagate(0)
        self.txt_pred = tk.Text(self.fm_pred, height = 57)
        self.txt_pred.grid(row = 0, column = 0)
        # ====================================================================
        # control bar
        # ====================================================================
        self.fm_control = tk.Frame(self.window, width=800, height=100, background = '#FFFFFF')
        self.fm_control.grid(row = 1, column=0, padx=10, pady=2)
        self.btn_prev_frame = tk.Button(self.fm_control, text='Prev Frame', command = self.__action_prev_frame, state = 'disabled')
        self.btn_prev_frame.grid(row = 0, column=0, padx=10, pady=2)
        self.lb_current_frame = tk.Label(self.fm_control, background = '#FFFFFF')
        self.lb_current_frame.grid(row = 0, column=1, padx=10, pady=2)
        self.lb_current_frame['text'] = '-/-'
        self.btn_next_frame = tk.Button(self.fm_control, text='Next Frame', command = self.__action_next_frame, state = 'disabled')
        self.btn_next_frame.grid(row = 0, column=2, padx=10, pady=2)

        # ====================================================================
        # status bar
        # ====================================================================
        self.fm_status = tk.Frame(self.window, width = 800, height = 100, background = '#FFFFFF')
        self.fm_status.grid(row = 1, column=1, padx=10, pady=2)
        self.btn_new = tk.Button(self.fm_status, text='New', command = self.__action_browse, state = 'disabled')
        self.btn_new.grid(row = 0, column=2, padx=10, pady=2)


        self.var=tk.StringVar() 
        r1=tk.Radiobutton(self.fm_status,text='English',variable=self.var,value='english',command=None)   
        r1.grid(row = 0, column=3, padx=10, pady=2)

        r2=tk.Radiobutton(self.fm_status,text='Chinese',variable=self.var,value='chinese',command=None)  
        r2.grid(row = 0, column=4, padx=10, pady=2)

        r3=tk.Radiobutton(self.fm_status,text='Korean',variable=self.var,value='korean',command=None)  
        r3.grid(row = 0, column=5, padx=10, pady=2)

        r4=tk.Radiobutton(self.fm_status,text='Japanese',variable=self.var,value='japanese',command=None)  
        r4.grid(row = 0, column=6, padx=10, pady=2)

        self.var.set("english")
        
        self.btn_load = tk.Button(self.fm_status, text = 'Load', command = self.__action_load, state = 'disabled')
        self.btn_load.grid(row = 0, column=7, padx=10, pady=2)
        self.lb_status = tk.Label(self.fm_status, background = '#FFFFFF')
        self.lb_status.grid(row = 0, column=8, padx=10, pady=2)
        self.cnt_status = STATE_UNINITIALIZED
        self.__update_status_bar()

        self.window.resizable(False, False)

    def __init_model(self):
        def init():
           
            self.cnt_status = STATE_INITIALIZING
            self.ext = Extractor()
            self.cnt_status = STATE_READY
            self.__update_btn_new()
            print("==========please select a video file for recognize===============")
        threading.Thread(target = init).start()

    def __action_prev_frame(self):
        self.cnt_current_frame -= 1
        self.__update_control_bar()
        self.__update_canvas_frame()

    def __action_next_frame(self):
        self.cnt_current_frame += 1
        self.__update_control_bar()
        self.__update_canvas_frame()

    def __action_load(self):
        dire = askdirectory()
        self.load(dire)
    def __action_browse(self):
        filename = askopenfilename(initialdir = '.',title = "choose your file",filetypes = (("mp4 files","*.mp4"), ("jpeg files","*.jpg"),("all files","*.*")))
        print(filename)
       
        if len(filename) == 0 or filename.strip().split('.')[-1].lower() not in ['mp4', 'avi', 'mov']:
            print(filename, "is not a video file")
            return
        if os.path.isfile(filename):
            selected_language = self.var.get()
            # print(len(selected_language))
            if len(selected_language) == 0:
                print("no language selected")
                return
            print("selected language " + selected_language)
            self.language_name = selected_language
            # return None
            self.do(filename, selected_language)
            # extension = filename.split('.')[-1]
            # res_dir = filename.replace(extension, '')
            # if os.path.exists(res_dir):
            #     # TODO: load history
            #     self.load(res_dir)
            # else:
            #     ans = askquestion('新任务', '未找到{}对应的处理结果，开始新任务？'.format(filename))
            #     if ans == 'yes':
            #         self.do(filename)

    def __update_canvas_frame(self):    
        img = self.collections_frame[self.cnt_current_frame]
        rgb = img.astype(np.uint8).copy()
        img = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        # img = Image.fromarray(img.astype(np.uint8))
        self.canvas.add(img)

        pred = self.collections_text[self.cnt_current_frame]
        self.txt_pred.delete(1.0, tk.END)
        self.txt_pred.insert(tk.END, pred)
        # self.update_txt_size()
    
    def update_txt_size(self):
        w = 0
        # pdb.set_trace()
        # for i in self.txt_pred.get(1.0, tk.END).split('\n'):
        for i in self.collections_text[self.cnt_current_frame].split('\n'):
            
            if len(i) > w:
                w = len(i) + 1
        self.txt_pred.configure(width = int(w*1.3))

    def __update_btn_new(self):
        if self.cnt_status == STATE_READY or self.cnt_status == STATE_FILE_ERROR:
            self.btn_new['state'] = 'normal'
            self.btn_load['state'] = 'normal'
        else:
            self.btn_new['state'] = 'disabled'
            self.btn_load['state'] = 'disabled'

    def __update_control_bar(self):
        if self.cnt_current_frame > 0:
            self.btn_prev_frame['state'] = 'normal'
        else:
            self.btn_prev_frame['state'] = 'disabled'

        if self.cnt_current_frame + 1< self.cnt_total_frame:
            self.btn_next_frame['state'] = 'normal'
        else:
            self.btn_next_frame['state'] = 'disabled'
        self.lb_current_frame['text'] = '{}/{}'.format(self.cnt_current_frame + 1,self.cnt_total_frame)
    
    def __update_status_bar(self):
        self.lb_status['text'] = self.cnt_status
        if self.cnt_status == STATE_READY or self.cnt_status == STATE_FILE_ERROR:
            self.btn_new['state'] = 'normal'
            self.btn_load['state'] = 'normal'
        else:
            self.btn_new['state'] = 'disabled'
            self.btn_load['state'] = 'disabled'
        self.lb_status.after(500, self.__update_status_bar)

    
    def load(self, res_dir):
        print(res_dir)
        if len(res_dir) == 0:
            return 
        frame_dir = os.path.join(res_dir, 'gui_frames')
        text_dir = os.path.join(res_dir, 'gui_preds')
        if not os.path.exists(frame_dir) or not os.path.exists(text_dir):
            return
        num = len(os.listdir(frame_dir))
        nums = len(os.listdir(text_dir))
        if num != nums or num <= 0:
            return
        frame_path_collections = [os.path.join(frame_dir, '{}.png'.format(i)) for i in range(num)]
        
        text_path_collections = [os.path.join(text_dir, '{}.txt'.format(i)) for i in range(num)]
        for i in range(num):
            if not os.path.exists(frame_path_collections[i]) or not os.path.exists(text_path_collections[i]):
                return
        self.collections_frame = [cv2.imread(i) for i in frame_path_collections]
        self.collections_text = [''.join(open(i,'r').readlines()) for i in text_path_collections]
        self.cnt_current_frame = 0
        self.cnt_total_frame = len(self.collections_frame)
        self.__update_control_bar()
        self.__update_canvas_frame()


    def do(self, filename, language_name):
        def pred():
            start_time4 = time.time()
            self.cnt_status = STATE_PREDICTING
            self.ext.from_video(filename, language_name)
            self.ext.save()
            frames = self.ext.gui(language_name)
            print("time: ", str(time.time() - start_time4))
            print(frames)
            if frames  == 9:
                self.cnt_status = STATE_FILE_ERROR
            else:
                self.cnt_status = STATE_READY
        threading.Thread(target = pred).start()
        self.wait_for_result()
    
    def wait_for_result(self):
        if self.cnt_status == STATE_READY or self.cnt_status == STATE_FILE_ERROR:
            self.collections_frame = self.ext.gui_frames
            if len(self.collections_frame)  <= 0:
                self.btn_next_frame['state'] = 'disabled' 
                self.btn_prev_frame['state'] = 'disabled'
                print("********this vodeo is passed*********please next******************")
                return
            self.collections_text = self.ext.gui_preds
            self.cnt_current_frame = 0
            self.cnt_total_frame = len(self.collections_frame)
            self.__update_control_bar()
            self.__update_canvas_frame()
        else:
            self.window.after(1000, self.wait_for_result)


    def launch(self):
        self.window.mainloop()


if __name__ == '__main__':
    ext = Extractor_GUI()
    ext.launch()
