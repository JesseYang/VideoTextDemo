import socket
import threading
import sys
import time
import struct
import cv2
import numpy as np
import uuid
from enum import Enum
import os
from queue import Queue
from threading import Thread

from demo_terminal_server import Extractor

# from visualize import VisualizeGUI
# from audio import AudioThread

from cfgs.config import cfg
import pdb

import json
import pickle

class ServerAccept:
    def __init__(self, result_queue, host='192.168.5.41', port=8117):
        print(os.getpid())
        self.host = host
        self.port= port
        self.bufsize = 1024*100
        self.language_name = "english" ## chinese:0 english:1 korean:2 japanese:3
        self.result_queue = result_queue
        self.socket_server()

    def receive_data(self, conn, addr):
        save_file = open("test.mp4", 'wb')
        print(os.getpid())
        try:
            conn.settimeout(120)
            print(os.getpid())
            buf = b""
            data_len = -1
            while True:
                tem_buf = conn.recv(self.bufsize)
                buf += tem_buf
                if len(buf) >=4 and data_len == -1:
                    data_len = (buf[0]<<24) + (buf[1]<<16) + (buf[2]<<8) + buf[3]
                if data_len != -1 and (data_len+5) == len(buf):
                    break
            language_idx = buf[4]
            if language_idx == 0:
                self.language_name = "chinese"
            elif language_idx == 1:
                self.language_name = "english"
            elif language_idx == 2:
                self.language_name = "korean"
            else:
                self.language_name = "japanese"

            save_file.write(buf[5:])
            print("received data size %d , language_name %s" % (data_len, self.language_name))
            conn.close()
        except socket.timeout:
            print("time output: {0}, has closed".format(addr))
            conn.close()

    def socket_server(self):
        try:
            self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
            try:
                self.s.bind((self.host,self.port))
            except socket.error as e:
                print("Bind failed")
                print(e)
                sys.exit()
            self.s.listen(5)
        except:
            print("init socket error!")
        print(os.getpid())

    def deal_video(self):

        ext.from_video("test.mp4", self.language_name)
        ext.save()
        frames = ext.gui(self.language_name)

    def send_result(self, addr, port):
        time.sleep(2)
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((addr, 6667))
        except socket.error as msg:
            print("client connect error")
            print(msg)
            sys.exit(1)

        while True:

            flage, gui_img, pre_txt = self.result_queue.get()

            if flage == 1:
                img_encode = cv2.imencode('.jpg', gui_img)[1]
                img_code = np.array(img_encode)
                str_encode = img_code.tostring()
                num=len(str_encode)
                struc_2 = "bbbbbbbbbbb%dsbbbbbbbbbb%ds" % (num, len(pre_txt))

                a=[]
                for i in str(num):
                    a.append(int(i))
                for i in range(10-len(str(num))):
                    a.append(127)

                b=[]
                for i in str(len(pre_txt)):
                    b.append(int(i))

                for i in range(10-len(b)):
                    b.append(127)

                print(len(str_encode), len(pre_txt))
                data2 = struct.pack(struc_2, flage, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], str_encode, b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], b[8], b[9] , pre_txt.encode('utf-8'))
                s.send(data2)
                print("send over, pid", os.getpid())
               
            elif flage == 0:
                struc_2 = "bb"
                data2 = struct.pack(struc_2, flage, gui_img)
                s.send(data2)
                print("send over, pid", os.getpid())
                
            else:
                struc_2 = "b"
                data2 = struct.pack(struc_2, flage)
                s.send(data2)
                print("send over, pid", os.getpid())
                break

        s.close()
            
    def deal_data(self):
        while True:
            print("waiting for connection...")
            conn, addr = self.s.accept()
            conn.settimeout(30)
            print("accept new connection from {0}".format(addr))
            self.receive_data(conn, addr)

            t=threading.Thread(target=self.deal_video)
            t.start()
           
            t1=threading.Thread(target=self.send_result, args=(addr))
            t1.start()
        

if __name__ == '__main__':
  
    result_queue = Queue(maxsize=cfg.max_queue_len)

    ext = Extractor(result_queue)

    server_accept = ServerAccept(result_queue)

    server_accept.deal_data()
