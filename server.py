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

# from visualize import VisualizeGUI
# from audio import AudioThread

from cfgs.config import cfg
import pdb

import json
import pickle

print("over")
class ServerAccept:
    def __init__(self, host='192.168.1.120', port=8117):
        print(os.getpid())
        self.host = host
        self.port= port
        self.bufsize = 14096

        # self.capture_queue = Queue(maxsize=cfg.max_queue_len)
        # self.result_queue = Queue(maxsize=cfg.max_queue_len)

        # self.detect_thread = DetectThread(self.capture_queue, self.result_queue)
        # self.detect_thread.start()
        self.output_path = "output_%s.mp4" % str(uuid.uuid4())
 
        self.socket_server()

    def receive_data(self, conn, addr):
        save_file = ("test.mp4", 'wb')
        print(os.getpid())
        try:
            conn.settimeout(600)
            print(os.getpid())
            receive_time = time.time()
            buf = b""
            data_len = -1
            while True:
                tem_buf = conn.recv(self.bufsize)
                buf += tem_buf
                if len(tem_buf) != self.bufsize:
                    break
            print(len(buf))
            save_file.wirte(buf)
            print("file save finished")
        except socket.timeout:
            print("Client connection interrupted: {0}".format(addr))
            print("{0} closed! ".format(addr))
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

    def deal_data(self):
        while 1:
            print("waiting for connection...")
            conn, addr = self.s.accept()
            conn.settimeout(30)
            print("accept new connection from {0}".format(addr))
            t = threading.Thread(target=self.receive_data, args=(conn, addr))
            t.start()
        # self.s.close()


if __name__ == '__main__':

    server_accept = ServerAccept()
    server_accept.deal_data()
