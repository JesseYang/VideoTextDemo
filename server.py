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
import argparse

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
        self.flage= False
        self.host = host
        self.port= port
        self.bufsize = 10240
        self.language_name = "chinese" ## chinese:0 english:1 korean:2 japanese:3
        self.result_queue = result_queue
        self.socket_server()

    def receive_data(self, conn, addr):
        a=1
        save_file = open("test.mp4", 'wb')
        print(os.getpid())
        try:
            print(os.getpid())
            buf = b""
            data_len = -1
            first_packet = True
            recv_len = 0
            while True:
                tem_buf = conn.recv(self.bufsize)
                recv_len += len(tem_buf)
               
                # print(recv_len)

                # buf += tem_buf
                # if len(buf) >=4 and data_len == -1:
                if first_packet == True:
                    data_len = (tem_buf[0]<<24) + (tem_buf[1]<<16) + (tem_buf[2]<<8) + tem_buf[3]
                    language_idx = tem_buf[4]

                if first_packet == True:
                    save_file.write(tem_buf[5:])
                    first_packet = False
                else:
                    save_file.write(tem_buf)

                if data_len != -1 and (data_len+5) == recv_len:
                    break

                
            # language_idx = buf[4]
            # print(buf[0:1024*3])
            if language_idx == 0:
                self.language_name = "chinese"
            elif language_idx == 1:
                self.language_name = "english"
            elif language_idx == 2:
                self.language_name = "korean"
            else:
                self.language_name = "japanese"

            save_file.write(buf[5:])
            print("received data size %d , %d, language_name %s" % (data_len,  len(buf[5:]), self.language_name))
            
        # except socket.timeout:
        #     print("time output: {0}, has closed".format(addr))
        #     conn.sendall(struct.pack('b',-2))
        #     conn.close()
        except socket.error as e:
            print("receive data error", e)
            a=0
            # conn.sendall(struct.pack('b',-2))
            # conn.close()
        return a

    def socket_server(self):
        print(self.host)
        
        try:
            self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
            try:
                self.s.bind((self.host,self.port))
            except socket.error as e:
                print("Bind failed")
                print(e)
                sys.exit(1)
                # self.result_queue.put(-1)
           
            self.flage=True
            self.s.listen(5)

        except:
            print("init socket error!")
            # self.result_queue.put(-2)

    def deal_video(self):
        print("deal video thread started")

        ext.from_video("test.mp4", self.language_name)
        self.result_queue.put(9)
        ext.save()
        frames = ext.gui(self.language_name)
        return frames
    def deal_data(self):
        print("waiting for connection...")
        while True:
            try:
                conn, addr = self.s.accept()
                print("accept new connection from {0}".format(addr))
            except Exception as e:
                # self.result_queue.put(-3)
                print(e)
                continue
            self.result_queue.queue.clear()

            
            is_success = self.receive_data(conn, addr)
        
        
            # break 
            if not is_success:
                self.result_queue.put(-4)
                conn.close()
                continue
            conn.close()

            t3=threading.Thread(target=self.socket_send_stage, args=(addr[0],))
            t3.start()
            time.sleep(1)
            flas=self.deal_video()
            if flas == 0:
                print("file error!!!,")
                print("waiting for connection...")
                self.result_queue.put(-5)
                continue
           
           
            time.sleep(0.5)
            t=threading.Thread(target=self.socket_send_img, args=(addr[0],))
            t.start()
           
            t1=threading.Thread(target=self.socket_send_txt, args=(addr[0],))
            t1.start()
            


            #self.send_result(addr[0], 6667)
            # self.socket_send_img(addr[0])
            # self.socket_send_txt(addr[0])

    def socket_send_img(self,addr):
        print(addr)
        self.result_queue.put(10)
        imgs = os.listdir(os.path.join("test_result","gui_frames" ))
        print("total frame", len(imgs))
        for im in range(len(imgs)):
            try:

                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
                #s.connect(('192.168.5.158', 10001))
                s.connect((addr, 10001))
            except socket.error as msg:
                print(msg)
                self.result_queue.put(-6)
                print(sys.exit(1))

            filepath = os.path.join("test_result","gui_frames", str(im)+".png")
            # filepath = 'test.png'
            # fhead = struct.pack(b'128sl', bytes(os.path.basename(filepath), encoding='utf-8'), os.stat(filepath).st_size)
            # s.send(fhead)
            print('client filepath: {0}'.format(filepath))

            #fp = open(filepath, 'rb')
            fp = open(filepath, 'rb')
            fileSize = os.path.getsize(filepath)
            print(filepath)
            print(fileSize)
            try:
                print('=========================================================img----{0}'.format(1 if im!=(len(imgs)-1) else 0))
                s.send(struct.pack('i',1 if im!=(len(imgs)-1) else 0))
                s.send(struct.pack('i',fileSize))
                print('waiting recv OK...')
                data = s.recv(2)
                print(' recv OK1...')
                print('data:')
                print(type(data))
                print(data)
                # print('start send 2.jpeg...')

                while 1:
                    data = fp.read()
                    if not data:
                        print('{0} file send over...'.format(filepath))
                        break
                    s.send(data)
                print(filepath, 'finished')
                print('start waitting recv OK2')
                data = s.recv(2)
                print(' recved OK2...')
                print('data:')
                print(type(data))
                print(data)
            except socket.error as msg:
                print("send data error", msg)
                self.result_queue.put(-7)
                s.close()
                sys.exit(1)
            s.close()
            # break
        print("img send over!!!")
        print("waiting for connection...")
        time.sleep(2)
        self.result_queue.put(-10)
        
    def socket_send_txt(self,addr):
        print(addr)
        
        imgs = os.listdir(os.path.join("test_result","gui_preds" ))
        print("total text", len(imgs))
        for im in range(len(imgs)):
            try:

                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
                #s.connect(('192.168.5.158', 10001))
                s.connect((addr, 10002))
            except socket.error as msg:
                print(msg)
                self.result_queue.put(-8)
                print(sys.exit(1))

            filepath = os.path.join("test_result","gui_preds", str(im)+".txt")
            # filepath = 'test.png'
            # fhead = struct.pack(b'128sl', bytes(os.path.basename(filepath), encoding='utf-8'), os.stat(filepath).st_size)
            # s.send(fhead)
            print('client filepath: {0}'.format(filepath))

            #fp = open(filepath, 'rb')
            fp = open(filepath, 'rb')
            fileSize = os.path.getsize(filepath)
            print(filepath)
            print(fileSize)

            try:
                print('=========================================================txt----{0}'.format(1 if im!=(len(imgs)-1) else 0))
                s.send(struct.pack('i',1 if im!=(len(imgs)-1) else 0))
                s.send(struct.pack('i',fileSize))
                print('waiting recv OK...')
                data = s.recv(2)
                print(' recv OK1...')
                print('data:')
                print(type(data))
                print(data)
                # print('start send 2.jpeg...')
                while 1:
                    data = fp.read()
                    if not data:
                        print('{0} file send over...'.format(filepath))
                        break
                    s.send(data)
                print(filepath, 'finished')
                print('start waitting recv OK2')
                data = s.recv(2)
                print(' recved OK2...')
                print('data:')
                print(type(data))
                print(data)
            except socket.error as msg:
                print("send data error", msg)
                self.result_queue.put(-9)
                s.close()
                sys.exit(1)
            s.close()
        print("txt send over!!!")
        print("waiting for connection...")
                       # break

    def socket_send_stage(self, addr):
        print(addr)
        while True:
            try:
                ss = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                ss.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)  
                ss.connect((addr, 10003))
                print("send stage connected ok")
            except socket.error as msg:
                print("connect server to send stage",msg)
                print(sys.exit(1))
 
            stage = self.result_queue.get()
            print("stage ====", stage)
            ss.sendall(struct.pack('b', stage))
            if stage < 0:
                # self.result_queue.queue.clear()
                ss.close()
                break
        # sys.exit(1)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ip', help='ip address of server', default='192.168.5.41')
    args = parser.parse_args()
    print('cfg.max_queue_len:')
   
    print(cfg.max_queue_len)
  
    result_queue = Queue(cfg.max_queue_len)
    # time.sleep(10000)
    ext = Extractor(result_queue)

    server_accept = ServerAccept(result_queue, host=args.ip)
   
    if server_accept.flage:
        # result_queue.queue.clear()
        server_accept.deal_data()
