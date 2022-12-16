# -*- coding:utf-8 -*-

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
import websocket
import hashlib
import base64
import hmac
import json
from urllib.parse import urlencode
import time
# 该模块为客户端和服务器端的网络套接字提供对传输层安全性(通常称为“安全套接字层”)
# 的加密和对等身份验证功能的访问。
import ssl
from wsgiref.handlers import format_date_time
from datetime import datetime
from time import mktime
from websocket import WebSocketApp
import _thread as thread
import pyaudio




class Ws_Param(object):
    # 初始化
    def __init__(self, APPID, APIKey, APISecret):
        self.APPID = APPID
        self.APIKey = APIKey
        self.APISecret = APISecret

        # 公共参数(common)
        self.CommonArgs = {"app_id": self.APPID}
        # 业务参数(business)，更多个性化参数可在官网查看
        self.BusinessArgs = {"domain": "iat", "language": "zh_cn", "accent": "mandarin", "vinfo": 1, "vad_eos": 10000}

    # 生成url
    def create_url(self):
        url = 'wss://ws-api.xfyun.cn/v2/iat'
        # 生成RFC1123格式的时间戳
        now = datetime.now()
        date = format_date_time(mktime(now.timetuple()))

        # 拼接字符串
        signature_origin = "host: " + "ws-api.xfyun.cn" + "\n"
        signature_origin += "date: " + date + "\n"
        signature_origin += "GET " + "/v2/iat " + "HTTP/1.1"
        # 进行hmac-sha256进行加密
        signature_sha = hmac.new(self.APISecret.encode('utf-8'), signature_origin.encode('utf-8'),
                                 digestmod=hashlib.sha256).digest()
        signature_sha = base64.b64encode(signature_sha).decode(encoding='utf-8')

        authorization_origin = "api_key=\"%s\", algorithm=\"%s\", headers=\"%s\", signature=\"%s\"" % (
            self.APIKey, "hmac-sha256", "host date request-line", signature_sha)
        authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode(encoding='utf-8')
        # 将请求的鉴权参数组合为字典
        v = {
            "authorization": authorization,
            "date": date,
            "host": "ws-api.xfyun.cn"
        }
        # 拼接鉴权参数，生成url
        url = url + '?' + urlencode(v)
        # print("date: ",date)
        # print("v: ",v)
        # 此处打印出建立连接时候的url,参考本demo的时候可取消上方打印的注释，比对相同参数时生成的url与自己代码生成的url是否一致
        # print('websocket url :', url)
        return url

class MyWebsocket(WebSocketApp):
    def __init__(self,url,on_message=None,on_error=None, on_close=None,img=None,color=None,flag=None):
        super(MyWebsocket, self).__init__(url)
        self.on_close=on_close
        self.on_message=on_message
        self.on_error=on_error
        self.img=img
        self.color=color
        self.flag=flag








# TODO 记得修改 APPID、APIKey、APISecret

from tkinter import *
import threading  # 多线程
import tkinter

"""
setDaemon()方法
我们在程序运行中，执行一个主线程，如果主线程又创建一个子线程，
主线程和子线程就分兵两路，分别运行，那么当主线程完成想退出时，会检验子线程是否完成。
如果子线程未完成，则主线程会等待子线程完成后再退出。
但是有时候我们需要的是，只要主线程完成了，不管子线程是否完成，
都要和主线程一起退出，这时就可以用setDaemon方法啦
"""


# t = threading.Thread(target=run)


def thread_it(func, *args):
    t = threading.Thread(target=func, args=args)
    t.setDaemon(True)
    t.start()


if __name__=='__main__':
    window = Tk()
    window.geometry('500x350')
    window.title('语音识别')
    t = Text(window, bg='pink')  # 文本框
    t.pack()

    # lambda匿名函数
    """
    def thread_it(run,):
        t=threading.Thread(target=run,args=)
        t.setDaemon(True)
        t.start()
    """
    tkinter.Button(window, text='开始', command=lambda: thread_it(run, )).place(x=230, y=315)

    window.mainloop()