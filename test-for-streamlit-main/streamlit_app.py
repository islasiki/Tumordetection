import _thread as thread
import base64
import json
import logging.handlers
import queue
import re
import ssl
import time
from pathlib import Path

import numpy as np
import pydub
import streamlit as st
import torch
import torchaudio
import websocket
from PIL import Image
from streamlit_webrtc import WebRtcMode, webrtc_streamer

import SpeechModel as SM
from networks.u_net import Baseline
from utils import helpers
from xunfei import Ws_Param, MyWebsocket

HERE = Path(__file__).parent

logger = logging.getLogger(__name__)

STATUS_FIRST_FRAME = 0  # 第一帧的标识
STATUS_CONTINUE_FRAME = 1  # 中间帧标识
STATUS_LAST_FRAME = 2  # 最后一帧的标识

FLAG = 0


# 收到websocket消息的处理
def on_message(ws, message):
    # st.write(ws.img)
    global FLAG
    try:

        # st.write(img)
        code = json.loads(message)["code"]
        sid = json.loads(message)["sid"]
        if code != 0:
            errMsg = json.loads(message)["message"]
            print("sid:%s call error:%s code is:%s" % (sid, errMsg, code))

        else:
            data = json.loads(message)["data"]["result"]["ws"]
            result = ""
            for i in data:
                for w in i["cw"]:
                    result += w["w"]

            # st.write(result)
            # print(result)

            if result == '。' or result == '.。' or result == ' .。' or result == ' 。':
                pass
            else:
                # st.write(result)

                if bool(re.search(r'[Rr]o.{,2}tation|[Rr]o.{,2}tate|[Ll]ocation|[Rr]oot', result)):
                    st.write('Rotating the picture...')
                    segmentation(ws.img.transpose(-1, -2), color=ws.color)

                if bool(re.search(r'[Cc]an you hear me', result)) is True:
                    st.write('Yes! Please tell me the instruction!')
                if bool(re.search(r'[Gg]reen', result)) is True:
                    ws.color = torch.tensor([0, 255, 0])
                    ws.flag = 'green'
                    st.write('Successfully change the color to ', ws.flag, '!')
                    # st.write(ws.color)
                if bool(re.search(r'[Rr]ed', result)) is True:
                    ws.color = torch.tensor([255, 0, 0])
                    ws.flag = 'red'
                    st.write('Successfully change the color to ', ws.flag, '!')
                    # st.write(ws.color)
                if bool(re.search(r'[Bb]lue', result)) is True:
                    ws.color = torch.tensor([0, 0, 255])
                    ws.flag = 'blue'
                    st.write('Successfully change the color to ', ws.flag, '!')
                    # st.write(ws.color)

                if bool(re.search(r'[dD]etect|[Oo]peration', result)) is True:
                    st.write('Detecting...')
                    if ws.flag is not None:
                        st.write('Changing color of mask to ', ws.flag)
                    if ws.color is not None:
                        segmentation(ws.img, color=ws.color)
                    else:
                        segmentation(ws.img)
                if bool(re.search(r'[Ss]top', result)) is True:
                    st.write('Thanks for using me! Goodbye!')
                    FLAG = 1
                    ws.close()

    except Exception as e:
        print("receive msg,but parse exception:", e)


# 收到websocket错误的处理
def on_error(ws, error):
    print("### error:", error)


# 收到websocket关闭的处理
def on_close(ws):
    pass
    # print("### closed ###")


# 收到websocket连接建立的处理
def on_open(ws):
    global webrtc_ctx

    def run(*args):
        if not webrtc_ctx.state.playing:
            return

        while True:
            # webrtc返回的原始的帧可能含有一些描述格式等信息的头，直接发给讯飞识别不了
            audio_frame = webrtc_ctx.audio_receiver.get_frame()

            # 因此需要初始化一个AudioSegment类提取其中的声音信息
            sound = pydub.AudioSegment(
                data=audio_frame.to_ndarray().tobytes(),
                sample_width=audio_frame.format.bytes,
                frame_rate=audio_frame.sample_rate,
                channels=len(audio_frame.layout.channels)
            )

            sound = sound.set_channels(1).set_frame_rate(16000)
            # 将声音信息转为数组，并且转为字节流
            frame = np.array(sound.get_array_of_samples()).tobytes()

            # 写成json格式将声音信息发送给讯飞服务器
            d = {"common": wsParam.CommonArgs,
                 "business": wsParam.BusinessArgs,
                 "data": {"status": 1, "format": "audio/L16;rate=16000",
                          "audio": str(base64.b64encode(frame), 'utf-8'),
                          "encoding": "raw"}}
            d = json.dumps(d)
            ws.send(d)

    thread.start_new_thread(run, ())


def app_sst(img):
    global webrtc_ctx

    if not webrtc_ctx.state.playing:
        return

    st.write(
        'Hello! I am your little instruction assistant! If you want to do some operation on the picture, tell me please!')
    global wsParam
    # 讯飞接口
    # 初始化一个websocket对象
    wsParam = Ws_Param(APPID='fb41b9d3',
                       APIKey='40d5c53cd3d5ab53ec63ebe67dd39622',
                       APISecret='MjE1ZDI2ZDJmYzhhZTE4OGM4OGUyNTE5')
    websocket.enableTrace(False)
    wsUrl = wsParam.create_url()
    ws = MyWebsocket(wsUrl, on_message=on_message, on_error=on_error, on_close=on_close, img=img)
    ws.on_open = on_open
    ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE}, ping_timeout=2)
    return

def segmentation(x, color=torch.tensor([255, 0, 0])):
    st.markdown("""
     the result of Tumor predict is
     """)

    net = Baseline(img_ch=1, num_classes=3, depth=2).cpu()
    net.load_state_dict(torch.load(
        "./test-for-streamlit-main/checkpoint/unet_depth=2_fold_2_dice_223135.pth" ,map_location=torch.device('cpu')))
    # net.half()
    net.eval()

    y = net(x)
    y = torch.sigmoid(y)

    output = y.detach().cpu()

    palette = [[0], [128], [255]]
    save_pred, z = helpers.onehot_to_mask(np.array(output.squeeze()).transpose([1, 2, 0]), palette)

    save_pred_png = helpers.array_to_img(save_pred)

    newpic = helpers.mask_to_onehot(np.array(torch.from_numpy(save_pred)), palette)
    # totensor
    newpic = newpic.transpose([2, 0, 1])
    newpic = torch.from_numpy(newpic)
    # print(newpic.size())torch.Size([3, 256, 256])

    # 分割
    bg, b1, b2 = newpic.split(1, 0)

    # RGB b1膀胱壁 b2癌症

    b1rgb = torch.tensor([16, 136, 246])
    newb1 = torch.zeros(3, 512, 512)

    for i in range(3):
        newb1[i] = b1rgb[i] * b1
    newb2 = torch.zeros(3, 512, 512)
    for i in range(3):
        newb2[i] = color[i] * b2
    # print(newb1.size())--------------------------torch.Size([3, 256, 256])

    # 叠加

    raw = torch.squeeze(x, 0)
    bg = torch.zeros(3, 512, 512)
    for i in range(3):
        bg[i] = raw
    newpicrgb = 1 * bg + 0.5 * newb1 + 0.5 * newb2

    # print(newpicrgb.size())-----------------------torch.Size([3, 256, 256])

    # from PIL import Image
    a = np.array(newpicrgb).transpose([1, 2, 0]).astype(int)
    a[a > 255] = 255
    st.image(a)


if __name__ == "__main__":
    st.header("TumorImageEditor")
    st.markdown(
        """
Tumor image editor is mainly a fusion of natural language and image segmentation, the purpose is to help doctors in clinical diagnosis.

"""
    )
    uploaded_file = st.file_uploader("Choose an image of tumor to test")
    img = 0
    webrtc_ctx = webrtc_streamer(
        key="speech-to-text",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=1024,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": False, "audio": True},

    )

    if uploaded_file is not None:
        # src_image = load_image(uploaded_file)
        uploaded_image = Image.open(uploaded_file)

        img = np.array(uploaded_image)

        img = torch.from_numpy(img)
        img = img.float()
        img = img.unsqueeze(0)

        st.image(uploaded_file, caption='Input Image', use_column_width=True)

        img = img.unsqueeze(0)
        # img=img.half()
        img = img.to('cpu')

        segmentation(img)

    app_sst(img)
