import _thread as thread
import base64
import json
import logging.handlers
import queue
import ssl
import time
import urllib.request
from pathlib import Path
import re

import numpy as np
import pyaudio
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
from xunfei import Ws_Param,MyWebsocket

HERE = Path(__file__).parent

logger = logging.getLogger(__name__)

STATUS_FIRST_FRAME = 0  # 第一帧的标识
STATUS_CONTINUE_FRAME = 1  # 中间帧标识
STATUS_LAST_FRAME = 2  # 最后一帧的标识


# 收到websocket消息的处理
def on_message(ws, message):

    #st.write(ws.img)
    try:

        #st.write(img)
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

            if result == '。' or result == '.。' or result == ' .。' or result == ' 。':
                pass
            else:
                # t.insert(END, result)  # 把上边的标点插入到result的最后
                # print("翻译结果: %s。" % (result))
                #st.write(result)
                #st.write('detect?',bool(re.search(r'[dD]etect',result)))
                if bool(re.search('[Cc]an you hear me',result)) is True:
                    st.write('Yes! Please tell me the instruction!')
                if bool(re.search('[Gg]reen',result)) is True:
                    ws.color=torch.tensor([0,255,0])
                    ws.flag='green'
                    st.write('Successfully change the color to ',ws.flag,'!')
                    #st.write(ws.color)
                if bool(re.search('[Rr]ed',result)) is True:
                    ws.color=torch.tensor([255,0,0])
                    ws.flag ='red'
                    st.write('Successfully change the color to ', ws.flag, '!')
                    #st.write(ws.color)
                if bool(re.search('[Bb]lue',result)) is True:
                    ws.color=torch.tensor([0,0,255])
                    ws.flag ='blue'
                    st.write('Successfully change the color to ', ws.flag, '!')
                    #st.write(ws.color)

                if bool(re.search(r'[dD]etect',result)) is True:
                    st.write('Detecting...')
                    if ws.flag is not None:
                        st.write('Changing color of mask to ',ws.flag)
                    if ws.color is not None:
                        segmentation(ws.img,color=ws.color)
                    else:
                        segmentation(ws.img)
                if bool(re.search(r'[Ss]top',result)) is True:
                    st.write('Thanks for using me! Goodbye!')
                    ws.close()





    except Exception as e:
        print("receive msg,but parse exception:", e)
    return result


# 收到websocket错误的处理
def on_error(ws, error):
    print("### error:", error)


# 收到websocket关闭的处理
def on_close(ws):
    pass
    # print("### closed ###")


# 收到websocket连接建立的处理
def on_open(ws):
    def run(*args):
        status = STATUS_FIRST_FRAME  # 音频的状态信息，标识音频是第一帧，还是中间帧、最后一帧
        CHUNK = 520  # 定义数据流块
        FORMAT = pyaudio.paInt16  # 16bit编码格式
        CHANNELS = 1  # 单声道
        RATE = 16000  # 16000采样频率
        # 实例化pyaudio对象
        p = pyaudio.PyAudio()  # 录音
        # 创建音频流
        # 使用这个对象去打开声卡，设置采样深度、通道数、采样率、输入和采样点缓存数量
        stream = p.open(format=FORMAT,  # 音频流wav格式
                        channels=CHANNELS,  # 单声道
                        rate=RATE,  # 采样率16000
                        input=True,
                        frames_per_buffer=CHUNK)

        print("- - - - - - - Start Recording ...- - - - - - - ")

        for i in range(0, int(RATE / CHUNK * 60)):
            # # 读出声卡缓冲区的音频数据
            buf = stream.read(CHUNK)
            if not buf:
                status = STATUS_LAST_FRAME
            if status == STATUS_FIRST_FRAME:

                d = {"common": wsParam.CommonArgs,
                     "business": wsParam.BusinessArgs,
                     "data": {"status": 0, "format": "audio/L16;rate=16000",
                              "audio": str(base64.b64encode(buf), 'utf-8'),
                              "encoding": "raw"}}
                d = json.dumps(d)
                ws.send(d)
                status = STATUS_CONTINUE_FRAME
                # 中间帧处理
            elif status == STATUS_CONTINUE_FRAME:
                d = {"data": {"status": 1, "format": "audio/L16;rate=16000",
                              "audio": str(base64.b64encode(buf), 'utf-8'),
                              "encoding": "raw"}}
                ws.send(json.dumps(d))

            # 最后一帧处理
            elif status == STATUS_LAST_FRAME:
                d = {"data": {"status": 2, "format": "audio/L16;rate=16000",
                              "audio": str(base64.b64encode(buf), 'utf-8'),
                              "encoding": "raw"}}
                ws.send(json.dumps(d))
                time.sleep(1)
                break

        stream.stop_stream()  # 暂停录制
        stream.close()  # 终止流
        p.terminate()  # 终止pyaudio会话
        ws.close()

    thread.start_new_thread(run, ())


def run(img):
    global wsParam
    # 讯飞接口
    wsParam = Ws_Param(APPID='fb41b9d3',
                       APIKey='40d5c53cd3d5ab53ec63ebe67dd39622',
                       APISecret='MjE1ZDI2ZDJmYzhhZTE4OGM4OGUyNTE5')
    websocket.enableTrace(False)
    wsUrl = wsParam.create_url()
    ws = MyWebsocket(wsUrl, on_message=on_message, on_error=on_error, on_close=on_close,img=img)
    ws.on_open = on_open
    ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE}, ping_timeout=2)




def app_sst1():
    global img
    webrtc_ctx = webrtc_streamer(
        key="speech-to-text",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=1024,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": False, "audio": True},
    )

    status_indicator = st.empty()

    if not webrtc_ctx.state.playing:
        return
    run(img)


def app_sst2():
    st.write('11111111111111111111111')

    webrtc_ctx = webrtc_streamer(
        key="speech-to-text",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=1024,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": False, "audio": True},
    )

    st.write('22222222222222222222222222222222')
    status_indicator = st.empty()

    st.write('3333333333333333333333333333333333333333')
    if not webrtc_ctx.state.playing:
        return
    run()
    model1 = SM.M5()
    model1.load_state_dict(torch.load('c:/users/napoleon/desktop/test-for-streamlit-main/weights.pth'))
    model1.eval()
    del (model1.ConvLayer[1])
    del (model1.ConvLayer[4])
    del (model1.ConvLayer[7])
    del (model1.ConvLayer[10])
    new_sample_rate = 8000
    transform = torchaudio.transforms.Resample(orig_freq=16000, new_freq=new_sample_rate)
    status_indicator.write("Loading...")
    text_output = st.empty()
    stream = None
    i = 0
    buffer_all = []
    # while True:
    st.write('55555555555555555555555555555555555555')
    while True:
        if webrtc_ctx.audio_receiver:
            sound_chunk = pydub.AudioSegment.empty()
            try:
                audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
            except queue.Empty:
                time.sleep(0.1)
                status_indicator.write("No frame arrived.")
                continue

            status_indicator.write("Running. Say something!")

            for audio_frame in audio_frames:
                sound = pydub.AudioSegment(
                    data=audio_frame.to_ndarray().tobytes(),
                    sample_width=audio_frame.format.bytes,
                    frame_rate=audio_frame.sample_rate,
                    channels=len(audio_frame.layout.channels),
                )
                sound_chunk += sound

            if len(sound_chunk) > 0:
                sound_chunk = sound_chunk.set_channels(1).set_frame_rate(
                    16000
                )
                a = len(sound_chunk)
                buffer = np.array(sound_chunk.get_array_of_samples())
                buffer_all = np.append(buffer_all, buffer)

                # text_output.markdown(f"**Text:** {text}")
        else:
            status_indicator.write("AudioReciver is not set. Abort.")
            break
        i += 1
        if i % 50 == 0:
            i = 0
            buffer_all = buffer_all[0:16000]
            input = torch.tensor(buffer_all, dtype=torch.float)
            input = transform(input)
            input = input.unsqueeze(0)
            input = input.unsqueeze(0)
            # st.write(buffer_all.shape)
            # st.audio(buffer_all, sample_rate=16000)
            output = model1(input)
            buffer_all = []
            label = SM.tensor2label(output)
            st.write(label)
            time.sleep(0.1)

    # x=torch.ones(2,1,512,512).cuda()
    return label

def app_sst(model_path: str, lm_path: str, lm_alpha: float, lm_beta: float, beam: int):
    webrtc_ctx = webrtc_streamer(
        key="speech-to-text",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=1024,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": False, "audio": True},
    )

    status_indicator = st.empty()

    if not webrtc_ctx.state.playing:
        return

    status_indicator.write("Loading...")
    text_output = st.empty()
    stream = None

    while True:
        if webrtc_ctx.audio_receiver:
            if stream is None:
                from deepspeech import Model

                model = Model(model_path)
                model.enableExternalScorer(lm_path)
                model.setScorerAlphaBeta(lm_alpha, lm_beta)
                model.setBeamWidth(beam)

                stream = model.createStream()

                status_indicator.write("Model loaded.")

            sound_chunk = pydub.AudioSegment.empty()
            try:
                audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
            except queue.Empty:
                time.sleep(0.1)
                status_indicator.write("No frame arrived.")
                continue

            status_indicator.write("Running. Say something!")

            for audio_frame in audio_frames:
                sound = pydub.AudioSegment(
                    data=audio_frame.to_ndarray().tobytes(),
                    sample_width=audio_frame.format.bytes,
                    frame_rate=audio_frame.sample_rate,
                    channels=len(audio_frame.layout.channels),
                )
                sound_chunk += sound

            if len(sound_chunk) > 0:
                sound_chunk = sound_chunk.set_channels(1).set_frame_rate(
                    model.sampleRate()
                )
                buffer = np.array(sound_chunk.get_array_of_samples())
                stream.feedAudioContent(buffer)
                text = stream.intermediateDecode()
                text_output.markdown(f"**Text:** {text}")
        else:
            status_indicator.write("AudioReciver is not set. Abort.")
            break


def segmentation(x,color=torch.tensor([255, 0, 0])):
    st.markdown("""
     the result of Tumor predict is
     """)

    net = Baseline(img_ch=1, num_classes=3, depth=2).cuda()
    net.load_state_dict(torch.load(
        "c:\\users\\napoleon\\desktop\\test-for-streamlit-main\\checkpoint\\unet_depth=2_fold_2_dice_223135.pth"))
    #net.half()
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
    # print(b1.size())torch.Size([1, 256, 256])
    # print(b2.size())torch.Size([1, 256, 256])

    # RGB b1膀胱壁 b2癌症
    #b2rgb = torch.tensor([0, 255, 0]).reshape(3, 1)
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

# if __name__ == "__main__":
#     # import os
#     #
#     # DEBUG = os.environ.get("DEBUG", "false").lower() not in ["false", "no", "0"]
#     #
#     # logging.basicConfig(
#     #     format="[%(asctime)s] %(levelname)7s from %(name)s in %(pathname)s:%(lineno)d: "
#     #            "%(message)s",
#     #     force=True,
#     # )
#     #
#     # logger.setLevel(level=logging.DEBUG if DEBUG else logging.INFO)
#     #
#     # st_webrtc_logger = logging.getLogger("streamlit_webrtc")
#     # st_webrtc_logger.setLevel(logging.DEBUG)
#     #
#     # fsevents_logger = logging.getLogger("fsevents")
#     # fsevents_logger.setLevel(logging.WARNING)
#
#     st.header("TumorImageEditor")
#     st.markdown(
#         """
# Tumor image editor is mainly a fusion of natural language and image segmentation, the purpose is to help doctors in clinical diagnosis.
#
# """
#     )
#     uploaded_file = st.file_uploader("Choose an image of tumor to test")
#
#     if app_sst1():
#         print()
#
#     if uploaded_file is not None:
#         # src_image = load_image(uploaded_file)
#         uploaded_image = Image.open(uploaded_file)
#
#         img = np.array(uploaded_image)
#
#         img = torch.from_numpy(img)
#         img = img.float()
#         img = img.unsqueeze(0)
#
#         st.image(uploaded_file, caption='Input Image', use_column_width=True)
#         img = img.unsqueeze(0)
#         # img=img.half()
#         img = img.to('cuda')
#
#         segmentation(img)

st.header("TumorImageEditor")
st.markdown(
        """
Tumor image editor is mainly a fusion of natural language and image segmentation, the purpose is to help doctors in clinical diagnosis.

"""
    )
uploaded_file = st.file_uploader("Choose an image of tumor to test")
img=0


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
    

    segmentation(img)

if app_sst1():
    print()
