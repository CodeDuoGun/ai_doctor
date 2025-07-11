import math
import torch
import numpy as np

#from .utils import *
import subprocess
from utils.log import logger
import os
import time
import cv2
import glob
import pickle
import copy

import queue
from queue import Queue
from threading import Thread, Event
from io import BytesIO
import multiprocessing as mp


from ttsreal import EdgeTTS,VoitsTTS,XTTS,CustomTTS
from lipasr import LipASR
import asyncio
from av import AudioFrame, VideoFrame

from model.wav2lip.wav2lip_model import Wav2Lip

from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} for inference.'.format(device))

def _load(checkpoint_path):
	if device == 'cuda':
		checkpoint = torch.load(checkpoint_path)
	else:
		checkpoint = torch.load(checkpoint_path,
								map_location=lambda storage, loc: storage)
	return checkpoint

def load_model(path):
	model = Wav2Lip()
	print("Load checkpoint from: {}".format(path))
	checkpoint = _load(path)
	s = checkpoint["state_dict"]
	new_s = {}
	for k, v in s.items():
		new_s[k.replace('module.', '')] = v
	model.load_state_dict(new_s)

	model = model.to(device)
	return model.eval()

def read_imgs(img_list):
    frames = []
    print('reading images...')
    for img_path in tqdm(img_list):
        frame = cv2.imread(img_path)
        frames.append(frame)
    return frames

def __mirror_index(size, index):
    #size = len(self.coord_list_cycle)
    turn = index // size
    res = index % size
    if turn % 2 == 0:
        return res
    else:
        return size - res - 1 

def inference(render_event,batch_size,face_imgs_path,audio_feat_queue,audio_out_queue,res_frame_queue):
    
    model = load_model("model/wav2lip/wav2lip.pth")
    input_face_list = glob.glob(os.path.join(face_imgs_path, '*.[jpJP][pnPN]*[gG]'))
    input_face_list = sorted(input_face_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    face_list_cycle = read_imgs(input_face_list)
    
    #input_latent_list_cycle = torch.load(latents_out_path)
    length = len(face_list_cycle)
    index = 0
    count=0
    counttime=0
    print('start inference')
    while True:
        if render_event.is_set():
            starttime=time.perf_counter()
            mel_batch = []
            try:
                mel_batch = audio_feat_queue.get(block=True, timeout=1)
            except queue.Empty:
                continue
                
            is_all_silence=True
            audio_frames = []
            for _ in range(batch_size*2):
                frame,type = audio_out_queue.get()
                audio_frames.append((frame,type))
                if type==0:
                    is_all_silence=False

            if is_all_silence:
                for i in range(batch_size):
                    res_frame_queue.put((None,__mirror_index(length,index),audio_frames[i*2:i*2+2]))
                    index = index + 1
            else:
                for i in range(batch_size):
                    res_frame_queue.put((None,__mirror_index(length,index),audio_frames[i*2:i*2+2]))
                    index = index + 1
            # else:
            #     # print('infer=======')
            #     t=time.perf_counter()
            #     img_batch = []
            #     for i in range(batch_size):
            #         idx = __mirror_index(length,index+i)
            #         face = face_list_cycle[idx]
            #         img_batch.append(face)
            #     img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

            #     # TODO: need new face paste back
            #     img_masked = img_batch.copy()
            #     img_masked[:, face.shape[0]//2:] = 0
            
            #     img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
            #     mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])
                
            #     img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
            #     mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

            #     with torch.no_grad():
            #         pred = model(mel_batch, img_batch)
            #     pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.

            #     counttime += (time.perf_counter() - t)
            #     count += batch_size
            #     #_totalframe += 1
            #     if count>=100:
            #         print(f"------actual avg infer fps:{count/counttime:.4f}")
            #         count=0
            #         counttime=0
            #     for i,res_frame in enumerate(pred):
            #         #self.__pushmedia(res_frame,loop,audio_track,video_track)
            #         res_frame_queue.put((res_frame,__mirror_index(length,index),audio_frames[i*2:i*2+2]))
            #         index = index + 1
                #print('total batch time:',time.perf_counter()-starttime)            
        else:
            time.sleep(1)
    print('musereal inference processor stop')

@torch.no_grad()
class LipReal:
    def __init__(self):
        self.W = 450
        self.H = 450
        self.fps = 25 # 20 ms per frame

        #### musetalk
        self.avatar_id = "wav2lip_avatar1"
        self.avatar_path = f"data/avatars/{self.avatar_id}"
        self.full_imgs_path = f"{self.avatar_path}/full_imgs" 
        self.face_imgs_path = f"{self.avatar_path}/face_imgs" 
        self.coords_path = f"{self.avatar_path}/coords.pkl"
        self.batch_size = 5
        self.idx = 0
        self.res_frame_queue = mp.Queue(self.batch_size*2)
        #self.__loadmodels()
        self.__loadavatar()

        self.asr = LipASR(self.fps, self.batch_size)
        self.asr.warm_up()
        #self.__warm_up()
        
        self.render_event = mp.Event()
        mp.Process(target=inference, args=(self.render_event,self.batch_size,self.face_imgs_path,
                                           self.asr.feat_queue,self.asr.output_queue,self.res_frame_queue,
                                           )).start()

    # def __loadmodels(self):
    #     # load model weights
    #     self.audio_processor, self.vae, self.unet, self.pe = load_all_model()
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     self.timesteps = torch.tensor([0], device=device)
    #     self.pe = self.pe.half()
    #     self.vae.vae = self.vae.vae.half()
    #     self.unet.model = self.unet.model.half()

    def __loadavatar(self):
        with open(self.coords_path, 'rb') as f:
            self.coord_list_cycle = pickle.load(f)
        input_img_list = glob.glob(os.path.join(self.full_imgs_path, '*.[jpJP][pnPN]*[gG]'))
        input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        self.frame_list_cycle = read_imgs(input_img_list)
        
    
    # def put_msg_txt(self,msg):
    #     self.tts.put_msg_txt(msg)
    
    def put_audio_frame(self,audio_chunk): #16khz 20ms pcm
        # logger.info(f"put audio to play asr ,size {len(audio_chunk)}")
        self.asr.put_audio_frame(audio_chunk)

    # TODO: 暂时无打断逻辑，不调用
    def pause_talk(self):
        # self.tts.pause_talk()
        self.asr.pause_talk()
      

    def process_frames(self,quit_event,loop=None,audio_track=None,video_track=None):
        
        while not quit_event.is_set():
            try:
                res_frame,idx,audio_frames = self.res_frame_queue.get(block=True, timeout=1)
            except queue.Empty:
                continue
            combine_frame = self.frame_list_cycle[idx]
            # TODO: 这里暂时注释，因为这个电脑垃圾，性能起不来
            # if audio_frames[0][1]==1 and audio_frames[1][1]==1: #全为静音数据，只需要取fullimg
            #     logger.info(f"全是静音数据")
            #     combine_frame = self.frame_list_cycle[idx]
            # else:
            #     logger.info(f"换脸了")
            #     bbox = self.coord_list_cycle[idx]
            #     combine_frame = copy.deepcopy(self.frame_list_cycle[idx])
            #     y1, y2, x1, x2 = bbox
            #     try:
            #         res_frame = cv2.resize(res_frame.astype(np.uint8),(x2-x1,y2-y1))
            #     except:
            #         continue
            #     #combine_frame = get_image(ori_frame,res_frame,bbox)
            #     #t=time.perf_counter()
            #     combine_frame[y1:y2, x1:x2] = res_frame
                #print('blending time:',time.perf_counter()-t)

            image = combine_frame #(outputs['image'] * 255).astype(np.uint8)
            new_frame = VideoFrame.from_ndarray(image, format="bgr24")
            asyncio.run_coroutine_threadsafe(video_track._queue.put(new_frame), loop) 

            for audio_frame in audio_frames:
                # logger.info(f"putting audio to audio track")
                frame,type = audio_frame
                frame = (frame * 32767).astype(np.int16)
                new_frame = AudioFrame(format='s16', layout='mono', samples=frame.shape[0])
                new_frame.planes[0].update(frame.tobytes())
                new_frame.sample_rate=16000
                # if audio_track._queue.qsize()>10:
                #     time.sleep(0.1)
                asyncio.run_coroutine_threadsafe(audio_track._queue.put(new_frame), loop)
        print('musereal process_frames thread stop') 
            
    def render(self,quit_event,loop=None,audio_track=None,video_track=None):
        #if self.opt.asr:
        #     self.asr.warm_up()

        # self.tts.render(quit_event)
        process_thread = Thread(target=self.process_frames, args=(quit_event,loop,audio_track,video_track))
        process_thread.start()

        self.render_event.set() #start infer process render
        count=0
        totaltime=0
        _starttime=time.perf_counter()
        #_totalframe=0
        while not quit_event.is_set(): 
            # update texture every frame
            # audio stream thread...
            t = time.perf_counter()
            self.asr.run_step()

            # if video_track._queue.qsize()>=2*self.opt.batch_size:
            #     print('sleep qsize=',video_track._queue.qsize())
            #     time.sleep(0.04*video_track._queue.qsize()*0.8)
            if video_track._queue.qsize()>=5:
                print('sleep qsize=',video_track._queue.qsize())
                time.sleep(0.04*video_track._queue.qsize()*0.8)
                
            # delay = _starttime+_totalframe*0.04-time.perf_counter() #40ms
            # if delay > 0:
            #     time.sleep(delay)
        self.render_event.clear() #end infer process render
        print('musereal thread stop')
            