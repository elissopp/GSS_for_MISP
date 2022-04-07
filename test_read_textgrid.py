#!/usr/bin/env python
# -- coding: UTF-8 
import os
import glob
import codecs
import argparse
from multiprocessing import Pool
import sys
from einops import rearrange

from nara_wpe.utils import stft, istft

# from pb_bss.distribution import CACGMMTrainer
# from pb_bss.evaluation import InputMetrics, OutputMetrics
# from dataclasses import dataclass
# from beamforming_wrapper import beamform_mvdr_souden_from_masks
# from pb_chime5.utils.numpy_utils import segment_axis_v2
# from read_textgrid import *
import soundfile as sf
import numpy as np
from test_gss import *
# from core_chime6 import GSS


signal = []
ob = '/yrfs2/cv1/hangchen2/data/MISP_121h_WPE_/R12/S201202/C02/R12_S201202_C02_I1_Middle_0.wav'
data, fs = sf.read(ob)
# signal.append(data[:1000])
signal.append(data)
ob = '/yrfs2/cv1/hangchen2/data/MISP_121h_WPE_/R12/S201202/C02/R12_S201202_C02_I1_Middle_1.wav'
data, fs = sf.read(ob)
# signal.append(data[:1000])
wavlen = len(data)
signal.append(data)
data = np.stack(signal, axis=0)
print(data.shape)
obstft = stft(data, 512, 256)
print(obstft.shape)
ac = []
text = '/raw7/cv1/hangchen2/misp2021_avsr/released_data/misp2021_avsr/eval_near_transcription/TextGrid/R53_S286287288289_C01_I1_Near_286.TextGrid'
ac.append(get_time_activity(text, wavlen, fs))
print(ac[0].count(True))
text = '/raw7/cv1/hangchen2/misp2021_avsr/released_data/misp2021_avsr/eval_near_transcription/TextGrid/R53_S286287288289_C01_I1_Near_286.TextGrid'
ac.append(get_time_activity(text, wavlen, fs))
ac.append([True]*wavlen)
fac = get_frequency_activity(ac, 512, 256)
print(fac.shape)
# gss = GSS(20, 0, True)
# masks = gss(obstft, fac)
# print(masks.shape)
# bf = Beamformer('mvdrSouden_ban', 'mask_mul')
# masks_bak = masks
# wavlist = []
# print('************')
# for i in range(masks.shape[0]):
#     print(masks.shape)
#     target_mask = masks[i]
#     print(target_mask.shape)
#     distortion_mask = np.sum(
#         np.delete(masks, i, axis=0),
#         axis=0,
#     )
#     print('******************')
#     print(target_mask.shape, distortion_mask.shape)
#     Xhat = bf(obstft, target_mask=target_mask, distortion_mask=distortion_mask)
#     print('Xhat shape:{}'.format(Xhat.shape))
#     xhat = istft(Xhat, 512, 256)
#     print('xhat shape:{}'.format(xhat.shape))
#     wavlist.append(xhat)
#     sf.write('tmp{}.wav'.format(i), xhat, fs)
#     masks = masks_bak
