# -*- coding: utf-8 -*-
"""
Created on Thu May 27 21:46:59 2021

@author: Muhammad Saad Saeed (18F-MS-CP-01)
"""
import model
import argparse
import os
import utils as ut
import numpy as np
from glob import glob
import pickle
import progressbar


parser = argparse.ArgumentParser()
# set up training configuration.
parser.add_argument('--gpu', default='', type=str)
# set up network configuration.
parser.add_argument('--net', default='resnet34s', choices=['resnet34s', 'resnet34l'], type=str)
parser.add_argument('--ghost_cluster', default=2, type=int)
parser.add_argument('--vlad_cluster', default=8, type=int)
parser.add_argument('--bottleneck_dim', default=512, type=int)
parser.add_argument('--aggregation_mode', default='gvlad', choices=['avg', 'vlad', 'gvlad'], type=str)
# set up learning rate, training loss and optimizer.
parser.add_argument('--loss', default='softmax', choices=['softmax', 'amsoftmax'], type=str)
parser.add_argument('--test_type', default='normal', choices=['normal', 'hard', 'extend'], type=str)

global args
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = ""

params = {'dim': (257, None, 1),
              'nfft': 512,
              'spec_len': 250,
              'win_length': 400,
              'hop_length': 160,
              'n_classes': 901,
              'sampling_rate': 16000,
              'normalize': True,
              }

network_eval = model.vggvox_resnet2d_icassp(input_dim=params['dim'],
                                                num_class=params['n_classes'],
                                                mode='eval', args=args)


def test_txt(txt_path):
    with open(txt_path, 'r+') as f:
        data = [dat.split(' ')[1] for count, dat in enumerate(f) if count%8==0]
    return data
    
def train_txt(txt_path):
    with open(txt_path, 'r+') as f:
        data = [dat.split(' ')[0] for dat in f]
    return data

model_path = '/home/shahpc/pycharm/emotion_recongition/resnet34_vlad8_ghost2_bdim512_deploy-20221009T104351Z-001/resnet34_vlad8_ghost2_bdim512_deploy/weights.h5'

if os.path.isfile(model_path):
    network_eval.load_weights(model_path, by_name=True)
else:
    print('Error-File not Found')

home_dir = '/home/shahpc/Downloads/archive/2/'
ids_list = os.listdir(home_dir)
i = 0

if not os.path.exists('voxFeats'):
    os.mkdir('voxFeats')

# data = train_txt('splits/voxlb1_train_all.txt')

# bar = progressbar.ProgressBar(max_value=153516)
for ids in ids_list:
    feats = []
    for wav in glob('%s%s/*/*.wav'%(home_dir, ids)):
        print wav
        #  v = network_eval.predict(specs)
        # exit()
        # wav = wav.replace('\\', '/')
        specs = ut.load_data(wav, win_length=params['win_length'], sr=params['sampling_rate'],
                  hop_length=params['hop_length'], n_fft=params['nfft' ],
                  spec_len=params['spec_len'], mode='eval')
        specs = np.expand_dims(np.expand_dims(specs, 0), -1)
        v = network_eval.predict(specs)
        # print v
        print len(v[0])
        label = wav.split('/')[-2]

        # pickle.dump([v, wav], f)
        feats.append([v, label])
        # print feats
        i+=1
        # bar.update(i)
            
    with open('voxFeats/%s.pkl'%('_2'), 'wb') as f:
        pickle.dump(feats, f)
