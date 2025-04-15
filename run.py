

import collections
import json
import os
import time

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from gensim.models import KeyedVectors

from models.Baselines import *
from models.FANVM import FANVMModel
from models.SVFEND import SVFENDModel
from models.SVFEND_transformer import SVFEND_transformer_original,Net,SVFEND_transformer_original1
# from models.SVFEND_transformer import SVFEND_tr
from models.TikTec import TikTecModel

from utils.dataloader import *
from models.Trainer import *
from models.Trainer_3set import *
from sklearn.model_selection import KFold
import numpy as np
from sklearn.model_selection import train_test_split

# from models.Trainer_3set import Trainer
from models.KDmodel import *


def pad_sequence(seq_len,lst, emb):
    result=[]
    for video in lst:
        if isinstance(video, list):
            video = torch.stack(video)
        ori_len=video.shape[0]
        if ori_len == 0:
            video = torch.zeros([seq_len,emb],dtype=torch.long)
        elif ori_len>=seq_len:
            if emb == 200:
                video=torch.FloatTensor(video[:seq_len])
            else:
                video=torch.LongTensor(video[:seq_len])
        else:
            video=torch.cat([video,torch.zeros([seq_len-ori_len,video.shape[1]],dtype=torch.long)],dim=0)
            if emb == 200:
                video=torch.FloatTensor(video)
            else:
                video=torch.LongTensor(video)
        result.append(video)
    return torch.stack(result)

def pad_sequence_bbox(seq_len,lst):
    result=[]
    for video in lst: 
        if isinstance(video, list):
            video = torch.stack(video)
        ori_len=video.shape[0]
        if ori_len == 0:
            video = torch.zeros([seq_len,45,4096],dtype=torch.float)
        elif ori_len>=seq_len:
            video=torch.FloatTensor(video[:seq_len])
        else:
            video=torch.cat([video,torch.zeros([seq_len-ori_len,45,4096],dtype=torch.float)],dim=0)
        result.append(video)
    return torch.stack(result)

def pad_frame_sequence(seq_len,lst):
    attention_masks = []
    result=[]
    for video in lst:
        video=torch.FloatTensor(video)
        ori_len=video.shape[0]
        if ori_len>=seq_len:
            gap=ori_len//seq_len
            video=video[::gap][:seq_len]
            mask = np.ones((seq_len))
        else:
            video=torch.cat((video,torch.zeros([seq_len-ori_len,video.shape[1]],dtype=torch.float)),dim=0)
            mask = np.append(np.ones(ori_len), np.zeros(seq_len-ori_len))
        result.append(video)
        mask = torch.IntTensor(mask)
        attention_masks.append(mask)
    return torch.stack(result), torch.stack(attention_masks)


def _init_fn(worker_id):
    np.random.seed(2022)

def SVFEND_collate_fn(batch): 
    num_comments = 23 
    num_frames = 83
    num_audioframes = 50 

    intro_inputid = [item['intro_inputid'] for item in batch]
    intro_mask = [item['intro_mask'] for item in batch]

    title_inputid = [item['title_inputid'] for item in batch]
    title_mask = [item['title_mask'] for item in batch]

    comments_like = [item['comments_like'] for item in batch]
    comments_inputid = [item['comments_inputid'] for item in batch]
    comments_mask = [item['comments_mask'] for item in batch]

    comments_inputid_resorted = [] 
    comments_mask_resorted = []
    comments_like_resorted = []

    for idx in range(len(comments_like)):
        comments_like_one = comments_like[idx]
        comments_inputid_one = comments_inputid[idx]
        comments_mask_one = comments_mask[idx]
        if comments_like_one.shape != torch.Size([0]):
            comments_inputid_one, comments_mask_one, comments_like_one = (list(t) for t in zip(*sorted(zip(comments_inputid_one, comments_mask_one, comments_like_one), key=lambda s: s[2], reverse=True)))
        comments_inputid_resorted.append(comments_inputid_one)
        comments_mask_resorted.append(comments_mask_one)
        comments_like_resorted.append(comments_like_one)
    
    comments_inputid = pad_sequence(num_comments,comments_inputid_resorted,250)
    comments_mask = pad_sequence(num_comments,comments_mask_resorted,250)
    comments_like=[]
    for idx in range(len(comments_like_resorted)):
        comments_like_resorted_one = comments_like_resorted[idx]
        if len(comments_like_resorted_one)>=num_comments:
            comments_like.append(torch.tensor(comments_like_resorted_one[:num_comments]))
        else:
            if isinstance(comments_like_resorted_one, list):
                comments_like.append(torch.tensor(comments_like_resorted_one+[0]*(num_comments-len(comments_like_resorted_one))))
            else:
                comments_like.append(torch.tensor(comments_like_resorted_one.tolist()+[0]*(num_comments-len(comments_like_resorted_one))))

    frames = [item['frames'] for item in batch]
    frames, frames_masks = pad_frame_sequence(num_frames, frames)

    audioframes  = [item['audioframes'] for item in batch]
    audioframes, audioframes_masks = pad_frame_sequence(num_audioframes, audioframes)

    c3d  = [item['c3d'] for item in batch]
    c3d, c3d_masks = pad_frame_sequence(num_frames, c3d)

    label = [item['label'] for item in batch]
    # vid = [item['vid'] for item in batch]
    
    #ocr
    gpt_data_inputid = [item['gpt_data_inputid'] for item in batch]
    gpt_data_mask = [item['gpt_data_mask'] for item in batch]
    
    #事件名
    keyword_inputid = [item['keyword_inputid'] for item in batch]
    keyword_mask = [item['keyword_mask'] for item in batch]
    
    # 揭穿视频文本特征
    # features_list = [item['features_list'] for item in batch]
    
    # cosine_similarity_ocr_c_inputid = [item['cosine_similarity_ocr_c_inputid'] for item in batch]
    # cosine_similarity_ocr_c_mask = [item['cosine_similarity_ocr_c_mask'] for item in batch]

    
#下面的代码未使用    
    #     emotion_features = [item['emotion_features'] for item in batch]


#     differ_cosine_inputid = [item['differ_cosine_inputid'] for item in batch]
#     differ_cosine_mask = [item['differ_cosine_mask'] for item in batch]
    
    
    
    # cosine_similarity_ocr_c_value=[item['cosine_similarity_ocr_c_value'] for item in batch]
    # print(cosine_similarity_ocr_c_value)
    
#     #获取发布日期
#     publish_time_inputid = [item['publish_time_inputid'] for item in batch]
#     publish_time_mask = [item['publish_time_mask'] for item in batch]
    # attn_text_cosine_inputid = [item['attn_text_cosine_inputid'] for item in batch]
    # attn_text_cosine_mask = [item['attn_text_cosine_mask'] for item in batch]
    
    # attn_ocr_inputid = [item['attn_ocr_inputid'] for item in batch]
    # attn_ocr_mask = [item['attn_ocr_mask'] for item in batch]
    # attn_keywords_inputid = [item['attn_keywords_inputid'] for item in batch]
    # attn_keywords_mask = [item['attn_keywords_mask'] for item in batch]
    # attn_title_inputid = [item['attn_title_inputid'] for item in batch]
    # attn_title_mask = [item['attn_title_mask'] for item in batch]
    
    # text_emoion_cosine = [item['text_emoion_cosine'] for item in batch]
    # print(123)
    
    # print(type(torch.stack(gpt_data_inputid)))

    return {
        'label': torch.stack(label),
        'intro_inputid': torch.stack(intro_inputid),
        'intro_mask': torch.stack(intro_mask),
        'title_inputid': torch.stack(title_inputid),
        'title_mask': torch.stack(title_mask),
        'comments_inputid': comments_inputid,
        'comments_mask': comments_mask,
        'comments_like': torch.stack(comments_like),
        'audioframes': audioframes,
        'audioframes_masks': audioframes_masks,
        'frames':frames,
        'frames_masks': frames_masks,
        'c3d': c3d,
        'c3d_masks': c3d_masks,
        # 'vid':vid,
        'gpt_data_inputid':torch.stack(gpt_data_inputid),
        'gpt_data_mask':torch.stack(gpt_data_mask),
        'keyword_inputid': torch.stack(keyword_inputid),
        'keyword_mask': torch.stack(keyword_mask),
        # 'features_list':features_list,
        # 'ocr_inputid': torch.stack(ocr_inputid),
        # 'ocr_mask': torch.stack(ocr_mask),
        # 'cosine_similarity_ocr_c_inputid':torch.stack(cosine_similarity_ocr_c_inputid),
        # 'cosine_similarity_ocr_c_mask':torch.stack(cosine_similarity_ocr_c_mask),
    }


def FANVM_collate_fn(batch): 
    num_comments = 23 
    num_frames = 83

    title_inputid = [item['title_inputid'] for item in batch]
    title_mask = [item['title_mask'] for item in batch]

    comments_like = [item['comments_like'] for item in batch]
    comments_inputid = [item['comments_inputid'] for item in batch]
    comments_mask = [item['comments_mask'] for item in batch]

    comments_inputid_resorted = [] 
    comments_mask_resorted = []
    comments_like_resorted = []

    for idx in range(len(comments_like)):
        comments_like_one = comments_like[idx]
        comments_inputid_one = comments_inputid[idx]
        comments_mask_one = comments_mask[idx]
        if comments_like_one.shape != torch.Size([0]):
            comments_inputid_one, comments_mask_one, comments_like_one = (list(t) for t in zip(*sorted(zip(comments_inputid_one, comments_mask_one, comments_like_one), key=lambda s: s[2], reverse=True)))
        comments_inputid_resorted.append(comments_inputid_one)
        comments_mask_resorted.append(comments_mask_one)
        comments_like_resorted.append(comments_like_one)
    
    comments_inputid = pad_sequence(num_comments,comments_inputid_resorted,250)
    comments_mask = pad_sequence(num_comments,comments_mask_resorted,250)
    comments_like=[]
    for idx in range(len(comments_like_resorted)):
        comments_like_resorted_one = comments_like_resorted[idx]
        if len(comments_like_resorted_one)>=num_comments:
            comments_like.append(torch.tensor(comments_like_resorted_one[:num_comments]))
        else:
            if isinstance(comments_like_resorted_one, list):
                comments_like.append(torch.tensor(comments_like_resorted_one+[0]*(num_comments-len(comments_like_resorted_one))))
            else:
                comments_like.append(torch.tensor(comments_like_resorted_one.tolist()+[0]*(num_comments-len(comments_like_resorted_one))))

    frames = [item['frames'] for item in batch]
    frames, frames_masks = pad_frame_sequence(num_frames, frames)
    # frame_thmub = [item['frame_thmub'] for item in batch]

    label = [item['label'] for item in batch]
    label_event = [item['label_event'] for item in batch]
    s = [item['s'] for item in batch]

    return {
        'label': torch.stack(label),
        'title_inputid': torch.stack(title_inputid),
        'title_mask': torch.stack(title_mask),
        'comments_inputid': comments_inputid,
        'comments_mask': comments_mask,
        'comments_like': torch.stack(comments_like),
        'frames':frames,
        'frames_masks': frames_masks,
        # 'frame_thmub': torch.stack(frame_thmub),
        's': torch.stack(s),
        'label_event':torch.stack(label_event),
    }

def bbox_collate_fn(batch): 
    num_frames = 83

    bbox_vgg = [item['bbox_vgg'] for item in batch] 
    bbox_vgg = pad_sequence_bbox(num_frames,bbox_vgg) 

    label = [item['label'] for item in batch]

    return {
        'label': torch.stack(label),
        'bbox_vgg': bbox_vgg,
    }

def c3d_collate_fn(batch):
    num_frames = 83

    c3d  = [item['c3d'] for item in batch]
    c3d, c3d_masks = pad_frame_sequence(num_frames, c3d)

    label = [item['label'] for item in batch]

    return {
        'label': torch.stack(label),
        'c3d': c3d,
        'c3d_masks': c3d_masks,
    }

def vgg_collate_fn(batch):
    num_frames = 83

    frames = [item['frames'] for item in batch]
    frames, frames_masks = pad_frame_sequence(num_frames, frames)

    label = [item['label'] for item in batch]

    return {
        'label': torch.stack(label),
        'frames':frames,
        'frames_masks': frames_masks,
    }

def comments_collate_fn(batch): 
    num_comments = 23 

    comments_like = [item['comments_like'] for item in batch]
    comments_inputid = [item['comments_inputid'] for item in batch]
    comments_mask = [item['comments_mask'] for item in batch]

    comments_inputid_resorted = [] 
    comments_mask_resorted = []
    comments_like_resorted = []

    for idx in range(len(comments_like)):
        comments_like_one = comments_like[idx]
        comments_inputid_one = comments_inputid[idx]
        comments_mask_one = comments_mask[idx]
        if comments_like_one.shape != torch.Size([0]):
            comments_inputid_one, comments_mask_one, comments_like_one = (list(t) for t in zip(*sorted(zip(comments_inputid_one, comments_mask_one, comments_like_one), key=lambda s: s[2], reverse=True)))
        comments_inputid_resorted.append(comments_inputid_one)
        comments_mask_resorted.append(comments_mask_one)
        comments_like_resorted.append(comments_like_one)
    
    comments_inputid = pad_sequence(num_comments,comments_inputid_resorted,250)
    comments_mask = pad_sequence(num_comments,comments_mask_resorted,250)
    comments_like=[]
    for idx in range(len(comments_like_resorted)):
        comments_like_resorted_one = comments_like_resorted[idx]
        if len(comments_like_resorted_one)>=num_comments:
            comments_like.append(torch.tensor(comments_like_resorted_one[:num_comments]))
        else:
            if isinstance(comments_like_resorted_one, list):
                comments_like.append(torch.tensor(comments_like_resorted_one+[0]*(num_comments-len(comments_like_resorted_one))))
            else:
                comments_like.append(torch.tensor(comments_like_resorted_one.tolist()+[0]*(num_comments-len(comments_like_resorted_one))))

    label = [item['label'] for item in batch]

    return {
        'label': torch.stack(label),
        'comments_inputid': comments_inputid,
        'comments_mask': comments_mask,
        'comments_like': torch.stack(comments_like),
    }

def title_w2v_collate_fn(batch):
    length_title = 128
    title_w2v = [item['title_w2v'] for item in batch]
    title_w2v = pad_sequence(length_title, title_w2v, 100)

    label = [item['label'] for item in batch]

    return {
        'label': torch.stack(label),
        'title_w2v': title_w2v,
    }


class Run():
    def __init__(self,
                 config
                 ):

        self.model_name = config['model_name']
        self.mode_eval = config['mode_eval']
        self.fold = config['fold']
        self.data_type = 'SVFEND'

        self.epoches = config['epoches']
        self.batch_size = config['batch_size']
        self.num_workers = config['num_workers']
        self.epoch_stop = config['epoch_stop']
        self.seed = config['seed']
        self.device = config['device']
        self.lr = config['lr']
        self.lambd=config['lambd']
        self.save_param_dir = config['path_param']
        self.path_tensorboard = config['path_tensorboard']
        self.dropout = config['dropout']
        self.weight_decay = config['weight_decay']
        self.event_num = 616 
        self.mode ='normal'
    

    def get_dataloader(self,data_type,data_fold):
        collate_fn=None
#         num_folds = 5
#         data_files = ['vid_fold_1.txt', 'vid_fold_2.txt', 'vid_fold_3.txt', 'vid_fold_4.txt', 'vid_fold_5.txt']
#         if data_type == 'SVFEND':
            

#             kf = KFold(n_splits=num_folds)
#             fold_indices = list(kf.split(data_files))  # 获取所有交叉验证的索引

#             # 选择特定五倍交叉验证的训练集和测试集
#             fold_info = fold_indices[data_fold - 1]
#             if len(fold_info) == 2:
#                 train_index, test_index = fold_info
#                 data_fold = data_fold

#                 # 当前折的训练数据文件
#                 train_files = [data_files[j] for j in train_index]

#                 # 初始化训练集的内容
#                 train_vid = []
#                 for path_vid in train_files:
#                     with open('/root/autodl-tmp/new_fakesv/FakeSV-main/dataset/data-split/event/' + path_vid, "r") as fr:
#                         for line in fr.readlines():
#                             train_vid.append(line.strip())

#                 # 当前折的测试数据文件
#                 test_files = [data_files[j] for j in test_index]

#                 # 初始化测试集的内容
#                 test_vid = []
#                 for path_vid in test_files:
#                     with open('/root/autodl-tmp/new_fakesv/FakeSV-main/dataset/data-split/event/' + path_vid, "r") as fr:
#                         for line in fr.readlines():
#                             test_vid.append(line.strip())
            
#             dataset_train = SVFENDDataset(train_vid)
#                 # 使用 dataset_train 进行训练

           
#             dataset_test = SVFENDDataset(test_vid)
#             # 使用 dataset_test 进行测试
                # 使用 dataset_test 进行测试
        if data_type == 'SVFEND':
            print('SVFEND')
            dataset_train = SVFENDDataset(f'vid_fold_no_{data_fold}.txt')
            dataset_test = SVFENDDataset(f'vid_fold_{data_fold}.txt')
            collate_fn=SVFEND_collate_fn
        
        elif data_type=='FANVM':
            print('FANVM')
            dataset_train = FANVMDataset_train(f'vid_fold_no_{data_fold}.txt')
            dataset_test = FANVMDataset_test(path_vid_train=f'vid_fold_no_{data_fold}.txt', path_vid_test=f'vid_fold_{data_fold}.txt')
            collate_fn = FANVM_collate_fn
        elif data_type=='c3d':
            dataset_train = C3DDataset(f'vid_fold_no_{data_fold}.txt')
            dataset_test = C3DDataset(f'vid_fold_{data_fold}.txt')
            collate_fn = c3d_collate_fn
        elif data_type=='vgg':
            dataset_train = VGGDataset(f'vid_fold_no_{data_fold}.txt')
            dataset_test = VGGDataset(f'vid_fold_{data_fold}.txt')
            collate_fn = vgg_collate_fn
        elif data_type=='bbox':
            dataset_train = BboxDataset('vid_fold_no1.txt')
            dataset_test = BboxDataset('vid_fold_1.txt')
            collate_fn = bbox_collate_fn
        elif data_type=='comments':
            dataset_train = CommentsDataset(f'vid_fold_no_{data_fold}.txt')
            dataset_test = CommentsDataset(f'vid_fold_{data_fold}.txt')
            collate_fn = comments_collate_fn
        elif data_type=='w2v':
            wv_from_text = KeyedVectors.load_word2vec_format("./stores/tencent-ailab-embedding-zh-d100-v0.2.0-s/tencent-ailab-embedding-zh-d100-v0.2.0-s.txt", binary=False)
            dataset_train = Title_W2V_Dataset(f'vid_fold_no{data_fold}.txt', wv_from_text)
            dataset_test = Title_W2V_Dataset(f'vid_fold_{data_fold}.txt', wv_from_text)
            collate_fn = title_w2v_collate_fn

        train_dataloader = DataLoader(dataset_train, batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True,
            worker_init_fn=_init_fn,
            collate_fn=collate_fn)

        test_dataloader=DataLoader(dataset_test, batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
            worker_init_fn=_init_fn,
            collate_fn=collate_fn)

        dataloaders =  dict(zip(['train', 'test'],[train_dataloader, test_dataloader]))

        return dataloaders


    def get_dataloader_temporal(self, data_type):
        collate_fn=None
        if data_type=='SVFEND':
            dataset_train = SVFENDDataset('vid_time3_train_new_2536.txt')
            dataset_val = SVFENDDataset('vid_time3_val_new_546.txt')
            dataset_test = SVFENDDataset('vid_time3_test_new_542.txt')
            
            
            collate_fn=SVFEND_collate_fn
        elif data_type=='FANVM':
            dataset_train = FANVMDataset_train('vid_time3_train_new_2536.txt')
            dataset_val = FANVMDataset_test(path_vid_train='vid_time3_train_new_2536.txt', path_vid_test='vid_time3_val_new_546.txt')
            dataset_test = FANVMDataset_test(path_vid_train='vid_time3_train_new_2536.txt', path_vid_test='vid_time3_test_new_542.txt')
            collate_fn = FANVM_collate_fn
        else:
            # can be added
            print ("Not available")

        train_dataloader = DataLoader(dataset_train, batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True,
            worker_init_fn=_init_fn,
            collate_fn=collate_fn,
            drop_last=True
                                     )
        val_dataloader = DataLoader(dataset_val, batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
            worker_init_fn=_init_fn,
            collate_fn=collate_fn,
            drop_last=True)
        test_dataloader=DataLoader(dataset_test, batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
            worker_init_fn=_init_fn,
            collate_fn=collate_fn,
            drop_last=True)
 
        dataloaders =  dict(zip(['train', 'val', 'test'],[train_dataloader, val_dataloader, test_dataloader]))

        return dataloaders


    def get_model(self):
        if self.model_name == 'SVFEND':
            self.model = SVFENDModel(bert_model='/root/autodl-tmp/new_fakesv/FakeSV-main/code/data/bert-base-chinese', fea_dim=128,dropout=self.dropout)
        elif self.model_name == 'SVFEND_transformer_original':
            self.model = SVFEND_transformer_original(bert_model='/root/autodl-tmp/new_fakesv/FakeSV-main/code/data/bert-base-chinese', fea_dim=128,dropout=self.dropout)
        elif self.model_name == 'SVFEND_transformer_original1':
            self.model = SVFEND_transformer_original1(bert_model='/root/autodl-tmp/new_fakesv/FakeSV-main/code/data/bert-base-chinese', fea_dim=128,dropout=self.dropout)
        elif self.model_name == 'Net':
            self.model = Net(bert_model='bert-base-chinese', fea_dim=128,dropout=self.dropout)
            
        elif self.model_name == 'KD':  #用知识蒸馏的方式训练模型融合各个特征
            self.model = ASF(bert_model='bert-base-chinese', fea_dim=128,dropout=self.dropout)
        elif self.model_name == 'FANVM':
            self.model = FANVMModel(bert_model='/root/autodl-tmp/new_fakesv/FakeSV-main/code/data/bert-base-chinese', fea_dim=128)
            self.data_type = "FANVM"
            self.mode = 'eann'
        elif self.model_name == 'C3D':
            self.model = bC3D(fea_dim=128)
            self.data_type = "c3d"
        elif self.model_name == 'VGG':
            self.model = bVGG(fea_dim=128)
            self.data_type = "vgg"
        elif self.model_name == 'Bbox':
            self.model = bBbox(fea_dim=128)
            self.data_type = "bbox"
        elif self.model_name == 'Vggish':
            self.model = bVggish(fea_dim=128)
        elif self.model_name == 'Bert':
            self.model = bBert(bert_model='bert-base-chinese', fea_dim=128,dropout=self.dropout)
        elif self.model_name == 'TextCNN':
            self.model = bTextCNN(fea_dim=128, vocab_size=100)
            self.data_type = "w2v"
        elif self.model_name == 'Comments':
            self.model = bComments(bert_model='bert-base-chinese', fea_dim=128)
            self.data_type = "comments"
        elif self.model_name == 'TikTec':
            self.model = TikTecModel(VCIF_dropout=self.dropout, MLP_dropout=self.dropout)
            self.data_type = 'TikTec'

        return self.model


    def main(self):
        if self.mode_eval == "nocv":
            self.model = self.get_model()
            dataloaders = self.get_dataloader(data_type=self.data_type, data_fold=self.fold)
            trainer = Trainer(model=self.model, device = self.device, lr = self.lr, dataloaders = dataloaders, epoches = self.epoches, dropout = self.dropout, weight_decay = self.weight_decay, mode = self.mode, model_name = self.model_name, event_num = self.event_num, 
                    epoch_stop = self.epoch_stop, save_param_path = self.save_param_dir+self.data_type+"/"+self.model_name+"/", writer = SummaryWriter(self.path_tensorboard))
            result=trainer.train()

        elif self.mode_eval == "temporal":
            self.model = self.get_model()
            dataloaders = self.get_dataloader_temporal(data_type=self.data_type)
            trainer = Trainer3KD_fusion(model=self.model, device = self.device, lr = self.lr, dataloaders = dataloaders, epoches = self.epoches, dropout = self.dropout, weight_decay = self.weight_decay, mode = self.mode, model_name = self.model_name, event_num = self.event_num, 
                    epoch_stop = self.epoch_stop, save_param_path = self.save_param_dir+self.data_type+"/"+self.model_name+"/", writer = SummaryWriter(self.path_tensorboard))
            result=trainer.train()
            return result
        
        elif self.mode_eval == "cv":
            collate_fn=None
            if self.model_name == 'TextCNN':
                wv_from_text = KeyedVectors.load_word2vec_format("./stores/tencent-ailab-embedding-zh-d100-v0.2.0-s/tencent-ailab-embedding-zh-d100-v0.2.0-s.txt", binary=False)

            history = collections.defaultdict(list) 
            for fold in range(5, 6): 
                print('-' * 50)
                print ('fold %d:' % fold)
                print('-' * 50)
                self.model = self.get_model()
                dataloaders = self.get_dataloader(data_type=self.data_type, data_fold=fold)
                
                trainer = TrainerKD_fusion(model = self.model, device = self.device, lr = self.lr, dataloaders = dataloaders, epoches = self.epoches, dropout = self.dropout, weight_decay = self.weight_decay, mode = self.mode, model_name = self.model_name, event_num = self.event_num, 
                    epoch_stop = self.epoch_stop, save_param_path = self.save_param_dir+self.data_type+"/"+self.model_name+"/", writer = SummaryWriter(self.path_tensorboard+"fold_"+str(fold)+"/"))
       
                result = trainer.train()

                history['auc'].append(result['auc'])
                history['f1'].append(result['f1'])
                history['recall'].append(result['recall'])
                history['precision'].append(result['precision'])
                history['acc'].append(result['acc'])
                
            print ('results on 5-fold cross-validation: ')
            for metric in ['acc', 'f1', 'precision', 'recall', 'auc']:
                # print ('%s : %.4f +/- %.4f' % (metric, np.mean(history[metric]), np.std(history[metric])))
                result_str = '%s : %.4f +/- %.4f' % (metric, np.mean(history[metric]), np.std(history[metric]))
                print(result_str)
                with open(r"D:\worksapce\new_fakesv\FakeSV-main\code\utils\result.txt", "a") as f:  #把运行结果保存到追加到txt文件
                    f.write(result_str)
        
        else:
            print ("Not Available")
