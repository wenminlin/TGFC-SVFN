import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os, sys
import math
import pandas as pd
from .SelfAttention import *

from transformers import RobertaTokenizer, RobertaModel, TimesformerModel, Data2VecAudioModel
from transformers import BertModel, BertTokenizer
from models.CrossmodalTransformer import MULTModel, MULTModel2

    
from transformers import BertModel, GPT2Model, GPT2Config    

from transformers import BertModel
from torch_geometric.nn import GATConv
import torch_geometric.utils as tg_utils
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse

    
#扩散模型的处理代码
 

from models.Multimodal_Model import Text_Noise_Pre, Image_Noise_Pre, Audio_Noise_Pre, Visual_Noise_Pre, GaussianDiffusionSampler, generate_noise_sequence
    
def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    device = t.device
    out = torch.gather(v, index=t, dim=0).float().to(device)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))   


 ###### teacher model
class Teacher_model(nn.Module):
    def __init__(self, bert_model, fea_dim, dropout):
        super(Teacher_model, self).__init__()
        self.text_dim=768
        self.dropout=dropout
        self.img_dim=4096
        self.comment_dim = 768
        
        self.bert = BertModel.from_pretrained(r"/root/autodl-tmp/new_fakesv/FakeSV-main/code/data/bert-base-chinese", local_files_only=True).requires_grad_(False)

        self.linear_comment = nn.Sequential(Attention(dim=self.comment_dim, heads=4, dropout=0.1),
                                            torch.nn.Linear(self.comment_dim, fea_dim), torch.nn.ReLU(),
                                            nn.Dropout(p=0.2))

        
        self.linear_text = nn.Sequential(Attention(dim=self.text_dim,heads=4,dropout=self.dropout),torch.nn.Linear(self.text_dim, fea_dim), torch.nn.ReLU(),nn.Dropout(p=self.dropout))
        
        self.linear_intro = nn.Sequential(Attention(dim=self.text_dim,heads=4,dropout=self.dropout),torch.nn.Linear(self.text_dim, fea_dim*2),torch.nn.ReLU(),nn.Dropout(p=self.dropout))
        #gpt
        self.linear_gpt = nn.Sequential(Attention(dim=self.text_dim,heads=4,dropout=0.1),torch.nn.Linear(self.text_dim, fea_dim), torch.nn.ReLU(),nn.Dropout(p=0.1))
        
        self.CrossmodalTransformer = MULTModel(128, 128, 128, 128, 0.1)
        
        self.gpt2_config = GPT2Config(
        n_embd=fea_dim,
        n_layer=6,
        n_head=8,
        use_cache=True,
        embd_pdrop=0.1,
        resid_pdrop=0.1,
        attn_pdrop=0.1
    )
        self.gpt2_model = GPT2Model(self.gpt2_config)
        
        
        
        # 扩散模型生成反事实特征
        
        self.beta_1 = 1e-4
        self.beta_T = 0.02
        self.T=1000
        self.register_buffer(
            'betas', torch.linspace(self.beta_1, self.beta_T, self.T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))
        
        self.model_t = Text_Noise_Pre(T=self.T, ch=256, dropout= self.dropout , in_ch=256)
        

        
        self.classifier = nn.Linear(256, 2)
        self.classifier_coun = nn.Linear(256, 2)

    def forward(self,**kwargs):
        
        # ### gpt ###
        gpt_data_inputid = kwargs['gpt_data_inputid']
        gpt_data_mask = kwargs['gpt_data_mask']
        fea_gpt = self.bert(gpt_data_inputid, attention_mask=gpt_data_mask)['last_hidden_state']
        fea_gpt = self.linear_gpt(fea_gpt)
        # fea_gpt=torch.mean(fea_gpt,dim=1)
        
        
         ## Title ###
        title_inputid = kwargs['title_inputid']#(batch,512)
        title_mask=kwargs['title_mask']#(batch,512)

        fea_text=self.bert(title_inputid,attention_mask=title_mask)['last_hidden_state'] #(batch,sequence,768)
        # print(fea_text.shape)
        fea_text=self.linear_text(fea_text)
        
        
        #        ### User Intro ###
        intro_inputid = kwargs['intro_inputid']
        intro_mask = kwargs['intro_mask']
        fea_intro = self.bert(intro_inputid, attention_mask=intro_mask)[1] #(batch, 768)  
        fea_intro = self.bert(intro_inputid, attention_mask=intro_mask)['last_hidden_state']  #当融合两个特征时，取隐藏层
        fea_intro = self.linear_intro(fea_intro)
        
        
        comments_inputid = kwargs['comments_inputid']  # (batch,20,250)
        comments_mask = kwargs['comments_mask']  # (batch,20,250)

        comments_like = kwargs['comments_like']
        comments_feature = []
        for i in range(comments_inputid.shape[0]):
            bert_fea = self.bert(comments_inputid[i], attention_mask=comments_mask[i])[1]
            comments_feature.append(bert_fea)
        comments_feature = torch.stack(comments_feature)  # (batch,seq,fea_dim)

        fea_comments = []
        for v in range(comments_like.shape[0]):
            comments_weight = torch.stack(
                [torch.true_divide((i + 1), (comments_like[v].shape[0] + comments_like[v].sum())) for i in
                 comments_like[v]])
            comments_fea_reweight = torch.sum(
                comments_feature[v] * (comments_weight.reshape(comments_weight.shape[0], 1)), dim=0)
            fea_comments.append(comments_fea_reweight)
        fea_comments = torch.stack(fea_comments)
        # print(fea_comments.shape)
        fea_comments = self.linear_comment(fea_comments)  # (batch,fea_dim)
        fea_comments=fea_comments.unsqueeze(1)
        
        
        # 使用gpt2推理文本信息
        fea_gpt = self.gpt2_model(inputs_embeds=fea_gpt)
        fea_gpt = fea_gpt.last_hidden_state
        
        fea_text = self.gpt2_model(inputs_embeds=fea_text)
        fea_text = fea_text.last_hidden_state

        fea_comments = self.gpt2_model(inputs_embeds=fea_comments)
        fea_comments = fea_comments.last_hidden_state
        
        
        fea_text, fea_gpt, fea_comments = self.CrossmodalTransformer(fea_text, fea_gpt, fea_comments)

        
        fea = torch.cat((fea_text.transpose(0, 1),fea_intro,fea_gpt.transpose(0, 1),fea_comments.transpose(0, 1)),dim=1)

        # print(fea.shape)
        
        
        # 构建反事实特征,使用扩散模型构建
        t_t = torch.randint(self.T, size=(fea.shape[0], ), device=fea.device) # batchsize (0->T-1)
        noise_t = torch.randn_like(fea) # 生成文本向量噪声
        x_tmp_t = (
            extract(self.sqrt_alphas_bar, t_t, fea.shape) * fea +
            extract(self.sqrt_one_minus_alphas_bar, t_t, fea.shape) * noise_t) # x_t 添加t步噪声 得到的 x_t_noise(t)
  
        # 无条件扩散,反向去噪过程
        x_t_pre = self.model_t(x_tmp_t, t_t, fea)  #torch.Size([64, 769, 256])
        # print(x_t_pre.shape)

        x_tmp_t=torch.mean(x_tmp_t,1)
        output_coun = self.classifier_coun(x_tmp_t)

        fea=torch.mean(fea,dim=1)
        output = self.classifier(fea)
        
        x_t_pre=torch.mean(x_t_pre,1)
        loss_t = F.mse_loss(x_t_pre, fea, reduction='none')

        fea_gpt=torch.mean(fea_gpt.transpose(0, 1),1)

        return  output, fea, fea_gpt.squeeze(1), output_coun,loss_t
 


 # Student model
class Student_Audio(nn.Module):
    def __init__(self, bert_model, fea_dim, dropout):
        super(Student_Audio, self).__init__()
        self.dropout=dropout
        self.text_dim=768

        self.vggish_layer = torch.hub.load(r'/root/autodl-tmp/hub/harritaylor_torchvggish_master', 'vggish', source="local")
        net_structure = list(self.vggish_layer.children())
        self.vggish_modified = nn.Sequential(*net_structure[-2:-1])

        self.linear_audio = nn.Sequential(Attention(dim=fea_dim, heads=4, dropout=self.dropout),
                                          torch.nn.Linear(fea_dim, fea_dim*2),torch.nn.ReLU(), nn.Dropout(p=0.15))
    
    ###### 用扩散模型添加反事实特征
    
        self.beta_1 = 1e-4
        self.beta_T = 0.02
        self.T=1000
        self.register_buffer(
            'betas', torch.linspace(self.beta_1, self.beta_T, self.T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))
    
        self.model_a = Audio_Noise_Pre(T=self.T, ch=256, dropout= self.dropout, in_ch=256)
        
        self.CrossmodalTransformer = MULTModel2(256, 256, 256, 0.1)

        self.classifier = nn.Linear(fea_dim*2, 2)
        self.classifier_coun = nn.Linear(fea_dim*2, 2)

    def forward(self, **kwargs):
        
        text = kwargs['t_fea']
        text = text.unsqueeze(1)
        
        
        audioframes = kwargs['audioframes']  # (batch,36,12288)
        audioframes_masks = kwargs['audioframes_masks']
        fea_audio = self.vggish_modified(audioframes)  # (batch, frames, 128)
        fea_audio = self.linear_audio(fea_audio)
        
        # 模态和文本模态对齐
        text,fea_audio = self.CrossmodalTransformer(text,fea_audio)
        fea_audio=fea_audio.transpose(0, 1)

        
        # 反事实特征构建
        # 构建反事实特征,使用扩散模型构建
        t_a = torch.randint(self.T, size=(fea_audio.shape[0], ), device=fea_audio.device) # batchsize (0->T-1)
        noise_a = torch.randn_like(fea_audio) # 生成文本向量噪声
        x_tmp_a = (
            extract(self.sqrt_alphas_bar, t_a, fea_audio.shape) * fea_audio +
            extract(self.sqrt_one_minus_alphas_bar, t_a, fea_audio.shape) * noise_a) # x_t 添加t步噪声 得到的 x_t_noise(t)
  
        # print(x_tmp_a.shape, t_a.shape, text.shape)
        # 无条件扩散,反向去噪过程
        x_a_pre = self.model_a(x_tmp_a, t_a, fea_audio)  #torch.Size([64, 769, 256])
        # print(x_t_pre.shape)

        
        # print(x_tmp_a.shape)
        x_tmp_a=torch.mean(x_tmp_a,1)
        # print(x_tmp_a.shape)
        output_coun = self.classifier_coun(x_tmp_a)
        
        fea_audio=torch.mean(fea_audio, -2)
        audio_logit=self.classifier(fea_audio)
        
        x_a_pre=torch.mean(x_a_pre,1)
        loss_a = F.mse_loss(x_a_pre, fea_audio, reduction='none')
        

        return audio_logit,fea_audio, output_coun, loss_a



    
    

    #student model
class Student_Frames(nn.Module):
    def __init__(self, bert_model, fea_dim, dropout):
        super(Student_Frames, self).__init__()
        self.dropout=dropout
        self.img_dim=4096
        self.video_dim=4096

        self.linear_img = nn.Sequential(Attention(dim=self.img_dim, heads=4, dropout=0.1),
                                        torch.nn.Linear(self.img_dim, fea_dim*2), torch.nn.ReLU(), nn.Dropout(p=0.1))
        
        self.linear_video = nn.Sequential(Attention(dim=self.video_dim,heads=4,dropout=0.1),torch.nn.Linear(self.video_dim, fea_dim*2), torch.nn.ReLU(),nn.Dropout(p=0.2))
        
        ####用扩散模型模拟反事实特征
        self.beta_1 = 1e-4
        self.beta_T = 0.02
        self.T=500
        self.register_buffer(
            'betas', torch.linspace(self.beta_1, self.beta_T, self.T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))
    
        self.model_v = Visual_Noise_Pre(T=self.T, ch=256, dropout= self.dropout, in_ch=256)
        
        self.CrossmodalTransformer = MULTModel(256, 256, 256, 128, 0.1)

        self.classifier = nn.Linear(fea_dim*2, 2)
        self.classifier_coun = nn.Linear(fea_dim*2, 2)

    def forward(self, **kwargs):
        text = kwargs['t_fea']
        text = text.unsqueeze(1)
        
        frames = kwargs['frames']  # (batch,30,4096)
        fea_frames = self.linear_img(frames)
        
        c3d = kwargs['c3d'] # (batch, 36, 4096)
        # c3d_masks = kwargs['c3d_masks']
        fea_video = self.linear_video(c3d)
        

        # 模态和文本模态对齐
        text,fea_frames,fea_video = self.CrossmodalTransformer(text,fea_frames,fea_video)
        fea_frames=fea_frames.transpose(0, 1)
        fea_video=fea_video.transpose(0, 1)
        # text=text.transpose(0, 1)
        
        
        fea=torch.cat((fea_frames,fea_video),dim=1)
        
        
        # 反事实特征构建
        # 构建反事实特征,使用扩散模型构建
        t_v = torch.randint(self.T, size=(fea.shape[0], ), device=fea.device) # batchsize (0->T-1)
        noise_v = torch.randn_like(fea) # 生成文本向量噪声
        x_tmp_v = (
            extract(self.sqrt_alphas_bar, t_v, fea.shape) * fea +
            extract(self.sqrt_one_minus_alphas_bar, t_v, fea.shape) * noise_v) # x_t 添加t步噪声 得到的 x_t_noise(t)
  
        # 无条件扩散,反向去噪过程
        x_v_pre = self.model_v(x_tmp_v, t_v, fea)  #torch.Size([64, 769, 256])
        # print(x_t_pre.shape)

        # print(x_tmp_a.shape)
        x_tmp_v=torch.mean(x_tmp_v,1)
        # print(x_tmp_a.shape)
        output_coun = self.classifier_coun(x_tmp_v)
        
   
        fea=torch.mean(fea, -2)

        frames_logit=self.classifier(fea)
        
        x_v_pre=torch.mean(x_v_pre,1)
        loss_v = F.mse_loss(x_v_pre, fea, reduction='none')

        return  frames_logit,fea, output_coun, loss_v
    

    

  ##fusion model  
    
class Fusion(nn.Module):  # 用多头自注意力机制融合三个训练好的模型推理出来的多模态特征
    def __init__(self, bert_model, fea_dim, dropout):
        super(Fusion, self).__init__()
        self.text_dim=768
        # self.dropout=dropout

        self.text_dim = 256
        self.visual_dim = 256
        self.audio_dim = 256
        self.multihead_attn = nn.MultiheadAttention(self.visual_dim + self.audio_dim, 4)

        self.W_hav =  nn.Linear(self.visual_dim + self.audio_dim + self.text_dim, self.text_dim)

        self.W_av = nn.Linear(self.visual_dim + self.audio_dim, self.text_dim)

        self.beta_shift = 2e-1 #视频和音频的比例

        self.LayerNorm = nn.LayerNorm(256)
        self.AV_LayerNorm = nn.LayerNorm(self.visual_dim + self.audio_dim)
        self.dropout = nn.Dropout(0.1)


        """Logit"""
        self.W = nn.Linear(256, 2)
        
        
        # 扩散模型生成反事实特征
        
        self.beta_1 = 1e-4
        self.beta_T = 0.02
        self.T=1000
        self.register_buffer(
            'betas', torch.linspace(self.beta_1, self.beta_T, self.T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))
        
        self.model_T = Text_Noise_Pre(T=self.T, ch=256, dropout= 0.1 , in_ch=256)
        
        self.classifier_coun = nn.Linear(fea_dim*2, 2)
        
        

    def forward(self, **kwargs):
        visual = kwargs['frames']
        acoustic = kwargs['audioframes']
        text_embedding = kwargs['text']
            
        
        eps = 1e-6
        nv_embedd = torch.cat((visual, acoustic), dim=-1)
        new_nv = self.multihead_attn(nv_embedd, nv_embedd, nv_embedd)[0] + nv_embedd

        av_embedd = self.dropout(self.AV_LayerNorm(new_nv))

        weight_av = F.relu(self.W_hav(torch.cat((av_embedd, text_embedding), dim=-1)))

        h_m = weight_av * self.W_av(av_embedd)

        em_norm = text_embedding.norm(2, dim=-1)
        hm_norm = h_m.norm(2, dim=-1)

        hm_norm_ones = torch.ones(hm_norm.shape, requires_grad=True).cuda()
        hm_norm = torch.where(hm_norm == 0, hm_norm_ones, hm_norm)

        thresh_hold = (em_norm / (hm_norm + eps)) * self.beta_shift

        ones = torch.ones(thresh_hold.shape, requires_grad=True).cuda()

        alpha = torch.min(thresh_hold, ones)
        alpha = alpha.unsqueeze(dim=-1)

        acoustic_vis_embedding = alpha * h_m

        embedding_output = self.dropout(
            self.LayerNorm(acoustic_vis_embedding + text_embedding)
        )
        #print(embedding_output.shape)

        logits = self.W(embedding_output)
        
        
        
        fea=embedding_output.unsqueeze(1)
        
        
        
        t_v = torch.randint(self.T, size=(fea.shape[0], ), device=fea.device) # batchsize (0->T-1)
        noise_v = torch.randn_like(fea) # 生成文本向量噪声
        x_tmp_v = (
            extract(self.sqrt_alphas_bar, t_v, fea.shape) * fea +
            extract(self.sqrt_one_minus_alphas_bar, t_v, fea.shape) * noise_v) # x_t 添加t步噪声 得到的 x_t_noise(t)
  
        # 无条件扩散,反向去噪过程
        x_v_pre = self.model_T(x_tmp_v, t_v, fea)  #torch.Size([64, 769, 256])
        # print(x_t_pre.shape)

        
        # print(x_tmp_a.shape)
        x_tmp_v=torch.mean(x_tmp_v,1)
        #print(x_tmp_v.shape)
        output_coun = self.classifier_coun(x_tmp_v)
        
   
    #         fea=torch.mean(fea, -2)

#         frames_logit=self.classifier(fea)
        
        x_v_pre=torch.mean(x_v_pre,1)
        loss_v = F.mse_loss(x_v_pre, fea, reduction='none')
 
        return logits, embedding_output, output_coun, loss_v
    

    

