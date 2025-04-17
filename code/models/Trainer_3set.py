import copy
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import tqdm
from sklearn.metrics import *
from tqdm import tqdm
from transformers import BertModel
from utils.metrics import *
from zmq import device

from .SelfAttention import *
from models.KDmodel import *
from models.meld_kd import *



    



# The training method of the teacher model
class Trainer3KD_T():
    def __init__(self,
                 model,
                 device,
                 lr,
                 dropout,
                 dataloaders,
                 weight_decay,
                 save_param_path,
                 writer,
                 epoch_stop,
                 epoches,
                 mode,
                 model_name,
                 event_num,
                 save_threshold=0.0,
                 start_epoch=0,
                 ):

        self.model = model
        self.device = device
        self.mode = mode
        self.model_name = model_name
        self.event_num = event_num

        self.dataloaders = dataloaders
        self.start_epoch = start_epoch
        self.num_epochs = epoches
        self.epoch_stop = epoch_stop
        self.save_threshold = save_threshold
        self.writer = writer

        if os.path.exists(save_param_path):
            self.save_param_path = save_param_path
        else:
            self.save_param_path = os.makedirs(save_param_path)
            self.save_param_path = save_param_path

        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout

        self.criterion = nn.CrossEntropyLoss()
        self.p = False
        self.best_train_fea = []
        self.best_test_fea = []
        self.feature_loss = Feature_Loss().cuda()

    

    def train(self):

        since = time.time()

        self.model.cuda()

        best_model_wts_test = copy.deepcopy(self.model.state_dict())
        best_acc_val = 0.0
        best_epoch_test = 0
        is_earlystop = False

        if self.mode == "eann":
            best_acc_test_event = 0.0
            best_epoch_test_event = 0

        for epoch in range(self.start_epoch, self.start_epoch + self.num_epochs):
            if is_earlystop:
                break
            print('-' * 50)
            print('Epoch {}/{}'.format(epoch + 1, self.start_epoch + self.num_epochs))
            print('-' * 50)

            p = float(epoch) / 100
            lr = self.lr / (1. + 10 * p) ** 0.75
            self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=lr)

            for phase in ['train', 'val', 'test']:
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()
                print('-' * 10)
                print(phase.upper())
                print('-' * 10)

                running_loss_fnd = 0.0
                running_loss = 0.0
                tpred = []
                tlabel = []

                if self.mode == "eann":
                    running_loss_event = 0.0
                    tpred_event = []
                    tlabel_event = []

                for batch in tqdm(self.dataloaders[phase]):
                    
                    
                                # 获取系统内存的使用情况
                    memory_info = psutil.virtual_memory()
                    print(f"Used Memory (MB): {memory_info.used / (1024 ** 2):.2f} MB")

                    # 查看当前GPU分配的内存（单位：字节）
                    allocated_memory = torch.cuda.memory_allocated()

                    # 转换为MB
                    allocated_memory_MB = allocated_memory / 1024 / 1024  # 转换为MB
                    print(f"Allocated memory: {allocated_memory_MB:.2f} MB")


                    time_elapsed = time.time() - since
                    print('Training complete in {:.2f}m {:.02f}s'.format(
                        time_elapsed // 60, time_elapsed % 60))
                    
                    
                    
                    batch_data = batch
                    for k, v in batch_data.items():
                        if isinstance(v, list):
                            # 如果 v 是一个列表，则将其中的每个张量移到 GPU
                             # 对列表中的每个元素进行检查和处理
                            batch_data[k] = [item.cuda() if isinstance(item, torch.Tensor) else item for item in v]
                        elif isinstance(v, torch.Tensor):
                      
                            # 否则，直接将张量移到 GPU
                            # print(type(k))
                            batch_data[k] = v.cuda()
                    label = batch_data['label']
                    if self.mode == "eann":
                        label_event = batch_data['label_event']

                    with torch.set_grad_enabled(phase == 'train'):
                        if self.mode == "eann":
                            outputs, outputs_event, fea = self.model(**batch_data)
                            loss_fnd = self.criterion(outputs, label)
                            loss_event = self.criterion(outputs_event, label_event)
                            loss = loss_fnd + loss_event
                            _, preds = torch.max(outputs, 1)
                            _, preds_event = torch.max(outputs_event, 1)
                        else:
                            # print(self.model)
                            outputs, fea , gpt_fea, output_coun ,loss_t = self.model(**batch_data)
                            
                            # print(outputs.shape)
                            _, preds = torch.max((outputs), 1)
                            # print(preds)

                            feature_loss = self.feature_loss(fea,gpt_fea)
                            
                            u=0.5
                            
                            loss = 0.3*(0.6*self.criterion(outputs, label) + 0.4*feature_loss) + 0.5*self.criterion(outputs-output_coun, label) + 0.2*loss_t.mean()

                            
                        if phase == 'train':
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                            self.optimizer.step()
                            
                            self.optimizer.zero_grad()

                    tlabel.extend(label.detach().cpu().numpy().tolist())
                    tpred.extend(preds.detach().cpu().numpy().tolist())
                    running_loss += loss.item() * label.size(0)

                    if self.mode == "eann":
                        tlabel_event.extend(label_event.detach().cpu().numpy().tolist())
                        tpred_event.extend(preds_event.detach().cpu().numpy().tolist())
                        running_loss_event += loss_event.item() * label_event.size(0)
                        running_loss_fnd += loss_fnd.item() * label.size(0)

                epoch_loss = running_loss / len(self.dataloaders[phase].dataset)
                print('Loss: {:.4f} '.format(epoch_loss))
                results = metrics(tlabel, tpred)
                print(results)
                self.writer.add_scalar('Loss/' + phase, epoch_loss, epoch + 1)
                self.writer.add_scalar('Acc/' + phase, results['acc'], epoch + 1)
                self.writer.add_scalar('F1/' + phase, results['f1'], epoch + 1)

                if self.mode == "eann":
                    epoch_loss_fnd = running_loss_fnd / len(self.dataloaders[phase].dataset)
                    print('Loss_fnd: {:.4f} '.format(epoch_loss_fnd))
                    epoch_loss_event = running_loss_event / len(self.dataloaders[phase].dataset)
                    print('Loss_event: {:.4f} '.format(epoch_loss_event))
                    self.writer.add_scalar('Loss_fnd/' + phase, epoch_loss_fnd, epoch + 1)
                    self.writer.add_scalar('Loss_event/' + phase, epoch_loss_event, epoch + 1)

                if phase == 'val' :
                    if results['acc'] > best_acc_val:
                        best_acc_val = results['acc']
                        best_model_wts_val = copy.deepcopy(self.model.state_dict())
                        best_epoch_val = epoch+1
                        if best_acc_val > self.save_threshold:
                            torch.save(self.model.state_dict(), self.save_param_path + "_val_epoch" + str(best_epoch_val) + "_{0:.4f}".format(best_acc_val))
                            print ("saved " + self.save_param_path + "_val_epoch" + str(best_epoch_val) + "_{0:.4f}".format(best_acc_val) )
                    else:
                        if epoch-best_epoch_val >= self.epoch_stop-1:
                            is_earlystop = True
                            print ("early stopping...")

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print("Best model on val: epoch" + str(best_epoch_val) + "_" + str(best_acc_val))

        if self.mode == "eann":
            print("Event: Best model on val: epoch" + str(best_epoch_val_event) + "_" + str(best_acc_val_event))


        self.model.load_state_dict(best_model_wts_val)

        print ("test result when using best model on val")
        return self.test()

    def test(self):
        # test_fea_list = []
        since = time.time()

        self.model.cuda()
        self.model.eval()

        pred = []
        label = []

        if self.mode == "eann":
            pred_event = []
            label_event = []

        for batch in tqdm(self.dataloaders['test']):
            with torch.no_grad():
                batch_data = batch
                for k, v in batch_data.items():
                    batch_data[k] = v.cuda()
                batch_label = batch_data['label']

                
                batch_outputs, fea, gpt_fea, output_coun ,loss_t = self.model(**batch_data)

                _, batch_preds = torch.max(batch_outputs, 1)

                label.extend(batch_label.detach().cpu().numpy().tolist())
                pred.extend(batch_preds.detach().cpu().numpy().tolist())

        print(get_confusionmatrix_fnd(np.array(pred), np.array(label)))
        print(metrics(label, pred))

        if self.mode == "eann" and self.model_name != "FANVM":
            print("event:")
            print(accuracy_score(np.array(label_event), np.array(pred_event)))


        return metrics(label, pred)



##蒸馏损失  
def CE_Loss(pred_outs, logit_t, hidden_s, hidden_t, labels,predst):  #此处的教师模型计算loss的方式不同于学生模型的计算方式,,logit_s, logit_t, hidden_s, hidden_t, batch_labels
    ori_loss = nn.CrossEntropyLoss()
    ori_loss = ori_loss(pred_outs, labels)   #pred_outs:学生模型的分类结果，计算分类损失
    logit_loss = Logit_Loss().cuda()
    # print(pred_outs.shape, logit_t.shape)
    logit_loss = logit_loss(pred_outs, logit_t)   #学生模型分类结果，教师模型推理结果，计算响应的蒸馏损失=类内损失+类间损失
       
    loss_val = ori_loss + logit_loss 
    
    return loss_val



# ########The training method of the student(frames/audio) model
# #在这一部分的前面需要加载教师模型，设置为测试模式，去训练学生模型
model_t = Teacher_model(bert_model=r"/root/autodl-tmp/new_fakesv/FakeSV-main/code/data/bert-base-chinese", fea_dim=128, dropout=0.1)
model_t.load_state_dict(torch.load('/root/autodl-tmp/new_fakesv/FakeSV-main/code/data/KDmodel_param/text/_val_epoch2_0.8320'))
for para in model_t.parameters():
    para.requires_grad = False
model_t = model_t.cuda()
model_t.eval()

class Trainer3KD_Sf():
    def __init__(self,
                 model,
                 device,
                 lr,
                 dropout,
                 dataloaders,
                 weight_decay,
                 save_param_path,
                 writer,
                 epoch_stop,
                 epoches,
                 mode,
                 model_name,
                 event_num,
                 save_threshold=0.0,
                 start_epoch=0,
                 ):

        self.model = model
        self.device = device
        self.mode = mode
        self.model_name = model_name
        self.event_num = event_num

        self.dataloaders = dataloaders
        self.start_epoch = start_epoch
        self.num_epochs = epoches
        self.epoch_stop = epoch_stop
        self.save_threshold = save_threshold
        self.writer = writer

        if os.path.exists(save_param_path):
            self.save_param_path = save_param_path
        else:
            self.save_param_path = os.makedirs(save_param_path)
            self.save_param_path = save_param_path

        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout

        self.criterion = nn.CrossEntropyLoss()
        self.p = False
        self.best_train_fea = []
        self.best_test_fea = []
        
        self.feature_loss = Feature_Loss().cuda()

    def train(self):

        since = time.time()

        self.model.cuda()

        best_model_wts_test = copy.deepcopy(self.model.state_dict())
        best_acc_val = 0.0
        # best_epoch_test = 0
        best_acc_test = 0.0
        best_epoch_test = 0
        is_earlystop = False

        if self.mode == "eann":
            best_acc_test_event = 0.0
            best_epoch_test_event = 0

        for epoch in range(self.start_epoch, self.start_epoch + self.num_epochs):
            if is_earlystop:
                break
            print('-' * 50)
            print('Epoch {}/{}'.format(epoch + 1, self.start_epoch + self.num_epochs))
            print('-' * 50)

            p = float(epoch) / 100
            lr = self.lr / (1. + 10 * p) ** 0.75
            self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=lr)

            for phase in ['train','val', 'test']:
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()
                print('-' * 10)
                print(phase.upper())
                print('-' * 10)

                running_loss_fnd = 0.0
                running_loss = 0.0
                tpred = []
                tlabel = []

                if self.mode == "eann":
                    running_loss_event = 0.0
                    tpred_event = []
                    tlabel_event = []

                for batch in tqdm(self.dataloaders[phase]):
                    batch_data = batch
                    for k, v in batch_data.items():
                        if isinstance(v, list):  #这里的分支选择语句是加入了gpt的数据的时候添加的
                            # 对列表中的每个元素进行检查和处理
                            batch_data[k] = [item.cuda() if isinstance(item, torch.Tensor) else item for item in v]
                        else:
                            # 直接对张量调用 cuda()
                            batch_data[k] = v.cuda()
                        # batch_data[k] = v.cuda()
                    label = batch_data['label']
                    if self.mode == "eann":
                        label_event = batch_data['label_event']

                    with torch.set_grad_enabled(phase == 'train'):
                        if self.mode == "eann":
                            outputs, outputs_event, fea = self.model(**batch_data)
                            loss_fnd = self.criterion(outputs, label)
                            loss_event = self.criterion(outputs_event, label_event)
                            loss = loss_fnd + loss_event
                            _, preds = torch.max(outputs, 1)
                            _, preds_event = torch.max(outputs_event, 1)
                        else:

                            
                            output_t, text, fea_gpt, output_coun,loss_t = model_t(**batch_data)
                            _, predst = torch.max(output_t, 1)
                            batch_data['t_fea'] = text
                            
                            
                            outputs,fea_audio, output_coun, loss_a = self.model(**batch_data)   #此处的模型是视频帧的模型或者是音频的模型，此时要在run。py的569行替换模型名称
                            _, preds = torch.max(outputs, 1)
                            

                            
                            # loss = 0.3*self.criterion(outputs, label) + 0.5*self.criterion(outputs-output_coun, label) + 0.2*loss_a.mean()  #终1，，0.3，0.4，0.3
                            
                            loss = 0.3*CE_Loss(outputs, output_t, fea_audio, text, label,predst) + 0.5*self.criterion(outputs-output_coun, label) + 0.2*loss_a.mean()  #终2
                            # loss = self.criterion(outputs, label)
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()
                            self.optimizer.zero_grad()

                    tlabel.extend(label.detach().cpu().numpy().tolist())
                    tpred.extend(preds.detach().cpu().numpy().tolist())
                    running_loss += loss.item() * label.size(0)

                    if self.mode == "eann":
                        tlabel_event.extend(label_event.detach().cpu().numpy().tolist())
                        tpred_event.extend(preds_event.detach().cpu().numpy().tolist())
                        running_loss_event += loss_event.item() * label_event.size(0)
                        running_loss_fnd += loss_fnd.item() * label.size(0)
                        
                        
                       

                epoch_loss = running_loss / len(self.dataloaders[phase].dataset)
                print('Loss: {:.4f} '.format(epoch_loss))
                results = metrics(tlabel, tpred)
                print(results)
                self.writer.add_scalar('Loss/' + phase, epoch_loss, epoch + 1)
                self.writer.add_scalar('Acc/' + phase, results['acc'], epoch + 1)
                self.writer.add_scalar('F1/' + phase, results['f1'], epoch + 1)

                if self.mode == "eann":
                    epoch_loss_fnd = running_loss_fnd / len(self.dataloaders[phase].dataset)
                    print('Loss_fnd: {:.4f} '.format(epoch_loss_fnd))
                    epoch_loss_event = running_loss_event / len(self.dataloaders[phase].dataset)
                    print('Loss_event: {:.4f} '.format(epoch_loss_event))
                    self.writer.add_scalar('Loss_fnd/' + phase, epoch_loss_fnd, epoch + 1)
                    self.writer.add_scalar('Loss_event/' + phase, epoch_loss_event, epoch + 1)

                if phase == 'val' :
                        if results['acc'] > best_acc_val:
                            best_acc_val = results['acc']
                            best_model_wts_val = copy.deepcopy(self.model.state_dict())
                            best_epoch_val = epoch+1
                            if best_acc_val > self.save_threshold:
                                torch.save(self.model.state_dict(), self.save_param_path + "_val_epoch" + str(best_epoch_val) + "_{0:.4f}".format(best_acc_val))
                                print ("saved " + self.save_param_path + "_val_epoch" + str(best_epoch_val) + "_{0:.4f}".format(best_acc_val) )
                        else:
                            if epoch-best_epoch_val >= self.epoch_stop-1:
                                is_earlystop = True
                                print ("early stopping...")

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print("Best model on val: epoch" + str(best_epoch_val) + "_" + str(best_acc_val))

        if self.mode == "eann":
            print("Event: Best model on val: epoch" + str(best_epoch_val_event) + "_" + str(best_acc_val_event))

    
        self.model.load_state_dict(best_model_wts_val)

        print ("test result when using best model on val")
        return self.test() 
    
    

    def test(self):
        # test_fea_list = []
        since = time.time()

        self.model.cuda()
        self.model.eval()

        pred = []
        label = []

        if self.mode == "eann":
            pred_event = []
            label_event = []

        for batch in tqdm(self.dataloaders['test']):
            with torch.no_grad():
                batch_data = batch
                for k, v in batch_data.items():
                    batch_data[k] = v.cuda()
                batch_label = batch_data['label']

            
                batch_outputs, fea, output_coun, loss_a = self.model(**batch_data)

                _, batch_preds = torch.max(batch_outputs, 1)

                label.extend(batch_label.detach().cpu().numpy().tolist())
                pred.extend(batch_preds.detach().cpu().numpy().tolist())

        print(get_confusionmatrix_fnd(np.array(pred), np.array(label)))
        print(metrics(label, pred))

        if self.mode == "eann" and self.model_name != "FANVM":
            print("event:")
            print(accuracy_score(np.array(label_event), np.array(pred_event)))


        return metrics(label, pred)



    
    
# # # # # ######## fusion model
# # # #在这一部分的前面需要加载教师模型，设置为测试模式，去训练学生模型
model_t = Teacher_model(bert_model=r"/root/autodl-tmp/new_fakesv/FakeSV-main/code/data/bert-base-chinese", fea_dim=128, dropout=0.1)
model_t.load_state_dict(torch.load('/root/autodl-tmp/new_fakesv/FakeSV-main/code/data/KDmodel_param/text/_val_epoch2_0.8320'))
for para in model_t.parameters():
    para.requires_grad = False
model_t = model_t.cuda()
model_t.eval()

model_a = Student_Audio(bert_model=r"/root/autodl-tmp/new_fakesv/FakeSV-main/code/data/bert-base-chinese", fea_dim=128, dropout=0.1)
model_a.load_state_dict(torch.load('/root/autodl-tmp/new_fakesv/FakeSV-main/code/checkpoints/SVFEND/KD/_val_epoch4_0.8359'))
for para in model_t.parameters():
    para.requires_grad = False
model_a = model_a.cuda()
model_a.eval()

model_f = Student_Frames(bert_model=r"/root/autodl-tmp/new_fakesv/FakeSV-main/code/data/bert-base-chinese", fea_dim=128, dropout=0.1)
model_f.load_state_dict(torch.load('/root/autodl-tmp/new_fakesv/FakeSV-main/code/checkpoints/SVFEND/KD/_val_epoch1_0.8398'))
for para in model_f.parameters():
    para.requires_grad = False
model_f = model_f.cuda()
model_f.eval()


class Trainer3KD_fusion():
    def __init__(self,
                 model,
                 device,
                 lr,
                 dropout,
                 dataloaders,
                 weight_decay,
                 save_param_path,
                 writer,
                 epoch_stop,
                 epoches,
                 mode,
                 model_name,
                 event_num,
                 save_threshold=0.0,
                 start_epoch=0,
                 ):

        self.model = model
        self.device = device
        self.mode = mode
        self.model_name = model_name
        self.event_num = event_num

        self.dataloaders = dataloaders
        self.start_epoch = start_epoch
        self.num_epochs = epoches
        self.epoch_stop = epoch_stop
        self.save_threshold = save_threshold
        self.writer = writer

        if os.path.exists(save_param_path):
            self.save_param_path = save_param_path
        else:
            self.save_param_path = os.makedirs(save_param_path)
            self.save_param_path = save_param_path

        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout

        self.criterion = nn.CrossEntropyLoss()
        self.p = False
        self.best_train_fea = []
        self.best_test_fea = []


    def train(self):
        
        
        #####读取gpt得到的json文件
        # data_test = pd.read_json(r'/root/autodl-tmp/new_fakesv/FakeSV-main/code/data/gptdata/gpt_time3_test_new_542.json', orient='records', dtype=False, lines=True)
        # vid2=data_test['新闻id'].astype(str)

        since = time.time()

        self.model.cuda()

        best_model_wts_test = copy.deepcopy(self.model.state_dict())
        best_acc_val = 0.0
        best_acc_test = 0.0
        best_epoch_test = 0
        is_earlystop = False

        if self.mode == "eann":
            best_acc_test_event = 0.0
            best_epoch_test_event = 0

        for epoch in range(self.start_epoch, self.start_epoch + self.num_epochs):
            if is_earlystop:
                break
            print('-' * 50)
            print('Epoch {}/{}'.format(epoch + 1, self.start_epoch + self.num_epochs))
            print('-' * 50)

            p = float(epoch) / 100
            lr = self.lr / (1. + 10 * p) ** 0.75
            self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=lr)  #原始优化器


            for phase in ['train','val','test']:
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()
                print('-' * 10)
                print(phase.upper())
                print('-' * 10)

                running_loss_fnd = 0.0
                running_loss = 0.0
                tpred = []
                tlabel = []
                
                # 添加计算新的损失的方法
                class_counts = torch.zeros(2).to("cuda")


                for batch in tqdm(self.dataloaders[phase]):
                    batch_data = batch
                    for k, v in batch_data.items():
                        if k!='vid' and k!='features_list':
                            batch_data[k] = v.cuda()
                    label = batch_data['label']
                    

                    with torch.set_grad_enabled(phase == 'train'):
                      
                        first_key = next(iter(batch_data))
                        length = batch_data[first_key].size(0)
                        # batch_data['text'] = torch.zeros(length, 128).cuda()
                          #此处的模型是视频帧的模型或者是音频的模型，此时要在run。py的569行替换模型名称
                        output_t, text, fea_gpt, output_coun,loss_t = model_t(**batch_data)
                        _, preds_t = torch.max(output_t, 1)
                        batch_data['t_fea'] = text


                        output_a,audio, output_coun, loss_a = model_a(**batch_data)
                        _, preds_a = torch.max(output_a, 1)

                        output_f,frames, output_coun, loss_a=model_f(**batch_data)
                        _, preds_f = torch.max(output_f, 1)

                        batch_data['frames'] = frames
                        batch_data['audioframes'] = audio
                        batch_data['text'] = text

                        # 计算 softmax 概率

                        outputs, fea , output_coun, loss_v = self.model(**batch_data)  
                        # outputs, fea  = self.model(**batch_data) 
                        _, preds = torch.max(outputs, 1)

                        # print(preds,label)

                        loss = 0.3*self.criterion(outputs, label) + 0.5*self.criterion(outputs - output_coun, label) + 0.2*loss_v.mean() #2

                    if phase == 'train':
                        loss.backward()
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                    tlabel.extend(label.detach().cpu().numpy().tolist())
                    tpred.extend(preds.detach().cpu().numpy().tolist())
                    running_loss += loss.item() * label.size(0)

                    if self.mode == "eann":
                        
                        tlabel_event.extend(label_event.detach().cpu().numpy().tolist())
                        tpred_event.extend(preds_event.detach().cpu().numpy().tolist())
                        running_loss_event += loss_event.item() * label_event.size(0)
                        running_loss_fnd += loss_fnd.item() * label.size(0)

                epoch_loss = running_loss / len(self.dataloaders[phase].dataset)
                print('Loss: {:.4f} '.format(epoch_loss))
                results = metrics(tlabel, tpred)
                print(results)
                self.writer.add_scalar('Loss/' + phase, epoch_loss, epoch + 1)
                self.writer.add_scalar('Acc/' + phase, results['acc'], epoch + 1)
                self.writer.add_scalar('F1/' + phase, results['f1'], epoch + 1)

                if self.mode == "eann":
                    epoch_loss_fnd = running_loss_fnd / len(self.dataloaders[phase].dataset)
                    print('Loss_fnd: {:.4f} '.format(epoch_loss_fnd))
                    epoch_loss_event = running_loss_event / len(self.dataloaders[phase].dataset)
                    print('Loss_event: {:.4f} '.format(epoch_loss_event))
                    self.writer.add_scalar('Loss_fnd/' + phase, epoch_loss_fnd, epoch + 1)
                    self.writer.add_scalar('Loss_event/' + phase, epoch_loss_event, epoch + 1)

                if phase == 'val' :
                        if results['acc'] > best_acc_val:
                            best_acc_val = results['acc']
                            best_model_wts_val = copy.deepcopy(self.model.state_dict())
                            best_epoch_val = epoch+1
                            if best_acc_val > self.save_threshold:
                                # torch.save(self.model.state_dict(), self.save_param_path + "_val_epoch" + str(best_epoch_val) + "_{0:.4f}".format(best_acc_val))
                                print ("saved " + self.save_param_path + "_val_epoch" + str(best_epoch_val) + "_{0:.4f}".format(best_acc_val) )
                        else:
                            if epoch-best_epoch_val >= self.epoch_stop-1:
                                is_earlystop = True
                                print ("early stopping...")

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print("Best model on val: epoch" + str(best_epoch_val) + "_" + str(best_acc_val))

        if self.mode == "eann":
            print("Event: Best model on val: epoch" + str(best_epoch_val_event) + "_" + str(best_acc_val_event))

    
        self.model.load_state_dict(best_model_wts_val)

        print ("test result when using best model on val")
        return self.test() 
    
    

    def test(self):
        # test_fea_list = []
        since = time.time()

        self.model.cuda()
        self.model.eval()

        pred = []
        label = []

        if self.mode == "eann":
            pred_event = []
            label_event = []

        for batch in tqdm(self.dataloaders['test']):
            with torch.no_grad():
                batch_data = batch
                for k, v in batch_data.items():
                    if k!='vid' and k!='features_list':
                        batch_data[k] = v.cuda()
                batch_label = batch_data['label']

                first_key = next(iter(batch_data))
                length = batch_data[first_key].size(0)
                # batch_data['text'] = torch.zeros(length, 128).cuda()
                    #此处的模型是视频帧的模型或者是音频的模型，此时要在run。py的569行替换模型名称
                output_t, text, fea_gpt, output_coun,loss_t = model_t(**batch_data)
                _, preds_t = torch.max(output_t, 1)
                batch_data['t_fea'] = text


                output_a,audio, output_coun, loss_a = model_a(**batch_data)
                _, preds_a = torch.max(output_a, 1)

                output_f,frames, output_coun, loss_a=model_f(**batch_data)
                _, preds_f = torch.max(output_f, 1)

                batch_data['frames'] = frames
                batch_data['audioframes'] = audio
                batch_data['text'] = text

                batch_outputs, fea, output_coun, loss_v = self.model(**batch_data)

                _, batch_preds = torch.max(batch_outputs, 1)

                label.extend(batch_label.detach().cpu().numpy().tolist())
                pred.extend(batch_preds.detach().cpu().numpy().tolist())

        print(get_confusionmatrix_fnd(np.array(pred), np.array(label)))
        print(metrics(label, pred))

        if self.mode == "eann" and self.model_name != "FANVM":
            print("event:")
            print(accuracy_score(np.array(label_event), np.array(pred_event)))


        return metrics(label, pred)
    
    
