#!/usr/bin/env python
# _*_ coding:utf-8 _*_

"""Based on DiaASQ:https://github.com/something678/TodKat"""
# from src.Roberta import MultiHeadAttention
from transformers import AutoModel, AutoConfig
from src.common import MultiHeadAttention
import pdb
import numpy as np
import torch
import torch.nn as nn
from itertools import accumulate
from src.GAT_model import *
from src.gcn import *
from scipy.linalg import block_diag
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
class BertWordPair(nn.Module):
    def __init__(self, cfg,args):
        super(BertWordPair, self).__init__()
        self.args = args
        self.bert = AutoModel.from_pretrained(cfg.bert_path)
        bert_config = AutoConfig.from_pretrained(cfg.bert_path)

        self.dense_layers = nn.ModuleDict({
            'ent': nn.Linear(bert_config.hidden_size, cfg.inner_dim * 4 * 6),
            'rel': nn.Linear(bert_config.hidden_size, cfg.inner_dim * 4 * 3),
            'pol': nn.Linear(bert_config.hidden_size, cfg.inner_dim * 4 * 4),
        })
        self.linear = nn.Linear(cfg.hidden_dim*3,cfg.hidden_dim) 
        self.dropout = nn.Dropout(bert_config.hidden_dropout_prob)

        att_head_size = int(bert_config.hidden_size / bert_config.num_attention_heads)

        self.cfg = cfg 
        self.pos_embedding = torch.nn.Embedding(5, cfg.pos_dim)
        self.bilstm_encoder = torch.nn.LSTM(cfg.pos_dim, cfg.lstm_dim, num_layers=1, batch_first=True,
                                            bidirectional=True)
        # GCN Module
        self.gcn_layer = GCNModel(cfg) 
    def _get_pos_embedding(self, sentence_poses, mask):
        pos_embs = []
        st = 0
        ed = 0
        for i,pos in enumerate(sentence_poses):
            pos = pos.to(torch.int)
            pos_embedding = self.pos_embedding(pos) #[6,139,100] 
            ed+=len(pos)
            embedding = pos_embedding * mask[st:ed].unsqueeze(2).float().expand_as(pos_embedding)
            st = ed
            pos_embs.append(embedding)
        pos_embs = torch.stack(pos_embs)
        return pos_embs
    def _lstm_feature(self, embeddings):
        contexts = []
        for i,embedding in enumerate(embeddings):
            context, _ = self.bilstm_encoder(embedding) #[6,55,lstm_dim*2]
            contexts.append(context)
        context = torch.stack(contexts)
        return contexts
    def thread_classify_matrix(self, kwargs, sequence_outputs, mat_name='ent'):
        thread_masks = kwargs['thread_sentence_masks'] if mat_name == 'ent' else kwargs['thread_full_masks']
        matrix = kwargs[f'{mat_name}_matrix']
        dense_layer = self.dense_layers[mat_name]
        thread_lengths = kwargs['thread_input_masks'].sum(1) #[70,32]
        thread_utterance_spans = kwargs['thread_utterance_spans'] #[[(1, 12), (15, 49), (52, 55), (58, 74)], [(1, 12), (77, 81), (84, 96)]]
        thread_utterance_spans_raw = kwargs['thread_utterance_spans_raw']
        thread_loss = 0.0
        input_labels = []
        active_labels = []
        pred_logits = []
        for i,output in enumerate(sequence_outputs):
            length = thread_lengths[i]
            output = dense_layer(output)
            output = torch.split(output,self.cfg.inner_dim*4, dim=-1)
            output = torch.stack(output,dim=-2) # [165, 6,1024]
           
            q_token,q_utterance,k_token,k_utterance = torch.split(output,self.cfg.inner_dim,dim=-1)
            t_pred_logits = torch.einsum('bmh,xnh->bxn', q_token, k_token)
            t_pred_logits = t_pred_logits[1:length-1,1:length-1,:] #有效的窗口内容,不包括cls
            t_span = thread_utterance_spans[i]
            input_label = self.get_thread_labels(t_span,matrix)
            active_loss =  thread_masks[i,1:length-1,1:length-1].reshape(-1) == 1 
            active_label = input_label.reshape(-1)[active_loss]   
            active_logit = t_pred_logits.reshape(-1, t_pred_logits.shape[-1])[active_loss] #[121,6]     
            nums = t_pred_logits.shape[-1]
            criterion = nn.CrossEntropyLoss(output.new_tensor([1.0] + [self.cfg.loss_weight[mat_name]] * (nums - 1)))
            t_loss = criterion(active_logit,active_label)
            thread_loss += t_loss
        
            pred_logits.append(t_pred_logits)
            input_labels.append(input_label)
            active_labels.append(active_label)
        loss,tags = self.inference(pred_logits,input_labels,thread_utterance_spans_raw,criterion)
        return loss,tags
    def inference(self,pred_logits,gold_logits,thread_utterance_spans_raw,criterion):
        final_logits = self.thread2dialogue(pred_logits,thread_utterance_spans_raw)
        final_labels = self.thread2dialogue(gold_logits,thread_utterance_spans_raw,False)
        active_logits = final_logits.reshape(-1, final_logits.shape[-1])
        active_labels = final_labels.reshape(-1) 
        loss = criterion(active_logits,active_labels)
        return loss,final_logits
    def thread2dialogue(self,final_logits,thread_spans,pred=True):
        
        max_lens=0
        thread_nums = len(final_logits)
        thread_spans = [[num for span in sublist for num in span] for sublist in thread_spans]
        max_lens = max(max(sublist) for sublist in thread_spans) + 1
        if pred==False: #gold
            matrix = torch.zeros([max_lens, max_lens],dtype=torch.long).to(self.cfg.device)
        else:
            matrix = torch.zeros([max_lens, max_lens,final_logits[0].shape[-1]]).to(self.cfg.device)
        for i,logit in enumerate(final_logits):
            span = thread_spans[i]#[0, 11, 12, 46, 47, 50, 51, 67]
            if(i==0): #ABCD 可以直接划分完整区域
                matrix[span[0]:span[-1]+1,span[0]:span[-1]+1] = logit
            else: #默认线程至少2个句子
                matrix[:span[1]+1,span[2]:span[-1]+1] = logit[:span[1]+1,span[1]+1:] #行坐标为root的范围，列坐标为DE的范围
                matrix[span[2]:span[-1]+1,:span[1]+1] = logit[span[1]+1:,:span[1]+1] #行坐标为其余句子的范围，列坐标为root的范围
                matrix[span[2]:span[-1]+1,span[2]:span[-1]+1] = logit[span[1]+1:,span[1]+1:] # DE的正方形面积 
        # pdb.set_trace()
        return matrix
    
    def get_thread_labels(self,t_span,matrix):
        input_label = []
        for (x1,y1)in t_span: 
            region = []
            for (x2,y2) in t_span:
                region1 = matrix[0,x1:y1+1,x2:y2+1] 
                region.append(region1)
            region = torch.cat(region, axis=1) 
            input_label.append(region)  
        input_label = torch.cat(input_label, axis=0)  
        return input_label
    
    def forward(self,args,**kwargs):
        input_ids, input_masks, input_segments = [kwargs[w] for w in ['input_ids', 'input_masks', 'input_segments']]
        thread_input_ids,thread_input_masks, thread_input_segments =  [kwargs[w] for w in ['thread_input_ids', 'thread_input_masks', 'thread_input_segments']]
        
        reply_masks, speaker_masks, thread_masks, dialogue_length = [kwargs[w] for w in ['reply_masks', 'speaker_masks', 'thread_masks', 'dialogue_length']]

        dis_adj, spk_adj,pos,dep_matrix = [kwargs[w] for w in ['dis_adj','spk_adj','pos','dep_matrix']] 
        
        sequence_outputs = self.bert(input_ids, token_type_ids=input_segments, attention_mask=input_masks)[0] #[29, 61, 768]
        thread_sequence_outputs = self.bert(thread_input_ids,token_type_ids=thread_input_segments,attention_mask=thread_input_masks)[0]
       #--------------------------------------------#
        GAT_model = DualGATs(args) 
        GAT_model.to(torch.device("cuda"))
        hidden_size = sequence_outputs.shape[2] 
        num_sentences = dis_adj.size(1)
        cls_features = torch.zeros((dis_adj.size(0),num_sentences, hidden_size)).to(torch.device("cuda")) #[B,max_U,emd_dim] [4,9,768]
        st = 0
        for i in range(dis_adj.size(0)):#batch 
            for j in range(dialogue_length[i]):#utterance numbers
                cls_features[i, j, :] = sequence_outputs[st, 0, :]  # 0 表示CLS特征
                st += 1
      
        gat_outputs,gat_loss = GAT_model(cls_features,dis_adj,spk_adj) #[4,9,768]
        gat_outputs = gat_outputs.unsqueeze(2).expand(-1,-1,sequence_outputs.shape[1],-1)
        
        pos_embedding = self._get_pos_embedding(pos, input_masks) 
        lstm_feature = self._lstm_feature(pos_embedding)
    
        # GCN encoder
        """
        dep_matrix:   [batch, 10,55,55]
        lstm_features:[batch,10,55,lstm_dim*2] [1,10,55,600]
        pos_embedding:[batch,sentence_nums,55,emb_dim] [1,10,55,100]
        input_masks:  [10,55],10为batch*sentence_nums
        gcn_feature:  [batch,sentence_nums,max_len,hidden_dim]:[1,10,55,768]
        sequence_outputs: [batch*sentence_nums,max_len,hidden_dim]: [10,55,768]
        """
        gcn_feature = self.gcn_layer(lstm_feature, dep_matrix, input_masks) ##[batch,sentence_nums,max_len,hidden_dim]:[1,7, 105,768]
       
        st = 0
        ed = 0
        new_sequence_outputs = []
        for i in range(dis_adj.size(0)):#batch
            ed += dialogue_length[i] #utterance numbers
            new_sequence_output = torch.cat((sequence_outputs[st:ed], gat_outputs[i],gcn_feature[i]),dim=-1) #句法信息+ DualGAT
            st = ed
            new_sequence_outputs.append(new_sequence_output)
        sequence_outputs = torch.stack(new_sequence_outputs) #[1, 7, 105, 2304]
        sequence_outputs = self.linear(sequence_outputs) #([1, 7, 105, 768])
        sequence_outputs = torch.squeeze(sequence_outputs,dim=0) ##([7, 105,768 )
        utterance_spans = kwargs['utterance_spans']
        thread_sents_index = kwargs['thread_sents_index']
        for i in range(dis_adj.size(0)):#batch
            for idx,thread_sent_output in enumerate(thread_sequence_outputs):
                st = 1 #跳过cls
                index = thread_sents_index[i][idx]
                # pdb.set_trace()
                for t_idx in index:
                    x = utterance_spans[i][t_idx][0]
                    y = utterance_spans[i][t_idx][1]
                    ed = st + y-x+1
                    thread_sequence_outputs[idx][st:ed] += sequence_outputs[t_idx][1:y-x+2]
                    st = ed
        
        loss0,tags0 = self.thread_classify_matrix(kwargs, thread_sequence_outputs, 'ent')
        loss1,tags1 = self.thread_classify_matrix(kwargs, thread_sequence_outputs, 'rel')
        loss2,tags2 = self.thread_classify_matrix(kwargs, thread_sequence_outputs, 'pol')
       
        return (loss0, loss1, loss2), (tags0, tags1, tags2)
    
