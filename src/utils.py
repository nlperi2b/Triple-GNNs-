#!/usr/bin/env python

import torch
import numpy as np
from attrdict import AttrDict
from scipy.linalg import block_diag
from collections import defaultdict
from attrdict import AttrDict 
import pdb
from torch.utils.data import Dataset, DataLoader
import os
import pickle as pkl
import random
from loguru import logger
import json

from src.common import WordPair
from src.preprocess import Preprocessor
# from src.run_eval1 import Template as Run_eval
from src.run_eval import Template as Run_eval

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

class MyDataLoader:
    def __init__(self, cfg):
        path = os.path.join(cfg.preprocessed_dir, '{}_{}.pkl'.format(cfg.lang, cfg.bert_path.replace('/', '-')))
        preprocessor = Preprocessor(cfg)
        
        data = None
        if not os.path.exists(path):
            logger.info('Preprocessing data...')
            data = preprocessor.forward()
            logger.info('Saving preprocessed data to {}'.format(path))
            if not os.path.exists(cfg.preprocessed_dir):
                os.makedirs(cfg.preprocessed_dir)
            pkl.dump(data, open(path, 'wb'))
        
        logger.info('Loading preprocessed data from {}'.format(path))
        self.data = pkl.load(open(path, 'rb')) if data is None else data

        self.kernel = WordPair()
        self.config = cfg 

    def worker_init(self, worked_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    def masks(self,thread_sentence_length):
        thread_sentence_masks = []
        mask= []
        max_lens = int(max(thread_sentence_length))
        for i,length in enumerate(thread_sentence_length): 
           
            mask = np.zeros([max_lens, max_lens], dtype=int)
            t_mask = [np.triu(np.ones([length, length ], dtype=int))]
            t_mask = block_diag(*t_mask)
            mask[:len(t_mask), :len(t_mask)] = t_mask
            mask =  mask.tolist()
            thread_sentence_masks.append(mask) 
        
        thread_full_masks = [] 
        for i,length in enumerate(thread_sentence_length): 
            cur_masks =np.zeros((max_lens, max_lens), dtype=np.int64)
            cur_mask = np.zeros((length, length), dtype=np.int64)
            cur_mask.fill(1)
            cur_masks[:length, :length] = cur_mask
            thread_full_masks.append(cur_masks.tolist())

        return thread_sentence_masks, thread_full_masks
    
    def collate_fn(self, lst):
        # pdb.set_trace()
        doc_id, input_ids, input_masks, input_segments,sentence_length, token2sents, utterance_index, \
            token_index, thread_length, token2speaker, reply_mask, speaker_mask, thread_mask, pieces2words, new2old, \
                triplets, pairs, entity_list, rel_list, polarity_list,speaker_adj,structure_adj,dep_matrix,pos_tags, \
                    thread_input_ids,thread_input_masks,thread_input_segments,utterance_spans,thread_sents_index,\
                        utterance_spans_raw,token2sents_raw,sentence_length_raw = zip(*lst)

        dialogue_length = list(map(len, input_ids))

        max_lens = max(map(lambda line: max(map(len, line)), input_ids))
        padding = lambda input_batch: [w + [0] * (max_lens - len(w)) for line in input_batch for w in line]
        input_ids, input_masks, input_segments = map(padding, [input_ids, input_masks, input_segments])
        #--------------thread----------------#
        thread_max_lens = max(map(lambda line: max(map(len, line)), thread_input_ids))
        thread_padding = lambda input_batch: [w + [0] * (thread_max_lens - len(w)) for line in input_batch for w in line]
        thread_input_ids, thread_input_masks, thread_input_segments = map(thread_padding, [thread_input_ids, thread_input_masks, thread_input_segments])
        #-------------thread-----------------#
        max_lens = max(map(len, token2sents))
        padding = lambda input_batch: [w + [0] * (max_lens - len(w)) for w in input_batch]
        token2sents, utterance_index, token_index, token2speaker = map(padding, [token2sents, utterance_index, token_index, token2speaker])
        max_lens2 = max(map(len, token2sents_raw))
        sentence_masks_raw = np.zeros([len(token2sents_raw), max_lens2, max_lens2], dtype=int)
        padding_list = lambda input_batch : [list(map(list, w)) + [[0, 0, 0]] * (max(map(len, input_batch)) - len(w)) for w in input_batch]
        entity_lists, rel_lists, polarity_lists = map(padding_list, [entity_list, rel_list, polarity_list])
       
        max_tri_num = max(map(len, triplets))
        triplet_masks = [[1] * len(w) + [0] * (max_tri_num - len(w)) for w in triplets]
        triplets = [list(map(list, w)) + [[0] * 7] * (max_tri_num - len(w)) for w in triplets]

        sentence_masks = np.zeros([len(token2sents), max_lens, max_lens], dtype=int)
        for i in range(len(sentence_length)):
            masks = [np.triu(np.ones([lens, lens], dtype=int)) for lens in sentence_length[i]]
            masks = block_diag(*masks)
            sentence_masks[i, :len(masks), :len(masks)] = masks
        sentence_masks = sentence_masks.tolist()

        flatten_length = list(map(sum, sentence_length))
        cur_masks = (np.expand_dims(np.arange(max(flatten_length)), 0) < np.expand_dims(flatten_length, 1)).astype(np.int64)
        full_masks = (np.expand_dims(cur_masks, 2) * np.expand_dims(cur_masks, 1)).tolist()
        
        for i in range(len(sentence_length_raw)):
            masks = [np.triu(np.ones([lens, lens], dtype=int)) for lens in sentence_length_raw[i]]
            masks = block_diag(*masks)
            sentence_masks_raw[i, :len(masks), :len(masks)] = masks
        sentence_masks_raw = sentence_masks_raw.tolist()

        flatten_length = list(map(sum, sentence_length_raw))
        cur_masks = (np.expand_dims(np.arange(max(flatten_length)), 0) < np.expand_dims(flatten_length, 1)).astype(np.int64)
        full_masks_raw = (np.expand_dims(cur_masks, 2) * np.expand_dims(cur_masks, 1)).tolist()

        entity_matrix = self.kernel.list2rel_matrix4batch(entity_lists, max_lens)
        rel_matrix = self.kernel.list2rel_matrix4batch(rel_lists, max_lens)
        polarity_matrix = self.kernel.list2rel_matrix4batch(polarity_lists, max_lens)

        new_reply_masks = np.zeros([len(reply_mask), max_lens, max_lens])
        for i in range(len(new_reply_masks)):
            lens = len(reply_mask[i])
            new_reply_masks[i, :lens, :lens] = reply_mask[i]

        new_speaker_masks = np.zeros([len(speaker_mask), max_lens, max_lens])
        for i in range(len(new_speaker_masks)):
            lens = len(speaker_mask[i])
            new_speaker_masks[i, :lens, :lens] = speaker_mask[i]

        new_thread_masks = np.zeros([len(thread_mask), max_lens, max_lens])
        for i in range(len(new_thread_masks)):
            lens = len(thread_mask[i])
            new_thread_masks[i, :lens, :lens] = thread_mask[i]
    
        #==================================ADD adj==================================#
        max_lens = max(map(len,speaker_adj))
        new_spk_adj = np.zeros([len(speaker_adj), max_lens, max_lens]) #[B,max_utterance_accounts(max_U),(max_U)]
        for i in range(len(new_spk_adj)):
            lens = len(speaker_adj[i])
            new_spk_adj[i, :lens, :lens] = speaker_adj[i]
        new_dis_adj = np.zeros([len(structure_adj), max_lens, max_lens]) #[1,6,6]
        for i in range(len(new_dis_adj)):
            lens = len(structure_adj[i])
            new_dis_adj[i, :lens, :lens] = structure_adj[i]
        #====================句法信息：adj和pos===========================#
        max_words = 0
        max_len = 0
        max_lens = max(map(len,dep_matrix))  #6
        for i,adj in enumerate(dep_matrix):
            for j, words in enumerate(adj):
                max_len = max(max_len,len(adj))
                max_words = max(max_words,len(dep_matrix[i][j]))
        new_dep_matrix = np.zeros([len(dep_matrix), max_lens, max_words,max_words])
        for i in range(len(new_dep_matrix)):
            for j in range(len(dep_matrix[i])):
                new_dep_matrix[i, j, :len(dep_matrix[i][j][0]),:len(dep_matrix[i][j][0])] = dep_matrix[i][j] #[9,49,49]
        new_dep_matrix = torch.tensor(new_dep_matrix).to(self.config.device)
        padded_new_dep_matrix =torch.zeros((new_dep_matrix.shape[0],new_dep_matrix.shape[1],new_dep_matrix.shape[2]+2,new_dep_matrix.shape[3]+2))
        padded_new_dep_matrix[:,:,1:-1,1:-1] = new_dep_matrix

        max_lens = max(map(len,pos_tags))
        new_pos_tags = np.zeros([len(pos_tags), max_lens,max_words]) #(1, 9, 93) [batch, sentences_nums, max_U_tokens]
        for i in range(len(new_pos_tags)):
            for j in range(len(pos_tags[i])):
                # pdb.set_trace()
                lens = len(pos_tags[i][j])
                new_pos_tags[i,j,:lens] = pos_tags[i][j]
        new_pos_tags = torch.tensor(new_pos_tags)
        padded_new_pos_tags = torch.cat((torch.zeros_like(new_pos_tags[:, :, :1]),new_pos_tags, torch.zeros_like(new_pos_tags[:, :, -1:])), dim=2)#头和尾加上[cls] [sep]占位符的mask
        thread_sentence_length = torch.tensor(thread_input_masks).sum(1)
        thread_sentence_masks,thread_full_masks = self.masks(thread_sentence_length)
        thread_utterance_spans = []
        thread_utterance_spans_raw = []
        for i in range(len(utterance_spans)): #batch
            for j in range(len(thread_sents_index[i])): #thread
                thread_utterance_span=[]
                thread_utterance_span_raw=[]
                for k in range(len(thread_sents_index[i][j])):# sentences
                    thread_utterance_span.append(utterance_spans[i][thread_sents_index[i][j][k]])
                    thread_utterance_span_raw.append(utterance_spans_raw[i][thread_sents_index[i][j][k]])
                thread_utterance_spans.append(thread_utterance_span)
                thread_utterance_spans_raw.append(thread_utterance_span_raw)
 #============================================================================#
        res = {
            "doc_id": doc_id,
			"input_ids": input_ids, "input_masks": input_masks, "input_segments": input_segments,
            'ent_matrix': entity_matrix,   'rel_matrix': rel_matrix, 'pol_matrix': polarity_matrix,
            'sentence_masks': sentence_masks, 'full_masks': full_masks,             
            'triplets': triplets, 'triplet_masks': triplet_masks, 'pairs': pairs,
            'token2sents': token2sents, 'dialogue_length': dialogue_length,
            'utterance_index': utterance_index, 'token_index': token_index,
            'thread_lengths': thread_length, 'token2speakers': token2speaker,
            'reply_masks': new_reply_masks, 'speaker_masks': new_speaker_masks, 'thread_masks': new_thread_masks,
            'pieces2words': pieces2words, 'new2old': new2old,
            'spk_adj':new_spk_adj, 'dis_adj':new_dis_adj,'dep_matrix':padded_new_dep_matrix,'pos':padded_new_pos_tags,
            'thread_input_ids':thread_input_ids,'thread_input_masks':thread_input_masks,'thread_input_segments':thread_input_segments,
            'utterance_spans':utterance_spans,'thread_sents_index':thread_sents_index,'thread_sentence_masks':thread_sentence_masks,'thread_full_masks':thread_full_masks,
            'thread_utterance_spans':thread_utterance_spans,'thread_utterance_spans_raw':thread_utterance_spans_raw,'token2sents_raw':token2sents_raw,
            'sentence_masks_raw': sentence_masks_raw, 'full_masks_raw': full_masks_raw
        }
        nocuda = ['thread_lengths', 'pairs', 'doc_id', 'pieces2words', 'new2old','new_pos_tags','utterance_spans','thread_sents_index','thread_utterance_spans','thread_utterance_spans_raw']
        res = {k: v if k in nocuda else torch.tensor(v).to(self.config.device) for k, v in res.items()}
        return res
       
    def getdata(self):
        
        load_data = lambda mode: DataLoader(MyDataset(self.data[mode]), num_workers=0, worker_init_fn=self.worker_init, 
                                                shuffle=(mode == 'train'),  batch_size=self.config.batch_size, collate_fn=self.collate_fn)
        
        train_loader, valid_loader, test_loader = map(load_data, 'train valid test'.split())

        line = 'polarity_dict target_dict aspect_dict opinion_dict entity_dict relation_dict'.split()
        for w, z in zip(line, self.data['label_dict']):
            self.config[w] = z

        res = (train_loader, valid_loader, test_loader, self.config)

        return res
    
class RelationMetric:
    def __init__(self, config):
        self.clear()
        self.kernel = WordPair()
        self.predict_result = defaultdict(list)
        self.config = config
    
    def trans2position(self, triplet, pieces2words):
        res = []
        """
        recover the position of entities in the original sentence

        new2old: transfer position from index with CLS and SEP to index without CLS and SEP
        pieces2words: transfer position from index of wordpiece to index of original words 

        Example:
        list0 (original sentence):"London is the capital of England"
        list1 (tokenized sentence): "Lon ##don is the capital of England"
        list2 (packed sentence): "[CLS] Lon #don is the capital of England [SEP]"
        predicted entity: (1, 2), denotes "Lon #don" in list2

        new2old: list2->list1
          = {'1': 0, '2': 1, '3': 2, '4': 3, '5': 4, ...}
        pieces2words: list1->list0
          = {'0': 0, '1': 0, '2': 1, '3': 2, '4': 3, ...}

        input  -> entity in list2: "Lon #don" (1, 2)
        middle -> entity in list1: "Lon #don" (0, 1)
        output -> entity in list0: "London"   (0, 0)
        """
        head = lambda x : pieces2words[x]
        tail = lambda x : pieces2words[x]


        triplet = list(triplet)
        for s0, e0, s1, e1, s2, e2, pol in triplet:
            ns0, ns1, ns2 = head(s0), head(s1), head(s2)
            ne0, ne1, ne2 = tail(e0), tail(e1), tail(e2)
            res.append([ns0, ne0, ns1, ne1, ns2, ne2, pol])
        return res
    
    def trans2pair(self, pred_pairs, pieces2words):
        new_pairs = {}
        # new_pos = lambda x : pieces2words[new2old[x]]
        new_pos = lambda x : pieces2words[x]
        for k, line in pred_pairs.items():
            new_line = []
            for s0, e0, s1, e1 in line:
                s0, e0, s1, e1 = new_pos(s0), new_pos(e0), new_pos(s1), new_pos(e1)
                new_line.append([s0, e0, s1, e1])
            new_pairs[k] = new_line
        return new_pairs

    def filter_entity(self, ent_list, pieces2words):
        res = []

        # If the entity is a sub-string of another entity, remove it
        # ent_list = sorted(ent_list, key=lambda x: (x[0], -x[1]))
        # ent_list = [w for i, w in enumerate(ent_list) if i == 0 or w[0] != ent_list[i-1][0]]

        for s, e, pol in ent_list:
            # ns, ne = pieces2words[new2old[s]], pieces2words[new2old[e]]
            ns, ne = pieces2words[s], pieces2words[e]
            res.append([ns, ne, pol])
        return res

    def add_instance(self, data, pred_ent_matrix, pred_rel_matrix, pred_pol_matrix):
        """
        input_matrix: [B, Seq, Seq]
        pred_matrix: [B, Seq, Seq, 6]
        input_masks: [B, Seq]
        """
        pred_ent_matrix = pred_ent_matrix.argmax(-1) * data['sentence_masks_raw']
        pred_rel_matrix = pred_rel_matrix.argmax(-1) * data['full_masks_raw']
        pred_pol_matrix = pred_pol_matrix.argmax(-1) * data['full_masks_raw'] 
        # token2sents = data['token2sents'].tolist()
        token2sents = data['token2sents_raw'].tolist() #不包括cls的token到句子的映射
        # new2old = data['new2old']
        pieces2words = data['pieces2words']
        doc_id = data['doc_id']

        pred_rel_matrix = np.array(pred_rel_matrix.tolist()) #(1,304,304) [batch,max_len,max_len]
        pred_ent_matrix = np.array(pred_ent_matrix.tolist())
        pred_pol_matrix = np.array(pred_pol_matrix.tolist())
        # pdb.set_trace()
        for i in range(len(pred_ent_matrix)): #batch级
            ent_matrix, rel_matrix, pol_matrix = pred_ent_matrix[i], pred_rel_matrix[i], pred_pol_matrix[i]
            pred_triplet, pred_pairs = self.kernel.get_triplets(ent_matrix, rel_matrix, pol_matrix, token2sents[i])
            pred_ents = self.kernel.rel_matrix2list(ent_matrix)

            """
            将上面解码得到的结果 重定向到原始sentence的index（既去除掉<s>和</s>的）
            pred_pairs
                前：{'ta': [(96, 98, 112, 113), (165, 167, 173, 173), (287, 287, 281, 282), (287, 287, 295, 298),...}
                后：{'ta': [[84, 84, 95, 95], [140, 140, 145, 145], [236, 236, 231, 231], [236, 236, 243, 245],...}
            pred_triplets:
                前：[(165, 167, 173, 173, 169, 172, 1), (287, 287, 281, 282, 290, 293, 2), (287, 287, 295, 298, 290, 293, 2), (287, 287, 297, 298, 290, 293, 2),...]
                后：
            
            对于我自己构建的matrix，因为本身就是不包含cls的，所以在下面的解码中不需要new2old这一步映射，直接拼接pieces2word就可以。
            
            """
            """上面的还是带着cls的"""
            pred_ents = self.filter_entity(pred_ents, pieces2words[i]) #[[3, 5, 3], [11, 11, 1], [17, 17, 3], [20, 20, 1], [24, 24, 2], ...]
            pred_pairs = self.trans2pair(pred_pairs,pieces2words[i])
            # pdb.set_trace()
            pred_triplet = self.trans2position(pred_triplet, pieces2words[i])
            self.predict_result[doc_id[i]].append(pred_ents)
            self.predict_result[doc_id[i]].append(pred_pairs)
            self.predict_result[doc_id[i]].append(pred_triplet)
    def clear(self):
        self.predict_result = defaultdict(list)

    def save2file(self, gold_file, pred_file):
        # pol_dict = {"O": 0, "pos": 1, "neg": 2, "other": 3}
        pol_dict = self.config.polarity_dict
        reverse_pol_dict = {v: k for k, v in pol_dict.items()}
        reverse_ent_dict = {v: k for k, v in self.config.entity_dict.items()}
                
        gold_file = open(gold_file, 'r', encoding='utf-8')

        data = json.load(gold_file)

        res = []
        for line in data:
            doc_id, sentence = line['doc_id'], line['sentences']
            if doc_id not in self.predict_result:
                continue
            doc = ' '.join(sentence).split()
            new_triples = []

            prediction = self.predict_result[doc_id]
            
            entities = defaultdict(list)
            for head, tail, tp in prediction[0]:
                tp = reverse_ent_dict[tp]
                head, tail = head, tail + 1
                tp_dict = {'ENT-T': 'targets', 'ENT-A': 'aspects', 'ENT-O': 'opinions'}
                entities[tp_dict[tp]].append([head, tail])

            pairs = defaultdict(list)
            for key in ['ta', 'to', 'ao']:
                for s0, e0, s1, e1 in prediction[1][key]:
                    e0, e1 = e0 + 1, e1 + 1
                    pairs[key].append([s0, e0, s1, e1])

            new_triples = []
            for s0, e0, s1, e1, s2, e2, pol in prediction[2]:
                pol = reverse_pol_dict[pol]
                e0, e1, e2 = e0 + 1, e1 + 1, e2 + 1
                new_triples.append([s0, e0, s1, e1, s2, e2, pol, ' '.join(doc[s0:e0]), ' '.join(doc[s1:e1]), ' '.join(doc[s2:e2])])

            res.append({'doc_id': doc_id, 'triplets': new_triples, \
                        'targets': entities['targets'], 'aspects': entities['aspects'], 'opinions': entities['opinions'],\
                        'ta': pairs['ta'], 'to': pairs['to'], 'ao': pairs['ao']})
        logger.info('Save prediction results to {}'.format(pred_file))
        json.dump(res, open(pred_file, 'w', encoding='utf-8'), ensure_ascii=False)
    
    def compute(self, name='valid'):
        # action: pred, make prediction, save to file 
        # action: eval, make prediction, save to file and evaluate 

        args = AttrDict({
            'pred_file': os.path.join(self.config.target_dir, 'pred_{}_{}.json'.format(self.config.lang, name)),
            'gold_file': os.path.join(self.config.json_path, '{}.json'.format(name))
        })
        self.save2file(args.gold_file, args.pred_file)

        micro, iden, res = Run_eval(args).forward()
        self.clear()
        return micro[2], res