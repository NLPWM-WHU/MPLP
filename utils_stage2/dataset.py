import json
import logging
import os
import torch
import pickle
import numpy as np
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from utils_stage1.dataset import get_input_format, set_emotion_dict, get_encoded_len, get_bart_feature

neu_dict = {
    'DailyDialog': 'none',
    'EmoryNLP': 'Neutral',
    'IEMOCAP': 'neu',
    'MELD': 'neutral'
}

def load_sentinet(senti_file_name):
    # load sentiwordnet
    f = open(senti_file_name, 'r')

    line_id = 0
    sentinet = {}

    for line in f.readlines():
        if line_id < 26:
            line_id += 1
            continue
        # if line_id == 26:
        # print(line)
        if line_id == 117685:
            # print(line)
            break
        line_split = line.strip().split('\t')
        pos_tag, pscore, nscore, term, gloss = line_split[0], float(line_split[2]), float(line_split[3]), line_split[4], \
                                               line_split[5]

        if "\"" in gloss:
            shop_pos = gloss.index('\"')
            gloss = gloss[: shop_pos - 2]
        each_term = term.split(' ')
        for ele in each_term:
            ele_split = ele.split('#')
            assert len(ele_split) == 2
            word, sn = ele_split[0], int(ele_split[1])
            if word not in sentinet:
                sentinet[word] = {}
            if pos_tag not in sentinet[word]:
                sentinet[word][pos_tag] = []
            sentinet[word][pos_tag].append([sn, pscore, nscore, gloss, line_id - 26])
        line_id += 1

    f.close()
    return sentinet

def get_gen_paraphrase(speaker, emotion_name):
    para = speaker + ' is ' + emotion_name
    return para

def get_emotion_name(u):
    if 'label' in u.keys():
        emotion_name = u['label']
    else:
        emotion_name = 'none'
    return emotion_name

def get_adj_gloss(adj_w, sentinet):
    best_sn = 100
    adj_best_gloss = ''
    for paraphrase in sentinet[adj_w]['a']:
        sn, pscore, nscore, gloss = paraphrase[0], paraphrase[1], paraphrase[2], paraphrase[3]
        # 与情感有一定关联
        if not (pscore == 0 and nscore == 0):
            if best_sn > sn:
                best_sn = sn
                adj_best_gloss = gloss
    return adj_best_gloss

def get_flated_data(data):
    flated_data = []
    for i, d in enumerate(tqdm(data)):
        for j, u in enumerate(d):
            flated_data.append(u)
    return flated_data

class ERCDataset(Dataset):

    def __init__(self, dataset_name, split, tokenizer, max_seq_length, device, args):
        self.args = args
        self.tokenizer = tokenizer
        self.device = device
        self.dataset_name = dataset_name
        self.max_seq_length = max_seq_length
        self.max_history_feat_len = args.max_history_feat_len

        self.emotion_dict = set_emotion_dict(dataset_name)
        self.sentinet = load_sentinet(senti_file_name='./data/SentiWordNet_3.0.0.txt')

        self.max_inp_len = 0
        self.max_tgt_len = 0
        self.auxi_max_len = 0
        self.data = self.read(dataset_name, split, args.conv_len)
        self.len = len(self.data)


    def read(self, dataset_name, split, conv_len):
        self.label_vocab = pickle.load(open('./data/%s/label_vocab.pkl' % dataset_name, 'rb'))
        self.speaker_vocab = pickle.load(open('./data/%s/speaker_vocab.pkl' % (dataset_name), 'rb'))
        raw_data = pickle.load(open(f'./data/%s/i_feature/%s_data_generation_len%d_feature.pkl' % (dataset_name, split, conv_len), 'rb'))

        # if self.args.use_exp_prompt is True:
        demon_features = pickle.load(open(f'./data/%s/i_feature/%s_top10_{self.args.feat_type}_demon_feat.pkl' % (dataset_name, split), 'rb'))
        print("Demonstrations loaded .", len(raw_data), len(demon_features))

        self.neu_label_id = self.label_vocab['stoi'][neu_dict[dataset_name]]

        dialogs = []
        total = 0
        for i, d in enumerate(tqdm(raw_data)):
            utterances = []
            speaker_feels_mask = []
            labels = []
            speakers = []

            pre_encoded_features = []
            dia_hist_feat_seqs = []
            dia_pre_feat_lens = []
            is_same_speaker = []

            pre_encoded_demon_exp = []
            auxi_tgt_sentences = []

            for j, u in enumerate(d):

                # input
                # the same text as the first phrase, with only max_seq_len tokens
                cur_text = get_input_format(u=u, tokenizer=self.tokenizer, max_seq_length=self.args.max_seq_len, text_hist_len=self.args.text_hist_len)

                # add prompts
                prefix = ''
                if self.args.use_exp_prompt is True:
                    prefix += u['speaker'] + f' may feel [exp_tok] , '
                if self.args.use_hist_prompt is True:
                    prefix += u['speaker'] + f' may feel [hist_tok] [SEP] '
                cur_text = prefix + cur_text

                # prepare paraphrases
                label_id = self.label_vocab['stoi'][u['label']] if 'label' in u.keys() else -100
                emotion_name = get_emotion_name(u)
                adj_gloss = get_adj_gloss(adj_w=self.emotion_dict[emotion_name][0], sentinet=self.sentinet)
                label_gloss = get_gen_paraphrase(u['speaker'], emotion_name) + ' , ' + adj_gloss
                labels.append(int(label_id))
                speakers.append(self.speaker_vocab['stoi'][u['speaker']])

                utterances.append(cur_text)
                speaker = u['speaker']
                # view2_prefix = f'How does {speaker} feel ? '
                view2_prefix = ''
                speaker_feel_target = view2_prefix + u['speaker'] + f' feels {self.tokenizer.mask_token}'
                speaker_feels_mask.append(speaker_feel_target)
                auxi_tgt_sentences.append(label_gloss)


                # 20 tokens for addding prompt
                # note that no more contextual tokens are added compared with staage1
                cur_len = get_encoded_len(text=utterances[-1], tokenizer=self.tokenizer, max_seq_length=self.max_seq_length+10)
                tgt_len = get_encoded_len(text=speaker_feels_mask[-1], tokenizer=self.tokenizer, max_seq_length=self.max_seq_length)
                aux_len = get_encoded_len(text=auxi_tgt_sentences[-1], tokenizer=self.tokenizer, max_seq_length=self.max_seq_length)
                if cur_len > self.max_inp_len:
                    self.max_inp_len = cur_len
                if tgt_len > self.max_tgt_len:
                    self.max_tgt_len = tgt_len
                if aux_len > self.auxi_max_len:
                    self.auxi_max_len = aux_len


                # prepare features
                cls = u['self_feat']
                pre_encoded_features.append(cls)

                # if self.args.use_hist_prompt is True:
                dia_pre_feats_len = len(u['dia_pre_feats'])
                dia_pre_feat_lens.append(dia_pre_feats_len)
                dia_hist_feat_seqs.append(torch.stack(u['dia_pre_feats'][:self.max_history_feat_len] + [torch.zeros(1024 * 1)] * (self.max_history_feat_len - dia_pre_feats_len)))
                is_same_speaker.append(u['is_same_speaker'][:self.max_history_feat_len] + [0] * (self.max_history_feat_len - dia_pre_feats_len) )
                assert len(u['is_same_speaker']) == dia_pre_feats_len, print(len(u['is_same_speaker']), dia_pre_feats_len)

                # if self.args.use_exp_prompt is True:
                demon_feat_list = []
                for d_num, demon_feat in enumerate(demon_features[total]):
                    if d_num == self.args.exp_demon_num:
                        break
                    demon_feat_list.append(demon_feat)
                pre_encoded_demon_exp.append(torch.stack(demon_feat_list))

                if len(dialogs) < 2:
                    print(" ============================================ ")
                    print('cur_text:', utterances[-1])
                    print('speaker_feels_mask:', speaker_feels_mask[-1])
                    print('label:', auxi_tgt_sentences[-1])
                    print()

                total += 1

            dialogs.append({
                'utterances': utterances,
                'speaker_feels_mask': speaker_feels_mask,
                'labels': labels,
                'speakers': speakers,

                'pre_encoded_features': pre_encoded_features,
                'dia_hist_feat_seqs': dia_hist_feat_seqs,
                'dia_pre_feat_lens': dia_pre_feat_lens,
                'is_same_speaker': is_same_speaker,
                'pre_encoded_demon_exp': pre_encoded_demon_exp,
                'auxi_tgt_sentences': auxi_tgt_sentences,

            })


        print(f" ******** max_len : {self.max_inp_len, self.max_tgt_len, self.max_seq_length} ********* ")
        return dialogs

    def __getitem__(self, index):
        """
        :param index:
        :return:
        """
        return self.data[index]['utterances'], self.data[index]['labels'], self.data[index]['speakers'], \
               self.data[index]['speaker_feels_mask'], \
               self.data[index]['pre_encoded_features'], self.data[index]['dia_hist_feat_seqs'], self.data[index]['is_same_speaker'], \
               self.data[index]['pre_encoded_demon_exp'], self.data[index]['auxi_tgt_sentences'], self.data[index]['dia_pre_feat_lens']


    def __len__(self):
        return self.len

    def collate_fn(self, datas):
        """
        :param datas:
        :return:
        inputs['input_ids'] (Batch_size, Utter_len, Sent_len)
        inputs['attention_mask'] (Batch_size, Utter_len, Sent_len)
        """
        inputs = {'input_ids': pad_sequence([get_bart_feature(data[0], self.tokenizer, max_seq_length=self.max_inp_len, device=self.device)['input_ids'] for data in datas], batch_first=True,
                                            padding_value=1),
                  'attention_mask': pad_sequence([get_bart_feature(data[0], self.tokenizer, max_seq_length=self.max_inp_len, device=self.device)['attention_mask'] for data in datas],
                                                 batch_first=True, padding_value=0)}

        inputs['labels'] = pad_sequence([torch.tensor(data[1], device=inputs['input_ids'].device) for data in datas], batch_first=True, padding_value=-100)
        inputs['speakers'] = pad_sequence([torch.tensor(data[2], device=inputs['input_ids'].device) for data in datas], batch_first=True, padding_value=-100)
        inputs['speaker_feels_mask'] = pad_sequence(
            [get_bart_feature(data[3], self.tokenizer, max_seq_length=40,device=self.device)['input_ids'] for data in datas],
            batch_first=True, padding_value=1)

        inputs['pre_encoded_features'] = pad_sequence([torch.stack(data[4]) for data in datas], batch_first=True).to(self.device)       # (B, N, demon_num, D)
        # if self.args.use_hist_prompt is True:
        inputs['dia_hist_feat_seqs'] = pad_sequence([torch.stack(data[5]) for data in datas], batch_first=True).to(self.device)         # (B, N, dia_len, D)
        inputs['is_self_seq'] = pad_sequence(
            [torch.tensor(data[6], device=inputs['input_ids'].device) for data in datas], batch_first=True,
            padding_value=0)       # (B, N, dia_len)
        # if self.args.use_exp_prompt is True:
        inputs['pre_encoded_demon_exp'] = pad_sequence([torch.stack(data[7]) for data in datas], batch_first=True).to(self.device)      # (B, N, demon_num, D)
        inputs['auxi_tgt_sentences'] = pad_sequence([get_bart_feature(data[8], self.tokenizer, max_seq_length=self.auxi_max_len, device=self.device)['input_ids'] for data in datas], batch_first=True, padding_value=1)
        # inputs['dia_pre_feat_lens'] = pad_sequence([torch.tensor(data[9], device=inputs['input_ids'].device) for data in datas], batch_first=True, padding_value=0)      # (B, N, 1)

        return inputs

