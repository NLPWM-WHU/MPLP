import json
import logging
import os
import torch
import pickle
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm


def get_input_format(u, tokenizer, max_seq_length, text_hist_len):
    history = u['history'][-text_hist_len:]
    cur_text = ' * ' + u['speaker'] + f' : ' + u['text'] + ' * '

    if len(history) == 0:
        pass
    else:
        prev_len = len(tokenizer.encode(cur_text))
        for j in range(len(history) - 1, -1, -1):
            id = tokenizer.encode(history[j] + ' ' + cur_text)
            if len(id) > max_seq_length:
                # truncate the last historical sentence if needed
                temp_id = tokenizer.encode(history[j])
                needed_temp_len = max_seq_length - prev_len
                temp_id = temp_id[1: needed_temp_len]
                temp = tokenizer.decode(temp_id)
                cur_text = temp + ' ' + cur_text
                break
            else:
                temp = history[j]
                cur_text = temp + ' ' + cur_text
            prev_len = len(id)

    prefix = ''
    return prefix + cur_text

def get_bart_feature(sentence, tokenizer, max_seq_length, device):
    inputs = tokenizer(sentence, max_length=max_seq_length, padding='max_length', truncation=True, return_tensors="pt")
    inputs = inputs.to(device)
    return inputs

def set_emotion_dict(dataset_name):
    emotion_dict_dd = {'happiness': ['happy', 'happiness'], 'neutral': ['neutral', 'neutrality'],
                       'anger': ['angry', 'anger'],
                       'sadness': ['sad', 'sadness'], 'fear': ['scared', 'fear'],
                       'surprise': ['surprised', 'surprise'], 'disgust': ['disgusted', 'disgust'],
                       'none': ['neutral', 'neutrality']}

    emotion_dict_iemocap = {'neu': ['neutral', 'neutrality'], 'hap': ['happy', 'happiness'],
                            'sad': ['sad', 'sadness'],
                            'ang': ['angry', 'anger'], 'fru': ['frustrated', 'frustration'],
                            'exc': ['excited', 'excitement'],
                            'none': ['neutral', 'neutrality']}

    emotion_dict_meld = {'neutral': ['neutral', 'neutrality'], 'joy': ['joyful', 'joy'],
                         'surprise': ['surprised', 'surprise'], 'sadness': ['sad', 'sadness'],
                         'anger': ['angry', 'anger'], 'disgust': ['disgusted', 'disgust'],
                         'fear': ['scared', 'fear']}

    if dataset_name == 'DailyDialog':
        emotion_dict = emotion_dict_dd
    elif dataset_name == 'IEMOCAP':
        emotion_dict = emotion_dict_iemocap
    elif dataset_name == 'MELD':
        emotion_dict = emotion_dict_meld
    else:
        assert 0 == 1, print("Assign Proper Dataset Name!")
    return emotion_dict

def get_encoded_len(text, tokenizer, max_seq_length):
    return min(len(tokenizer.encode(text)), max_seq_length)

class ERCDataset(Dataset):

    def __init__(self, dataset_name, split, tokenizer, max_seq_length, device, args):
        self.label_prefix = None
        self.label_start_positions = None
        self.label_vocab = None
        self.speaker_vocab = None

        self.args = args
        self.tokenizer = tokenizer
        self.device = device
        self.dataset_name = dataset_name
        self.max_seq_length = max_seq_length
        self.pad_token_id = tokenizer.pad_token_id

        self.emotion_dict = set_emotion_dict(dataset_name)
        self.max_inp_len = 0
        self.max_tgt_len = 0
        self.data = self.read(dataset_name, split, args.conv_len)
        self.len = len(self.data)

    def read(self, dataset_name, split, conv_len):
        with open('./data/%s/%s_data_generation_len%d.json' % (dataset_name, split, conv_len), encoding='utf-8') as f:
            raw_data = json.load(f)
        self.label_vocab = pickle.load(open('./data/%s/label_vocab.pkl' % dataset_name, 'rb'))
        self.speaker_vocab = pickle.load(open('./data/%s/speaker_vocab.pkl' % (dataset_name), 'rb'))

        dialogs = []
        for i, d in enumerate(tqdm(raw_data)):
            utterances = []
            labels = []
            speakers = []
            speaker_feels_mask = []
            for j, u in enumerate(d):
                # convert label from text to number
                label_id = self.label_vocab['stoi'][u['label']] if 'label' in u.keys() else -100

                labels.append(int(label_id))
                speakers.append(self.speaker_vocab['stoi'][u['speaker']])
                cur_text = get_input_format(u=u, tokenizer=self.tokenizer, max_seq_length=self.max_seq_length, text_hist_len=self.args.text_hist_len)

                speaker = u['speaker']
                utterances.append(cur_text)
                # view2_prefix = f'How does {speaker} feel ? '
                view2_prefix = ''
                speaker_feels_mask.append(view2_prefix + speaker + f' feels {self.tokenizer.mask_token}')

                inp_len = get_encoded_len(text=utterances[-1], tokenizer=self.tokenizer, max_seq_length=self.max_seq_length)
                tgt_len = get_encoded_len(text=speaker_feels_mask[-1], tokenizer=self.tokenizer, max_seq_length=self.max_seq_length)

                if inp_len > self.max_inp_len:
                    self.max_inp_len = inp_len
                if tgt_len > self.max_tgt_len:
                    self.max_tgt_len = tgt_len

                if len(dialogs) < 2:
                    print(" ============================================ ")
                    print('cur_text:', utterances[-1])
                    print('speaker_feels_mask:', speaker_feels_mask[-1])
                    print()

            dialogs.append({
                'utterances': utterances,
                'labels': labels,
                'speakers': speakers,
                'speaker_feels_mask': speaker_feels_mask,
            })
        print(f" ******** max_len input/target/limit : {self.max_inp_len, self.max_tgt_len, self.max_seq_length} ********* ")

        return dialogs

    def __getitem__(self, index):
        """
        :param index:
        :return:
        """
        return self.data[index]['utterances'], self.data[index]['labels'], self.data[index]['speakers'], self.data[index]['speaker_feels_mask']

    def __len__(self):
        return self.len

    def collate_fn(self, datas):
        """
        :param datas:
        :return:
        inputs['input_ids'] (Batch_size, Utter_len, Sent_len)
        inputs['attention_mask'] (Batch_size, Utter_len, Sent_len)
        """
        inputs = {'input_ids': pad_sequence([get_bart_feature(data[0], self.tokenizer, max_seq_length=self.max_inp_len, device=self.device)['input_ids'] for data in datas], batch_first=True, padding_value=1),
                  'attention_mask': pad_sequence([get_bart_feature(data[0], self.tokenizer, max_seq_length=self.max_inp_len, device=self.device)['attention_mask'] for data in datas], batch_first=True, padding_value=0)}
        inputs['labels'] = pad_sequence([torch.tensor(data[1], device=inputs['input_ids'].device) for data in datas], batch_first=True, padding_value=-100)
        inputs['speakers'] = pad_sequence([torch.tensor(data[2], device=inputs['input_ids'].device) for data in datas], batch_first=True, padding_value=-100)
        inputs['speaker_feels_mask'] = pad_sequence([get_bart_feature(data[3], self.tokenizer, max_seq_length=self.max_tgt_len, device=self.device)['input_ids'] for data in datas], batch_first=True, padding_value=1)

        return inputs

