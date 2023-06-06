import json
import random
from transformers import AutoTokenizer

def split_data(dataset_name, train_dev_pred, max_history_len=2, conv_len=8):
    print(f" ********* history len: {max_history_len} ********* ")
    with open('./data/%s/%s_data.json' % (dataset_name, train_dev_pred), encoding='utf-8') as f:
        data_raw = json.load(f)
    data_raw = sorted(data_raw, key=lambda x: len(x))

    final_json = []
    with open('./data/%s/%s_data_generation_len%d.json' % (dataset_name, train_dev_pred, conv_len), "w") as f:
        for context_index, context in enumerate(data_raw):
            context_len = len(context)
            new_context = []

            index = 0
            while index < context_len:
                sentence = context[index]
                if 'label' in sentence:
                    # history sentences
                    if index == 0:
                        history = []
                    else:
                        history = []
                        for j in range(max(index - max_history_len, 0), index):
                            past_sent = context[j]["speaker"] + " : " + context[j]["text"]
                            history.append(past_sent)                 # 按顺序存放history
                    new_sentence = {"text": sentence["text"], "speaker": sentence["speaker"],
                                    "label": sentence["label"], "history": history}
                    new_context.append(new_sentence)

                    if len(new_context) == conv_len:
                        final_json.append(new_context)
                        new_context = []
                index += 1

            if len(new_context) > 0:
                final_json.append(new_context)

        json_data = json.dumps(final_json)
        f.write(json_data)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', type=str, required=True, choices=['MELD', 'IEMOCAP', 'DailyDialog', 'EmoryNLP'])
    parser.add_argument('--max_history_len', type=int, required=True, default=5)
    parser.add_argument('--conv_len', type=int, required=True, default=8, help="1: train with speaker, 0: verse vice")
    args = parser.parse_args()
    print("Start preprocess data")

    split_data(args.task_name, 'train', args.max_history_len, args.conv_len)
    split_data(args.task_name, 'dev', args.max_history_len, args.conv_len)
    split_data(args.task_name, 'test', args.max_history_len, args.conv_len)

    print("Preprocess data complete")
