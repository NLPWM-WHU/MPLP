import os
import random
import torch
import logging
from typing import Optional
from dataclasses import dataclass, field
from utils_stage1.loggingHandler import LoggingHandler
from utils_stage2.train_or_eval import train, evaluate
from utils_stage2.dataloader import get_dataloaders, load_vocab
from model.modeling_bart_prompt import BartForERC_MPLP
from transformers import (AutoTokenizer,
                          HfArgumentParser,
                          TrainingArguments,
                          AutoConfig,
                          set_seed)

import fitlog
import numpy as np
logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model2/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model2 or model2 identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model2 for training and eval.
    """
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
                    "This argument is also used to override the ``max_length`` param of ``model2.generate``, which is used "
                    "during ``evaluate`` and ``predict``."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model2 maximum sentence length. "
                    "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                    "efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
                    "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                    "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                    "value if set."
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model2.generate``, "
                    "which is used during ``evaluate`` and ``predict``."
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default=None, metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )

def add_label_sp_token(tokenizer, label_vocab):
    label_tokens = label_vocab['stoi'].keys()
    sp_token_list = []
    for ltok in label_tokens:
        tokenizer.add_tokens(f'[{ltok}]')
        sp_token_list.append(f'[{ltok}]')
        print(f"adding [{ltok}] ...")
    return tokenizer, sp_token_list

if __name__ == '__main__':
    # fitlog.commit(__file__)
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    parser.add_argument('--task_name', type=str, required=True, choices=['MELD', 'IEMOCAP', 'DailyDialog'])
    parser.add_argument('--num_labels', type=int, required=False, default=7)
    parser.add_argument('--comment', type=str, required=True)
    parser.add_argument('--max_seq_len', type=int, required=True, default=128)
    parser.add_argument('--conv_len', type=int, default=8)
    parser.add_argument('--freeze_encoder', action='store_true')
    parser.add_argument('--freeze_decoder', action='store_true')
    parser.add_argument('--freeze_shared', action='store_true')
    parser.add_argument('--text_hist_len', type=int, required=True)
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--diff_lr', action='store_true')

    # hist-oriented prompt
    parser.add_argument('--use_hist_prompt', action='store_true')
    parser.add_argument('--max_history_feat_len', type=int, default=20)

    # exp-oriented prompt
    parser.add_argument('--feat_type', type=str, required=True)
    parser.add_argument('--use_exp_prompt', action='store_true')
    parser.add_argument('--exp_demon_num', type=int, default=3)

    # label paraphrasing
    parser.add_argument('--gen_ratio', type=float, required=True, default=0.3)


    model_args, data_args, training_args, other_args = parser.parse_args_into_dataclasses()
    # Create directories and set seed
    training_args.output_dir = training_args.output_dir + '/' + other_args.comment
    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)

    if not os.path.exists(f'./logs/{other_args.task_name}_{other_args.comment}'):
        os.makedirs(f'./logs/{other_args.task_name}_{other_args.comment}')
    fitlog.set_log_dir(f'./logs/{other_args.task_name}_{other_args.comment}')
    rnd_seed = fitlog.set_rng_seed() if training_args.seed == -1 else fitlog.set_rng_seed(training_args.seed)
    set_seed(rnd_seed)
    logger.info("The random seed is %d" % rnd_seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    speaker_vocab, label_vocab = load_vocab(other_args.task_name)
    num_labels = len(label_vocab['stoi'])
    other_args.num_labels = num_labels

    tokenizer = AutoTokenizer.from_pretrained(
        "facebook/bart-large",
        cache_dir=None,
        use_fast=True)
    logger.info("********* Before length: %s", str(len(tokenizer)))
    tokenizer.add_tokens('[SEP]')
    tokenizer.add_tokens('[hist_tok]')
    tokenizer.add_tokens('[exp_tok]')
    logger.info("********* After length: %s", str(len(tokenizer)))

    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    train_dataloader, eval_dataloader, test_dataloader = get_dataloaders(args=other_args,
                                                                         tokenizer=tokenizer, task_name=other_args.task_name,
                                                                         train_batch_size=training_args.per_device_train_batch_size,
                                                                         eval_batch_size=training_args.per_device_eval_batch_size,
                                                                         device=device,
                                                                         max_seq_length=other_args.max_seq_len)

    param_dict = {}
    for k, v in vars(training_args).items():
        param_dict[k] = v
    for k, v in vars(data_args).items():
        param_dict[k] = v
    for k, v in vars(model_args).items():
        param_dict[k] = v
    for k, v in vars(other_args).items():
        param_dict[k] = v

    fitlog.add_hyper(param_dict)
    fitlog.add_hyper_in_file(__file__)

    logger.info("********* Training... *********** ")
    try:
        qs = [None]
        if training_args.do_train:
            config = AutoConfig.from_pretrained(
                model_args.model_name_or_path,
                num_labels=num_labels,
                finetuning_task=other_args.task_name,
                cache_dir=None,
                revision=None,
                use_auth_token=None
            )
            model = BartForERC_MPLP.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                cache_dir=None,
                revision=None,
                use_auth_token=None,
                args=other_args,
                tokenizer=tokenizer
            )
            model.resize_token_embeddings(len(tokenizer))
            model = model.to(device)
            train(train_dataloader, eval_dataloader, test_dataloader, model, training_args, other_args, label_vocab, tokenizer)
            fitlog.finish()
    except KeyboardInterrupt:
        print("Catch keyboard interrupt.")

    logger.info("********* Eval / Test -ing... *********** ")
    if training_args.do_train:
        best_model_path = os.path.join(training_args.output_dir, "best_model_%d" % training_args.seed)
    else:
        best_model_path = model_args.model_name_or_path
    if training_args.do_eval or training_args.do_predict:
        config = AutoConfig.from_pretrained(best_model_path)
        model = BartForERC_MPLP.from_pretrained(
            best_model_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            args=other_args,
            tokenizer=tokenizer
            )
        model.resize_token_embeddings(len(tokenizer))
        model = model.to(device)
        if training_args.do_eval:
            results = evaluate(training_args, other_args, eval_dataloader, model, "evaluate", label_vocab, tokenizer)
            print(results)

        if training_args.do_predict:
            results = evaluate(training_args, other_args, test_dataloader, model, "predict", label_vocab, tokenizer)
            print(results)
