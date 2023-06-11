export TASK=$1
export seed=$2
export comment=$3
export max_seq_len=$4

EPOCHS=3
TRAIN_BATCH_SIZE=2
LOGGING_STEPS=50
WARMUP_RATIO=0.1
LEARNING_RATE=2e-5

if [ $TASK = 'MELD' ]; then
    EPOCHS=4
    TRAIN_BATCH_SIZE=1
    LOGGING_STEPS=80
    WARMUP_RATIO=0.6
    LEARNING_RATE=2e-5
elif [ $TASK = 'DailyDialog' ]; then
    EPOCHS=4
    TRAIN_BATCH_SIZE=1
    LOGGING_STEPS=500
    WARMUP_RATIO=0.3
    LEARNING_RATE=2e-5
elif [ $TASK = 'IEMOCAP' ]; then
    EPOCHS=10
    TRAIN_BATCH_SIZE=1
    LOGGING_STEPS=80
    WARMUP_RATIO=0.8
    LEARNING_RATE=2e-5
fi

EVAL_BATCH_SIZE=$(expr 3 \* $TRAIN_BATCH_SIZE)

python utils_stage1/data_process.py \
  --task_name $TASK \
  --max_history_len 5 \
  --conv_len 8

python main_stage1.py \
--model_name_or_path facebook/bart-large \
--do_train \
--task_name $TASK \
--num_train_epochs $EPOCHS \
--learning_rate $LEARNING_RATE \
--output_dir ./save/$TASK \
--overwrite_output_dir \
--per_device_train_batch_size $TRAIN_BATCH_SIZE \
--per_device_eval_batch_size $EVAL_BATCH_SIZE \
--logging_steps $LOGGING_STEPS \
--warmup_ratio $WARMUP_RATIO \
--adam_epsilon 1e-6 \
--weight_decay 0.01 \
--seed $seed \
--comment $comment \
--max_seq_len $max_seq_len



