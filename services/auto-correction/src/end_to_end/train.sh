export lr="5e-5"
export s="100"
echo "${lr}"
export PROJECT_PATH=/Users/manred1997/armed2003/search-engine/services/auto-correction
export MODEL_DIR=scRoberta
export MODEL_DIR=$MODEL_DIR"/"$lr"/"$s
echo "${MODEL_DIR}"
python3 src/end_to_end/main.py --token_level syllable_level  \
        --model_type scRoberta   \
        --model_dir $MODEL_DIR  \
        --data_dir ../../resources/auto-correction/mini-VNTC  \
        --is_self_supervised_learning \
        --seed $s   \
        --do_train  \
        --do_eval   \
        --save_steps 100    \
        --logging_steps 100 \
        --num_train_epochs 100  \
        --tuning_metric loss    \
        --gpu_id 0  \
        --learning_rate $lr \
        --train_batch_size 16   \
        --eval_batch_size 64   \
        --max_seq_len 128   \
        --early_stopping 25