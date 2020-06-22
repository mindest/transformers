#export TRAIN_FILE=/workspace/data/WIKI-2/wikitext-2/wiki.train.tokens
#export TEST_FILE=/workspace/data/WIKI-2/wikitext-2/wiki.test.tokens
export TRAIN_DIR=/bingads/pretrain/hdf5

RUN_FILE=/workspace/transformers/examples/run_language_modeling_ort.py
# RUN_FILE=/workspace/bert/benchmark/transformers/examples/run_language_modeling.py

RUN_CMD="mpirun -n 2 --allow-run-as-root python $RUN_FILE --ort_trainer True --output_dir=output-ort"
# RUN_CMD="python $RUN_FILE --output_dir=output"

$RUN_CMD \
    --model_type=gpt2 \
    --tokenizer_name=gpt2 \
    --config_name=gpt2 \
    --do_train \
    --train_data_dir=$TRAIN_DIR \
    --seq_length=50 \
    --vocab_size=70000 \
    --per_gpu_train_batch_size=8 \
    --max_steps=200 \
    --block_size 1024  \
    --weight_decay 0.01 \
    --overwrite_output_dir \
    --logging_steps 5 \
    --logging_first_step True \
    --logging_dir /bingads/models \
    --fp16 \
    # --do_eval \
    # --train_data_file=$TRAIN_FILE \
    # --eval_data_file=$TEST_FILE \
    # --per_gpu_eval_batch_size=4 \
    # --model_name_or_path=gpt2 \
    # --num_train_epochs 1\

# EVAL_OUTPUT_DIR='/workspace/bert/benchmark/ashbhandare/transformers/output-ort'
# EVAL_RUN_CMD="python $RUN_FILE --output_dir=$EVAL_OUTPUT_DIR"
# $EVAL_RUN_CMD \
#     --model_type=gpt2 \
#     --model_name_or_path=$EVAL_OUTPUT_DIR \
#     --tokenizer_name=gpt2  \
#     --config_name=gpt2  \
#     --do_eval \
#     --train_data_file=$TRAIN_FILE \
#     --eval_data_file=$TEST_FILE \
#     --per_gpu_train_batch_size=1  \
#     --per_gpu_eval_batch_size=4  \
#     --block_size 1024  \
#     --weight_decay 0.01 \
#     --overwrite_output_dir \
#     --logging_steps 500 \
#     --logging_first_step True \
#     # --num_train_epochs 1\
