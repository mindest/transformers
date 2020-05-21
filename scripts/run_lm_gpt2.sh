export TRAIN_FILE=/workspace/data/WIKI-2/wikitext-2/wiki.train.tokens
export TEST_FILE=/workspace/data/WIKI-2/wikitext-2/wiki.test.tokens

RUN_FILE=/workspace/bert/benchmark/transformers/examples/run_language_modeling_ort.py
# RUN_FILE=/workspace/bert/benchmark/transformers/examples/run_language_modeling.py

RUN_CMD="mpirun -n 8 --allow-run-as-root python $RUN_FILE --ort_trainer True "
# RUN_CMD="python $RUN_FILE"

# mpirun -n 8 --allow-run-as-root python $RUN_FILE \
$RUN_CMD \
    --output_dir=output-ort \
    --model_type=gpt2-medium \
    --model_name_or_path=gpt2-medium \
    --tokenizer_name=gpt2-medium  \
    --config_name=gpt2-medium  \
    --do_eval \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --eval_data_file=$TEST_FILE \
    --per_gpu_train_batch_size=1  \
    --per_gpu_eval_batch_size=4  \
    --block_size 1024  \
    --weight_decay 0.01 \
    --overwrite_output_dir \
    --logging_steps 500 \
    --logging_first_step True \




    # --do_eval \
    # 