ROOT=$(pwd)
export PYTHONPATH=$PYTHONPATH:$ROOT
mkdir -p snapshot
# CUDA_VISIBLE_DEVICES=0 python -u $ROOT/main.py --log loguvg.txt --testuvg_tree --pretrain snapshot/dvc_pretrain512.model --config config.json
CUDA_VISIBLE_DEVICES=0 python -u $ROOT/main.py --log loguvg.txt --testuvg_tree --pretrain snapshot/openlifter_pretrain512.model --config config.json
