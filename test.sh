ROOT=$(pwd)
export PYTHONPATH=$PYTHONPATH:$ROOT
mkdir -p snapshot
CUDA_VISIBLE_DEVICES=0 python -u $ROOT/main.py --log loguvg.txt --testuvg --pretrain snapshot/dvc_pretrain2048.model --config config.json
