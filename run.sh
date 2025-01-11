ROOT=$(pwd)
export PYTHONPATH=$PYTHONPATH:$ROOT
mkdir -p snapshot
CUDA_VISIBLE_DEVICES=0,1 python -u $ROOT/main.py --log log.txt --config config.json

