
torchrun --nnodes=1 --nproc_per_node=$1 --rdzv_backend=c10d --rdzv_endpoint=localhost:$2 ct_train.py  \
    --outdir=ct-runs --data=datasets/ffhq-64x64.zip  \
    --cond=0 --arch=ddpmpp --metrics=fid50k_full --cres=1,2,2,2        \
    --transfer=/path/to/cmt/network-snapshot-003000.pkl    \
    --duration=12.8 --tick=12.8 --double=250 --batch=128 --lr=0.0001 --optim=RAdam --dropout=0.2 --augment=0.0 \
    -q 512 --double 10000 --ema_beta 0.9999 --snap 1000 --sch 1     \
    ${@:3}
