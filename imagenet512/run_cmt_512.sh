
torchrun --nnodes=1 --nproc_per_node=$1 --rdzv_backend=c10d --rdzv_endpoint=localhost:$2 cmt_512.py        \
    --outdir=cmt-runs --cond=1 --arch=edm2 --preset=edm2-img512-xxl --fp16=0 --mid_t=1.526     \
    --duration=12.8 --tick=6.4 --batch=128 --batch-gpu=16    \
    --dump 25 --ckpt 2000 --sample_every 20000 --eval_every 2000 --optim Adam --wt snrpk             \
    --lr=2e-4 --ema_beta=0.999 --loss_metric=ELatentLPIPS             \
    --transfer=https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/edm2-img512-xxl-0939524-0.070.pkl \
    ${@:3}
