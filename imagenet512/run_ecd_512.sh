
torchrun --nnodes=1 --nproc_per_node=$1 --rdzv_backend=c10d --rdzv_endpoint=localhost:$2 cd_train_512.py  \
    --outdir=cd-runs --cond=1 --arch=edm2 --preset=edm2-img512-xxl-safe --fp16=0 --mid_t=1.526     \
    --data=/path/to/img512-sd.zip \
    --duration=100000000 --tick=6.4 --batch=128 --batch-gpu=16    \
    --dump 25 --ckpt 5000 --sample_every 100000 --eval_every 250 --optim Adam --wt snrpk             \
    --lr=1e-6 --ema_beta=0.9999 \
    --double 50000 -q 1024 --sch Constant -solver heun -guid 2.05 \
    --transfer=https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/edm2-img512-xxl-0939524-0.075.pkl \
    --transfer2=/path/to/initialization-ckpt.pkl \
    --gnet=https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/edm2-img512-m-0268435-0.155.pkl \
    ${@:3}