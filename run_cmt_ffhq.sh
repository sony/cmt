
torchrun --nnodes=1 --nproc_per_node=$1 --rdzv_backend=c10d --rdzv_endpoint=localhost:$2 run_cmt.py  \
    --outdir=cmt-runs --data=datasets/ffhq-64x64.zip  \
    --cond=0 --arch=ddpmpp --metrics=fid50k_full --cres=1,2,2,2        \
    --transfer=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-ffhq-64x64-uncond-vp.pkl    \
    --duration=38.4 --tick=12.8 --batch=128 --nfe=16 --lr=0.0002 --optim=RAdam --dropout=0.2 --augment=0.0 --mid_t=7.7569e-01 \
    --ema_beta 0.999 --eval_every 500 --ckpt 3000 --sch 1    \
    ${@:3}
