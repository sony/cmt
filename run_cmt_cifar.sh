
torchrun --nnodes=1 --nproc_per_node=$1 --rdzv_backend=c10d --rdzv_endpoint=localhost:$2 run_cmt.py  \
    --outdir=cmt-runs --data=datasets/cifar10-32x32.zip  \
    --cond=0 --arch=ddpmpp --metrics=fid50k_full        \
    --transfer=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-uncond-vp.pkl    \
    --duration=38.4 --tick=12.8 --batch=128 --nfe=16 --lr=0.0002 --optim=RAdam --dropout=0.2 --augment=0.0 --mid_t=7.7569e-01 \
    --ema_beta 0.999 --eval_every 500 --ckpt 3000 --sch 1    \
    ${@:3}
