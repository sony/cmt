
torchrun --nnodes=1 --nproc_per_node=$1 --rdzv_backend=c10d --rdzv_endpoint=localhost:$2 ct_eval.py  \
    --outdir=ct-evals --data=datasets/ffhq-64x64.zip  \
    --cond=0 --arch=ddpmpp --metrics=fid50k_full,pr50k3_full --cres=1,2,2,2        \
    ${@:3}
