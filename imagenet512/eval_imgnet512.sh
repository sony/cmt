
torchrun --nnodes=1 --nproc_per_node=$1 --rdzv_backend=c10d --rdzv_endpoint=localhost:$2 ct_eval_512.py        \
    --outdir=ct-evals --cond=1 --arch=edm2 --preset=edm2-img512-xxl --fp16=0 --mid_t=1.526  \
    --data=/data2/img512-sd.zip \
    --encoder_pkl=https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/edm2-img512-m-2147483-0.100.pkl \
    --resume_pkl=XXLnet_2500_3.46_1.84.pkl \
    ${@:3}
