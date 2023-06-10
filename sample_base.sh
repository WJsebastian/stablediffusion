# prompt=("a professional photograph of an astronaut riding a horse")
CKPT_PATH=/home/jwang/stable-diffusion-pretrain/512-base-ema.ckpt
CUDA_VISIBLE_DEVICES=5,6,7 \
python scripts/txt2img.py \
    --prompt "a professional photograph of an astronaut riding a horse" \
    --ckpt $CKPT_PATH \
    --config configs/stable-diffusion/v2-inference-v.yaml \
    --H 512 --W 512 \
    --device cuda \
    --steps 500 \
    --n_samples 1 \
    2>&1 | tee error.log