# prompt=("a professional photograph of an astronaut riding a horse")
CKPT_PATH=/home/jwang/stable-diffusion-pretrain/v2-1_768-ema-pruned.ckpt
CUDA_VISIBLE_DEVICES=5 \
python scripts/txt2img.py \
    --prompt "a professional photograph of an astronaut riding a horse" \
    --ckpt $CKPT_PATH \
    --config configs/stable-diffusion/v2-inference-v.yaml \
    --H 512 --W 512 \
    --device cuda \
    2>&1 | tee error.log