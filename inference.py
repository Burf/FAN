import os
import numpy as np
import torch
import argparse
from fan import FAN, personalized_t2i_encoder
from diffusers import DiffusionPipeline, EulerDiscreteScheduler, StableDiffusion3Pipeline, FluxPipeline

def main():
    parser = argparse.ArgumentParser(description="Foundation Encoders Are All You Need for Preference-Aware Personalization")
    parser.add_argument("--model", type=str, default="black-forest-labs/FLUX.1-dev", help="T2I models id")
    parser.add_argument("--sample_size", type=float, default=0.0, help="Sampling ratio for profiling")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--prompt", nargs='+', default=["A photograph of an astronaut riding a horse"], help="Prompt for personalized generation")
    parser.add_argument("--ref", nargs='+', default=["A retro-futuristic space exploration movie poster with bold, vibrant colors"], help="Reference descriptions for conditioning")
    parser.add_argument("--weight", nargs='+', type=float, default=[1], help="Weight for the reference descriptions")
    parser.add_argument("--alpha", type=float, default=0.4, help="Alpha scaling factor (range: 0-1)")
    parser.add_argument("--size", type=int, default=0, help="Image size (0 automatically sets the default value for each model)")
    parser.add_argument("--num_inference_steps", type=int, default=0, help="Number of inference steps (0 automatically sets the default value for each model)")
    parser.add_argument("--skip", type=int, default=0, help="Clip skip layer (i.e., -1, -2) (0 automatically sets the default value for each model)")
    parser.add_argument("--save_path", type=str, default="./image", help="Directory to save generated images")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Data type")

    args = parser.parse_args()
    sample_size = 0.1 if args.sample_size else 0
    args.weight = list(args.weight) * len(args.ref) if len(args.weight) == 1 else args.weight
    generator = torch.Generator("cuda").manual_seed(args.seed)
    try:
        dtype = getattr(torch, args.dtype)
    except:
        dtype = getattr(torch, "bfloat16")

    # ============================
    # Load T2I pipeline
    # ============================
    if "flux" in args.model.lower():
        pipeline = FluxPipeline.from_pretrained(args.model, torch_dtype = dtype)
        size = 1024 if args.size == 0 else args.size
        num_inference_steps = 28 if args.num_inference_steps == 0 else args.num_inference_steps
        skip = -2 if args.skip == 0 else args.skip
    elif "stable-diffusion-3" in args.model.lower():
        pipeline = StableDiffusion3Pipeline.from_pretrained(args.model, torch_dtype=dtype)
        size = 1024 if args.size == 0 else args.size
        num_inference_steps = 28 if args.num_inference_steps == 0 else args.num_inference_steps
        skip = -2 if args.skip == 0 else args.skip
    else:
        pipeline = DiffusionPipeline.from_pretrained(args.model, torch_dtype=dtype)
        if "xl-" in args.model.lower():
            size = 1024 if args.size == 0 else args.size
            skip = -2 if args.skip == 0 else args.skip
        else:
            size = 512 if args.size == 0 else args.size
            skip = -1 if args.skip == 0 else args.skip
        num_inference_steps = 50 if args.num_inference_steps == 0 else args.num_inference_steps
    pipeline = pipeline.to("cuda")
    #pipeline.safety_checker = lambda images, clip_input: (images, [False] * len(images))

    # ============================
    # Load FAN tailored T2I pipeline
    # ============================
    encoder = personalized_t2i_encoder(pipeline, "./weight")

    # ============================
    # Preference-Aware Personalization in T2I diffusion models
    # ============================
    with torch.no_grad():
        cond, pool_cond = encoder(args.prompt, args.ref, weight = args.weight, alpha = args.alpha, skip = skip, sample_size = sample_size)
        images = pipeline(
            prompt_embeds = cond.type(dtype),
            pooled_prompt_embeds = pool_cond.type(dtype) if pool_cond is not None else pool_cond,
            num_images_per_prompt = 1,
            num_inference_steps = num_inference_steps,
            generator = generator,
            height = size,
            width = size,
        ).images

    # ============================
    # Save Results
    # ============================
    os.makedirs(args.save_path, exist_ok = True)
    for i, img in enumerate(images):
        img.save(os.path.join(args.save_path, f"{i:05d}.png"))

    print(f"Inference complete. Images saved to {args.save_path}")

if __name__ == "__main__":
    main()