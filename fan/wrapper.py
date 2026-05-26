import os

import torch

from .model import FAN

def stable_diffusion(large):
    """
    openai/clip-vit-large-patch14, CLIPTextModel, skip -1
    """
    def inference(prompt, ref_prompt = None, weight = None, alpha = 0.4, sample_size = 0, skip = -1, batch_size = 64, skip_pa = 0, use_attn_mask = False, **kwargs):
        return large(prompt, ref_prompt, pooling = False, weight = weight, alpha = alpha, sample_size = sample_size, skip = skip, batch_size = batch_size, skip_pa = skip_pa, use_attn_mask = use_attn_mask, **kwargs), None
    return inference

def stable_diffusion_v2(huge):
    """
    openai/clip-vit-huge-patch14, CLIPTextModel, skip -1
    """
    def inference(prompt, ref_prompt = None, weight = None, alpha = 0.4, sample_size = 0, skip = -1, batch_size = 64, skip_pa = 0, use_attn_mask = False, **kwargs):
        return huge(prompt, ref_prompt, pooling = False, weight = weight, alpha = alpha, sample_size = sample_size, skip = skip, batch_size = batch_size, skip_pa = skip_pa, use_attn_mask = use_attn_mask, **kwargs), None
    return inference

def stable_diffusion_xl(large, bigG):
    """
    openai/clip-vit-large-patch14, CLIPTextModel, skip -2, unnorm
    laion/CLIP-ViT-bigG-14-laion2B-39B-b160k, CLIPTextModelWithProjection, skip -2, unnorm, pooling + proj
    """
    def inference(prompt, ref_prompt = None, weight = None, alpha = 0.4, sample_size = 0, skip = -2, batch_size = 64, skip_pa = 0, use_attn_mask = False, **kwargs):
        hidden_state = large(prompt, ref_prompt, pooling = False, weight = weight, alpha = alpha, sample_size = sample_size, skip = skip, batch_size = batch_size, skip_pa = skip_pa, use_attn_mask = use_attn_mask, normalize = False, **kwargs)
        if skip == -1:
            hidden_state2, pool_hidden_state = bigG(prompt, ref_prompt, pooling = True, weight = weight, alpha = alpha, sample_size = sample_size, skip = skip, batch_size = batch_size, skip_pa = skip_pa, use_attn_mask = use_attn_mask, normalize = False, normalize_pool = True, **kwargs)
        else:
            hidden_state2 = bigG(prompt, ref_prompt, pooling = False, weight = weight, alpha = alpha, sample_size = sample_size, skip = skip, batch_size = batch_size, skip_pa = skip_pa, use_attn_mask = use_attn_mask, normalize = False, **kwargs)
            pool_hidden_state = bigG(prompt, ref_prompt, pooling = True, weight = weight, alpha = alpha, sample_size = sample_size, skip = -1, batch_size = batch_size, skip_pa = skip_pa, use_attn_mask = use_attn_mask, normalize = False, normalize_pool = True, **kwargs)[1]
        hidden_state = torch.cat([hidden_state.to(hidden_state2.device), hidden_state2], dim = -1)
        pool_hidden_state = bigG.projection_text_hidden_state(pool_hidden_state)
        return hidden_state.type(pool_hidden_state.dtype), pool_hidden_state
    return inference

def stable_diffusion_v3(large, bigG, t5 = None):
    """
    openai/clip-vit-large-patch14, CLIPTextModelWithProjection, skip -2, unnorm, pooling + proj
    laion/CLIP-ViT-bigG-14-laion2B-39B-b160k, CLIPTextModelWithProjection, skip -2, unnorm, pooling + proj
    t5-v1_1-xxl, T5EncoderModel
    """
    def inference(prompt, ref_prompt = None, weight = None, alpha = 0.4, sample_size = 0, skip = -2, batch_size = 64, skip_pa = 0, use_attn_mask = False, **kwargs):
        if skip == -1:
            hidden_state, pool_hidden_state = large(prompt, ref_prompt, pooling = True, weight = weight, alpha = alpha, sample_size = sample_size, skip = skip, batch_size = batch_size, skip_pa = skip_pa, use_attn_mask = use_attn_mask, normalize = False, normalize_pool = True, **kwargs)
            hidden_state2, pool_hidden_state2 = bigG(prompt, ref_prompt, pooling = True, weight = weight, alpha = alpha, sample_size = sample_size, skip = skip, batch_size = batch_size, skip_pa = skip_pa, use_attn_mask = use_attn_mask, normalize = False, normalize_pool = True, **kwargs)
        else:
            hidden_state = large(prompt, ref_prompt, pooling = False, weight = weight, alpha = alpha, sample_size = sample_size, skip = skip, batch_size = batch_size, skip_pa = skip_pa, use_attn_mask = use_attn_mask, normalize = False, **kwargs)
            hidden_state2 = bigG(prompt, ref_prompt, pooling = False, weight = weight, alpha = alpha, sample_size = sample_size, skip = skip, batch_size = batch_size, skip_pa = skip_pa, use_attn_mask = use_attn_mask, normalize = False, **kwargs)
            pool_hidden_state = large(prompt, ref_prompt, pooling = True, weight = weight, alpha = alpha, sample_size = sample_size, skip = -1, batch_size = batch_size, skip_pa = skip_pa, use_attn_mask = use_attn_mask, normalize = False, normalize_pool = True, **kwargs)[1]
            pool_hidden_state2 = bigG(prompt, ref_prompt, pooling = True, weight = weight, alpha = alpha, sample_size = sample_size, skip = -1, batch_size = batch_size, skip_pa = skip_pa, use_attn_mask = use_attn_mask,  normalize = False, normalize_pool = True, **kwargs)[1]
        hidden_state = torch.cat([hidden_state, hidden_state2], dim = -1)
        pool_hidden_state = large.projection_text_hidden_state(pool_hidden_state)
        pool_hidden_state2 = bigG.projection_text_hidden_state(pool_hidden_state2)
        if t5 is not None:
            hidden_state3 = t5(prompt, ref_prompt, pooling = False, weight = weight, alpha = alpha, sample_size = sample_size, skip_pa = skip_pa, use_attn_mask = use_attn_mask, normalize = False, **kwargs)
        else:
            hidden_state3 = torch.zeros((hidden_state.shape[0], large.n_token, 4096), dtype = hidden_state2.dtype, device = hidden_state2.device)
        hidden_state = torch.nn.functional.pad(hidden_state, (0, hidden_state3.shape[-1] - hidden_state.shape[-1]))
        hidden_state = torch.cat([hidden_state.to(hidden_state2.device), hidden_state3.to(hidden_state2.device)], dim = -2)
        pool_hidden_state = torch.cat([pool_hidden_state.to(hidden_state2.device), pool_hidden_state2], dim = -1)
        return hidden_state.type(pool_hidden_state.dtype), pool_hidden_state
    return inference

def flux(large, t5):
    """
    openai/clip-vit-large-patch14, CLIPTextModel, pooling
    t5-v1_1-xxl, T5EncoderModel
    """
    def inference(prompt, ref_prompt = None, weight = None, alpha = 0.4, sample_size = 0, skip = None, batch_size = 64, skip_pa = 0, use_attn_mask = False, **kwargs):
        hidden_state = t5(prompt, ref_prompt, pooling = False, weight = weight, alpha = alpha, sample_size = sample_size, batch_size = batch_size, skip_pa = skip_pa, use_attn_mask = use_attn_mask, normalize = False, **kwargs)
        pool_hidden_state = large(prompt, ref_prompt, pooling = True, weight = weight, alpha = alpha, sample_size = sample_size, skip = -1, batch_size = batch_size, skip_pa = skip_pa, use_attn_mask = use_attn_mask, normalize = False, normalize_pool = True, **kwargs)[1]
        return hidden_state.type(pool_hidden_state.dtype), pool_hidden_state
    return inference

def personalized_t2i_encoder(pipeline, decoder = None):
    dir_path = decoder if decoder is not None and os.path.isdir(decoder) else "./weight"
    
    if "FLUX" in pipeline.name_or_path:
        model = pipeline.text_encoder
        processor = pipeline.tokenizer
        model2 = pipeline.text_encoder_2
        processor2 = pipeline.tokenizer_2
        
        large = FAN(model, processor, decoder = os.path.join(dir_path, "L.pth")).eval()
        t5 = FAN(model2, processor2, max_token_size = 512).eval()
        
        encoder = flux(large, t5)
    elif "stable-diffusion-3" in pipeline.name_or_path: #sd v3
        model = pipeline.text_encoder
        processor = pipeline.tokenizer
        model2 = pipeline.text_encoder_2
        processor2 = pipeline.tokenizer_2
        model3 = pipeline.text_encoder_3
        processor3 = pipeline.tokenizer_3
        
        large = FAN(model, processor, decoder = os.path.join(dir_path, "L.pth")).eval()
        bigG = FAN(model2, processor2, decoder = os.path.join(dir_path, "bigG.pth")).eval()
        if model3 is not None:
            t5 = FAN(model3, processor3, max_token_size = 256).eval()
        
        encoder = stable_diffusion_v3(large, bigG, t5 if model3 is not None else None)
    elif "xl-" in pipeline.name_or_path: #sd xl
        model = pipeline.text_encoder
        processor = pipeline.tokenizer
        model2 = pipeline.text_encoder_2
        processor2 = pipeline.tokenizer_2
        
        large = FAN(model, processor, decoder = os.path.join(dir_path, "L.pth")).eval()
        bigG = FAN(model2, processor2, decoder = os.path.join(dir_path, "bigG.pth")).eval()
        
        encoder = stable_diffusion_xl(large, bigG)
    else: #sd
        model = pipeline.text_encoder
        processor = pipeline.tokenizer
        
        large = FAN(model, processor, decoder = os.path.join(dir_path, "L.pth")).eval()
        
        encoder = stable_diffusion(large)
    return encoder