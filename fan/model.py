import os
import types

import PIL
import numpy as np
import torch

def sample_reference(target, context, weight = None, sample_size = 0, batch_size = 64):
    b, n_token, d_model = target.shape
    ref_size = context.shape[1] // (n_token if context.dim() == 3 else 1)
    sample_size = max(min(round(ref_size * sample_size) if isinstance(sample_size, float) else sample_size, ref_size), 1)
    epsilon = torch.finfo(target.dtype).eps
    
    indices = None
    if sample_size < ref_size:
        aT = torch.nn.functional.normalize(target, dim = -1)
        context = context.view(b, ref_size, n_token, d_model)
        
        score = []
        for (start, end) in [(i * batch_size, min(ref_size, (i + 1) * batch_size)) for i in range(int(np.ceil(ref_size / batch_size)))]:
            aC = context[:, start:end]
            score.append(torch.einsum("btd,brsd->brst", aT, torch.nn.functional.normalize(aC, dim = -1)).mean(dim = (-2, -1)))
        score = torch.hstack(score)
        if weight is not None:
            score = score * weight
            
        pre_sample_size = np.round(np.sqrt(sample_size / ref_size) * ref_size).astype(int)
        score, indices = torch.topk(score, k = pre_sample_size, dim = -1, largest = True, sorted = False)
        
        bT = torch.nn.functional.adaptive_avg_pool2d(target.unsqueeze(1), (1, d_model)).squeeze(1)
        bT = torch.nn.functional.normalize(bT, dim = -1)
            
        bC, score = [], []
        context = torch.gather(context, dim = 1, index = indices.view(b, pre_sample_size, -1, 1).expand(-1, -1, n_token, d_model))
        for (start, end) in [(i * batch_size, min(pre_sample_size, (i + 1) * batch_size)) for i in range(int(np.ceil(pre_sample_size / batch_size)))]:
            _bC = context[:, start:end]
            _bC = torch.nn.functional.adaptive_avg_pool2d(_bC, (1, d_model))
            _bC = torch.nn.functional.normalize(_bC, dim = -1)
            bC.append(_bC)
            score.append(torch.einsum("btd,brsd->brst", bT, _bC).mean(dim = (-2, -1)))
        bC = torch.hstack(bC)
        score = torch.hstack(score)
        if weight is not None:
            bW = 1 / (torch.gather(weight, dim = 1, index = indices) + epsilon)
            score = score * bW
            
        indices2 = []
        bI = torch.arange(b).to(score.device)
        for i in range(sample_size):
            sample_index = torch.argmin(score, dim = 1)
            indices2.append(sample_index)
            if len(indices2) == sample_size:
                break

            sample = torch.gather(bC, dim = 1, index = sample_index.view(b, 1, 1, 1).expand(-1, -1, 1, d_model)).squeeze(1)
            sample_score = []
            for (start, end) in [(i * batch_size, min(pre_sample_size, (i + 1) * batch_size)) for i in range(int(np.ceil(pre_sample_size / batch_size)))]:
                sample_score.append(torch.einsum("btd,brsd->brst", sample, bC[:, start:end]).mean(dim = (-2, -1)))
            sample_score = torch.hstack(sample_score)
            if weight is not None:
                sample_score = sample_score * bW
                
            score = torch.maximum(score, sample_score)
            score[bI, sample_index] = float("inf")
        indices2 = torch.stack(indices2, dim = 1)
        indices = torch.gather(indices, dim = 1, index = indices2)
    return indices

def personalized_attention(query, key, value, mask = None, weight = 1, alpha = 0.4, n_token = 77, scale = 1, dropout = 0.0, training = False, **kwargs):
    b, h, s = query.shape[:3]
    b2, _, s2 = key.shape[:3]
    epsilon = torch.finfo(query.dtype).eps
        
    score = torch.matmul(query, key.transpose(-1, -2)) * scale
    if mask is not None:
        score = score + mask
    
    if weight is not None and torch.is_tensor(weight):
        weight = weight.unsqueeze(1)
        if weight.dim() != score.dim():
            weight = weight.unsqueeze(2)
    
    if training or n_token == s2:
        w = torch.nn.functional.softmax(score, dim = -1, dtype = torch.float32).to(query.dtype)
        if weight is not None and torch.is_tensor(weight):
            w = w * weight
            w = w / (w.sum(dim = -1, keepdim = True) + epsilon)
    else:
        target, ref = torch.split(score, [n_token, max(s2 - n_token, 1)], dim = -1)
        target = torch.nn.functional.softmax(target, dim = -1, dtype = torch.float32).to(query.dtype)
        if weight is not None and torch.is_tensor(weight):
            ws = weight.shape[-1] #Target token || number of ref
            target_weight, ref_weight = torch.split(weight, [n_token, max(ws - n_token, 1)], dim = -1)
            if ws != s2: #2 < token_size
                ref = ref.view(b2, h, s, ws - n_token, -1)
                ref = torch.nn.functional.softmax(ref, dim = -1, dtype = torch.float32).to(query.dtype)
                ref_weight = ref_weight.unsqueeze(-1)
            else: #token_size == 1
                ref = torch.ones_like(ref)
            target = target * target_weight
            ref = ref * ref_weight
            ref = ref.view(b2, h, s, -1)
        else:
            ref = torch.nn.functional.softmax(ref, dim = -1, dtype = torch.float32).to(query.dtype)
        if alpha is not None:
            target = target * (1 - alpha)
            ref = (ref / (ref.sum(dim = -1, keepdim = True) + epsilon)) * alpha
        w = torch.cat([target, ref], dim = -1)
        w = w / (w.sum(dim = -1, keepdim = True) + epsilon)
    w = torch.nn.functional.dropout(w, p = dropout, training = training)
    return torch.matmul(w, value)

def wrapper_forward(old_forward, batch_size = 1, weight = 1, alpha = 0.4, n_token = 77, memory_batch_size = 64):
    global_weight = weight
    def new_forward(self, hidden_states, **kwargs):
        b, s = hidden_states.shape[:2]
        out = None
        #if batch_size * n_token < b * s:
        if batch_size < b:
            hidden_states = hidden_states.view(batch_size, -1, self.embed_dim) #b * r, s, f > b, r * s, f
            k = v = hidden_states[:, s:] #b, (r - 1) * s, f
            weight = global_weight
            q = hidden_states[:, :s] #b, s, f
            hidden_states = hidden_states[:, s:].reshape(-1, s, self.embed_dim) #b * (r - 1), s, f

            b, s = q.shape[:2]
            b2, s2 = k.shape[:2]
            
            q = self.q_proj(q) #b, s, f
            k = self.k_proj(k) #b, (r - 1) * s, f
            v = self.v_proj(v)
            
            q = q.view(b, s, -1, self.head_dim).transpose(1, 2) #b, h, s, hf
            k = k.view(b2, s2, -1, self.head_dim).transpose(1, 2) #b, h, s, hf
            v = v.view(b2, s2, -1, self.head_dim).transpose(1, 2) #b, h, s, hf
            
            mask, mask_key = None, []
            if "causal_attention_mask" in kwargs and kwargs["causal_attention_mask"] is not None:
                mask_key.append("causal_attention_mask")
            if "attention_mask" in kwargs and kwargs["attention_mask"] is not None:
                mask_key.append("attention_mask")
                
            for m_key in mask_key:
                mask = kwargs[m_key].view(batch_size, -1, s, s)[:, 1:]
                kwargs[m_key] = mask.reshape(-1, 1, s, s)#.contiguous()
            if mask is not None:
                mask = mask.transpose(1, 2).reshape(batch_size, 1, s, -1)#.contiguous()
            
            out = personalized_attention(q, k, v, mask = mask, weight = weight, alpha = alpha, n_token = n_token, scale = self.scale)
            out = out.transpose(1, 2).reshape(b, s, -1).contiguous()
            out = self.out_proj(out) #b, s, f

        b = hidden_states.shape[0]
        attn_output, attn_weights = [], []
        for (start, end) in [(i * memory_batch_size, min(b, (i + 1) * memory_batch_size)) for i in range(int(np.ceil(b / memory_batch_size)))]:
            a, b = old_forward(hidden_states = hidden_states[start:end], **{k:v[start:end] if torch.is_tensor(v) and 2 < v.dim() else v for k, v in kwargs.items()})
            attn_output.append(a)
            attn_weights.append(b)
            del a, b
        attn_output = torch.vstack(attn_output)
        attn_weights = torch.vstack(attn_weights) if attn_weights[0] is not None else None
        if out is not None:
            attn_output = attn_output.view(batch_size, -1, self.embed_dim) #b * (r - 1), s, f > b, (r - 1) * s, f
            attn_output = torch.hstack([out, attn_output]) #b, r * s, f
            attn_output = attn_output.view(-1, s, self.embed_dim).contiguous()
        return attn_output, attn_weights
    return new_forward

def wrapper_t5_forward(old_forward, batch_size = 1, weight = 1, alpha = 0.4, n_token = 256, memory_batch_size = 64):
    global_weight = weight
    def new_forward(self, hidden_states, **kwargs):
        b, s = hidden_states.shape[:2]
        out = None
        #if batch_size * n_token < b * s:
        if batch_size < b:
            hidden_states = hidden_states.view(batch_size, -1, self.d_model) #b * r, s, f > b, r * s, f
            k = v = hidden_states[:, s:] #b, (r - 1) * s, f
            weight = global_weight
            q = hidden_states[:, :s] #b, s, f
            hidden_states = hidden_states[:, s:].reshape(-1, s, self.d_model) #b * (r - 1), s, f

            b, s = q.shape[:2]
            b2, s2 = k.shape[:2]

            q = self.q(q) #b, s, f
            k = self.k(k) #b, (r - 1) * s, f
            v = self.v(v)
            
            q = q.view(b, s, self.n_heads, self.key_value_proj_dim).transpose(1, 2) #b, h, s, hf
            k = k.view(b2, s2, self.n_heads, self.key_value_proj_dim).transpose(1, 2) #b, h, s, hf
            v = v.view(b2, s2, self.n_heads, self.key_value_proj_dim).transpose(1, 2) #b, h, s, hf
            
            causal_mask = None
            if "mask" in kwargs and kwargs["mask"] is not None:
                causal_mask = kwargs["mask"].view(batch_size, -1, 1, s)[:, 1:]
                kwargs["mask"] = causal_mask.reshape(-1, 1, 1, s)
                
            mask = None
            if "position_bias" in kwargs:
                if kwargs["position_bias"] is None:
                    if not self.has_relative_attention_bias:
                        position_bias = torch.zeros((1, self.n_heads, s, k.shape[-2]), device = q.device, dtype = q.dtype)
                        if self.gradient_checkpointing and self.training:
                            position_bias.requires_grad = True
                    else:
                        position_bias = self.compute_bias(s, s, device = q.device)
                        position_bias = position_bias.repeat(1, 1, 1, k.shape[-2] // s)[:, :, -s:, :]
                    if causal_mask is not None:
                        position_bias = position_bias + causal_mask.view(batch_size, 1, 1, -1)
                else:
                    position_bias = kwargs["position_bias"].view(batch_size, -1, self.n_heads, s, s)
                    if hidden_states.shape[0] < kwargs["position_bias"].shape[0]:
                        position_bias = position_bias[:, 1:]
                        kwargs["position_bias"] = position_bias.view(-1, self.n_heads, s, s)
                    position_bias = position_bias.permute([0, 2, 3, 1, 4])
                    position_bias = position_bias.reshape(batch_size, self.n_heads, s, -1).contiguous()
                if self.pruned_heads:
                    m = torch.ones(position_bias.shape[1], device = position_bias.device)
                    m[list(self.pruned_heads)] = 0
                    mask = position_bias[:, m.bool()]
                else:
                    mask = position_bias
             
            out = personalized_attention(q, k, v, mask = mask, weight = weight, alpha = alpha, n_token = n_token, scale = 1)
            out = out.transpose(1, 2).reshape(b, s, -1).contiguous()
            out = self.o(out) #b, s, f
            
        b = hidden_states.shape[0]
        result = []
        for (start, end) in [(i * memory_batch_size, min(b, (i + 1) * memory_batch_size)) for i in range(int(np.ceil(b / memory_batch_size)))]:
            r = old_forward(hidden_states = hidden_states[start:end], **{k:v[start:end] if torch.is_tensor(v) and 2 < v.dim() else v for k, v in kwargs.items()})
            for i, r in enumerate(r):
                if len(result) < i + 1:
                    result.append([])
                result[i].append(r)
            del r
        result = [torch.vstack(r) if torch.is_tensor(r[0]) else r[0] for r in result]
        if out is not None:
            result[0] = result[0].view(batch_size, -1, self.d_model) #b * (r - 1), s, f > b, (r - 1) * s, f
            result[0] = torch.hstack([out, result[0]]) #b, r * s, f
            result[0] = result[0].view(-1, s, self.d_model).contiguous()
        return tuple(result)
    return new_forward

class QuickGELU(torch.nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)

class ClassTokenDecoder(torch.nn.Module):
    def __init__(self, emb_dim = 768):
        super().__init__()
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, emb_dim // 4),
            QuickGELU(),
            torch.nn.Linear(emb_dim // 4, 1)
        )
    
    def forward(self, x, training = False):
        out = self.classifier(x.to(dtype = self.classifier[0].weight.dtype)).squeeze(-1) #B, T, E > B, T
        if training:
            return out
        return out.argmax(dim = -1)

class FAN:
    def __init__(self, model, processor, decoder = None, max_token_size = 256):
        text_config = model.config.text_config if hasattr(model.config, "text_config") else model.config
        vision_config = model.config.vision_config if hasattr(model.config, "vision_config") else model.config
        if hasattr(text_config, "model_type") and text_config.model_type == "t5":
            self.d_model = text_config.d_model
            self.n_token = min(processor.model_max_length, max_token_size)
            self.image_size = self.n_image_token = None
            self.clip = False
        else:
            self.n_token = text_config.max_position_embeddings if hasattr(text_config, "max_position_embeddings") else 77
            self.image_size = vision_config.image_size if hasattr(vision_config, "image_size") else None
            self.n_image_token = (self.image_size // vision_config.patch_size) ** 2 + 1 if hasattr(vision_config, "patch_size") else None
            self.clip = True
        self.model = model
        self.processor = processor
        
        self.decoder = None
        if self.clip and decoder is not None:
            self.decoder = ClassTokenDecoder(text_config.hidden_size).to(model.device)
            self.decoder.load_state_dict(torch.load(decoder, map_location=torch.device(model.device)))
        
        self.eval()
        self.to(model.device)
    
    def preprocess(self, x = None, text = None, image = None, return_tensors = "pt", padding = "max_length", truncation = True, **kwargs):
        if x is not None:
            x = [x] if np.ndim(x) in [0, 3] else x
            text, image = (None, x) if (isinstance(x[0], str) and os.path.splitext(x[0])[1][1:].lower() in ["jpg", "jpeg", "png", "gif", "bmp"]) or isinstance(x[0], PIL.Image.Image) or np.ndim(x) == 4 else (x, None)
                
        feed = {"return_tensors":return_tensors, **kwargs}
        if text is not None:
            feed.update({"text":[text] if np.ndim(text) == 0 else list(text),
                         "max_length":self.n_token,
                         "padding":padding,
                         "truncation":truncation})
        if image is not None:
            image = [image] if np.ndim(image) in [0, 3] else image
            feed["images"] = [load_image(img) if isinstance(img, str) else img for img in image] if isinstance(image[0], str) else image
        if not self.clip:
            feed["add_special_tokens"] = True
        return self.processor(**feed)
    
    def pool_text_hidden_state(self, hidden_state, x, padding = "max_length", truncation = True, **kwargs):
        if not self.clip:
            raise TypeError("This encoder does not support this function (pool_text_hidden_state).")
        if not hasattr(x, "items"):
            x = self.preprocess(x, padding = padding, truncation = truncation, **kwargs)
        if self.model.text_model.eos_token_id == 2:
            out = hidden_state[torch.arange(hidden_state.shape[0], device = hidden_state.device),
                              x["input_ids"].to(dtype = torch.int, device = hidden_state.device).argmax(dim = -1),]
        else:
            out = hidden_state[torch.arange(hidden_state.shape[0], device = hidden_state.device),
                              (x["input_ids"].to(dtype = torch.int, device = hidden_state.device) == self.model.text_model.eos_token_id).int().argmax(dim = -1),]
        return out
    
    def pool_image_hidden_state(self, hidden_state, x = None, **kwargs):
        if not self.clip:
            raise TypeError("This encoder does not support this function (pool_image_hidden_state).")
        out = hidden_state[:, 0, :]
        return out
    
    def normalize_text_hidden_state(self, hidden_state):
        out = self.model.text_model.final_layer_norm(hidden_state.type(self.model.dtype)) if self.clip and hasattr(self.model.text_model, "final_layer_norm") else hidden_state
        return out
    
    def normalize_image_hidden_state(self, hidden_state):
        out = self.model.vision_model.post_layernorm(hidden_state.type(self.model.dtype)) if self.clip and hasattr(self.model.vision_model, "post_layernorm") else hidden_state
        return out
    
    def projection_text_hidden_state(self, hidden_state):
        out = self.model.text_projection(hidden_state.type(self.model.dtype)) if self.clip and hasattr(self.model, "text_projection") else hidden_state
        return out
    
    def projection_image_hidden_state(self, hidden_state):
        out = self.model.visual_projection(hidden_state.type(self.model.dtype)) if self.clip and hasattr(self.model, "visual_projection") else hidden_state
        return out
    
    def get_text_feature(self, x, ref_x = None, weight = None, alpha = 0.4, skip = -1, batch_size = 64, skip_pa = 0, padding = "max_length", truncation = True, use_attn_mask = False, **kwargs):
        if not self.clip:
            raise TypeError("This encoder does not support this function (get_text_feature).")
        pool_hidden_state = self(x, ref_x, weight = weight, alpha = alpha, pooling = True, skip_pool = skip, batch_size = batch_size, skip_pa = skip_pa, padding = padding, truncation = truncation, use_attn_mask = use_attn_mask, normalize = False, normalize_pool = True, **kwargs)[1]
        result = self.projection_text_hidden_state(pool_hidden_state)
        return result
    
    def get_image_feature(self, x, ref_x = None, weight = None, alpha = 0.4, skip = -1, batch_size = 64, skip_pa = 0, **kwargs):
        if not self.clip:
            raise TypeError("This encoder does not support this function (get_image_feature).")
        pool_hidden_state = self(x, ref_x, weight = weight, alpha = alpha, pooling = True, skip_pool = skip, batch_size = batch_size, skip_pa = skip_pa, normalize = False, normalize_pool = True, **kwargs)[1]
        result = self.projection_image_hidden_state(pool_hidden_state)
        return result
    
    def encode_prompt(self, x, pooling = True, skip = -1, skip_pool = None, padding = "max_length", truncation = True, use_attn_mask = False, normalize = True, normalize_pool = True, return_mask = False, **kwargs):
        if not hasattr(x, "items"):
            x = self.preprocess(x, padding = padding, truncation = truncation, **kwargs)
        input_ids = x["input_ids"].to(self.device)
        attn_mask = x["attention_mask"].to(self.device) if use_attn_mask or return_mask else None
        with torch.no_grad():
            if self.clip:
                hidden_state = self.model.text_model(output_hidden_states = True, input_ids = input_ids, attention_mask = attn_mask if use_attn_mask else None)["hidden_states"]
                pool, hidden_state = hidden_state[skip_pool if skip_pool is not None else skip], hidden_state[skip]
                hidden_state = self.normalize_text_hidden_state(hidden_state) if normalize else hidden_state
            else:
                hidden_state = self.model(input_ids = input_ids, attention_mask = attn_mask if use_attn_mask else None)[0]
                pool = None
            if pooling:
                if self.clip:
                    pool = self.pool_text_hidden_state(self.normalize_text_hidden_state(pool) if normalize_pool else pool, x)
                hidden_state = (hidden_state, pool)
        return ((hidden_state + (attn_mask,)) if isinstance(hidden_state, tuple) else (hidden_state, attn_mask)) if return_mask else hidden_state
    
    def encode_image(self, x, pooling = True, skip = -1, skip_pool = None, use_attn_mask = False, normalize = True, normalize_pool = True, return_mask = False, **kwargs):
        if not self.clip:
            raise TypeError("This encoder does not support this function (encode_image).")
        if not hasattr(x, "items"):
            x = self.preprocess(x, **kwargs)
        pixel_values = x["pixel_values"].to(self.device, dtype = self.model.dtype)
        with torch.no_grad():
            hidden_state = self.model.vision_model(output_hidden_states = True, pixel_values = pixel_values)["hidden_states"]
            pool, hidden_state = hidden_state[skip_pool if skip_pool is not None else skip], hidden_state[skip]
            hidden_state = self.normalize_image_hidden_state(hidden_state) if normalize else hidden_state
            if pooling:
                pool = self.pool_image_hidden_state(pool)
                pool = self.normalize_image_hidden_state(pool) if normalize_pool else pool
                hidden_state = (hidden_state, pool)
        return ((hidden_state + (None,)) if isinstance(hidden_state, tuple) else (hidden_state, None)) if return_mask else hidden_state
   
    def encode_context(self, x, pooling = False, skip = -1, skip_pool = None, batch_size = 64, padding = "max_length", truncation = True, use_attn_mask = False, normalize = False, normalize_pool = False, return_mask = False, **kwargs):
        if not hasattr(x, "items") and np.ndim(x) in [0, 1, 3, 4]:
            shape_reduce = True
        elif hasattr(x, "items"):
            shape_reduce = list(x.values())[0].dim() in [2, 4]
        else:
            shape_reduce = False
        
        x = [x] if not hasattr(x, "items") and np.ndim(x) in [0, 3] else x
        if (hasattr(x, "items") and "pixel_values" in x) or (not hasattr(x, "items") and ((isinstance(x[0], str) and os.path.splitext(x[0])[1][1:].lower() in ["jpg", "jpeg", "png", "gif", "bmp"]) or isinstance(x[0], PIL.Image.Image))):
            keys, n_token, encode_func = ["pixel_values"], self.n_image_token, self.encode_image
            if not self.clip or n_token is None:
                raise TypeError("This encoder does not support vision-based this function (encode_context).")
        else:
            keys, n_token, encode_func = ["input_ids", "attention_mask"], self.n_token, self.encode_prompt
                                
        if not hasattr(x, "items"):
            x = [x] if np.ndim(x) in [1, 4] else x
            b, ref_size, shape = len(x), len(x[0]), np.shape(x[0][0])
            #x = np.reshape(x, [b * ref_size, -1])
            x = np.reshape(x, [b * ref_size, *shape])
            x = self.preprocess(x, padding = padding, truncation = truncation, **kwargs)
        else:
            b, ref_size, shape = (1, x[keys[0]].shape[0], x[keys[0]].shape[1:]) if x[keys[0]].dim() in [2, 4] else (*x[keys[0]].shape[:2], x[keys[0]].shape[2:])
            x = {k:v.reshape(b * ref_size, *shape) for k, v in x.items()}
        x = {k:v.to(self.device) for k, v in x.items()}
        attn_mask = x["attention_mask"].view(b, ref_size, -1) if return_mask and "attention_mask" in x else None
        
        hidden_state, pool_hidden_state = [], []
        batch_indices = [(i * batch_size, min((b * ref_size), (i + 1) * batch_size)) for i in range(int(np.ceil((b * ref_size) / batch_size)))]
        for start, end in batch_indices:
            out = encode_func({k:v[start:end] for k, v in x.items()}, pooling = pooling, skip = skip, skip_pool = skip_pool, use_attn_mask = use_attn_mask, normalize = normalize, normalize_pool = normalize_pool, **kwargs)
            if isinstance(out, tuple):
                hidden_state.append(out[0])
                pool_hidden_state.append(out[1])
            else:
                hidden_state.append(out)
        with torch.no_grad():
            hidden_state = torch.cat(hidden_state, dim = 0) if 1 < len(hidden_state) else hidden_state[0]
            pool_hidden_state = torch.cat(pool_hidden_state, dim = 0) if 1 < len(pool_hidden_state) else (pool_hidden_state[0] if len(pool_hidden_state) == 1 else None)
            hidden_state = hidden_state.view(b, ref_size, hidden_state.shape[1], -1).contiguous()
            if pooling:
                if self.clip:
                    pool_hidden_state = pool_hidden_state.view(b, ref_size, -1).contiguous()
                hidden_state = (hidden_state, pool_hidden_state)
        result = ((hidden_state + (attn_mask,)) if isinstance(hidden_state, tuple) else (hidden_state, attn_mask)) if return_mask else hidden_state
        if shape_reduce:
            result = tuple([r.view(-1, *r.shape[2:]) for r in result]) if isinstance(result, tuple) else result.view(-1, *result.shape[2:])
        return result
    
    def __call__(self, x, ref_x = None, weight = None, alpha = 0.4,
                 pooling = True, sample_size = 0,
                 skip = -1, skip_pool = None, batch_size = 64, skip_pa = 0,
                 padding = "max_length", truncation = True, use_attn_mask = False,
                 normalize = True, normalize_pool = True, **kwargs):
        x = [x] if not hasattr(x, "items") and np.ndim(x) in [0, 3] else x
        if (hasattr(x, "items") and "pixel_values" in x) or (not hasattr(x, "items") and ((isinstance(x[0], str) and os.path.splitext(x[0])[1][1:].lower() in ["jpg", "jpeg", "png", "gif", "bmp"]) or isinstance(x[0], PIL.Image.Image) or np.ndim(x) == 4)):
            vision, keys, n_token = True, ["pixel_values"], self.n_image_token
            model, norm_func, pool_func = self.model.vision_model, self.normalize_image_hidden_state, self.pool_image_hidden_state
            if not self.clip or n_token is None:
                raise TypeError("This encoder does not support vision-based conditioning.")
        else:
            vision, keys, n_token = False, ["input_ids", "attention_mask"], self.n_token
            model, norm_func, pool_func = self.model.text_model if self.clip else self.model, self.normalize_text_hidden_state, self.decoder
        
        if ref_x is not None:
            total_indices = indices = list(range(len(model.encoder.layers if self.clip else model.encoder.block)))
            if skip_pa is not None:
                skip_pa = [indices[i] for i in ([skip_pa] if not isinstance(skip_pa, (tuple, list)) else skip_pa)]
                indices = sorted(np.unique([i for i in indices if i not in skip_pa]))
            
            if not hasattr(x, "items"):
                x = self.preprocess(x, padding = padding, truncation = truncation, **kwargs)
            b1 = x[keys[0]].shape[0]

            if not hasattr(ref_x, "items"):
                if np.ndim(ref_x) in [0, 3]:
                    ref_x = [[ref_x]]
                elif np.ndim(ref_x) in [1, 4]:
                    ref_x = [ref_x]
                b2 = len(ref_x)
                ref_x = self.preprocess(np.reshape(ref_x, [-1, *np.shape(ref_x[0][0])] if np.ndim(ref_x) in [2, 5] else [-1]), padding = padding, truncation = truncation, **kwargs)
                ref_x = {k:v.view(b2, -1, *v.shape[1:]) for k, v in ref_x.items()}
            b2, ref_size = ref_x[keys[0]].shape[:2]

            if b1 == 1 and b1 != b2:
                x = {k:v.repeat_interleave(b2, dim = 0) for k, v in x.items()}
                b1 = b2
            if b2 == 1 and b2 != b1:
                ref_x = {k:v.repeat_interleave(b1, dim = 0) for k, v in ref_x.items()}
                b2 = b1
            
            if weight is not None:
                if not torch.is_tensor(weight):
                    weight = torch.tensor(weight)
                if weight.dim() == 0:
                    weight = weight.unsqueeze(0).unsqueeze(0)
                elif weight.dim() == 1:
                    weight = weight.unsqueeze(0)
                weight = weight.to(dtype = self.model.dtype, device = self.device)
            else:
                weight = torch.ones((1, ref_size), dtype = self.model.dtype, device = self.device)
            if weight.shape[0] == 1 and weight.shape[0] != b1:
                weight = weight.repeat_interleave(b1, dim = 0)
            extra_weight = torch.full((weight.shape[0], n_token), 1, dtype = self.model.dtype, device = self.device)
            
            if 0 < sample_size:
                target = (self.encode_image if vision else self.encode_prompt)(x, pooling = False, skip = -1, use_attn_mask = use_attn_mask)
                context = self.encode_context(ref_x, pooling = False, skip = -1, batch_size = batch_size, use_attn_mask = use_attn_mask)
                sample_indices = sample_reference(target, context, weight = weight, sample_size = sample_size, batch_size = batch_size)
                del target, context
            
                if sample_indices is not None:
                    ref_x = {k:torch.gather(v, dim = 1, index = sample_indices.to(v.device).view(b1, sample_indices.shape[1], *([1] * max(v.dim() - 2, 1))).expand(-1, -1, *v.shape[2:])) for k, v in ref_x.items()}
                    weight = torch.gather(weight, dim = 1, index = sample_indices)

            base_query = x
            shape = [-1, 3, self.image_size, self.image_size] if vision else [-1, n_token]
            x = {k:torch.cat([base_query[k].unsqueeze(1), x[k].unsqueeze(1), ref_x[k]], dim = 1).view(*shape).contiguous() for k in keys}
            weight = torch.cat([extra_weight, weight], dim = 1)
            
            try:
                x = {k:v for k, v in x.items() if k != "attention_mask"} if not use_attn_mask and "attention_mask" in x else x
                total_size = x[keys[0]].shape[0]

                old_forward = {}
                for index in total_indices:
                    target = model.encoder.layers[index].self_attn if self.clip else model.encoder.block[index].layer[0].SelfAttention
                    old_forward[index] = target.forward
                    target.forward = types.MethodType((wrapper_forward if self.clip else wrapper_t5_forward)(target.forward, batch_size = b1 if index in indices else total_size, weight = weight, alpha = alpha, n_token = n_token, memory_batch_size = batch_size), target)
                
                x = {k:v.to(self.device) for k, v in x.items()}
                if self.clip:
                    unnorm_hidden_state = model(output_hidden_states = True, **x)["hidden_states"]
                    unnorm_hidden_state, pool = unnorm_hidden_state[skip], unnorm_hidden_state if pooling else None
                else:
                    unnorm_hidden_state, pool = model(**x)[0], None
                unnorm_hidden_state = unnorm_hidden_state.view(b1, -1, unnorm_hidden_state.shape[-1])[:, :n_token].contiguous() #b * r, s, f > b, r * s, f > #b, s, f
            except Exception as e:
                raise e
            finally:
                for index in total_indices:
                    if index in old_forward:
                        target = model.encoder.layers[index].self_attn if self.clip else model.encoder.block[index].layer[0].SelfAttention
                        target.forward = old_forward.pop(index)
                del old_forward
            
            hidden_state = norm_func(unnorm_hidden_state) if normalize else unnorm_hidden_state
            if pooling:
                if pool is not None:
                    if vision:
                        pool = pool[skip_pool if skip_pool is not None else skip].view(b1, -1, hidden_state.shape[-1])[:, :n_token].contiguous()
                        pool = pool_func(pool)
                    else:
                        cand = pool[-1].view(b1, -1, hidden_state.shape[-1])[:, :n_token].contiguous()
                        pool = pool[skip_pool if skip_pool is not None else skip].view(b1, -1, hidden_state.shape[-1])[:, :n_token].contiguous() if (skip_pool if skip_pool is not None else skip) != -1 else cand
                        pool = pool[torch.arange(hidden_state.shape[0], device = hidden_state.device), pool_func(cand)]
                    pool = norm_func(pool) if normalize_pool else pool
                return (hidden_state, pool)
            return hidden_state
        else:
            result = self.encode_context(x, pooling = pooling, skip = skip, skip_pool = skip_pool, batch_size = batch_size, padding = padding, truncation = truncation, use_attn_mask = use_attn_mask, normalize = normalize, normalize_pool = normalize_pool, **kwargs)
            if pooling:
                result = (result, None) if not isinstance(result, tuple) else result
            elif isinstance(result, tuple):
                result = x[0]
            return result
    
    def to(self, device, pipeline = True):
        if pipeline and self.model.device != device:
            self.model.to(device)
        self.device = device
        if self.clip and self.decoder is not None:
            self.decoder.to(device)
        return self
        
    def eval(self):
        self.model.eval()
        if self.clip and self.decoder is not None:
            self.decoder.eval()
        return self
        
    def train(self):
        self.model.train()
        if self.clip:
            self.decoder.train()
        return self
        
    def parameters(self):
        return list(self.model.parameters())
    
    def named_parameters(self):
        return list(self.model.named_parameters())
