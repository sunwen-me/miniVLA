from .VLMConfig import VLMConfig
from .model import *
from typing import Optional, Tuple, List
from torch import nn
import warnings
from transformers import CLIPProcessor, CLIPModel
from transformers import AutoModel, AutoProcessor
import torch

warnings.filterwarnings('ignore')

class Position2DEncoder(nn.Module):
    def __init__(self, num_patches: int, dim: int, height: int, width: int):
        super().__init__()
        self.height = height
        self.width = width
        assert num_patches == height * width
        self.row_embed = nn.Embedding(height, dim)
        self.col_embed = nn.Embedding(width, dim)

    def forward(self, batch_size: int, device=None):
        if device is None:
            device = next(self.row_embed.parameters()).device

        rows = torch.arange(self.height, device=device).unsqueeze(1).expand(self.height, self.width)
        cols = torch.arange(self.width, device=device).unsqueeze(0).expand(self.height, self.width)

        pos = self.row_embed(rows) + self.col_embed(cols)  # [H, W, D]
        pos = pos.reshape(-1, pos.shape[-1])  # [H*W, D]
        pos = pos.unsqueeze(0).expand(batch_size, -1, -1)  # [B, H*W, D]
        return pos

class VisionProj(nn.Module):
    def __init__(self, ve_dim=768, lm_dim=512):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(ve_dim, lm_dim),
            nn.ReLU(),
            nn.Linear(lm_dim, lm_dim)
        )
        self.rescale = nn.Linear(ve_dim, lm_dim)

    def forward(self, x):
        return self.proj(x) + self.rescale(x)


# 继承自语言模型
class MiniMindVLM(MiniMindLM):
    config_class = VLMConfig

    def __init__(self, params: VLMConfig = None):
        super().__init__(params)
        if not params: params = VLMConfig()
        self.params = params
        self.vision_encoder, self.processor = self.__class__.get_vision_model()
        self.vision_proj = VisionProj(lm_dim=params.dim)

        self.pos2d_encoder = Position2DEncoder(
            num_patches=196,  # eg. 14x14 patches
            dim=768,  # 与视觉 token 维度对齐
            height=14,
            width=14
        )
    @staticmethod
    def get_vision_model(model_path="./model/vision_model/siglip2-base-patch16-224"):
        model = AutoModel.from_pretrained(model_path)
        processor = AutoProcessor.from_pretrained(model_path)
        # 冻结 vision_encoder 的所有参数
        for param in model.parameters():
            param.requires_grad = False
        return model.eval(), processor

    @staticmethod
    def image2tensor(image, processor):
        if image.mode in ['RGBA', 'LA']: image = image.convert('RGB')
        inputs = processor(images=image, return_tensors="pt")['pixel_values']
        return inputs

    @staticmethod
    def get_image_embeddings(image_tensors, vision_model,pos_encoder=None):
        with torch.no_grad():
            outputs = vision_model.vision_model(pixel_values=image_tensors)
        img_embedding = outputs.last_hidden_state[:, 0:, :].squeeze()
        if pos_encoder:
            pos_embed = pos_encoder(batch_size=img_embedding.size(0), device=img_embedding.device)
            img_embedding = img_embedding + pos_embed

        return img_embedding.squeeze()

    def count_vision_proj(self, tokens, h, vision_tensors=None, seqlen=1024):
        def find_indices(tokens, image_ids):
            image_ids_tensor = torch.tensor(image_ids).to(tokens.device)
            len_image_ids = len(image_ids)
            if len_image_ids > tokens.size(1):
                return None
            tokens_view = tokens.unfold(1, len_image_ids, 1)
            matches = (tokens_view == image_ids_tensor).all(dim=2)
            return {
                batch_idx: [(idx.item(), idx.item() + len_image_ids - 1) for idx in
                            matches[batch_idx].nonzero(as_tuple=True)[0]]
                for batch_idx in range(tokens.size(0)) if matches[batch_idx].any()
            } or None

        image_indices = find_indices(tokens, self.params.image_ids)
        if vision_tensors is not None and image_indices:
            vision_proj = self.vision_proj(vision_tensors)
            if len(vision_proj.shape) == 3:
                vision_proj = vision_proj.unsqueeze(0)
            new_h = []
            for i in range(h.size(0)):
                if i in image_indices:
                    h_i = h[i]
                    img_idx = 0
                    for start_idx, end_idx in image_indices[i]:
                        if img_idx < vision_proj.size(1):
                            h_i = torch.cat((h_i[:start_idx], vision_proj[i][img_idx], h_i[end_idx + 1:]), dim=0)[
                                  :seqlen]
                            img_idx += 1
                    new_h.append(h_i)
                else:
                    new_h.append(h[i])
            return torch.stack(new_h, dim=0)
        return h

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                **args):
        start_pos = args.get('start_pos', 0)
        pixel_tensors = args.get('pixel_tensors', None)
        h = self.tok_embeddings(input_ids)

        if pixel_tensors is not None and start_pos == 0:
            if len(pixel_tensors.shape) == 6:
                pixel_tensors = pixel_tensors.squeeze(2)
            bs, num, c, im_h, im_w = pixel_tensors.shape
            stack_dim = 1 if bs > 1 else 0
            vision_tensors = torch.stack([
                MiniMindVLM.get_image_embeddings(pixel_tensors[:, i, :, :, :], self.vision_encoder, self.pos2d_encoder)
                for i in range(num)
            ], dim=stack_dim)
            h = self.count_vision_proj(tokens=input_ids, h=h, vision_tensors=vision_tensors, seqlen=input_ids.shape[1])

        pos_cis = self.pos_cis[start_pos:start_pos + input_ids.shape[1]]
        past_kvs = []
        for l, layer in enumerate(self.layers):
            h, past_kv = layer(
                h, pos_cis,
                past_key_value=past_key_values[l] if past_key_values else None,
                use_cache=use_cache
            )
            past_kvs.append(past_kv)

        logits = self.output(self.norm(h))
        aux_loss = sum(l.feed_forward.aux_loss for l in self.layers if isinstance(l.feed_forward, MOEFeedForward))

        self.OUT.__setitem__('logits', logits)
        self.OUT.__setitem__('aux_loss', aux_loss)
        self.OUT.__setitem__('past_key_values', past_kvs)
        return self.OUT
