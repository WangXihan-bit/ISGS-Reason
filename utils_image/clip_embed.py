import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import open_clip
from dataclasses import dataclass, field
from typing import Type

@dataclass
class OpenCLIPNetworkConfig:
    _target: Type = field(default_factory=lambda: OpenCLIPNetwork)
    clip_model_type: str = "ViT-B-16"
    clip_model_pretrained: str = "open_clip/open_clip_pytorch_model.bin"
    clip_n_dims: int = 512
   
class OpenCLIPNetwork(nn.Module):
    def __init__(self, args, config: OpenCLIPNetworkConfig):
        super().__init__()
        self.config = config
        self.process = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )
        self.device = args.device
        model, _, _ = open_clip.create_model_and_transforms(
            self.config.clip_model_type, 
            self.config.clip_model_pretrained,
            precision="fp16",
        )
        model.eval()
        self.tokenizer = open_clip.get_tokenizer(self.config.clip_model_type)
        self.model = model.to(args.device)
        self.clip_n_dims = self.config.clip_n_dims

    @property
    def name(self) -> str:
        return "openclip_{}_{}".format(self.config.clip_model_type, self.config.clip_model_pretrained)

    @property
    def embedding_dim(self) -> int:
        return self.config.clip_n_dims
    
    def encode_image(self, input):
        processed_input = self.process(input).half()
        return self.model.encode_image(processed_input)
    
    def encode_texts(self, texts):
        with torch.no_grad():
            tokenized_texts = torch.cat([self.tokenizer(text) for text in texts]).to(self.device)
            text_feats = self.model.encode_text(tokenized_texts)
        text_feats /= text_feats.norm(dim=-1, keepdim=True)
        return text_feats
    