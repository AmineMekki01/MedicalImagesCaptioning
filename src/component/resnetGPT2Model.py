
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from types import SimpleNamespace
from src.component.model import GPT2Block
from torchvision.models import resnet50

class ResnetImageEncoder(nn.Module):
    
    def __init__(self, config: SimpleNamespace):
        super(ResnetImageEncoder, self).__init__()
        self.config = config

        resnet = resnet50(pretrained=True)

        for param in resnet.parameters():
            param.requires_grad = not self.config.fine_tune
        
        self.resnet50 = nn.Sequential(*list(resnet.children())[:-2])
        self.conv = nn.Conv2d(2048, 768, kernel_size=1)

        for param in self.conv.parameters():
            param.requires_grad = True

    def forward(self, images):
        features = self.resnet50(images) # [batch_size, 2048, 7]
        features = self.conv(features) # [batch_size, 768, 7, 7]
        features = features.permute(0,2,3,1) # [batch_size, 7, 7, 768]
        features = features.view(features.size(0), -1, features.size(3)) # [batch_size, 49, 768]
        return features
    

class ResnetGPT2Model(nn.Module):
    def __init__(self, config : SimpleNamespace):
        super().__init__()
        
        self.config = config
        self.resnet_encoder = ResnetImageEncoder(self.config)  

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size,config.embed_dim),
            wpe = nn.Embedding(config.seq_len,config.embed_dim),
            drop = nn.Dropout(config.emb_dropout),
            h = nn.ModuleList([GPT2Block(config) for _ in range(config.depth)]),
            ln_f = nn.LayerNorm(config.embed_dim)
        ))
        self.lm_head = nn.Linear(config.embed_dim,config.vocab_size,bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        
    def pretrained_layers_trainable(self, trainable=False):
        gpt_layers = [
            self.transformer.wte, 
            self.transformer.wpe, 
            self.transformer.ln_f, 
            self.lm_head
        ]

        for i in range(self.config.depth):
            gpt_layers.extend([
                self.transformer.h[i].ln_1, 
                self.transformer.h[i].ln_2,
                self.transformer.h[i].attn, 
                self.transformer.h[i].mlp
            ])

        for layer in gpt_layers:
            for p in layer.parameters():
                p.requires_grad = trainable

        total_frozen_params = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        print(f'{total_frozen_params=}')

    def unfreeze_gpt_layers(self):
        for i in range(self.config.depth):
            layers_to_unfreeze = [
                self.transformer.h[i].ln_1, 
                self.transformer.h[i].ln_2,
                self.transformer.h[i].attn, 
                self.transformer.h[i].mlp
            ]
            
            for layer in layers_to_unfreeze:
                for p in layer.parameters():
                    p.requires_grad = True
                    
    @classmethod
    def from_pretrained(cls, config):
        model = ResnetGPT2Model(config)
        sd = model.state_dict()
        keys = sd.keys()

        gpt2_small = GPT2LMHeadModel.from_pretrained('gpt2')
        sd_hf = gpt2_small.state_dict()
        hf_keys = sd_hf.keys()
        hf_keys = [key for key in hf_keys if not key.endswith('.attn.masked_bias')]
        hf_keys = [key for key in hf_keys if not key.endswith('.attn.bias')]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        for key in hf_keys:
            if key in keys:
                if any(key.endswith(w) for w in transposed):
                    assert sd_hf[key].shape[::-1] == sd[key].shape
                    with torch.no_grad():
                        sd[key].copy_(sd_hf[key].t())
                else:
                    assert sd_hf[key].shape == sd[key].shape
                    with torch.no_grad():
                        sd[key].copy_(sd_hf[key])

        model.load_state_dict(sd)

        return model

    
    def forward(self, image, input_ids, labels=None):
        image_features = self.resnet_encoder(image)

        token_embeddings = self.transformer.wte(input_ids) # batch x seq_len
        pos_embs = torch.arange(0, input_ids.size(1)).to(input_ids.device)
        positional_embeddings = self.transformer.wpe(pos_embs)
        input_ids = self.transformer.drop(token_embeddings + positional_embeddings)

        for i in range(self.config.depth):
            input_ids = self.transformer.h[i](input_ids, image_features)

        input_ids = self.transformer.ln_f(input_ids)

        if labels is not None:
            lm_logits = self.lm_head(input_ids)
            loss = F.cross_entropy(lm_logits.view(-1, lm_logits.shape[-1]), labels.view(-1))
            return loss

        lm_logits = self.lm_head(input_ids[:, [-1], :])
        return lm_logits
    
    # import image
    
    @torch.no_grad()
    def generate(self, 
                 image: torch.Tensor, 
                 sequence: torch.Tensor, 
                 max_tokens: int = 50, 
                 temperature: float = 1.0, 
                 deterministic: bool = False
                ) -> torch.Tensor:
        """
        Generate a caption for an image.    
    
        Parameters: 
        ----------- 
        image (torch.Tensor):   
            The image tensor.   
        sequence (torch.Tensor):    
            The input sequence tensor.  
        max_tokens (int):       
            The maximum number of tokens to generate.   
        temperature (float):        
            The temperature for the sampling distribution.  
        deterministic (bool):   
            Whether to use deterministic sampling.  
            
        Returns:    
        --------
        torch.Tensor:   
            The generated caption tensor.
        """
        generated = sequence
        for _ in range(max_tokens):
            out = self(image, generated)
            out = out[:, -1, :] / temperature
            probs = F.softmax(out, dim=-1)
            if deterministic:
                next_token = torch.argmax(probs, dim=-1, keepdim=True)
            else:
                next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)

            eos_reached = (next_token == self.tokenizer.eos_token_id).view(-1)
            if eos_reached.all():
                break

        return generated.cpu()