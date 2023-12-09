import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, GPT2Config
from types import SimpleNamespace
from timm import create_model
from typing import Tuple, Union
from src.component.model import GPT2Block
from src.logs import logger 
class VisionGPT2Model(nn.Module):
    def __init__(self, config: SimpleNamespace) -> None:
        super().__init__()

        self.config = config

        vit = create_model('vit_base_patch16_224',
                           pretrained=True, num_classes=0)
        self.patch_embed = vit.patch_embed
        logger.info(f" patch embeddings {self.patch_embed}")
        self.cls_token = vit.cls_token
        self.pos_embed = vit.pos_embed
        self.pos_drop = nn.Dropout(p=0.)

        self.blocks = nn.ModuleList([vit.blocks[i]
                                    for i in range(config.depth)])

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.embed_dim),
            wpe=nn.Embedding(config.seq_len, config.embed_dim),
            drop=nn.Dropout(config.emb_dropout),
            h=nn.ModuleList([GPT2Block(config) for _ in range(config.depth)]),
            ln_f=nn.LayerNorm(config.embed_dim)
        ))
        self.lm_head = nn.Linear(
            config.embed_dim, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

    def _pos_embed(self, x: torch.Tensor) -> torch.Tensor:
        pos_embed = self.pos_embed
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + pos_embed
        return self.pos_drop(x)

    def pretrained_layers_trainable(self, trainable: bool = False) -> None:
        """
        This method freezes the pretrained layers of the Vision Transformer backbone.   
    
        Parameters: 
        ----------- 
        trainable (bool):   
            Whether to freeze the pretrained layers or not. 
        
        Returns:    
        --------    
        None    
        """
        layers = [
            self.cls_token, self.patch_embed, self.pos_embed, self.blocks,
            self.transformer.wte, self.transformer.wpe,
            self.transformer.ln_f, self.lm_head
        ]
        gpt_layers = [[
            self.transformer.h[i].ln_1, self.transformer.h[i].ln_2,
            self.transformer.h[i].attn, self.transformer.h[i].mlp
        ] for i in range(self.config.depth)]
        for l in gpt_layers:
            layers.extend(l)

        for layer in layers:
            if not isinstance(layer, nn.Parameter):
                for p in layer.parameters():
                    p.requires_grad = trainable
            else:
                layer.requires_grad = trainable

        total_frozen_params = sum(
            [p.numel() for p in self.parameters() if not p.requires_grad])
        print(f'{total_frozen_params=}')

    def unfreeze_gpt_layers(self) -> None:
        """
        This method unfreezes the GPT2 layers for fine-tuning.
        """
        gpt_layers = [[
            self.transformer.h[i].ln_1, self.transformer.h[i].ln_2,
            self.transformer.h[i].attn, self.transformer.h[i].mlp
        ] for i in range(self.config.depth)]
        flatten = []
        for l in gpt_layers:
            flatten.extend(l)

        for layer in flatten:
            if not isinstance(layer, nn.Parameter):
                for p in layer.parameters():
                    p.requires_grad = True
            else:
                layer.requires_grad = True

    @classmethod
    def from_pretrained(cls, config: GPT2Config) -> 'VisionGPT2Model':
        """
        This method loads the pretrained GPT2 model from HuggingFace and initializes the weights of the Vision Transformer backbone with the pretrained weights of the GPT2 model.
        
        Parameters: 
        -----------
        config (GPT2Config):    
            The GPT2Config object.  
        
        Returns:    
        --------
        VisionGPT2Model:   
            The VisionGPT2Model object. 
        """
        model = VisionGPT2Model(config)
        sd = model.state_dict()
        ignore_matches = ['blocks.', 'cross_attn.', 'ln_3',
                          'cls_token', 'pos_embed', 'patch_embed.', '.attn.mask']
        gpt2_small = GPT2LMHeadModel.from_pretrained('gpt2')
        sd_hf = gpt2_small.state_dict()
        hf_keys = sd_hf.keys()
        hf_keys = [key for key in hf_keys if not key.endswith(
            '.attn.masked_bias')]
        hf_keys = [key for key in hf_keys if not key.endswith('.attn.bias')]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight',
                      'mlp.c_fc.weight', 'mlp.c_proj.weight']

        for key in hf_keys:
            if any(match in key for match in ignore_matches):
                continue
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

    def forward(self, 
                image: torch.Tensor, 
                input_ids: torch.Tensor, 
                labels: Union[torch.Tensor, None] = None
               ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass of the model.
    
        Parameters:
        ----------- 
        image (torch.Tensor):   
            The image tensor.   
        input_ids (torch.Tensor):   
            The input sequence tensor.
        labels (torch.Tensor):      
            The labels tensor.
        
        Returns:
        --------    
        lm_logits (torch.Tensor):   
            The logits tensor.  
        """
        image = self.patch_embed(image)
        image = self._pos_embed(image)

        token_embeddings = self.transformer.wte(input_ids) 
        pos_embs = torch.arange(0, input_ids.size(1)).to(input_ids.device)
        positional_embeddings = self.transformer.wpe(pos_embs)
        input_ids = self.transformer.drop(
            token_embeddings+positional_embeddings)

        for i in range(self.config.depth):
            image = self.blocks[i](image)
            input_ids = self.transformer.h[i](input_ids, image)

        input_ids = self.transformer.ln_f(input_ids)

        if labels is not None:
            lm_logits = self.lm_head(input_ids)
            loss = F.cross_entropy(
                lm_logits.view(-1, lm_logits.shape[-1]), labels.view(-1))
            return loss

        lm_logits = self.lm_head(input_ids[:, [-1], :])
        return lm_logits

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

