import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from types import SimpleNamespace
from timm import create_model


class GPT2Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.n_heads = config.num_heads
        assert self.embed_dim % self.n_heads == 0, 'embedding dimension by be divisible by number of heads'
        self.head_size = self.embed_dim // self.n_heads
        self.seq_len = config.seq_len

        self.c_attn = nn.Linear(
            self.embed_dim, self.head_size * self.n_heads * 3, bias=True)
        self.scale = self.head_size ** -0.5

        self.register_buffer('mask', torch.tril(
            torch.ones(1, 1, self.seq_len, self.seq_len)))

        self.c_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

        self.attn_dropout = nn.Dropout(config.attention_dropout)
        self.resid_dropout = nn.Dropout(config.residual_dropout)

    def forward(self, x):
        batch_size, sequence_len, feature_size = x.shape
        # query,key,value shape individually: batch_size * seq_len * embed_dim
        # we know that qk_t = query * k_t, where query=batch_size * sequence_len * head_dim, k_t=batch_size * head_tim * sequence_len
        query, key, value = self.c_attn(x).chunk(3, dim=-1)
        query = query.view(batch_size, sequence_len, self.n_heads, self.head_size).permute(
            0, 2, 1, 3)  # batch * n_heads * seq_len * head_dim
        key = key.view(batch_size, sequence_len, self.n_heads,
                       self.head_size).permute(0, 2, 1, 3)
        value = value.view(batch_size, sequence_len,
                           self.n_heads, self.head_size).permute(0, 2, 1, 3)

        qk_t = (query@key.transpose(-2, -1)) * self.scale
        qk_t = qk_t.masked_fill(
            self.mask[:, :, :sequence_len, :sequence_len] == 0, float('-inf'))
        qk_t = F.softmax(qk_t, dim=-1)
        weights = self.attn_dropout(qk_t)

        attention = weights @ value  # batch * n_heads * sequence_len * head_size
        attention = attention.permute(0, 2, 1, 3).contiguous().view(
            batch_size, sequence_len, feature_size)  # batch * sequence_len * embed_dim

        out = self.c_proj(attention)
        out = self.resid_dropout(out)

        return out


class GPT2CrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.n_heads = config.num_heads
        assert self.embed_dim % self.n_heads == 0, 'embedding dimension by be divisible by number of heads'
        self.head_size = self.embed_dim // self.n_heads
        self.seq_len = config.seq_len

        self.query = nn.Linear(self.embed_dim, self.embed_dim)
        self.key = nn.Linear(self.embed_dim, self.embed_dim)
        self.value = nn.Linear(self.embed_dim, self.embed_dim)
        self.scale = self.head_size ** -0.5

        self.c_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

        self.attn_dropout = nn.Dropout(config.attention_dropout)
        self.resid_dropout = nn.Dropout(config.residual_dropout)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, query, key, value):
        batch_size, sequence_len, feature_size = query.shape

        query = self.query(query)
        key = self.key(key)
        value = self.value(value)

        query = query.view(batch_size, query.size(1), self.n_heads, self.head_size).permute(
            0, 2, 1, 3)  # batch * n_heads * seq_len * head_dim
        key = key.view(batch_size, key.size(1), self.n_heads,
                       self.head_size).permute(0, 2, 1, 3)
        value = value.view(batch_size, value.size(
            1), self.n_heads, self.head_size).permute(0, 2, 1, 3)

        qk_t = (query@key.transpose(-2, -1)) * self.scale
        qk_t = F.softmax(qk_t, dim=-1)
        weights = self.attn_dropout(qk_t)

        attention = weights @ value  # batch * n_heads * sequence_len * head_size
        attention = attention.permute(0, 2, 1, 3).contiguous().view(
            batch_size, sequence_len, feature_size)  # batch * sequence_len * embed_dim

        out = self.c_proj(attention)
        out = self.resid_dropout(out)

        return out


class GPT2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.mlp_ratio = config.mlp_ratio
        self.mlp_dropout = config.mlp_dropout

        self.c_fc = nn.Linear(self.embed_dim, self.embed_dim*self.mlp_ratio)
        self.c_proj = nn.Linear(self.embed_dim*self.mlp_ratio, self.embed_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(self.mlp_dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x





class GPT2Block(nn.Module):
    def __init__(self, config, num_cross_attention_layers=1):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.ln_1 = nn.LayerNorm(self.embed_dim)
        self.attn = GPT2Attention(config)
        self.ln_2 = nn.LayerNorm(self.embed_dim)
        self.cross_attn_layers = nn.ModuleList([
            GPT2CrossAttention(config) for _ in range(num_cross_attention_layers)
        ])
        self.ln_cross_attn = nn.ModuleList([
            nn.LayerNorm(self.embed_dim) for _ in range(num_cross_attention_layers)
        ])
        self.mlp = GPT2MLP(config)
        self.ln_3 = nn.LayerNorm(self.embed_dim)

    def forward(self, x, enc_out):
        x = x + self.attn(self.ln_1(x))
        for cross_attn, ln in zip(self.cross_attn_layers, self.ln_cross_attn):
            x = x + cross_attn(ln(x), enc_out, enc_out)
        x = x + self.mlp(self.ln_3(x))
        return x


class VisionGPT2Model(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        vit = create_model('vit_base_patch16_224',
                           pretrained=True, num_classes=0)
        self.patch_embed = vit.patch_embed
        num_patches = self.patch_embed.num_patches

        self.cls_token = vit.cls_token
        embed_len = num_patches + vit.num_prefix_tokens
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

    def _pos_embed(self, x):
        pos_embed = self.pos_embed
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + pos_embed
        return self.pos_drop(x)

    def pretrained_layers_trainable(self, trainable=False):
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

    def unfreeze_gpt_layers(self,):
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
    def from_pretrained(self, config):
        model = VisionGPT2Model(config)
        sd = model.state_dict()
        keys = sd.keys()
        ignore_matches = ['blocks.', 'cross_attn.', 'ln_3',
                          'cls_token', 'pos_embed', 'patch_embed.', '.attn.mask']
        vit_keys = [key for key in keys if any(
            match in key for match in ignore_matches)]
        gpt_keys = [key for key in keys if key not in vit_keys]

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

    def forward(self, image, input_ids, labels=None):

        image = self.patch_embed(image)
        image = self._pos_embed(image)

        token_embeddings = self.transformer.wte(input_ids)  # batch x seq_len
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

    def generate(self, image, sequence, max_tokens=50, temperature=1.0, deterministic=False):
        batch_size = sequence.shape[0]
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

            # Check if all sequences in the batch have reached EOS
            eos_reached = (next_token == self.tokenizer.eos_token_id).view(-1)
            if eos_reached.all():
                break

        return generated.cpu()

