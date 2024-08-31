# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from ..modules.transformer import Transformer, TransformerConfig
from ..utils import sampling


@dataclass
class DepFormerConfig:
    transformer: TransformerConfig
    num_slices: int


@dataclass
class LmConfig:
    transformer: TransformerConfig
    depformer: DepFormerConfig
    text_in_vocab_size: int
    text_out_vocab_size: int
    audio_vocab_size: int
    audio_codebooks: int
    audio_delays: List[int]

    @property 
    def audio_eos_token(self) -> int:
        return self.audio_vocab_size - 2

    @property 
    def audio_padding_token(self) -> int:
        return self.audio_vocab_size - 1

class DepFormerSlice(nn.Module):
    def __init__(self, in_vocab_size: int, out_vocab_size: int, main_transformer_dim: int, cfg: TransformerConfig):
        super().__init__()

        dim = cfg.d_model
        self.emb = nn.Embedding(in_vocab_size, dim)
        self.linear_in = nn.Linear(main_transformer_dim, dim, bias=False)
        self.linear_out = nn.Linear(dim, out_vocab_size, bias=False)
        self.transformer = Transformer(cfg)

    def __call__(self, _: mx.array) -> mx.array:
        raise ValueError("not implemented")


class DepFormer(nn.Module):
    def __init__(self, cfg: LmConfig):
        super().__init__()

        self.slices: List[DepFormerSlice] = []
        for slice_idx in range(cfg.depformer.num_slices):
            in_vs = cfg.text_in_vocab_size if slice_idx == 0 else cfg.audio_vocab_size
            slice = DepFormerSlice(
                in_vs,
                cfg.audio_vocab_size - 1,
                main_transformer_dim=cfg.transformer.d_model,
                cfg=cfg.depformer.transformer,
            )
            self.slices.append(slice)

    def __call__(self, _: mx.array) -> mx.array:
        raise ValueError("not implemented")

    def sample(
        self,
        main_transformer_out: mx.array,
        sampler: sampling.Sampler,
        text_token: mx.array,
    ) -> mx.array:
        tokens = []
        last_token = text_token
        cache = None
        for slice in self.slices:
            xs = slice.linear_in(main_transformer_out) + slice.emb(last_token)
            xs, cache = slice.transformer(xs, cache=cache)
            logits = slice.linear_out(xs)
            last_token, _ = sampler(logits[0])
            tokens.append(last_token)
        tokens = mx.concatenate(tokens)
        return tokens


class Lm(nn.Module):
    def __init__(self, cfg: LmConfig):
        super().__init__()

        dim = cfg.transformer.d_model
        self.transformer = Transformer(cfg.transformer)
        self.depformer = DepFormer(cfg)
        self.text_emb = nn.Embedding(cfg.text_in_vocab_size, dim)
        self.cfg: LmConfig = cfg

        if cfg.transformer.norm == "layer_norm":
            self.out_norm = nn.LayerNorm(dim, 1e-5)
        elif cfg.transformer.norm == "rms_norm":
            self.out_norm = nn.RMSNorm(dim, 1e-8)
        else:
            raise ValueError(f"unsupported norm type {cfg.transformer.norm}")

        self.text_linear = nn.Linear(dim, cfg.text_out_vocab_size, bias=False)
        self.audio_embs = [nn.Embedding(cfg.audio_vocab_size, dim) for _ in range(cfg.audio_codebooks)]


    def __call__(
        self,
        token_ids: mx.array,
        cache: Optional[List[Tuple[mx.array, mx.array]]] = None,
    ) -> Tuple[mx.array, List[Tuple[mx.array, mx.array]]]:
        # Note that this does not apply the depformer.
        xs = self.text_emb(token_ids)
        transformer_out, upd_cache = self.transformer(xs, cache=cache)
        transformer_out = self.out_norm(transformer_out)
        text_logits = self.text_linear(transformer_out)
        return text_logits, upd_cache

    def sample(
        self,
        text_token_ids: mx.array,
        audio_token_ids: List[mx.array],
        text_sampler: sampling.Sampler,
        audio_sampler: sampling.Sampler,
        cache: Optional[List[Tuple[mx.array, mx.array]]] = None,
    ) -> Tuple[mx.array, mx.array, List[Tuple[mx.array, mx.array]]]:
        xs = self.text_emb(text_token_ids)
        for (token_ids, emb) in zip(audio_token_ids, self.audio_embs):
            xs = xs + emb(token_ids)
        transformer_out, upd_cache = self.transformer(xs, cache=cache)
        transformer_out = self.out_norm(transformer_out)
        text_logits = self.text_linear(transformer_out)
        text_token, _ = text_sampler(text_logits[:, 0])
        audio_tokens = self.depformer.sample(
            transformer_out,
            audio_sampler,
            text_token,
        )
        return text_token, audio_tokens, upd_cache


def config_v0_1() -> LmConfig:
    transformer = TransformerConfig(
        d_model=4096,
        num_heads=32,
        num_layers=32,
        dim_feedforward=4096 * 4, # dim * hidden_scale
        causal=True,
        norm_first=True,
        bias_ff=False,
        bias_attn=False,
        layer_scale=None,
        context=3000,
        max_period=10000,
        use_conv_block=False,
        use_conv_bias=True,
        cross_attention=False,
        gating=True,
        norm="rms_norm",
        positional_embedding="rope",
        conv_layout=False,
        conv_kernel_size=3,
        kv_repeat=1,
        max_seq_len=4096,
    )
    depformer = DepFormerConfig(
        transformer=TransformerConfig(
            d_model=1024,
            num_heads=16,
            num_layers=6,
            dim_feedforward=1024 * 4, # dim * hidden_scale
            causal=True,
            norm_first=True,
            bias_ff=False,
            bias_attn=False,
            layer_scale=None,
            context=8,
            max_period=10000,
            use_conv_block=False,
            use_conv_bias=True,
            cross_attention=False,
            gating=True,
            norm="rms_norm",
            positional_embedding="none",
            conv_layout=False,
            conv_kernel_size=3,
            kv_repeat=1,
            max_seq_len=4096,
        ),
        num_slices=8,
    )
    return LmConfig(
        transformer=transformer,
        depformer=depformer,
        audio_vocab_size=2049,
        text_in_vocab_size=32001,
        text_out_vocab_size=32000,
        audio_codebooks=16,
        audio_delays=([1] + [0] * 7) * 2,
    )

