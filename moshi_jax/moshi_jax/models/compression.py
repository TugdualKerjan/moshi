# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Part of this file is adapted from encodec.py in https://github.com/facebookresearch/audiocraft
# released under the following license.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Compression models or wrapper around existing models. In particular, provides the implementation
for Mimi. Also defines the main interface that a model must follow to be usable as an audio tokenizer.
"""

from abc import abstractmethod
from contextlib import nullcontext
from dataclasses import dataclass
import logging
import typing as tp

import equinox as eqx
import equinox.nn as nn
import jax
import jax.experimental
import jax.numpy as jnp

from ..quantization import (
    BaseQuantizer,
    SplitResidualVectorQuantizer,
    ResidualVectorQuantizer,
)
from ..modules.resample import ConvDownsample1d, ConvTrUpsample1d

logger = logging.getLogger()


class CompressionModel(eqx.Module):
    """Base API for all compression model that aim at being used as audio tokenizers
    with a language model.
    """

    @abstractmethod
    def __call__(self, x: jax.Array) -> tp.Tuple: ...

    @abstractmethod
    def encode(self, x: jax.Array) -> jax.Array:
        """See `MimiModel.encode`."""
        ...

    @abstractmethod
    def decode(self, codes: jax.Array) -> jax.Array:
        """See `MimiModel.decode`."""
        ...

    @abstractmethod
    def decode_latent(self, codes: jax.Array) -> jax.Array:
        """Decode from the discrete codes to continuous latent space."""
        ...

    @property
    @abstractmethod
    def channels(self) -> int: ...

    @property
    @abstractmethod
    def frame_rate(self) -> float: ...

    @property
    @abstractmethod
    def sample_rate(self) -> int: ...

    @property
    @abstractmethod
    def cardinality(self) -> int: ...

    @property
    @abstractmethod
    def num_codebooks(self) -> int: ...

    @property
    @abstractmethod
    def total_codebooks(self) -> int: ...

    @abstractmethod
    def set_num_codebooks(self, n: int):
        """Set the active number of codebooks used by the quantizer."""
        ...

class MimiModel(CompressionModel):
    """Mimi model operating on the raw waveform.

    Args:
        encoder (nn.Module): Encoder network.
        decoder (nn.Module): Decoder network.
        quantizer (qt.BaseQuantizer): Quantizer network.
        frame_rate (float): Final frame rate of the quantized representatiopn.
        encoder_frame_rate (float): frame rate of the encoder model. Note that if `frame_rate != encopder_frame_rate`,
            the latent will be resampled linearly to match the desired `frame_rate` before and after quantization.
        sample_rate (int): Audio sample rate.
        channels (int): Number of audio channels.
        causal (bool): Whether to use a causal version of the model.
        encoder_transformer (nn.Module or None): optional transformer for the encoder.
        decoder_transformer (nn.Module or None): optional transformer for the decoder.
        resample_method (str): method to use for resampling the latent space before the quantizer.
        upsample_channel_wise_bug (bool): controls whether the upsampling is channel wise.
            Defaults to true to reproduce bug in original implementation.
        freeze_encoder: whether to freeze the encoder weights.
        freeze_quantizer: whether to freeze the quantizer weights.
        freeze_quantizer_level: If positive, freeze the quantizer up to this level.
    """

    encoder: eqx.Module
    decoder: eqx.Module
    decoder_transformer: tp.Optional[eqx.Module]
    encoder_transformer: tp.Optional[eqx.Module]
    quantizer: BaseQuantizer
    _frame_rate: float
    _sample_rate: int
    _channels: int
    encoder_frame_rate: float
    downsample: ConvDownsample1d
    upsample: ConvTrUpsample1d
    dimension: int 
    resample_method: str
    
    def __init__(
        self,
        encoder: eqx.Module,
        decoder: eqx.Module,
        quantizer: BaseQuantizer,
        frame_rate: float,
        encoder_frame_rate: float,
        sample_rate: int,
        channels: int,
        causal: bool = False,
        encoder_transformer: tp.Optional[eqx.Module] = None,
        decoder_transformer: tp.Optional[eqx.Module] = None,
        resample_method: str = "interpolate",
        upsample_channel_wise_bug: bool = True,
        freeze_encoder: bool = False,
        freeze_quantizer: bool = False,
        freeze_quantizer_level: int = -1,
        key: jax.Array = None # type: ignore
    ):
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_transformer = encoder_transformer
        self.decoder_transformer = decoder_transformer
        self.quantizer = quantizer
        self._frame_rate = frame_rate
        self._sample_rate = sample_rate
        self._channels = channels
        self.encoder_frame_rate = encoder_frame_rate

        # if freeze_encoder:
        #     for p in self.encoder.parameters():
        #         p.requires_grad = False
        #     if self.encoder_transformer is not None:
        #         for p in self.encoder_transformer.parameters():
        #             p.requires_grad = False
        #     for name, p in self.quantizer.named_parameters():
        #         if name.endswith("input_proj.weight"):
        #             p.requires_grad = False
        # if freeze_quantizer:
        #     self.quantizer.ema_frozen_(True)
        # self.freeze_quantizer = freeze_quantizer
        # self.freeze_quantizer_level = (
        #     freeze_quantizer_level
        #     if freeze_quantizer_level > 0
        #     else self.quantizer.num_codebooks
        # )

        # We will need the dimension for the resampling. In general the encoder will be a SeanetEncoder
        # which exposes a `dimension` attribute.
        dimension = encoder.dimension # type: ignore
        assert isinstance(
            dimension, int
        ), f"Dimension should be int, got {dimension} of type {type(dimension)}."
        self.dimension = dimension

        assert resample_method in [
            "interpolate",
            "conv",
            "avg_pool",
        ], f"Invalid resample_method {resample_method}"
        self.resample_method = resample_method
        if encoder_frame_rate != frame_rate:
            assert not (
                causal and resample_method == "interpolate"
            ), "Cannot interpolate with causal model."
            if resample_method in ["conv", "avg_pool"]:
                assert (
                    self.encoder_frame_rate > self._frame_rate
                ), "Cannot upsample with conv."
                print(self.encoder_frame_rate)
                print(self._frame_rate)
                downsample_stride = self.encoder_frame_rate / self._frame_rate
                assert downsample_stride == int(
                    downsample_stride
                ), f"Only integer strides are supported, got {downsample_stride}"
                learnt = resample_method == "conv"
                key0, key1 = jax.random.split(key)
                self.downsample = ConvDownsample1d(
                    int(downsample_stride),
                    dimension=dimension,
                    learnt=learnt,
                    causal=causal,
                    key=key0
                )
                # if freeze_encoder:
                #     for p in self.downsample.parameters():
                #         p.requires_grad = False
                self.upsample = ConvTrUpsample1d(
                    int(downsample_stride),
                    dimension=dimension,
                    learnt=learnt,
                    causal=causal,
                    channel_wise=upsample_channel_wise_bug,
                    key=key1
                )

    @property
    def channels(self) -> int:
        return self._channels

    @property
    def frame_rate(self) -> float:
        return self._frame_rate

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def total_codebooks(self):
        """Total number of quantizer codebooks available."""
        return self.quantizer.total_codebooks

    @property
    def num_codebooks(self):
        """Active number of codebooks used by the quantizer."""
        return self.quantizer.num_codebooks

    def set_num_codebooks(self, n: int):
        """Set the active number of codebooks used by the quantizer."""
        self.quantizer.set_num_codebooks(n)

    @property
    def cardinality(self):
        """Cardinality of each codebook."""
        return self.quantizer.cardinality

    def _to_framerate(self, x: jax.Array):
        # Convert from the encoder frame rate to the overall framerate.
        _, length = x.shape
        frame_rate = self.encoder_frame_rate
        new_frame_rate = self.frame_rate
        if frame_rate == new_frame_rate:
            return x
        if self.resample_method == "interpolate":
            target_length = int(length * new_frame_rate / frame_rate)
            return jnp.expand_dims(jnp.interp(jnp.arange(target_length),jnp.arange(length), x[0, 0]), (0, 1))
        else:
            return self.downsample(x)

    def _to_encoder_framerate(self, x: jax.Array):
        # Convert from overall framerate to the encoder frame rate.
        _, length = x.shape
        frame_rate = self.encoder_frame_rate
        new_frame_rate = self.frame_rate
        if frame_rate == new_frame_rate:
            return x
        if self.resample_method == "interpolate":
            target_length = int(length * new_frame_rate / frame_rate)
            return jnp.expand_dims(jnp.interp(jnp.arange(target_length),jnp.arange(length), x[0, 0]), (0, 1))
        else:
            return self.upsample(x)

    @eqx.filter_jit
    def __call__(self, x: jax.Array) -> tp.Tuple:
        assert len(x.shape) == 2
        length = x.shape[-1]
        extra_metrics: tp.Dict[str, jax.Array] = {}

        # if self.freeze_quantizer:
        #     if isinstance(self.quantizer, SplitResidualVectorQuantizer):
        #         self.quantizer.rvq_first.eval()
        #         for i in range(
        #             self.freeze_quantizer_level - self.quantizer.n_q_semantic
        #         ):
        #             self.quantizer.rvq_rest.vq.layers[i].eval()
        #     elif isinstance(self.quantizer, ResidualVectorQuantizer):
        #         for i in range(self.freeze_quantizer_level):
        #             self.quantizer.vq.layers[i].eval()
        #     else:
        #         raise ValueError(f"Unsupported quantizer type {type(self.quantizer)}")

        emb = self.encoder(x) # type: ignore
        if self.encoder_transformer is not None:
            (emb,) = self.encoder_transformer(emb) # type: ignore
        emb = self._to_framerate(emb)
        expected_length = self.frame_rate * length / self.sample_rate
        # Checking that we have the proper length given the advertised frame rate.
        assert abs(emb.shape[-1] - expected_length) < 1, (
            emb.shape[-1],
            expected_length,
        )

        (emb, codes, bw, metrics) = self.quantizer(emb, self.frame_rate)
        emb = self._to_encoder_framerate(emb)
        if self.decoder_transformer is not None:
            (emb,) = self.decoder_transformer(emb) # type: ignore

        out = self.decoder(emb) # type: ignore

        # remove extra padding added by the encoder and decoder
        assert out.shape[-1] >= length, (out.shape[-1], length)
        out = out[..., :length]

        return (out, codes, bw, metrics.update(extra_metrics))

    def _encode_to_unquantized_latent(self, x: jax.Array) -> jax.Array:
        """Projects a batch of waveforms to unquantized latent space.

        Args:
            x (jax.Array): Float tensor of shape [B, C, T].

        Returns:
            Unquantized embeddings.
        """
        assert (
            len(x.shape) == 2
        ), f"CompressionModel._encode_to_unquantized_latent expects audio of shape [C, T] but got {x.shape}"
        emb = self.encoder(x) # type: ignore
        if self.encoder_transformer is not None:
            (emb,) = self.encoder_transformer(emb) # type: ignore
        emb = self._to_framerate(emb)
        return emb

    def encode(self, x: jax.Array) -> jax.Array:
        """Encode the given input tensor to quantized representation.

        Args:
            x (jax.Array): Float tensor of shape [B, C, T]

        Returns:
            codes (jax.Array): an int tensor of shape [B, K, T]
                with K the number of codebooks used and T the timestep.
        """
        emb = self._encode_to_unquantized_latent(x)
        codes = self.quantizer.encode(emb)
        return codes

    def encode_to_latent(self, x: jax.Array, quantize: bool = True) -> jax.Array:
        """Projects a batch of waveforms to latent space.

        Args:
            x (jax.Array): Float tensor of shape [B, C, T].

        Returns:
            Embeddings, either quantized or not.
        """
        emb = self._encode_to_unquantized_latent(x)
        if not quantize:
            return emb
        else:
            codes = self.quantizer.encode(emb)
            return self.decode_latent(codes)

    def decode(self, codes: jax.Array):
        """Decode the given codes to a reconstructed representation.

        Args:
            codes (jax.Array): Int tensor of shape [B, K, T]

        Returns:
            out (jax.Array): Float tensor of shape [B, C, T], the reconstructed audio.
        """
        emb = self.decode_latent(codes)
        emb = self._to_encoder_framerate(emb)
        if self.decoder_transformer is not None:
            (emb,) = self.decoder_transformer(emb) # type: ignore
        out = self.decoder(emb) # type: ignore
        # out contains extra padding added by the encoder and decoder
        return out

    def decode_latent(self, codes: jax.Array) -> jax.Array:
        """Decode from the discrete codes to continuous latent space."""
        return self.quantizer.decode(codes)


class WrapperCompressionModel(CompressionModel):
    """Base API for CompressionModel wrappers that do not depend on external frameworks."""

    def __init__(self, model: CompressionModel):
        super().__init__()
        self.model = model

    def __call__(self, x: jax.Array) -> tp.Tuple:
        return self.model.__call__(x)

    def encode(self, x: jax.Array) -> jax.Array:
        return self.model.encode(x)

    def decode(self, codes: jax.Array) -> jax.Array:
        return self.model.decode(codes)

    def decode_latent(self, codes: jax.Array) -> jax.Array:
        return self.model.decode_latent(codes)

    def set_num_codebooks(self, n: int):
        self.model.set_num_codebooks(n)

    @property
    def quantizer(self):
        return self.model.quantizer

    @property
    def channels(self) -> int:
        return self.model.channels

    @property
    def frame_rate(self) -> float:
        return self.model.frame_rate

    @property
    def sample_rate(self) -> int:
        return self.model.sample_rate

    @property
    def cardinality(self) -> int:
        return self.model.cardinality

    @property
    def num_codebooks(self) -> int:
        return self.model.num_codebooks

    @property
    def total_codebooks(self) -> int:
        return self.model.total_codebooks