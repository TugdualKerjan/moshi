# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp

from einops import rearrange
import jax
import equinox as eqx
import equinox.nn as nn
import jax.numpy as jnp

from .conv import StreamingConv1d, StreamingConvTranspose1d


class ConvDownsample1d(eqx.Module):
    """
    Downsampling by some integer amount `stride` using convolutions
    with a kernel size of twice the stride.
    If `causal` is True, the output uses a causal convolution.
    """

    learnt: bool
    channel_wise: bool
    conv: StreamingConv1d

    def __init__(
        self,
        stride: int,
        dimension: tp.Optional[int] = None,
        causal: bool = False,
        learnt: bool = False,
        channel_wise: bool = False,
        key:jax.Array = None,
    ):
        self.learnt = learnt
        self.channel_wise = channel_wise
        groups = 1
        if learnt:
            assert dimension is not None, "Dimension required for learnt convolutions."
            in_channels = dimension
            out_channels = dimension
            if channel_wise:
                groups = dimension
        else:
            in_channels = 1
            out_channels = 1

        conv = StreamingConv1d(
            in_channels,
            out_channels,
            kernel_size=2 * stride,
            stride=stride,
            causal=causal,
            groups=groups,
            bias=False,
            pad_mode="edge",
            key=key
        )
        if not learnt:
            where = lambda c: c.conv.conv.weight
            # TODO how to make this static ??
            conv = eqx.tree_at(where, conv, jnp.ones_like(conv.conv.conv.weight) / (2*stride))
        self.conv = conv

    @eqx.filter_jit
    def __call__(self, x: jax.Array):
        batch_size = len(x)
        if not self.learnt:
            x = rearrange(x, "b c t -> (b c) () t")
        y = self.conv(x)
        if not self.learnt:
            y = rearrange(y, "(b c) () t -> b c t", b=batch_size)
        return y


class ConvTrUpsample1d(eqx.Module):
    """
    Upsample by some integer amount `stride` using transposed convolutions.
    """
    learnt: bool
    channel_wise: bool
    convtr: StreamingConvTranspose1d

    def __init__(
        self,
        stride: int,
        dimension: tp.Optional[int] = None,
        causal: bool = False,
        learnt: bool = False,
        channel_wise: bool = False,
        key:jax.Array=None,
    ):
        self.learnt = learnt
        self.channel_wise = channel_wise
        groups = 1
        if learnt:
            assert dimension is not None, "Dimension required for learnt convolutions."
            in_channels = dimension
            out_channels = dimension
            if channel_wise:
                groups = dimension
        else:
            in_channels = 1
            out_channels = 1

        convtr = StreamingConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size=2 * stride,
            stride=stride,
            causal=causal,
            groups=groups,
            bias=False,
            key=key
        )
        if not learnt:
            where = lambda c: c.convtr.convtr.weight
            # TODO how to make this static ??
            convtr = eqx.tree_at(where, convtr, jnp.ones_like(convtr.convtr.convtr.weight))
        self.convtr = convtr

    # @eqx.filter_jit
    def __call__(self, x: jax.Array):
        batch_size = len(x)
        if not self.learnt:
            x = rearrange(x, "b c t -> (b c) () t")
        y = self.convtr(x)
        if not self.learnt:
            x_for_normalization = jnp.ones_like(x[:1])
            normalization = self.convtr(x_for_normalization)
            y = y / normalization
            y = rearrange(y, "(b c) () t -> b c t", b=batch_size)
        return y
