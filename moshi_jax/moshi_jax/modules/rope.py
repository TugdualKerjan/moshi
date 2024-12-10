# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import jax
import jax.numpy as jnp
import math
import equinox as eqx


class RotaryEmbedding(eqx.Module):
    """Rotary positional embedding (RoPE) from [Su et al 2022](https://arxiv.org/abs/2104.09864).

    Args:
        max_period (float): Maximum period of the rotation frequencies.
    """
    max_period:float

    def __init__(self, max_period: float = 10000.0):
        self.max_period = max_period


    def apply_rope(
        self,
        q: jax.Array,
        k: jax.Array,
        offset: jax.Array,
        max_period: float = 10_000,
        # time_before_heads: bool = False,
    ):
        """
        Args:
            q (jax.Array): queries, shape `[T, H, D]`.
            k (jax.Array): keys, shape `[T, H, D]`.
            offset (int): current offset, e.g. when streaming.
            max_period (float): maximum period for the cos and sin.
            time_before_heads (bool):  if True, expected [T, H, D], else [H, T ,D]
        """
        print(f"Shape for rotary embeddings: {q.shape}")
        T, D = q.shape
        assert k.shape == q.shape
        assert D > 0
        assert D % 2 == 0
        assert max_period > 0

        ds = jnp.arange(D // 2, dtype="float32")
        freqs = jnp.exp(ds * (-math.log(max_period) * 2 / D))
        # if time_before_heads:
        #     ts = jnp.expand_dims(jnp.arange(T), (1, 2))
        # else:
        ts = jnp.expand_dims(jnp.arange(T), (0, 2))
        print(f"Shape of ts: {ts.shape}")



        q = jnp.reshape(q, (*(q.shape[:-1]), D//2, 2))
        k = jnp.reshape(k, (*(k.shape[:-1]), D//2, 2))

        # convention is `r` suffix is real part, `i` is imaginary.
        qr, qi = jnp.unstack(q, axis=-1)
        kr, ki = jnp.unstack(k, axis=-1)

        rotr = jnp.cos(freqs * ts)
        roti = jnp.sin(freqs * ts)
        qor = qr * rotr - qi * roti
        qoi = qr * roti + qi * rotr

        kor = kr * rotr - ki * roti
        koi = kr * roti + ki * rotr

        qo = jnp.stack([qor, qoi], axis=-1)
        ko = jnp.stack([kor, koi], axis=-1)

        return jnp.reshape(qo, (-1, D)), jnp.reshape(ko, (-1, D))

    @eqx.filter_jit
    def __call__(
        self,
        q: jax.Array,
        k: jax.Array,
        offset: jax.Array,
    ):
        """Apply rope rotation to query or key tensor."""
        return self.apply_rope(q, k, offset, self.max_period)
