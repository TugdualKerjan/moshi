# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import jax
import jax.numpy as jnp
import equinox as eqx
import equinox.nn as nn
import typing as tp 


class ActivationGating(eqx.Module):
    """
    Gating FFN layer, using the given activation.
    Args:
        dim (int): dimension of the input and output of the transformer.
        activation (any callable Tensor to Tensor): activation function to use.
: other kwargs passed to the linear layer, in particular device and dtype.
    """
    linear_in: nn.Linear
    linear_out: nn.Linear
    activation: tp.Callable

    def __init__(self, dim: int, dim_feedforward: int, activation:str = "elu", key: jax.Array = None): # type: ignore
        # We should have 8 d^2 param, instead we will have
        # 2 * h * d + h * d = 3 h * d = 8 d^2
        # so h = 8 d / 3 but following Hervé's advice we use 21 / 8 as an approx.
        key0, key1 = jax.random.split(key)
        if dim_feedforward == 4 * dim:
            hidden = (21 * dim) // 8
        else:
            hidden = (2 * dim_feedforward) // 3
        self.linear_in = nn.Linear(dim, 2 * hidden, use_bias=False, key=key0)
        self.linear_out = nn.Linear(hidden, dim, use_bias=False, key= key1)
        self.activation = self._get_activation(activation)

    # TODO this should be in a utils
    def _get_activation(self, name: str):
        if name in ["sigmoid", "tanh", "relu"]:
            return getattr(jax.nn, name)
        elif name in ["leaky_relu", "elu", "gelu", "silu", "mish", "softsign"]:
            return getattr(jax.nn, name)
        elif name == "identity":
            return eqx.nn.Identity()
        else:
            raise ValueError(f"Unknown activation {name}")

    @eqx.filter_jit
    def __call__(self, x: jax.Array):
        x = jax.vmap(self.linear_in)(x)
        T, _ = x.shape
        x = jnp.reshape(x, (T, 2, -1))
        x = self.activation(x[..., 0, :]) * x[..., 1, :]
        return jax.vmap(self.linear_out)(x)




# def _make_gating(
#     name: str, dim: int, dim_feedforward: int
# ) -> nn.Module:
#     return ActivationGating(
#         dim, dim_feedforward, _get_activation(name)
#     )


# def make_gating(
#     name: str, dim: int, dim_feedforward: int
# ) -> eqx.Module:
#     gating = ActivationGating(
#         dim, dim_feedforward, _get_activation(name)
#     )
#     # max_params = 2 * dim * dim_feedforward
#     # params = sum(p.numel() for p in gating.parameters())
#     # assert (
#     #     params <= max_params
#     # ), f"{name} gating has {params} params, max is {max_params}"
#     return gating
