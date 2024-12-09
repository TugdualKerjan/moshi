
import equinox as eqx
import equinox.nn as nn
import jax
import jax.numpy as jnp
import typing as tp

from .gating import ActivationGating
from .rope import RotaryEmbedding


class LayerScale(eqx.Module):
    """Layer scale from [Touvron et al 2021] (https://arxiv.org/pdf/2103.17239.pdf).
    This rescales diagonally the residual outputs close to 0, with a learnt scale.

    Args:
        channels (int): Number of channels.
        init (float): Initial scale.
        channel_last (bool): If True, expect `[*, C]` shaped tensors, otherwise, `[*, C, T]`.
        device (torch.device or str, optional): Device on which to initialize the module.
        dtype (torch.dtype, optional): dtype to use to initialize the module.
    """

    scale: jax.Array
    channel_last:bool
    
    def __init__(
        self,
        channels: int,
        init: float = 1e-4,
        channel_last: bool = True,
    ):
        super().__init__()
        self.channel_last = channel_last
        self.scale = jnp.full(
                (channels,), init, 
            )

    @eqx.filter_jit
    def __call__(self, x: jax.Array):
        if self.channel_last:
            return self.scale * x
        else:
            return self.scale[:, None] * x

class LayerNormF32(eqx.Module):
    def __init__(self, *args, **kwargs):
        self.norm = eqx.nn.LayerNorm(*args, **kwargs)

    def __call__(
        self,
        x: jax.Array
    ):
        x_f32 = jnp.float32(x)
        out_f32 = self.norm(x_f32)
        return out_f32.astype(x.dtype)

def create_norm_fn(norm_type: str, dim: int, **kwargs) -> eqx.Module:
    """Create normalization module for transformer encoder layer.

    Args:
        norm_type (str): Normalization method.
        dim (int): Dimension of the normalized layer.
        **kwargs (dict): Additional parameters for normalization layer.
    Returns:
        eqx.Module: Normalization module.
    """
    if norm_type == "layer_norm":
        return nn.LayerNorm(dim, eps=1e-5, **kwargs)
    elif norm_type == "layer_norm_f32":
        kwargs.pop("dtype", None)
        return LayerNormF32(dim, eps=1e-8, **kwargs)
    elif norm_type in {"rms_norm"}:
        return nn.RMSNorm(dim, eps=1e-5, **kwargs)
    elif norm_type in {"rms_norm_f32"}:
        kwargs.pop("dtype", None)
        return nn.RMSNorm(dim, eps=1e-8, dtype=jnp.float32, **kwargs)
    else:
        raise ValueError(f"Unknown norm type: {norm_type}")

def create_sin_embedding(
    positions: jax.Array,
    dim: int,
    max_period: float = 10000,
) -> jax.Array:
    """Create sinusoidal positional embedding, with shape `[T, C]`.

    Args:
        positions (torch.Tensor): jax.Array of positions.
        dim (int): Dimension of the embedding.
        max_period (float): Maximum period of the cosine/sine functions.
    Returns:
        jax.Array: Sinusoidal positional embedding.
    """
    # We aim for BTC format
    assert dim % 2 == 0
    half_dim = dim // 2
    adim = jnp.expand_dims(jnp.arange(half_dim), 0) #(1, 1000)
    max_period_tensor = jnp.full(        [], max_period    )  # avoid sync point
    phase = positions / (max_period_tensor ** (adim / (half_dim - 1)))
    return jnp.concat([jnp.cos(phase), jnp.sin(phase)], axis=-1)

# class MultiLinear(eqx.Module):
#     lin: jax.Array
    
#     def __init__(self, num_linear, in_dim, out_dim, key: jax.Array = None):
#         self.lin = jax.random.normal(num_linear, in_dim, out_dim, key)

#     def __call__(self, x):
#         out = jax.vmap(self.lin)(x)
def multi_linear(*args, **kwargs):
    pass

class KVCacheResult(tp.NamedTuple):
    keys: jax.Array
    values: jax.Array
    positions: jax.Array

    @staticmethod
    def from_kv(keys: jax.Array, values: jax.Array) -> "KVCacheResult":
        H, T, D = keys.shape
        assert tuple(values.shape[:-1]) == (H, T)
        positions = jnp.arange(T)
        return KVCacheResult(keys, values, positions)

class StreamingMultiheadAttention(eqx.Module):
    """Similar to `nn.MultiheadAttention` but with support for streaming, causal evaluation.

    Args:
        embed_dim (int): Dimension to project to.
        num_heads (int): Number of heads.
        causal (bool): Causal mask applied automatically.
        context (int, optional): Number of time steps the attention can access to.
            When causal, can access `context` time steps into the past, and when non causal,
            can access `context // 2` steps in the past, and the same in the future.
        rope (`RotaryEmbedding`, optional): Rope embedding to use.
        weights_per_step (int): use different weights per time step. If non zero, should correspond to the
            number of possible time steps.
        device (torch.device, optional): Device on which to initialize.
        dtype (torch.dtype, optional): dtype to use.
    """

    embed_dim : int
    causal : bool
    context : tp.Optional[int]
    rope : tp.Optional[RotaryEmbedding]
    num_heads : int

    weights_per_step: int

    in_proj: nn.Linear
    out_proj: nn.Linear
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        causal: bool = False,
        context: tp.Optional[int] = None,
        rope: tp.Optional[RotaryEmbedding] = None,
        weights_per_step: int = 0,
        key: jax.Array = None # type: ignore
    ):
        self.embed_dim = embed_dim
        self.causal = causal
        self.context = context
        self.rope = rope
        self.num_heads = num_heads

        out_dim = embed_dim
        out_dim = 3 * embed_dim
        mult = 1
        self.weights_per_step = weights_per_step
        if weights_per_step:
            mult = weights_per_step
        key0, key1 = jax.random.split(key)
            
        self.in_proj = nn.Linear(embed_dim, mult * out_dim, use_bias=False, key=key0)
        # We try to follow the default PyTorch MHA convention, to easily compare results.
        # self.in_proj_weight = in_proj.weight
        # self.in_proj_bias = in_proj.bias
        self.out_proj = nn.Linear(
            embed_dim, mult * embed_dim, use_bias=False, key=key1
        )

    def _complete_kv(self, k, v) -> KVCacheResult:
        return KVCacheResult.from_kv(k, v)

    @eqx.filter_jit
    def __call__(self, x: jax.Array):
        T = x.shape[0]
        if self.weights_per_step:
            projected = multi_linear(
                self.weights_per_step, self.in_proj, x
            )
        else:
            print(x.shape)
            print(self.in_proj.weight.shape)
            projected = jax.vmap(self.in_proj)(x)
        q, k, v = jnp.split(projected, 3, axis=-1) # type: ignore

        if self.rope:
            q, k = self.rope(q, k, 0) # type: ignore

        kq = jnp.matmul(k, jnp.transpose(q))
        # Dim of (seq, seq), a matrix showing which tokens are interested in each other.
        # Mask to make causal
        # mask = jax.lax.stop_gradient(self.mask)
        mask = jnp.tril(jnp.ones((T, T)))
        kq = jnp.where(
            jnp.equal(jax.lax.stop_gradient(mask), 0), -jnp.inf, kq
        )  # Trick to lower compute
        kq = jax.nn.softmax(kq, axis=-1)
        # Add att dropout
        x = jnp.matmul(kq, v)
        if self.weights_per_step:
            x = multi_linear(self.weights_per_step, self.out_proj.weight, x) # type: ignore
        else:
            x = jax.vmap(self.out_proj)(x)
        return x

class StreamingTransformerLayer(eqx.Module):
    """TransformerLayer with Streaming / Causal support.

    Args:
        d_model (int): Dimension of the data.
        num_heads (int): Number of heads.
        dim_feedforward (int): Intermediate dimension of FF module.
        causal (bool): Causal mask applied automatically.
        context (int, optional): Receptive field for the causal mask, infinite if None.
        custom (bool): Use custom MHA implementation, for testing / benchmarking.
        rope (`RotaryEmbedding`, optional): Rope embedding to use.
        norm (str): Normalization to use. Currently, only 'layer_norm' is supported.
        layer_scale (float, optional): If not None, LayerScale will be used with the given value as initial scale.
        gating (str): if provided, replaces FFN with special gating, like GLU, GSiGLU etc.
        weights_per_step (int): use different weights per time step. If non zero, should correspond to the
            number of possible time steps.
        skip_self_attn: If true, skips the self attention module and the norm
        device (torch.device, optional): Device on which to initialize.
        dtype (torch.dtype, optional): dtype to use.
    """

    _fsdp_final = True

    gating: list
    linear1: tp.Optional[eqx.Module]
    linear2: tp.Optional[eqx.Module]
    self_attn: StreamingMultiheadAttention
    norm1: tp.Callable
    norm2: tp.Callable
    skip_self_attn:bool
    activation: tp.Callable
    layer_scale_1: eqx.Module
    layer_scale_2: eqx.Module
    weights_per_step: int
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int | list[int] = 2048,
        causal: bool = False,
        context: tp.Optional[int] = None,
        rope: tp.Optional[RotaryEmbedding] = None,
        norm: str = "layer_norm",
        layer_scale: tp.Optional[float] = None,
        gating: str = "none",
        weights_per_step: int = 0,
        activation=jax.nn.gelu,
        skip_self_attn: bool = False,
        key:jax.Array= None # type: ignore
    ):
        key0, key1, key2, key3 = jax.random.split(key, 4)

        # Redefine self_attn to our streaming multi-head attention
        attn_kwargs: tp.Dict[str, tp.Any] = {
            "embed_dim": d_model,
            "num_heads": num_heads,
        }
        
        self.self_attn: StreamingMultiheadAttention = StreamingMultiheadAttention(
            causal=causal,
            context=context,
            rope=rope,
            weights_per_step=weights_per_step,
            key=key3,
            **attn_kwargs,
        )
        self.norm1 = create_norm_fn(norm, d_model) # type: ignore
        self.norm2 = create_norm_fn(norm, d_model) # type: ignore

        # Redefine feedforward layers to expose bias parameter
        self.weights_per_step = weights_per_step
        self.activation = activation
        self.skip_self_attn = skip_self_attn

        if isinstance(dim_feedforward, list):
            assert dim_feedforward
            assert len(dim_feedforward) == weights_per_step, (
                "Length of dim_feedforward must match weights_per_step,"
                f" got {len(dim_feedforward)} != {weights_per_step}"
            )

        
        self.gating = None # type: ignore
        
        if gating == "none":
                
            assert (
                not weights_per_step
            ), "weights_per_step without gating not supported for now."
            assert not isinstance(
                dim_feedforward, list
            ), "List dim_feedforward without gating not supported for now."
            self.linear1 = nn.Linear(
                d_model, dim_feedforward, use_bias=False, key=key0
            )
            self.linear2 = nn.Linear(
                dim_feedforward, d_model, use_bias=False, key=key1
            )
        else:
            self.linear1 = None
            self.linear2 = None
            if weights_per_step:
                if isinstance(dim_feedforward, int):
                    dim_feedforward = [dim_feedforward] * weights_per_step
                assert isinstance(dim_feedforward, list), dim_feedforward
                self.gating = [
                        ActivationGating(dim, d_model, gating,key=k)
                        for dim,k in zip(dim_feedforward, jax.random.split(key2, len(dim_feedforward)))
                    ]
                
            else:
                assert isinstance(dim_feedforward, int)
                self.gating = [ActivationGating(dim_feedforward, d_model, gating,key=key2)]
    
    
        if layer_scale is None:
            self.layer_scale_1 = nn.Identity()
            self.layer_scale_2 = nn.Identity()
        else:
            self.layer_scale_1 = LayerScale(d_model, layer_scale, )  # type: ignore
            self.layer_scale_2 = LayerScale(d_model, layer_scale, )  # type: ignore

    @eqx.filter_jit
    def _ff_block(self, x: jax.Array) -> jax.Array:
        x_orig = x
        x = jax.vmap(self.norm2)(x)
        if self.gating is None:
            assert self.linear1 is not None
            assert self.linear2 is not None
            y = jax.vmap(self.linear1)(x)
            y = self.activation(y)
            update = jax.vmap(self.linear2)(y) # type: ignore
        else:
            if self.weights_per_step:
                assert isinstance(self.gating, tp.List)
                T, D = x.shape
                ys = []
                for t in range(T):
                    y = self.gating[t](x[:, t : t + 1])
                    ys.append(y)
                update = jnp.concat(ys, axis=1)
            else:
                update = self.gating[0](x) # type: ignore
        return x_orig + self.layer_scale_2(update) # type: ignore

    @eqx.filter_jit
    def _sa_block(self, x: jax.Array):
        if self.skip_self_attn:
            return x
        x_orig = x
        x = jax.vmap(self.norm1)(x)
        update = self.self_attn(x)
        return x_orig + self.layer_scale_1(update) # type: ignore

    # @eqx.filter_jit
    def __call__(self, x: jax.Array):
        print(f"Ours: {x[0, :3]}")
        x = self._sa_block(x)
        print(f"Ours 2: {x[0, :3]}")

        x = self._ff_block(x)
        return x


class StreamingTransformer(eqx.Module):
    """Transformer with Streaming / Causal support.

    Args:
        d_model (int): Dimension of the data.
        num_heads (int): Number of heads.
        dim_feedforward (int): Intermediate dimension of FF module.
        causal (bool): Causal mask applied automatically.
        context (int, optional): Receptive field for the causal mask, infinite if None.
        layer_scale (float, optional): If not None, LayerScale will be used
            with the given value as initial scale.
        positional_embedding (str): Positional embedding strategy (sin, rope, sin_rope, or none).
        max_period (float): Maximum period of the time embedding.
        positional_scale (float): Scale of positional embedding, set to 0 to deactivate.
        layer_class: (subclass of `StreamingTransformerLayer): class to use
            to initialize the layers, allowing further customization outside of AudioCraft.
        device (torch.device, optional): Device on which to initialize.
        dtype (torch.dtype, optional): dtype to use.
        **kwargs: See `StreamingTransformerLayer`.
    """


    positional_embedding: str
    max_period: float
    positional_scale: float
    betas: tp.Optional[tp.Tuple[float, float]]
    rope: tp.Optional[RotaryEmbedding]
    layers: list

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int,
        dim_feedforward: int | list[int] = 2048,
        causal: bool = False,
        context: tp.Optional[int] = None,
        positional_embedding: str = "sin",
        max_period: float = 10_000,
        positional_scale: float = 1.0,
        betas: tp.Optional[tp.Tuple[float, float]] = None,
        layer_class: tp.Type[StreamingTransformerLayer] = StreamingTransformerLayer,
        key:jax.Array=None, # type: ignore
        **kwargs,
    ):
        key0, key1 = jax.random.split(key, 2)
    
        assert d_model % num_heads == 0

        self.positional_embedding = positional_embedding
        self.max_period = max_period
        self.positional_scale = positional_scale
        self.betas = betas

        assert positional_embedding in {"sin", "rope", "sin_rope", "none"}
        self.rope: tp.Optional[RotaryEmbedding] = None
        if self.positional_embedding in {"rope", "sin_rope"}:
            self.rope = RotaryEmbedding(max_period=max_period)

        self.layers = [layer_class(
                    d_model=d_model,
                    num_heads=num_heads,
                    dim_feedforward=dim_feedforward,
                    causal=causal,
                    context=context,
                    rope=self.rope,
                    key=k,
                    **kwargs,
                )for k in jax.random.split(key1, num_layers)]

    # @eqx.filter_jit
    def __call__(self, x: jax.Array, *args, **kwargs):
        T, C = x.shape
        
        if self.positional_embedding in {"sin", "sin_rope"}:
            positions = jnp.expand_dims(jnp.arange(T), -1)
            positions = positions
            pos_emb = create_sin_embedding(
                positions, C, max_period=self.max_period
            )
            x = x + self.positional_scale * pos_emb

        for layer in self.layers:
            x = layer(x, *args, **kwargs)

        return x

class ProjectedTransformer(eqx.Module):
    """Transformer with optional projections of the input and output to different dimensions when needed.
    Supports multiple outputs.

    Args:
        input_dimension (int): dimension of the input.
        output_dimensions (tuple[int]): dimensions of the outputs.
        d_model (int): inner dimension of the Transformer.
        conv_layout (bool): If True, expects `[B, C, T]` shaped tensors, otherwise, `[B, T, C]`.
            Similarly, the output will have the same layout.
    """

    transformer: StreamingTransformer
    input_dimension: int
    output_dimensions: tp.Tuple[int, ...]
    output_projs: list
    input_proj: eqx.Module
    conv_layout: bool

    def __init__(
        self,
        input_dimension: int,
        output_dimensions: tp.Tuple[int, ...],
        d_model: int,
        *,
        conv_layout: bool = False,
        key:jax.Array= None, # type: ignore
        **kwargs,
    ):
        key0, key1 = jax.random.split(key)
    
        self.transformer = StreamingTransformer(d_model=d_model, **kwargs, key=key0)
        self.input_dimension = input_dimension
        self.output_dimensions = output_dimensions
        self.conv_layout = conv_layout
        if d_model != input_dimension:
            self.input_proj = nn.Linear(input_dimension, d_model, use_bias=False, key=key1)
        else: 
            self.input_proj = nn.Identity()

        self.output_projs = [nn.Identity() if d_model == output_dimension else nn.Linear(d_model, output_dimension, use_bias=False, key=k) for output_dimension, k in zip(output_dimensions, jax.random.split(key1, len(output_dimensions)))]

    # @eqx.filter_jit
    def __call__(self, x, *args, **kwargs):
        if self.conv_layout:
            x = jnp.transpose(x, (1, 0))
        print(x.shape)
        x = jax.vmap(self.input_proj)(x) # type: ignore
        z = self.transformer(x, *args, **kwargs)
        ys = []
        for output_proj in self.output_projs:
            y = jax.vmap(output_proj)(z)
            if self.conv_layout:
                y = jnp.transpose(y, (1, 0))
            ys.append(y)
        return ys
