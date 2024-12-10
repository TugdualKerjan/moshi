from operator import xor
import warnings
import jax 
import jax.numpy as jnp
import equinox as eqx
import equinox.nn as nn 
import typing as tp 
import math

from .streaming import RawStreamingConv1d, RawStreamingConvTranspose1d

CONV_NORMALIZATIONS = frozenset(["none", "weight_norm"])

def apply_parametrization_norm(module: eqx.Module, norm: str = "none"):
    assert norm in CONV_NORMALIZATIONS
    if norm == "weight_norm":
        return nn.WeightNorm(module)
    else:
        # We already check was in CONV_NORMALIZATION, so any other choice
        # doesn't need reparametrization.
        return module
    


def get_extra_padding_for_conv1d(
    x: jax.Array, kernel_size: int, stride: int, padding_total: int = 0
) -> int:
    """See `pad_for_conv1d`."""
    length = x.shape[-1]
    n_frames = (length - kernel_size + padding_total) / stride + 1
    ideal_length = (math.ceil(n_frames) - 1) * stride + (kernel_size - padding_total)
    return ideal_length - length


def pad_for_conv1d(
    x: jax.Array, kernel_size: int, stride: int, padding_total: int = 0
):
    """Pad for a convolution to make sure that the last window is full.
    Extra padding is added at the end. This is required to ensure that we can rebuild
    an output of the same length, as otherwise, even with padding, some time steps
    might get removed.
    For instance, with total padding = 4, kernel size = 4, stride = 2:
        0 0 1 2 3 4 5 0 0   # (0s are padding)
        1   2   3           # (output frames of a convolution, last 0 is never used)
        0 0 1 2 3 4 5 0     # (output of tr. conv., but pos. 5 is going to get removed as padding)
            1 2 3 4         # once you removed padding, we are missing one time step !
    """
    extra_padding = get_extra_padding_for_conv1d(x, kernel_size, stride, padding_total)
    return jnp.pad(x, ((0,0),(0, extra_padding)))


def pad1d(
    x: jax.Array,
    paddings: tp.Tuple[int, int],
    mode: str = "constant",
    value: float = 0.0,
):
    """Tiny wrapper around F.pad, just to allow for reflect padding on small input.
    If this is the case, we insert extra 0 padding to the right before the reflection happen.
    """
    length = x.shape[-1]
    padding_left, padding_right = paddings
    assert padding_left >= 0 and padding_right >= 0, (padding_left, padding_right)
    if mode == "reflect":
        max_pad = max(padding_left, padding_right)
        extra_pad = 0
        if length <= max_pad:
            extra_pad = max_pad - length + 1
            x = jnp.pad(x, ((0,0),(0, extra_pad)))
        padded = jnp.pad(x, ( (0,0), paddings), mode, constant_values=value)
        end = padded.shape[-1] - extra_pad
        return padded[..., :end]
    else:
        return jnp.pad(x,( (0,0), paddings), mode, constant_values=value)


def unpad1d(x: jax.Array, paddings: tp.Tuple[int, int]):
    """Remove padding from x, handling properly zero padding. Only for 1d!"""
    padding_left, padding_right = paddings
    assert padding_left >= 0 and padding_right >= 0, (padding_left, padding_right)
    assert (padding_left + padding_right) <= x.shape[-1]
    end = x.shape[-1] - padding_right
    return x[..., padding_left:end]

class NormConv1d(eqx.Module):
    """Wrapper around Conv1d and normalization applied to this conv
    to provide a uniform interface across normalization approaches.
    """

    conv: nn.Conv1d
    norm_type: str

    def __init__(
        self,
        *args,
        causal: bool = False,
        norm: str = "none",
        norm_kwargs: tp.Dict[str, tp.Any] = {},
        **kwargs,
    ):
        super().__init__()
        self.conv = apply_parametrization_norm(
            nn.Conv1d(*args, **kwargs), norm
        ) # type: ignore
        self.norm_type = norm

    def __call__(self, x):
        return self.conv(x)
    
class NormConvTranspose1d(eqx.Module):
    """Wrapper around ConvTranspose1d and normalization applied to this conv
    to provide a uniform interface across normalization approaches.
    """
    convtr: nn.ConvTranspose1d
    norm_type: str
    
    def __init__(
        self,
        *args,
        causal: bool = False,
        norm: str = "none",
        norm_kwargs: tp.Dict[str, tp.Any] = {},
        **kwargs,
    ):
        super().__init__()
        self.convtr = apply_parametrization_norm(
            nn.ConvTranspose1d(*args, **kwargs), norm
        ) # type: ignore
        self.norm_type = norm

    def __call__(self, x):
        return self.convtr(x)



class StreamingConv1d(eqx.Module):
    """Conv1d with some builtin handling of asymmetric or causal padding
    and normalization.
    """
    
    conv: NormConv1d
    causal: bool
    pad_mode: str
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        causal: bool = False,
        norm: str = "none",
        norm_kwargs: tp.Dict[str, tp.Any] = {},
        pad_mode: str = "reflect",
        key: jax.Array=None # type: ignore
    ):
        super().__init__()
        # warn user on unusual setup between dilation and stride
        if stride > 1 and dilation > 1:
            warnings.warn(
                "StreamingConv1d has been initialized with stride > 1 and dilation > 1"
                f" (kernel_size={kernel_size} stride={stride}, dilation={dilation})."
            )
        self.conv = NormConv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            dilation=dilation,
            groups=groups,
            use_bias=bias,
            causal=causal,
            norm=norm,
            norm_kwargs=norm_kwargs,
            key=key
        )
        self.causal = causal
        self.pad_mode = pad_mode

    @property
    def _stride(self) -> int:
        return self.conv.conv.stride[0]

    @property
    def _kernel_size(self) -> int:
        return self.conv.conv.kernel_size[0]

    @property
    def _effective_kernel_size(self) -> int:
        dilation = self.conv.conv.dilation[0]
        return (
            self._kernel_size - 1
        ) * dilation + 1  # effective kernel size with dilations

    @property
    def _padding_total(self) -> int:
        return self._effective_kernel_size - self._stride

    # def _init_streaming_state(self, batch_size: int) -> _StreamingConv1dState:
    #     assert self.causal, "streaming is only supported for causal convs"
    #     return _StreamingConv1dState(self._padding_total, self._padding_total)

    # @eqx.filter_jit
    def __call__(self, x):
        padding_total = self._padding_total
        extra_padding = get_extra_padding_for_conv1d(
            x, self._effective_kernel_size, self._stride, padding_total
        )
        # state = self._streaming_state
        # if state is None:
        if self.causal:
            # Left padding for causal
            x = pad1d(x, (padding_total, extra_padding), mode=self.pad_mode)
        else:
            # Asymmetric padding required for odd strides
            padding_right = padding_total // 2
            padding_left = padding_total - padding_right
            x = pad1d(
                x, (padding_left, padding_right + extra_padding), mode=self.pad_mode
            )
        # else:
        #     if state.padding_to_add > 0 and x.shape[-1] > 0:
        #         x = pad1d(x, (state.padding_to_add, 0), mode=self.pad_mode)
        #         state.padding_to_add = 0
        x= self.conv(x)
        return x


class StreamingConvTranspose1d(eqx.Module):
    """ConvTranspose1d with some builtin handling of asymmetric or causal padding
    and normalization.
    """
    convtr: NormConvTranspose1d
    causal: bool
    trim_right_ratio: float

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size:int,
        stride:int =1,
        groups:int =1,
        bias: bool = True,
        causal: bool = False,
        norm: str = "none",
        trim_right_ratio: float = 1.0,
        norm_kwargs: tp.Dict[str, tp.Any] = {},
        key: jax.Array = None # type: ignore
    ):
        self.convtr = NormConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            groups=groups,
            use_bias=bias,
            causal=causal,
            norm=norm,
            norm_kwargs=norm_kwargs,
            key=key
        )

        self.causal = causal
        self.trim_right_ratio = trim_right_ratio
        assert (
            self.causal or self.trim_right_ratio == 1.0
        ), "`trim_right_ratio` != 1.0 only makes sense for causal convolutions"
        assert self.trim_right_ratio >= 0.0 and self.trim_right_ratio <= 1.0
    
    # @eqx.filter_jit
    def __call__(self, x):
        kernel_size = self.convtr.convtr.kernel_size[0]
        stride = self.convtr.convtr.stride[0]
        padding_total = kernel_size - stride

        y = self.convtr(x)
        print(f"Ours : {y[0, :10]}")

        # if not self.is_streaming:
            # We will only trim fixed padding. Extra padding from `pad_for_conv1d` would be
            # removed at the very end, when keeping only the right length for the output,
            # as removing it here would require also passing the length at the matching layer
            # in the encoder.
        if self.causal:
            # Trim the padding on the right according to the specified ratio
            # if trim_right_ratio = 1.0, trim everything from right
            padding_right = math.ceil(padding_total * self.trim_right_ratio)
            padding_left = padding_total - padding_right
            y = unpad1d(y, (padding_left, padding_right))
        else:
            # Asymmetric padding required for odd strides
            padding_right = padding_total // 2
            padding_left = padding_total - padding_right
            y = unpad1d(y, (padding_left, padding_right))
        return y