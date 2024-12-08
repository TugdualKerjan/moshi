import abc
from contextlib import contextmanager
from dataclasses import dataclass
import itertools
import typing as tp
import math
import jax
import equinox as eqx
import equinox.nn as nn

# class Resetable(tp.Protocol):
#     def reset(self) -> None:
#         pass
    
# State = tp.TypeVar("State", bound=Resetable)

# @dataclass
# class _StreamingConvTrState:
#     partial: torch.Tensor | None = None

#     def reset(self):
#         self.partial = None

class RawStreamingConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.padding[0] == 0, "Padding should be handled outside."
        assert (
            self.stride[0] <= self.kernel_size[0]
        ), "stride must be less than kernel_size."

    # def _init_streaming_state(self, batch_size: int) -> _StreamingConvState:
    #     return _StreamingConvState()

    @eqx.filter_jit
    def __call__(self, x: jax.Array) -> jax.Array:
        # stride = self.stride[0]
        # Effective kernel size accounting for dilation.
        # kernel = (self.kernel_size[0] - 1) * self.dilation[0] + 1
        # if self._streaming_state is None:
        return super().__call__(x)
        # else:
        #     # Due to the potential overlap, we might have some cache of the previous time steps.
        #     previous = self._streaming_state.previous
        #     if previous is not None:
        #         input = torch.cat([previous, input], dim=-1)
        #     B, C, T = input.shape
        #     # We now compute the number of full convolution frames, i.e. the frames
        #     # that are ready to be computed.
        #     num_frames = max(0, int(math.floor((T - kernel) / stride) + 1))
        #     offset = num_frames * stride
        #     # We will compute `num_frames` outputs, and we are advancing by `stride`
        #     # for each of the frame, so we know the data before `stride * num_frames`
        #     # will never be used again.
        #     self._streaming_state.previous = input[..., offset:]
        #     if num_frames > 0:
        #         input_length = (num_frames - 1) * stride + kernel
        #         out = super().forward(input[..., :input_length])
        #     else:
        #         # Not enough data as this point to output some new frames.
        #         out = torch.empty(
        #             B, self.out_channels, 0, device=input.device, dtype=input.dtype
        #         )
        #     return out
        
class RawStreamingConvTranspose1d(
    nn.ConvTranspose1d
):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.padding[0] == 0, "Padding should be handled outside."
        assert self.dilation[0] == 1, "No dilation for now"
        assert (
            self.stride[0] <= self.kernel_size[0]
        ), "stride must be less than kernel_size."
        assert self.output_padding[0] == 0, "Output padding not supported."

    # def _init_streaming_state(self, batch_size: int) -> _StreamingConvTrState:
    #     return _StreamingConvTrState()

    def __call__(self, x: jax.Array) -> jax.Array:  # type: ignore
        # B, C, T = x.shape
        # stride = self.stride[0]
        # kernel = self.kernel_size[0]
        # if self._streaming_state is None:
        return super().__call__(x)
        # else:
        #     if T == 0:
        #         return torch.empty(
        #             B, self.out_channels, 0, device=x.device, dtype=x.dtype
        #         )
        #     out = super().forward(x)
        #     OT = out.shape[-1]
        #     partial = self._streaming_state.partial
        #     if partial is not None:
        #         # Due to the potential overlap, the rightmost output of the conv transpose is not
        #         # ready to be output, as it will receive contributions from the next input frames.
        #         # Here we recover those `partial` output frames. We know that the first time step
        #         # of the `partial` tensor corresponds to the first time step of `out` as anything
        #         # coming before the first time step of `out` would have been already flushed.
        #         PT = partial.shape[-1]
        #         if self.bias is not None:
        #             out[..., :PT] += partial - self.bias[:, None]
        #         else:
        #             out[..., :PT] += partial
        #     # The input is T, the output is S * (T - 1) + K.
        #     # The offset of the left of the next frame will be S * T
        #     # so everything between 0 and S * T is ready to be output, and we need
        #     # to keep in the internal state everything beyond that, i.e. S (T - 1) + K - S T = K - S
        #     invalid_steps = kernel - stride
        #     partial = out[..., OT - invalid_steps :]
        #     out = out[..., : OT - invalid_steps]
        #     self._streaming_state.partial = partial
        #     return out
