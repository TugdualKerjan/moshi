import equinox as eqx
import jax
import jax.numpy as jnp
import equinox.nn as nn
import typing as tp
from .conv import StreamingConv1d, StreamingConvTranspose1d


class SEANetResnetBlock(eqx.Module):
    """Residual block from SEANet model.

    Args:
        dim (int): Dimension of the input/output.
        kernel_sizes (list): List of kernel sizes for the convolutions.
        dilations (list): List of dilations for the convolutions.
        activation (str): Activation function.
        activation_params (dict): Parameters to provide to the activation function.
        norm (str): Normalization method.
        norm_params (dict): Parameters to provide to the underlying normalization used along with the convolution.
        causal (bool): Whether to use fully causal convolution.
        pad_mode (str): Padding mode for the convolutions.
        compress (int): Reduced dimensionality in residual branches (from Demucs v3).
        true_skip (bool): Whether to use true skip connection or a simple
            (streamable) convolution as the skip connection.
    """

    act: tp.Callable
    blocks: list
    # add: StreamingAdd
    shortcut: eqx.Module
    true_skip: bool

    def __init__(
        self,
        dim: int,
        kernel_sizes: tp.List[int] = [3, 1],
        dilations: tp.List[int] = [1, 1],
        activation: str = "ELU",
        activation_params: dict = {"alpha": 1.0},
        norm: str = "none",
        norm_kwargs: tp.Dict[str, tp.Any] = {},
        causal: bool = False,
        pad_mode: str = "reflect",
        compress: int = 2,
        true_skip: bool = True,
        key: jax.Array = None,  # type: ignore
    ):
        assert len(kernel_sizes) == len(
            dilations
        ), "Number of kernel sizes should match number of dilations"
        self.true_skip = true_skip
        self.act = getattr(jax.nn, activation.lower())
        hidden_dim = dim // compress

        key0, key1, key2 = jax.random.split(key, 3)

        self.blocks = (
            [
                StreamingConv1d(
                    dim,
                    hidden_dim,
                    kernel_size=kernel_sizes[0],
                    dilation=dilations[0],
                    norm=norm,
                    norm_kwargs=norm_kwargs,
                    causal=causal,
                    pad_mode=pad_mode,
                    key=key0,
                )
            ]
            + [
                StreamingConv1d(
                    hidden_dim,
                    hidden_dim,
                    kernel_size=kernel_sizes[i + 1],
                    dilation=dilations[i + 1],
                    norm=norm,
                    norm_kwargs=norm_kwargs,
                    causal=causal,
                    pad_mode=pad_mode,
                    key=k,
                )
                for i, k in enumerate(jax.random.split(key1, len(kernel_sizes) - 2))
            ]
            + [
                StreamingConv1d(
                    hidden_dim,
                    dim,
                    kernel_size=kernel_sizes[-1],
                    dilation=dilations[-1],
                    norm=norm,
                    norm_kwargs=norm_kwargs,
                    causal=causal,
                    pad_mode=pad_mode,
                    key=key2,
                )
            ]
        )

        # self.add = StreamingAdd()

        if self.true_skip:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = StreamingConv1d(
                dim,
                dim,
                kernel_size=1,
                norm=norm,
                norm_kwargs=norm_kwargs,
                causal=causal,
                pad_mode=pad_mode,
                key=key,
            )

    # @eqx.filter_jit
    def __call__(self, x):
        out = x

        for block in self.blocks:
            out = self.act(out)
            out = block(out)

        return out + self.shortcut(x)


class SEANetEncoder(eqx.Module):
    """SEANet encoder.

    Args:
        channels (int): Audio channels.
        dimension (int): Intermediate representation dimension.
        n_filters (int): Base width for the model.
        n_residual_layers (int): nb of residual layers.
        ratios (Sequence[int]): kernel size and stride ratios. The encoder uses downsampling ratios instead of
            upsampling ratios, hence it will use the ratios in the reverse order to the ones specified here
            that must match the decoder order. We use the decoder order as some models may only employ the decoder.
        activation (str): Activation function.
        activation_params (dict): Parameters to provide to the activation function.
        norm (str): Normalization method.
        norm_params (dict): Parameters to provide to the underlying normalization used along with the convolution.
        kernel_size (int): Kernel size for the initial convolution.
        last_kernel_size (int): Kernel size for the initial convolution.
        residual_kernel_size (int): Kernel size for the residual layers.
        dilation_base (int): How much to increase the dilation with each layer.
        causal (bool): Whether to use fully causal convolution.
        pad_mode (str): Padding mode for the convolutions.
        true_skip (bool): Whether to use true skip connection or a simple
            (streamable) convolution as the skip connection in the residual network blocks.
        compress (int): Reduced dimensionality in residual branches (from Demucs v3).
        disable_norm_outer_blocks (int): Number of blocks for which we don't apply norm.
            For the encoder, it corresponds to the N first blocks.
        mask_fn (nn.Module): Optional mask function to apply after convolution layers.
        mask_position (int): Position of the mask function, with mask_position == 0 for the first convolution layer,
            mask_position == 1 for the first conv block, etc.
    """

    channels: int
    dimension: int
    n_filters: int
    ratios: list
    n_residual_layers: int
    hop_length: int
    n_blocks: int
    disable_norm_outer_blocks: int
    act: tp.Callable
    first_layer: StreamingConv1d
    blocks: list
    last_layer: StreamingConv1d

    def __init__(
        self,
        channels: int = 1,
        dimension: int = 128,
        n_filters: int = 32,
        n_residual_layers: int = 3,
        ratios: tp.List[int] = [8, 5, 4, 2],
        activation: str = "elu",
        activation_params: dict = {"alpha": 1.0},
        norm: str = "none",
        norm_kwargs: tp.Dict[str, tp.Any] = {},
        kernel_size: int = 7,
        last_kernel_size: int = 7,
        residual_kernel_size: int = 3,
        dilation_base: int = 2,
        causal: bool = False,
        pad_mode: str = "reflect",
        true_skip: bool = True,
        compress: int = 2,
        disable_norm_outer_blocks: int = 0,
        mask_fn: tp.Optional[eqx.Module] = None,
        mask_position: tp.Optional[int] = None,
        key: jax.Array = None,  # type: ignore
    ):
        self.channels = channels
        self.dimension = dimension
        self.n_filters = n_filters
        self.ratios = list(reversed(ratios))
        del ratios
        self.n_residual_layers = n_residual_layers
        self.hop_length = int(jnp.prod(jnp.array(self.ratios)))
        self.n_blocks = len(self.ratios) + 2  # first and last conv + residual blocks
        self.disable_norm_outer_blocks = disable_norm_outer_blocks
        assert (
            self.disable_norm_outer_blocks >= 0
            and self.disable_norm_outer_blocks <= self.n_blocks
        ), (
            "Number of blocks for which to disable norm is invalid."
            "It should be lower or equal to the actual number of blocks in the network and greater or equal to 0."
        )
        key0, k1, key2 = jax.random.split(key, 3)

        self.act = getattr(jax.nn, activation.lower())
        mult = 1
        self.first_layer = StreamingConv1d(
            channels,
            mult * n_filters,
            kernel_size,
            norm="none" if disable_norm_outer_blocks >= 1 else norm,
            norm_kwargs=norm_kwargs,
            causal=causal,
            pad_mode=pad_mode,
            key=key0
        )
        if mask_fn is not None and mask_position == 0:
            self.blocks += [mask_fn]
        self.blocks = []
        for i, ratio in enumerate(self.ratios):
            k1, k2 = jax.random.split(k1)
            block_norm = "none" if disable_norm_outer_blocks >= i + 2 else norm
            self.blocks.append(
                (
                    [
                        SEANetResnetBlock(
                            mult * n_filters,
                            kernel_sizes=[residual_kernel_size, 1],
                            dilations=[dilation_base**j, 1],
                            norm=block_norm,
                            norm_kwargs=norm_kwargs,
                            activation=activation,
                            activation_params=activation_params,
                            causal=causal,
                            pad_mode=pad_mode,
                            compress=compress,
                            true_skip=true_skip,
                            key=k
                        )
                        for j, k in enumerate(jax.random.split(k2, n_residual_layers))
                    ],
                    StreamingConv1d(
                        mult * n_filters,
                        mult * n_filters * 2,
                        kernel_size=ratio * 2,
                        stride=ratio,
                        norm=block_norm,
                        norm_kwargs=norm_kwargs,
                        causal=causal,
                        pad_mode=pad_mode,
                        key=k1
                    ),
                )
            )
            mult *= 2
            if mask_fn is not None and mask_position == i + 1:
                self.blocks += [mask_fn]
        self.last_layer = StreamingConv1d(
            mult * n_filters,
            dimension,
            last_kernel_size,
            norm=("none" if disable_norm_outer_blocks == self.n_blocks else norm),
            norm_kwargs=norm_kwargs,
            causal=causal,
            pad_mode=pad_mode,
            key=key2
        )

    @eqx.filter_jit
    def __call__(self, x):
        y = self.first_layer(x)

        for resnetBlocks, down in self.blocks:
            print(f"Ours : {y[0, :10]}")

            for block in resnetBlocks:
                y = block(y)

            y = self.act(y)
            y = down(y)

        y = self.act(y)
        print(f"Ours : {y[0, :10]}")
        return self.last_layer(y)


class SEANetDecoder(eqx.Module):
    """SEANet decoder.

    Args:
        channels (int): Audio channels.
        dimension (int): Intermediate representation dimension.
        n_filters (int): Base width for the model.
        n_residual_layers (int): nb of residual layers.
        ratios (Sequence[int]): kernel size and stride ratios.
        activation (str): Activation function.
        activation_params (dict): Parameters to provide to the activation function.
        final_activation (str): Final activation function after all convolutions.
        final_activation_params (dict): Parameters to provide to the activation function.
        norm (str): Normalization method.
        norm_params (dict): Parameters to provide to the underlying normalization used along with the convolution.
        kernel_size (int): Kernel size for the initial convolution.
        last_kernel_size (int): Kernel size for the initial convolution.
        residual_kernel_size (int): Kernel size for the residual layers.
        dilation_base (int): How much to increase the dilation with each layer.
        causal (bool): Whether to use fully causal convolution.
        pad_mode (str): Padding mode for the convolutions.
        true_skip (bool): Whether to use true skip connection or a simple.
            (streamable) convolution as the skip connection in the residual network blocks.
        compress (int): Reduced dimensionality in residual branches (from Demucs v3).
        disable_norm_outer_blocks (int): Number of blocks for which we don't apply norm.
            For the decoder, it corresponds to the N last blocks.
        trim_right_ratio (float): Ratio for trimming at the right of the transposed convolution under the causal setup.
            If equal to 1.0, it means that all the trimming is done at the right.
    """

    channels: int
    dimension: int
    n_filters: int
    ratios: list
    n_residual_layers: int
    hop_length: int
    n_blocks: int
    disable_norm_outer_blocks: int
    act: tp.Callable
    first_layer: StreamingConv1d
    blocks: list
    last_layer: StreamingConv1d
    final_act: tp.Callable

    def __init__(
        self,
        channels: int = 1,
        dimension: int = 128,
        n_filters: int = 32,
        n_residual_layers: int = 3,
        ratios: tp.List[int] = [8, 5, 4, 2],
        activation: str = "elu",
        activation_params: dict = {"alpha": 1.0},
        final_activation: tp.Optional[str] = None,
        final_activation_params: tp.Optional[dict] = None,
        norm: str = "none",
        norm_kwargs: tp.Dict[str, tp.Any] = {},
        kernel_size: int = 7,
        last_kernel_size: int = 7,
        residual_kernel_size: int = 3,
        dilation_base: int = 2,
        causal: bool = False,
        pad_mode: str = "reflect",
        true_skip: bool = True,
        compress: int = 2,
        disable_norm_outer_blocks: int = 0,
        trim_right_ratio: float = 1.0,
        key: jax.Array = None,  # type: ignore
    ):
        self.dimension = dimension
        self.channels = channels
        self.n_filters = n_filters
        self.ratios = ratios
        del ratios
        self.n_residual_layers = n_residual_layers
        self.hop_length = int(jnp.prod(jnp.array(self.ratios)))
        self.n_blocks = len(self.ratios) + 2  # first and last conv + residual blocks
        self.disable_norm_outer_blocks = disable_norm_outer_blocks
        assert (
            disable_norm_outer_blocks >= 0
            and disable_norm_outer_blocks <= self.n_blocks
        ), (
            "Number of blocks for which to disable norm is invalid."
            "It should be lower or equal to the actual number of blocks in the network and greater or equal to 0."
        )

        key0, k1, key2 = jax.random.split(key, 3)

        self.act = getattr(jax.nn, activation.lower())
        mult = int(2 ** len(self.ratios))
        self.first_layer = StreamingConv1d(
            dimension,
            mult * n_filters,
            kernel_size,
            norm=("none" if disable_norm_outer_blocks == self.n_blocks else norm),
            norm_kwargs=norm_kwargs,
            causal=causal,
            pad_mode=pad_mode,
            key=key0,
        )
        self.blocks = []
        for i, ratio in enumerate(self.ratios):

            k1, k2 = jax.random.split(k1)
            block_norm = (
                "none" if disable_norm_outer_blocks >= self.n_blocks - (i + 1) else norm
            )
            self.blocks.append(
                (
                    StreamingConvTranspose1d(
                        mult * n_filters,
                        mult * n_filters // 2,
                        kernel_size=ratio * 2,
                        stride=ratio,
                        norm=block_norm,
                        norm_kwargs=norm_kwargs,
                        causal=causal,
                        trim_right_ratio=trim_right_ratio,
                        key=k1,
                    ),
                    [
                        SEANetResnetBlock(
                            mult * n_filters // 2,
                            kernel_sizes=[residual_kernel_size, 1],
                            dilations=[dilation_base**j, 1],
                            activation=activation,
                            activation_params=activation_params,
                            norm=block_norm,
                            norm_kwargs=norm_kwargs,
                            causal=causal,
                            pad_mode=pad_mode,
                            compress=compress,
                            true_skip=true_skip,
                            key=k,
                        )
                        for j, k in enumerate(jax.random.split(k2, n_residual_layers))
                    ]
                )
            )
            mult //= 2
        self.last_layer = StreamingConv1d(
            n_filters,
            channels,
            last_kernel_size,
            norm="none" if disable_norm_outer_blocks >= 1 else norm,
            norm_kwargs=norm_kwargs,
            causal=causal,
            pad_mode=pad_mode,
            key=key2,
        )
        if final_activation is not None:
            final_act = getattr(jax.nn, final_activation)
            final_activation_params = final_activation_params or {}
            self.final_act = final_act(**final_activation_params)
        else:
            self.final_act = nn.Identity()

    # @eqx.filter_jit
    def __call__(self, x):
        y = self.first_layer(x)
        for down, resnetBlocks in self.blocks:
            y = self.act(y)
            y = down(y)
            for block in resnetBlocks:
                y = block(y)
        y = self.act(y)
        y = self.last_layer(y)
        return self.final_act(y)
