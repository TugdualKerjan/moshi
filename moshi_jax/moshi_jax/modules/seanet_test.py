import functools
import pytest
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from .seanet import SEANetResnetBlock, SEANetDecoder


SEANET_RESNET_DATA = [
    # batch_size, dim, res_layer_index, seq_len, kernel_size
    # pytest.param(
    #     3, 2, 1, 4, 2,
    #     id='small resnet test 1',
    # ),
    # pytest.param(
    #     4, 5, 2, 10, 7,
    #     id='small resnet test 2',
    # ),
    # pytest.param(
    #     5, 6, 4, 10, 2,
    #     id='small resnet test 3',
    # ),
    # pytest.param(
    #     1, 512, 2, 256, 7,
    #     id='large resnet test 1',
    # ),
]
NUM_TIMESTEPS_DATA = [
    # pytest.param(
    #     1,
    #     id='length 1',
    # ),
    pytest.param(
        2,
        id='length 2',
    ),
    # pytest.param(
    #     10,
    #     id='length 10',
    # ),
    # pytest.param(
    #     100,
    #     id='length 100',
    # ),
]

SEANET_KWARGS_DATA = [
    pytest.param(
        {
            "channels": 1,
            "dimension": 8,
            "causal": True,
            "n_filters": 2,
            "n_residual_layers": 1,
            "activation": "elu",
            "compress": 2,
            "dilation_base": 2,
            "disable_norm_outer_blocks": 0,
            "kernel_size": 7,
            "residual_kernel_size": 3,
            "last_kernel_size": 3,
            # We train using weight_norm but then the weights are pre-processed for inference so
            # that we can use a normal convolution.
            "norm": "none",
            "pad_mode": "constant",
            "ratios": [5],
            "true_skip": True,
        },
        id='Tiny SEANet',
    ),

    # pytest.param(
    #     {
    #         "channels": 1,
    #         "dimension": 512,
    #         "causal": True,
    #         "n_filters": 64,
    #         "n_residual_layers": 1,
    #         "activation": "elu",
    #         "compress": 2,
    #         "dilation_base": 2,
    #         "disable_norm_outer_blocks": 0,
    #         "kernel_size": 7,
    #         "residual_kernel_size": 3,
    #         "last_kernel_size": 3,
    #         # We train using weight_norm but then the weights are pre-processed for inference so
    #         # that we can use a normal convolution.
    #         "norm": "none",
    #         "pad_mode": "constant",
    #         "ratios": [8, 6, 5, 4],
    #         "true_skip": True,
    #     },
    #     id='Large SEANet',
    # ),
]

key = jax.random.key(1)
def _init_weights(path, x):
    global key
    key, k = jax.random.split(key)
    path = jtu.keystr(path)
    if "weight" in path:
        return jnp.ones(x.shape)
        # return jax.nn.initializers.xavier_uniform()(k, shape=x.shape)
    elif "bias" in path:
        return jnp.ones(x.shape)

        return jax.nn.initializers.constant(0)(k, x.shape)
    else:
        print(path)
        return x
        # nn.init.xavier_uniform_(param, generator=generator)

# 3 , 4, 1, 10, 6
# @pytest.mark.parametrize("batch_size, dim, res_layer_index, seq_len, kernel_size", SEANET_RESNET_DATA)
# def test_resnet(batch_size, dim, res_layer_index, seq_len, kernel_size):
#     """Test that SEANetResnetBlock() calls are causal. Having new inputs does not change the previous output."""
#     assert seq_len > kernel_size

#     dilation_base = 2
#     model = SEANetResnetBlock(dim=dim, dilations=[dilation_base**res_layer_index, 1], pad_mode="constant", causal=True, key=jax.random.key(1))

#     model = jtu.tree_map_with_path(_init_weights, model)

#     shape = (batch_size, dim, seq_len)
#     input_hidden_states = jax.random.normal(jax.random.key(1), shape=shape)

#     expected_output = jax.vmap(model)(input_hidden_states)

#     for end_index in range(kernel_size, seq_len + 1):
#         actual_output = jax.vmap(model)(input_hidden_states[..., :end_index])
#         assert jnp.allclose(actual_output, expected_output[..., :actual_output.shape[-1]]), lambda original_msg: f"Failed at end_index={end_index}: \n{original_msg}"


# @pytest.mark.parametrize("batch_size, dim, res_layer_index, seq_len, kernel_size", SEANET_RESNET_DATA)
# def test_resnet_streaming(batch_size, dim, res_layer_index, seq_len, kernel_size):
#     """Test that SEANetResnetBlock() streaming works as expected."""
#     assert seq_len > kernel_size

#     dilation_base = 2
#     model = SEANetResnetBlock(dim=dim, dilations=[dilation_base**res_layer_index, 1], pad_mode="constant", causal=True, key=jax.random.key(1))

#     model = jtu.tree_map_with_path(_init_weights, model)

#     shape = (batch_size, dim, seq_len,)
#     input_hidden_states = jax.random.normal(jax.random.key(1), shape=shape)

#     expected_output = jax.vmap(model)(input_hidden_states)

#     start_index = 0
#     actual_outputs = []
#     with model.streaming(batch_size=batch_size):
#         for end_index in range(kernel_size, seq_len + 1):
#             actual_output = jax.vmap(model)(input_hidden_states[..., start_index:end_index])
#             start_index = end_index
#             actual_outputs.append(actual_output)
#     actual_outputs = jnp.concat(actual_outputs, axis=-1)

#     jnp.allclose(actual_outputs, expected_output)


@pytest.mark.parametrize("num_timesteps", NUM_TIMESTEPS_DATA)
@pytest.mark.parametrize("seanet_kwargs", SEANET_KWARGS_DATA)
def test_nonstreaming_causal_decode(num_timesteps, seanet_kwargs):
    """Test that the SEANetDecoder does not depend on future inputs."""

    model = SEANetDecoder(**seanet_kwargs,key=jax.random.key(1))

    model = jtu.tree_map_with_path(_init_weights, model)
    # with torch.no_grad():
        # [B, K = 8, T]
    codes = jax.random.normal(jax.random.key(1), shape=(1, seanet_kwargs['dimension'], num_timesteps))
    # codes = jnp.stack([jnp.ones(shape=(1, seanet_kwargs['dimension'])),jnp.zeros(shape=(1, seanet_kwargs['dimension']))], axis=-1)
    print(codes.shape)
    print("Expect")
    expected_decoded = jax.vmap(model)(codes)

    num_timesteps = codes.shape[-1]
    for t in range(num_timesteps):
        current_codes = codes[..., :t + 1]
        print(f"Timestep {t}")
        actual_decoded = jax.vmap(model)(current_codes)
        assert jnp.allclose(expected_decoded[..., :actual_decoded.shape[-1]], actual_decoded), lambda original_msg: f"Failed at t={t}: \n{original_msg}"
