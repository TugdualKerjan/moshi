import math
import jax
import jax.numpy as jnp
import typing as tp
import equinox as eqx
import equinox.nn as nn


def _ema_inplace(moving_avg: jax.Array, new: jax.Array, decay: float) -> jax.Array:
    return moving_avg * decay + new * (1 - decay)


def _sample_vectors(samples: jax.Array, num: int, key: jax.Array = None) -> jax.Array:  # type: ignore
    num_samples = samples.shape[0]

    if num_samples >= num:
        indices = jax.random.permutation(key, num_samples)
        # indices = jnp.repeat(indices,( num_samples // indices.shape[0]) + 1)
        # print(indices.shape)
    else:
        indices = jax.random.permutation(key, num_samples)

    return samples[indices]


def _compute_entropy(usage: jax.Array) -> jax.Array:
    # Usage is some unnormalized distribution.
    proba = usage / jnp.sum(usage)
    p_log_p = jnp.where(proba == 0, 0, proba * jnp.log(proba))
    return -p_log_p.sum()


def _run_kmeans(samples: jax.Array, num_clusters: int, num_iters: int = 50, key: jax.Array = None) -> tp.Tuple[jax.Array, jax.Array]:  # type: ignore

    k1, key = jax.random.split(key)
    dim = samples.shape[-1]
    means = _sample_vectors(samples, num_clusters, key=k1)
    bins = None

    for _ in range(num_iters):
        k1, key = jax.random.split(key)

        dists = jnp.linalg.norm(samples[:, None] - means[None, :], ord=2, axis=-1)
        buckets = jnp.argmin(dists, axis=-1)
        bins = jnp.bincount(buckets, minlength=num_clusters)
        zero_mask = bins == 0
        jnp.clip(bins, min=1)

        new_means = jnp.zeros_like(means)

        # Use scatter_add to add updates to new_means at the specified indices
        # TODO implement efficient version with scatter add from XLA
        new_means = new_means.at[buckets].add(samples)
        new_means /= bins[..., None]
        resampled = _sample_vectors(samples, num_clusters, key=k1)
        means = jnp.where(jnp.expand_dims(zero_mask, -1), resampled, new_means)

    assert bins is not None
    return means, bins


class EuclideanCodebook(eqx.Module):
    """Codebook with Euclidean distance.

    Args:
        dim (int): Dimension.
        codebook_size (int): Codebook size.
        decay (float): Decay for exponential moving average over the codebooks.
        epsilon (float): Epsilon value for numerical stability.
        threshold_usage_ratio (float): Defines the threshold for the cluster usage under which a centroid
            is replaced. This is expressed as a fraction of the usage a centroid would get under
            a uniform distribution, so that it doesn't depend on the batch size etc.
        replaced_usage_ratio (float): When replacing a centroid, use this as an initial centroid usage,
            to avoid the centroid getting replaced too quickly.
        check_unused_every (int): Check for unused centroids every `check_unused_every` iterations.
            This is to avoid too many synchronization points.

    Buffers:
        cluster_usage (torch.Tensor): EMA of the cluster usage per batch, e.g. this will
            be dependent on the batch size etc.
        embedding_sum (torch.Tensor): EMA of the sum of the assigned points to each cluster.
            In particular, this can be normalized by `cluster_usage` to obtain the
            actual cluster centroids.
    """

    cluster_usage: jax.Array = eqx.field(static=True)
    embedding_sum: jax.Array = eqx.field(static=True)
    embedding: jax.Array

    dim: int
    codebook_size: int
    decay: float
    epsilon: float
    threshold_usage_ratio: float
    replaced_usage_ratio: float
    check_unused_every: int

    _cached_initialized: bool
    _next_unused_check: int

    def __init__(
        self,
        dim: int,
        codebook_size: int,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        threshold_usage_ratio: float = 0.1,
        replaced_usage_ratio: float = 1.0,
        check_unused_every: int = 5,
        data: jax.Array = None,  # type: ignore
        key: jax.Array = None,
    ):  # type: ignore

        self.decay = decay

        self.dim = dim
        self.codebook_size = codebook_size

        self.epsilon = epsilon
        self.threshold_usage_ratio = threshold_usage_ratio
        self.replaced_usage_ratio = replaced_usage_ratio
        self.check_unused_every = check_unused_every
        self._next_unused_check = check_unused_every
        self._cached_initialized = False

        self._init_embedding(data, key)

    def _init_embedding(self, data: jax.Array, key: jax.Array = None) -> None:  # type: ignore
        embedding, cluster_usage = _run_kmeans(data, self.codebook_size, key=key)
        self.embedding_sum = embedding * cluster_usage[:, None]
        self.cluster_usage = cluster_usage
        self.embedding = (
            self.embedding_sum / jnp.clip(cluster_usage, min=self.epsilon)[:, None]
        )

    def _replace_expired_codes(self, samples: jax.Array, new_embedding_sum: jax.Array, new_cluster_usage : jax.Array, mask: jax.Array, key: jax.Array =None):
        # Replaces expired centroids, as indicated by `mask` (a true value indicate the code needs to be replaced).
        # The new codes are sampled from the batch `samples`.
        # new_vectors = _sample_vectors(samples, self.codebook_size, key)
        # print(new_vectors.shape)
        replace_cluster_usage = (
            self.replaced_usage_ratio * self.cluster_usage / self.codebook_size
        )

        embedding_sum = jnp.where(
            mask[:, None], replace_cluster_usage[:, None] * jax.random.normal(key, shape=(self.codebook_size, self.dim)), new_embedding_sum
        )
        cluster_usage = jnp.where(
            mask, replace_cluster_usage, new_cluster_usage
        )

        embedding = embedding_sum / jnp.clip(cluster_usage, min=self.epsilon)[:, None]
        where = lambda q: (q.embedding_sum, q.cluster_usage, q.embedding)

        return eqx.tree_at(where, self, (embedding_sum, cluster_usage, embedding))

    def _check_expired_codes(self, x: jax.Array, codes: jax.Array, key: jax.Array=None):
        cluster_usage = jnp.zeros_like(self.cluster_usage)
        cluster_usage = cluster_usage.at[codes].add(jnp.ones_like(codes))
        
        cluster_usage = _ema_inplace(self.cluster_usage, cluster_usage, self.decay)

        embedding_sum = jnp.zeros_like(self.embedding_sum)
        embedding_sum = embedding_sum.at[codes].add(x)
        
        embedding_sum = _ema_inplace(self.embedding_sum, embedding_sum, self.decay)

        threshold_cluster_usage = (
            self.threshold_usage_ratio
            * jnp.sum(cluster_usage)
            / self.codebook_size
        )
        expired_codes = self.cluster_usage < threshold_cluster_usage

        return self._replace_expired_codes(
            x, embedding_sum, cluster_usage, expired_codes, key
        )

    def _reshape_input(self, x: jax.Array) -> jax.Array:
        # Flattens all the dimensions but the last one, e.g. return a vector of shape `[N, D]`.
        return jnp.reshape(x, (-1, x.shape[-1]))

    def _reshape_codes(self, codes: jax.Array, shape: tp.Tuple) -> jax.Array:
        return jnp.reshape(codes, shape[0])

    def _quantize(self, x: jax.Array) -> jax.Array:
        # Projects each vector in `x` over the nearest centroid and return its index.
        # `x` should be `[N, D]` with `N` the number of input vectors and `D` the dimension.
        assert len(x.shape) == 2
        a_squared = jnp.sum(x**2, axis=-1, keepdims=True)
        b_squared = jnp.transpose(jnp.sum(self.embedding**2, axis=-1, keepdims=True))
        distance = (
            a_squared + b_squared - 2 * jnp.matmul(x, jnp.transpose(self.embedding))
        )

        return jnp.argmin(distance, axis=-1)

    def encode(self, x: jax.Array) -> jax.Array:
        """Given a tensor `x` of shape `[*, D]`, returns a tensor of integer codes of shape `[*]`.
        The codes are defined as the indexes of the centroids nearest to each vector in `x`.
        """
        # assert x.dtype.is_floating_point, f"Input should be floats, got {x.dtype}"
        shape = x.shape
        x = self._reshape_input(x)
        codes = self._quantize(x)
        # codes = self._reshape_codes(codes, shape)
        return codes

    def decode(self, codes: jax.Array) -> jax.Array:
        """Given a tensor of codes of shape `[*]`, returns a tensor of shape `[*, D]`,
        corresponding to the centroids associated to each code index.
        """
        # assert (
        #     not issubclass(codes.dtype, jnp.float16)
        # ), f"Codes should be integers, got {codes.dtype}"
        quantized = self.embedding[codes]
        return quantized

    # @eqx.filter_jit
    def __call__(self, x):
        x = self._reshape_input(x)
        
        codes = self._quantize(x)
        # codes = self._reshape_codes(flat_codes, shape)
        quantized = self.decode(codes)
        # metrics = {'rvq_entropy': _compute_entropy(self.cluster_usage) / math.log(self.codebook_size)}
        return (quantized, (x, codes), {})

class VectorQuantization(eqx.Module):
    """Vector quantization implementation.
    Currently supports only euclidean distance.

    Args:
        dim (int): Dimension
        codebook_size (int): Codebook size
        codebook_dim (int): Codebook dimension. If not defined, uses the specified dimension in dim.
        decay (float): Decay for exponential moving average over the codebooks.
        epsilon (float): Epsilon value for numerical stability.
        threshold_usage_ratio (float): Defines the threshold for the cluster usage under which a centroid
            is replaced. This is expressed as a fraction of the usage a centroid would get under
            a uniform distribution, so that it doesn't depend on the batch size etc.
        replaced_usage_ratio (float): When replacing a centroid, use this as an initial centroid usage,
            to avoid the centroid getting replaced too quickly.
        check_unused_every (int): Check for unused centroids every `check_unused_every` iterations.
            This is to avoid too many synchronization points.
    """

    project_in: eqx.Module
    project_out: eqx.Module
    epsilon: float
    _codebook: EuclideanCodebook
    codebook_size: int

    def __init__(
        self,
        dim: int,
        codebook_size: int,
        codebook_dim: tp.Optional[int] = None,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        threshold_usage_ratio: float = 0.1,
        key: jax.Array = None,  # type: ignore
        **kwargs,
    ):
        if codebook_dim is None:
            codebook_dim = dim

        requires_projection = codebook_dim != dim
        key0, key1, key2 = jax.random.split(key, 3)
        self.project_in = (
            nn.Linear(dim, codebook_dim, key=key0)
            if requires_projection
            else nn.Identity()
        )

        self.project_out = (
            nn.Linear(codebook_dim, dim, key=key1)
            if requires_projection
            else nn.Identity()
        )

        self.epsilon = epsilon
        self._codebook = EuclideanCodebook(
            dim=codebook_dim,
            codebook_size=codebook_size,
            decay=decay,
            epsilon=epsilon,
            threshold_usage_ratio=threshold_usage_ratio,
            data=jax.random.normal(key2, shape=(codebook_size, dim)),
            key=key2,
            **kwargs,
        )
        self.codebook_size = codebook_size

    @property
    def embedding(self):
        return self._codebook.embedding

    def _rearrange_input(self, x):
        x = jnp.transpose(x, (1, 0))
        return x

    def _rearrange_output(self, quantized):
        quantized = jnp.transpose(quantized, (1, 0))
        return quantized

    def encode(self, x: jax.Array) -> jax.Array:
        """Encodes `x` into discrete integer codes."""
        x = self._rearrange_input(x)
        x = jax.vmap(self.project_in)(x)  # type: ignore
        codes = self._codebook.encode(x)
        return codes

    def decode(self, codes: jax.Array) -> jax.Array:
        """Converts integer codes into quantized vectors."""
        quantized = self._codebook.decode(codes)
        quantized = jax.vmap(self.project_out)(quantized)  # type: ignore
        quantized = self._rearrange_output(quantized)
        return quantized

    def __call__(self, x: jax.Array):
        x = self._rearrange_input(x)
        x = jax.vmap(self.project_in)(x)  # type: ignore
        
        quantized, codes, metrics = self._codebook(x)

        quantized = x + jax.lax.stop_gradient(quantized - x)

        quantized = jax.vmap(self.project_out)(quantized)  # type: ignore
        quantized = self._rearrange_output(quantized)

        return (quantized, codes, metrics)


class ResidualVectorQuantization(eqx.Module):
    """Residual vector quantization implementation.

    Follows Algorithm 1. in https://arxiv.org/pdf/2107.03312.pdf
    """

    layers: list
    codebook_offset: int

    def __init__(self, *, num_quantizers: int, codebook_offset: int, key: jax.Array = None, **kwargs):  # type: ignore
        keys = jax.random.split(key, num_quantizers)
        self.layers = [VectorQuantization(**kwargs, key=k) for k in keys]
        self.codebook_offset = codebook_offset

    # @eqx.filter_jit
    def __call__(self, x: jax.Array, n_q: tp.Optional[int] = None):
        """
        Args:
            x (jax.Array): input tensor to quantize, of shape `[C, T]`.
            n_q (int or None): if provided, number of codebook levels to use in RVQ.
        """

        quantized_out = 0
        residual = x

        all_codes = []
        all_metrics: tp.Dict[str, jax.Array] = {}

        n_q = n_q or len(self.layers)

        for i, layer in enumerate(self.layers[:n_q]):  # type: ignore

            quantized, codes, metrics = layer(residual)

            residual = residual - quantized
            quantized_out = quantized_out + quantized

            all_codes.append(codes)

            for key, value in metrics.items():
                if key in all_metrics:
                    all_metrics[key] += value / n_q
                else:
                    all_metrics[key] = value / n_q
                all_metrics[key + f"_{i + self.codebook_offset}"] = value

        quantized_out = x + jax.lax.stop_gradient(quantized_out - x)

        # out_codes = jnp.stack(all_codes)
        return (quantized_out, all_codes, all_metrics)

    def encode(self, x: jax.Array, n_q=None) -> jax.Array:
        """Encodes `x` into discrete integer codes. If `n_q` is provided, only uses the first `n_q` codebook levels."""
        residual = x
        all_indices = []
        n_q = n_q or len(self.layers)
        for layer in self.layers[:n_q]:  # type: ignore
            indices = layer.encode(residual)
            quantized = layer.decode(indices)
            residual = residual - quantized
            all_indices.append(indices)
        out_indices = jnp.stack(all_indices)
        return out_indices

    def decode(self, codes: jax.Array) -> jax.Array:
        """Converts the integer codes into quantized vectors."""
        quantized = jnp.empty(codes.shape[0], self.layers[0].dim)
        for idx, layer_codes in enumerate(codes):
            layer = self.layers[idx]
            quantized = quantized + layer.decode(layer_codes)
        return quantized
