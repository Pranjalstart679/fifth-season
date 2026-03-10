from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import Model, layers


def _keras_serializable(fn_or_cls):
    """Register with Keras, silently skipping if already registered (e.g. on module reload)."""
    try:
        return tf.keras.utils.register_keras_serializable(package="fifth_season")(fn_or_cls)
    except ValueError:
        return fn_or_cls


@_keras_serializable
def _ensure_time_axis(tensor: tf.Tensor) -> tf.Tensor:
    if tensor.shape.rank == 4:
        return tf.expand_dims(tensor, axis=1)
    return tensor


@_keras_serializable
def ssim_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    y_true = _ensure_time_axis(tf.cast(y_true, tf.float32))
    y_pred = _ensure_time_axis(tf.cast(y_pred, tf.float32))
    y_true = tf.squeeze(y_true, axis=1)
    y_pred = tf.squeeze(y_pred, axis=1)
    return 1.0 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))


@_keras_serializable
def combined_regression_loss(y_true: tf.Tensor, y_pred: tf.Tensor, alpha: float = 0.80) -> tf.Tensor:
    """MSE + SSIM regression loss. alpha controls SSIM weight (higher = less blurry predictions)."""
    mse = tf.reduce_mean(tf.math.squared_difference(tf.cast(y_true, tf.float32), tf.cast(y_pred, tf.float32)))
    return (1.0 - alpha) * mse + alpha * ssim_loss(y_true, y_pred)


@_keras_serializable
def focal_crossentropy_loss(y_true: tf.Tensor, y_pred: tf.Tensor, gamma: float = 2.0) -> tf.Tensor:
    """Sparse focal cross-entropy for spatial segmentation.

    Downweights easy (well-classified) pixels so training focuses on the
    hard minority aerosol classes rather than collapsing to the majority class.

    y_true: integer class labels  (...,)
    y_pred: softmax probabilities (..., num_classes)
    """
    num_classes = tf.shape(y_pred)[-1]
    y_true_flat = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
    y_pred_flat = tf.reshape(y_pred, [-1, num_classes])
    p_true = tf.reduce_sum(
        tf.one_hot(y_true_flat, num_classes) * y_pred_flat,
        axis=-1,
    )
    p_true = tf.clip_by_value(p_true, 1e-7, 1.0 - 1e-7)
    focal_weight = tf.pow(1.0 - p_true, gamma)
    return tf.reduce_mean(-focal_weight * tf.math.log(p_true))


@_keras_serializable
class AverageSSIM(tf.keras.metrics.Metric):
    def __init__(self, name: str = "average_ssim", **kwargs) -> None:
        super().__init__(name=name, **kwargs)
        self.total = self.add_weight(name="total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight: tf.Tensor | None = None) -> None:
        y_true = _ensure_time_axis(tf.cast(y_true, tf.float32))
        y_pred = _ensure_time_axis(tf.cast(y_pred, tf.float32))
        y_true = tf.squeeze(y_true, axis=1)
        y_pred = tf.squeeze(y_pred, axis=1)
        ssim_scores = tf.image.ssim(y_true, y_pred, max_val=1.0)
        ssim_scores = tf.cast(ssim_scores, self.dtype)

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            ssim_scores *= sample_weight

        self.total.assign_add(tf.reduce_sum(ssim_scores))
        self.count.assign_add(tf.cast(tf.size(ssim_scores), self.dtype))

    def result(self) -> tf.Tensor:
        return tf.math.divide_no_nan(self.total, self.count)

    def reset_state(self) -> None:
        self.total.assign(0.0)
        self.count.assign(0.0)


@_keras_serializable
def flatten_spatial(x: tf.Tensor) -> tf.Tensor:
    shape = tf.shape(x)
    batch = shape[0]
    timesteps = shape[1]
    height = shape[2]
    width = shape[3]
    channels = shape[4]
    return tf.reshape(x, (batch * timesteps, height * width, channels))


@_keras_serializable
def restore_spatial(inputs: tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
    flattened, reference = inputs
    reference_shape = tf.shape(reference)
    return tf.reshape(
        flattened,
        (
            reference_shape[0],
            reference_shape[1],
            reference_shape[2],
            reference_shape[3],
            tf.shape(flattened)[-1],
        ),
    )


@_keras_serializable
def expand_time_axis(x: tf.Tensor) -> tf.Tensor:
    return tf.expand_dims(x, axis=1)


@_keras_serializable
def _flatten_spatial_output_shape(input_shape: tuple[int | None, ...]) -> tuple[int | None, int | None, int | None]:
    _, timesteps, height, width, channels = input_shape
    flattened_steps = None if timesteps is None else height * width
    merged_batch = None
    if timesteps is not None:
        merged_batch = None
    return (merged_batch, flattened_steps, channels)


@_keras_serializable
def _restore_spatial_output_shape(
    input_shape: tuple[tuple[int | None, ...], tuple[int | None, ...]],
) -> tuple[int | None, int | None, int | None, int | None, int | None]:
    _, reference_shape = input_shape
    return reference_shape


@_keras_serializable
def _expand_time_axis_output_shape(
    input_shape: tuple[int | None, int | None, int | None, int | None],
) -> tuple[int | None, int | None, int | None, int | None, int | None]:
    batch, height, width, channels = input_shape
    return (batch, 1, height, width, channels)


def _resize_skip_connection(source: tf.Tensor, reference: tf.Tensor) -> tf.Tensor:
    target_height = reference.shape[2]
    target_width = reference.shape[3]
    if target_height is None or target_width is None:
        raise ValueError("Spatial dimensions must be statically known for the ConvLSTM U-Net skip connections.")
    return layers.TimeDistributed(
        layers.Resizing(target_height, target_width, interpolation="bilinear")
    )(source)


def _convlstm_block(
    tensor: tf.Tensor,
    filters: int,
    return_sequences: bool,
    name_prefix: str,
    dropout_rate: float = 0.1,
) -> tf.Tensor:
    tensor = layers.ConvLSTM2D(
        filters=filters,
        kernel_size=(3, 3),
        padding="same",
        return_sequences=return_sequences,
        dropout=dropout_rate,
        recurrent_dropout=0.0,
        name=f"{name_prefix}_convlstm",
    )(tensor)
    tensor = layers.BatchNormalization(name=f"{name_prefix}_bn")(tensor)
    return tensor


def _spatial_attention_block(tensor: tf.Tensor, num_heads: int = 4, name_prefix: str = "attention") -> tf.Tensor:
    """Channel squeeze-excitation attention applied per timestep.

    Replaces full spatial MHA (O(H²W²) memory — OOM at full resolution) with
    SE-style channel attention (O(C)), which is both memory-safe and effective
    for ConvLSTM feature maps where channel correlations carry the signal.
    """
    channels = int(tensor.shape[-1] or 32)
    reduction = max(channels // 4, 4)

    # Global average-pool over H×W per timestep → (B, T, 1, 1, C)
    squeezed = layers.TimeDistributed(
        layers.GlobalAveragePooling2D(keepdims=True),
        name=f"{name_prefix}_gap",
    )(tensor)

    # Squeeze: reduce channel dimension
    excitation = layers.TimeDistributed(
        layers.Conv2D(reduction, kernel_size=1, activation="relu", use_bias=True),
        name=f"{name_prefix}_squeeze",
    )(squeezed)

    # Excite: restore channel dimension with sigmoid gate
    excitation = layers.TimeDistributed(
        layers.Conv2D(channels, kernel_size=1, activation="sigmoid", use_bias=True),
        name=f"{name_prefix}_excite",
    )(excitation)

    # Scale: broadcast (B, T, 1, 1, C) across spatial dims of (B, T, H, W, C)
    attended = layers.Multiply(name=f"{name_prefix}_scale")([tensor, excitation])
    return layers.BatchNormalization(name=f"{name_prefix}_bn")(attended)


def build_mtl_unet_convlstm(
    input_shape: tuple[int, int, int, int],
    num_classes: int = 5,
    base_filters: int = 16,
) -> Model:
    inputs = layers.Input(shape=input_shape, name="input_sequence")

    encoder_1 = _convlstm_block(inputs, base_filters, return_sequences=True, name_prefix="encoder1")
    pool_1 = layers.TimeDistributed(layers.MaxPooling2D(pool_size=(2, 2)), name="pool1")(encoder_1)

    encoder_2 = _convlstm_block(pool_1, base_filters * 2, return_sequences=True, name_prefix="encoder2")
    pool_2 = layers.TimeDistributed(layers.MaxPooling2D(pool_size=(2, 2)), name="pool2")(encoder_2)

    bottleneck = _convlstm_block(pool_2, base_filters * 4, return_sequences=True, name_prefix="bottleneck")
    bottleneck = _spatial_attention_block(bottleneck, num_heads=4, name_prefix="bottleneck_attention")

    up_2 = layers.TimeDistributed(layers.UpSampling2D(size=(2, 2)), name="up2")(bottleneck)
    up_2 = _resize_skip_connection(up_2, encoder_2)
    merge_2 = layers.Concatenate(axis=-1, name="skip2_concat")([up_2, encoder_2])
    decoder_2 = _convlstm_block(merge_2, base_filters * 2, return_sequences=True, name_prefix="decoder2")

    up_1 = layers.TimeDistributed(layers.UpSampling2D(size=(2, 2)), name="up1")(decoder_2)
    up_1 = _resize_skip_connection(up_1, encoder_1)
    merge_1 = layers.Concatenate(axis=-1, name="skip1_concat")([up_1, encoder_1])
    decoder_1 = _convlstm_block(merge_1, base_filters, return_sequences=False, name_prefix="decoder1")

    decoder_1 = layers.Conv2D(base_filters, kernel_size=(3, 3), padding="same", activation="relu", name="decoder_refine")(decoder_1)
    head_input = layers.Lambda(
        expand_time_axis,
        output_shape=_expand_time_axis_output_shape,
        name="expand_time_axis",
    )(decoder_1)

    regression_output = layers.Conv3D(
        filters=1,
        kernel_size=(1, 1, 1),
        activation="sigmoid",
        padding="same",
        name="severity_output",
    )(head_input)
    classification_output = layers.Conv3D(
        filters=num_classes,
        kernel_size=(1, 1, 1),
        activation="softmax",
        padding="same",
        name="identity_output",
    )(head_input)

    return Model(inputs=inputs, outputs=[regression_output, classification_output], name="mtl_unet_convlstm")


def compile_mtl_unet_convlstm(
    input_shape: tuple[int, int, int, int],
    num_classes: int = 5,
    learning_rate: float = 1e-5,
    base_filters: int = 16,
) -> Model:
    model = build_mtl_unet_convlstm(input_shape=input_shape, num_classes=num_classes, base_filters=base_filters)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0),
        loss={
            "severity_output": combined_regression_loss,
            "identity_output": focal_crossentropy_loss,
        },
        loss_weights={
            "severity_output": 2.0,
            "identity_output": 1.0,
        },
        metrics={
            "severity_output": [tf.keras.metrics.MeanSquaredError(name="mse"), AverageSSIM()],
            "identity_output": [tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
        },
    )
    return model


if __name__ == "__main__":
    example_model = compile_mtl_unet_convlstm((5, 291, 512, 11), num_classes=5)
    example_model.summary()
