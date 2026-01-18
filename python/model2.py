# --- STANDARD ---
import copy

# --- PROJECT ---

# --- THIRD-PARTY ---
import tensorflow as tf
from tensorflow.keras import layers, models  # type: ignore


CONST = {
    "gru_dropout": 0.1,
    "post_gru_dropout": 0.1,
    "symbol_attention_bottleneck_ratio": 0.5,  # low-rank constraint
    "use_symbol_layer_norm": True,
    "dense_activation": "relu",
    "gate_activation": "sigmoid",
    "news_time_axis": 1,  # time dimension
    "output_dtype": "float32",
    "optimizer": "Adam",
}


VAR = {
    "stock_embed": 32,
    "news_embed": 32,
    "stock_attention_heads": 4,
    "shared_trunk_size_1": 2048,
    "shared_trunk_size_2": 1024,
}

if VAR["stock_embed"] % VAR["stock_attention_heads"] != 0:
    raise ValueError("stock_embed must be divisible by stock_attention_heads")


if (VAR["stock_embed"] * CONST["symbol_attention_bottleneck_ratio"]) % VAR[
    "stock_attention_heads"
] != 0:
    raise ValueError(
        "(stock_embed * symbol_attention_bottleneck_ratio) must be divisible by stock_attention_heads"
    )


def build(
    *, time_steps: int, stock_feature_dim: int, news_feature_dim: int, symbol_count: int
) -> tuple[models.Model, dict]:

    # ============================================================
    # INPUTS
    # ============================================================
    stock_in = layers.Input(
        shape=(time_steps, symbol_count, stock_feature_dim), name="stock"
    )
    news_in = layers.Input(shape=(time_steps, news_feature_dim), name="news")

    # ============================================================
    # STOCK: PER-SYMBOL PROJECTION
    # ============================================================
    x = layers.TimeDistributed(
        layers.TimeDistributed(
            layers.Dense(
                VAR["stock_embed"],
                activation=CONST["dense_activation"],
            )
        )
    )(stock_in)
    # (batch, time, symbols, embed)

    # ============================================================
    # STOCK: TEMPORAL REGIME (GRU)
    # ============================================================

    x = layers.Permute((2, 1, 3), name="to_symbol_major")(x)
    # (batch, symbols, time, embed)

    x = layers.Lambda(
        lambda t: tf.reshape(t, (-1, time_steps, VAR["stock_embed"])),
        name="merge_batch_and_symbols",
    )(x)
    # (batch * symbols, time, embed)

    x = layers.GRU(
        VAR["stock_embed"],
        return_sequences=False,
        dropout=CONST["gru_dropout"],
        name="stock_gru",
    )(x)

    x = layers.Dropout(CONST["post_gru_dropout"])(x)

    x = layers.Lambda(
        lambda t: tf.reshape(t, (-1, symbol_count, VAR["stock_embed"])),
        name="restore_symbol_axis",
    )(x)
    # (batch, symbols, embed)

    # ============================================================
    # CROSS-SYMBOL ATTENTION (LOW-RANK, STABILIZED)
    # ============================================================
    if CONST["use_symbol_layer_norm"]:
        x = layers.LayerNormalization(name="symbol_ln")(x)

    bottleneck_dim = int(
        VAR["stock_embed"] * CONST["symbol_attention_bottleneck_ratio"]
    )

    bottleneck = layers.Dense(
        bottleneck_dim,
        activation=CONST["dense_activation"],
        name="symbol_bottleneck",
    )(x)

    key_dim = bottleneck_dim // VAR["stock_attention_heads"]

    x = layers.MultiHeadAttention(
        num_heads=VAR["stock_attention_heads"],
        key_dim=key_dim,
        name="symbol_attention",
    )(bottleneck, bottleneck)

    stock_repr = layers.Dense(
        VAR["stock_embed"],
        activation=CONST["dense_activation"],
        name="stock_post_attention",
    )(x)
    # (batch, symbols, embed)

    # ============================================================
    # SYMBOL GATING (ARCHITECTURAL)
    # ============================================================
    symbol_gate = layers.Dense(
        1,
        activation=CONST["gate_activation"],
        name="symbol_gate",
    )(stock_repr)

    stock_repr = layers.Multiply(name="stock_gated")([stock_repr, symbol_gate])

    # ============================================================
    # NEWS: TEMPORAL DECAY / GATING
    # ============================================================
    gate_logits = layers.Dense(
        1,
        name="news_time_gate",
    )(news_in)

    gate = layers.Softmax(
        axis=CONST["news_time_axis"],
        name="news_time_softmax",
    )(gate_logits)

    gated_news = layers.Multiply(name="news_time_gated")([news_in, gate])

    news_base = layers.Dense(
        VAR["news_embed"],
        activation=CONST["dense_activation"],
        name="news_base_embed",
    )(gated_news)

    news_base = layers.GlobalAveragePooling1D(name="news_time_pool")(news_base)
    # (batch, news_embed)

    # ============================================================
    # NEWS: REGIME CONDITIONING
    # ============================================================
    regime = layers.GlobalAveragePooling1D(name="regime_pool")(stock_repr)

    regime_gate = layers.Dense(
        VAR["news_embed"],
        activation=CONST["gate_activation"],
        name="news_regime_gate",
    )(regime)

    news_conditioned = layers.Multiply(name="news_conditioned")(
        [news_base, regime_gate]
    )

    # ============================================================
    # NEWS: SYMBOL-AWARE ATTRIBUTION
    # ============================================================
    news_sym = layers.Dense(
        symbol_count * VAR["news_embed"],
        activation=CONST["dense_activation"],
        name="news_symbol_projection",
    )(news_conditioned)

    news_sym = layers.Reshape(
        (symbol_count, VAR["news_embed"]),
        name="news_per_symbol",
    )(news_sym)

    # ============================================================
    # FUSION
    # ============================================================
    fused = layers.Concatenate(
        axis=-1,
        name="symbol_fusion",
    )([stock_repr, news_sym])

    fused = layers.Flatten(name="fusion_flatten")(fused)

    # ============================================================
    # SHARED TRUNK
    # ============================================================
    shared = layers.Dense(
        VAR["shared_trunk_size_1"],
        activation=CONST["dense_activation"],
        name="shared_dense_1",
    )(fused)

    shared = layers.Dense(
        VAR["shared_trunk_size_2"],
        activation=CONST["dense_activation"],
        name="shared_dense_2",
    )(shared)

    # ============================================================
    # HEADS
    # ============================================================
    def head(name, activation=None):
        return layers.Dense(
            symbol_count,
            activation=activation,
            name=name,
            dtype=CONST["output_dtype"],
        )(shared)

    outputs = {
        "zscore_1d": head("zscore_1d"),
        "rank_1d": head("rank_1d"),
        "direction_1d": head("direction_1d", activation=CONST["gate_activation"]),
        "log_return_1d": head("log_return_1d"),
        "log_return_3d": head("log_return_3d"),
        "log_return_5d": head("log_return_5d"),
    }

    model = models.Model(
        inputs=[stock_in, news_in],
        outputs=outputs,
    )

    config = {
        "constants": copy.deepcopy(CONST),
        "variables": copy.deepcopy(VAR),
    }

    return model, config
