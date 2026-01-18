# --- STANDARD ---
import os

# --- PROJECT ---

# --- THIRD-PARTY ---
import tensorflow as tf
from tensorflow.keras import layers, losses, models, optimizers  # type: ignore


STOCK_ATTENTION_HEADS = int(os.getenv("SYMBOL_ATTENTION_HEADS", "4"))
STOCK_EMBED = int(os.getenv("SYMBOL_EMBED", "32"))

if STOCK_EMBED % STOCK_ATTENTION_HEADS != 0:
    raise ValueError("STOCK_EMBED must be divisible by STOCK_ATTENTION_HEADS")

NEWS_ATTENTION_HEADS = int(os.getenv("NEWS_ATTENTION_HEADS", "4"))
NEWS_EMBED = int(os.getenv("NEWS_EMBED", "32"))

if NEWS_EMBED % NEWS_ATTENTION_HEADS != 0:
    raise ValueError("NEWS_EMBED must be divisible by NEWS_ATTENTION_HEADS")

SHARED_TRUNK_SIZE_1 = int(os.getenv("SHARED_TRUNK_SIZE_1", "2048"))
SHARED_TRUNK_SIZE_2 = int(os.getenv("SHARED_TRUNK_SIZE_2", "1024"))

LEARNING_RATE = float(os.getenv("LEARNING_RATE", "1e-3"))

LOSSES_HUBER_DELTA = float(os.getenv("LOSSES_HUBER_DELTA", "0.1"))

LOSS_ZSCORE_1D = os.getenv("LOSS_ZSCORE_1D", "mse")
LOSS_RANK_1D = losses.Huber(delta=LOSSES_HUBER_DELTA)
LOSS_DIRECTION_1D = os.getenv("LOSS_DIRECTION_1D", "binary_crossentropy")
LOSS_LOG_RETURN_1D = os.getenv("LOSS_LOG_RETURN_1D", "mse")
LOSS_LOG_RETURN_3D = os.getenv("LOSS_LOG_RETURN_3D", "mse")
LOSS_LOG_RETURN_5D = os.getenv("LOSS_LOG_RETURN_5D", "mse")

LOSS_WEIGHT_ZSCORE_1D = float(os.getenv("LOSS_WEIGHT_ZSCORE_1D", "1.0"))
LOSS_WEIGHT_RANK_1D = float(os.getenv("LOSS_WEIGHT_RANK_1D", "0.3"))
LOSS_WEIGHT_DIRECTION_1D = float(os.getenv("LOSS_WEIGHT_DIRECTION_1D", "0.3"))
LOSS_WEIGHT_LOG_RETURN_1D = float(os.getenv("LOSS_WEIGHT_LOG_RETURN_1D", "0.05"))
LOSS_WEIGHT_LOG_RETURN_3D = float(os.getenv("LOSS_WEIGHT_LOG_RETURN_3D", "0.05"))
LOSS_WEIGHT_LOG_RETURN_5D = float(os.getenv("LOSS_WEIGHT_LOG_RETURN_5D", "0.05"))


def build(
    *, time_steps: int, stock_feature_dim: int, news_feature_dim: int, symbol_count: int
) -> tuple[models.Model, dict]:

    config = {
        "stock_attention_heads": STOCK_ATTENTION_HEADS,
        "stock_embed": STOCK_EMBED,
        "news_attention_heads": NEWS_ATTENTION_HEADS,
        "news_embed": NEWS_EMBED,
        "shared_trunk_size_1": SHARED_TRUNK_SIZE_1,
        "shared_trunk_size_2": SHARED_TRUNK_SIZE_2,
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "loss_zscore_1d": LOSS_ZSCORE_1D,
        "loss_rank_1d": {
            "type": "Huber",
            "delta": LOSSES_HUBER_DELTA,
        },
        "loss_direction_1d": LOSS_DIRECTION_1D,
        "loss_log_return_1d": LOSS_LOG_RETURN_1D,
        "loss_log_return_3d": LOSS_LOG_RETURN_3D,
        "loss_log_return_5d": LOSS_LOG_RETURN_5D,
        "loss_weight_zscore_1d": LOSS_WEIGHT_ZSCORE_1D,
        "loss_weight_rank_1d": LOSS_WEIGHT_RANK_1D,
        "loss_weight_direction_1d": LOSS_WEIGHT_DIRECTION_1D,
        "loss_weight_log_return_1d": LOSS_WEIGHT_LOG_RETURN_1D,
        "loss_weight_log_return_3d": LOSS_WEIGHT_LOG_RETURN_3D,
        "loss_weight_log_return_5d": LOSS_WEIGHT_LOG_RETURN_5D,
    }

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
        layers.TimeDistributed(layers.Dense(config["stock_embed"], activation="relu"))
    )(stock_in)
    # (batch, time, symbols, embed)

    # ============================================================
    # STOCK: TEMPORAL REGIME (GRU)
    # ============================================================
    x = layers.Reshape((-1, time_steps, config["stock_embed"]))(x)
    x = layers.GRU(
        config["stock_embed"],
        return_sequences=False,
        dropout=0.1,
        recurrent_dropout=0.0,
        name="stock_gru",
    )(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Reshape((-1, symbol_count, config["stock_embed"]))(x)
    # (batch, symbols, embed)

    # ============================================================
    # SECTOR EMBEDDINGS (SYMBOL INDUCTIVE BIAS)
    # ============================================================
    # assumes sector indices are provided externally in fixed order
    sector_ids = tf.range(symbol_count)
    sector_embed = layers.Embedding(
        input_dim=symbol_count,
        output_dim=config["stock_embed"],
        name="sector_embedding",
    )(sector_ids)
    sector_embed = tf.expand_dims(sector_embed, axis=0)

    x = layers.Add()([x, sector_embed])

    # ============================================================
    # CROSS-SYMBOL ATTENTION (LOW-RANK)
    # ============================================================
    bottleneck = layers.Dense(
        config["stock_embed"] // 2, activation="relu", name="symbol_bottleneck"
    )(x)

    x = layers.MultiHeadAttention(
        num_heads=2,
        key_dim=(config["stock_embed"] // 2) // 2,
        name="symbol_attention",
    )(bottleneck, bottleneck)

    x = layers.Dense(config["stock_embed"], activation="relu")(x)
    stock_repr = x
    # (batch, symbols, embed)

    # ============================================================
    # NEWS: TEMPORAL DECAY / GATING
    # ============================================================
    gate_logits = layers.Dense(1, name="news_time_gate")(news_in)
    gate = layers.Softmax(axis=1)(gate_logits)
    gated_news = layers.Multiply()([news_in, gate])

    news_base = layers.Dense(
        config["news_embed"], activation="relu", name="news_base_embed"
    )(gated_news)
    news_base = layers.GlobalAveragePooling1D()(news_base)
    # (batch, news_embed)

    # ============================================================
    # NEWS: REGIME CONDITIONING
    # ============================================================
    regime = layers.GlobalAveragePooling1D()(stock_repr)
    regime_gate = layers.Dense(
        config["news_embed"], activation="sigmoid", name="news_regime_gate"
    )(regime)

    news_conditioned = layers.Multiply()([news_base, regime_gate])

    # ============================================================
    # NEWS: SYMBOL-AWARE ATTRIBUTION
    # ============================================================
    news_sym = layers.Dense(
        symbol_count * config["news_embed"],
        activation="relu",
        name="news_symbol_projection",
    )(news_conditioned)

    news_sym = layers.Reshape((symbol_count, config["news_embed"]))(news_sym)

    # ============================================================
    # FUSION (SYMBOL-WISE)
    # ============================================================
    fused = layers.Concatenate(axis=-1)([stock_repr, news_sym])

    # ============================================================
    # SYMBOL GATING BEFORE COMPRESSION
    # ============================================================
    symbol_gate = layers.Dense(1, activation="sigmoid", name="symbol_gate")(fused)
    fused = layers.Multiply()([fused, symbol_gate])

    fused = layers.Flatten()(fused)

    # ============================================================
    # SHARED TRUNK
    # ============================================================
    shared = layers.Dense(config["shared_trunk_size_1"], activation="relu")(fused)
    shared = layers.Dense(config["shared_trunk_size_2"], activation="relu")(shared)

    # ============================================================
    # HEADS
    # ============================================================
    def head(name, activation=None):
        return layers.Dense(
            symbol_count,
            activation=activation,
            name=name,
            dtype="float32",
        )(shared)

    outputs = {
        "zscore_1d": head("zscore_1d"),
        "rank_1d": head("rank_1d"),
        "direction_1d": head("direction_1d", activation="sigmoid"),
        "log_return_1d": head("log_return_1d"),
        "log_return_3d": head("log_return_3d"),
        "log_return_5d": head("log_return_5d"),
    }

    model = models.Model(
        inputs=[stock_in, news_in],
        outputs=outputs,
    )

    model.compile(
        optimizer=optimizers.Adam(config["learning_rate"]),
        loss={
            "zscore_1d": config["loss_zscore_1d"],
            "rank_1d": config["loss_rank_1d"],
            "direction_1d": config["loss_direction_1d"],
            "log_return_1d": config["loss_log_return_1d"],
            "log_return_3d": config["loss_log_return_3d"],
            "log_return_5d": config["loss_log_return_5d"],
        },
        loss_weights={
            "zscore_1d": config["loss_weight_zscore_1d"],
            "rank_1d": config["loss_weight_rank_1d"],
            "direction_1d": config["loss_weight_direction_1d"],
            "log_return_1d": config["loss_weight_log_return_1d"],
            "log_return_3d": config["loss_weight_log_return_3d"],
            "log_return_5d": config["loss_weight_log_return_5d"],
        },
    )

    return model, config
