# --- STANDARD ---

# --- PROJECT ---
import shared_model as sm

# --- THIRD-PARTY ---
from tensorflow.keras import layers, models  # type: ignore

VAR = {
    "stock_attention_heads": 4,
    "stock_embed": 32,
    "news_attention_heads": 4,
    "news_embed": 32,
    "shared_trunk_size_1": 2048,
    "shared_trunk_size_2": 1024,
}


if VAR["stock_embed"] % VAR["stock_attention_heads"] != 0:
    raise ValueError("stock_embed must be divisible by stock_attention_heads")


if VAR["news_embed"] % VAR["news_attention_heads"] != 0:
    raise ValueError("news_embed must be divisible by news_attention_heads")


def build(
    *, time_steps: int, stock_feature_dim: int, news_feature_dim: int, symbol_count: int
) -> tuple[models.Model, dict]:

    stock_in = layers.Input(
        shape=(time_steps, symbol_count, stock_feature_dim), name="stock"
    )
    news_in = layers.Input(shape=(time_steps, news_feature_dim), name="news")

    # --- per-symbol projection ---
    x = layers.TimeDistributed(
        layers.TimeDistributed(layers.Dense(VAR["stock_embed"], activation="relu"))
    )(stock_in)
    # (batch, time, symbols, embed)

    # --- pool symbols ---
    x = sm.MeanOverSymbols(name="pool_symbols")(x)

    # (batch, time, embed)

    # --- temporal attention only ---
    x = layers.MultiHeadAttention(
        num_heads=VAR["stock_attention_heads"],
        key_dim=VAR["stock_embed"] // VAR["stock_attention_heads"],
    )(x, x)
    # (batch, time, embed)

    # --- pool time ---
    x = layers.GlobalAveragePooling1D()(x)
    # (batch, embed)

    # --- news encoder ---
    y = layers.MultiHeadAttention(
        num_heads=VAR["news_attention_heads"],
        key_dim=VAR["news_embed"],
    )(news_in, news_in)
    y = layers.GlobalAveragePooling1D()(y)

    # --- fuse ---
    x = layers.Concatenate()([x, y])

    # --- shared trunk ---
    shared = layers.Dense(VAR["shared_trunk_size_1"], activation="relu")(x)
    shared = layers.Dense(VAR["shared_trunk_size_2"], activation="relu")(shared)

    # --- heads ---
    def head(name, activation=None):
        return layers.Dense(
            symbol_count,
            activation=activation,
            name=name,
            dtype="float32",
        )(shared)

    model = models.Model(
        inputs=[stock_in, news_in],
        outputs=[
            head("zscore_1d"),
            head("rank_1d"),
            head("direction_1d", activation="sigmoid"),
            head("log_return_1d"),
            head("log_return_3d"),
            head("log_return_5d"),
        ],
    )

    config = {"variables": VAR}

    return model, config
