# %% [markdown]
# # Cleaning ETH and RTH Market Data
# We combine extended trading hours (ETH) and regular trading hours (RTH) datasets into tidy
# assets and option dataframes ready for downstream analysis.

# %% [markdown]
# ## Dataset Overview
# - **RTH vs ETH**: RTH covers the standard 09:30–16:00 ET NYSE session, while ETH captures the
#   electronic session surrounding the close and open; both are needed to understand intraday shocks.
# - **SPX**: The S&P 500 index level observed from cash equities.
# - **VIX**: CBOE Volatility Index, capturing market-implied 30-day variance.
# - **ES futures**: Front-month E-mini S&P 500 futures with bid/ask quotes that proxy overnight SPX moves.
# - **SPX options**: Listed SPX index options with call/put strikes across expiries, quoted with bid/ask prices.

# %% [markdown]
# ## Imports and Paths

# %%
from pathlib import Path

import pandas as pd

pd.options.display.width = 140
pd.options.display.max_columns = 20

PROJECT_ROOT = Path(".")
SESSION_FOLDERS = {"ETH": PROJECT_ROOT / "ETH", "RTH": PROJECT_ROOT / "RTH"}

# %% [markdown]
# ## Helper Functions


# %%
# Helper routines: parsing CSVs, melting option quotes, and orchestrating session-level loads.
def parse_timestamped_csv(csv_path: Path) -> pd.DataFrame:
    """Load a CSV whose first column stores timestamps and return a tidy timestamp column."""
    frame = pd.read_csv(csv_path, index_col=0)
    frame.index = pd.to_datetime(frame.index)
    frame = frame.reset_index().rename(columns={"index": "timestamp"})
    return frame


def melt_option_quotes(frame: pd.DataFrame, value_name: str) -> pd.DataFrame:
    """Convert wide strike columns into long format for a single quote side."""
    melted = frame.melt(id_vars="timestamp", var_name="strike", value_name=value_name)
    melted["strike"] = pd.to_numeric(melted["strike"], errors="coerce")
    melted[value_name] = pd.to_numeric(melted[value_name], errors="coerce")
    return melted


def load_option_book(option_type: str) -> pd.DataFrame:
    """Load and merge bid/ask quotes for the requested option type across sessions."""
    records: list[pd.DataFrame] = []
    for session, folder in SESSION_FOLDERS.items():
        ask_files = sorted(folder.glob(f"*_{option_type}_ask_*.csv"))
        for ask_path in ask_files:
            stem_parts = ask_path.stem.split("_")
            quote_date = pd.to_datetime(stem_parts[0])
            expiry = pd.to_datetime(stem_parts[-1])
            bid_path = ask_path.with_name(ask_path.name.replace("_ask_", "_bid_"))
            if not bid_path.exists():
                continue

            ask_frame = melt_option_quotes(parse_timestamped_csv(ask_path), "ask")
            bid_frame = melt_option_quotes(parse_timestamped_csv(bid_path), "bid")
            merged = pd.merge(
                ask_frame, bid_frame, on=["timestamp", "strike"], how="outer"
            )

            merged["expiry"] = expiry
            merged["session"] = session
            merged["mid"] = merged[["bid", "ask"]].mean(axis=1)
            # merged = merged.dropna(subset=["strike"])
            # merged = merged.dropna(subset=["bid", "ask"], how="all")

            records.append(merged)

    long_df = pd.concat(records, ignore_index=True)
    long_df = long_df.sort_values(["timestamp", "strike"]).reset_index(drop=True)
    return long_df


def load_assets() -> pd.DataFrame:
    """Load SPX/ES/VIX snapshots from both sessions."""
    frames: list[pd.DataFrame] = []
    for session, folder in SESSION_FOLDERS.items():
        for csv_path in sorted(folder.glob("*_assets.csv")):
            frame = parse_timestamped_csv(csv_path)
            frame["session"] = session
            frames.append(frame)

    assets = pd.concat(frames, ignore_index=True)
    assets = assets.astype(
        {
            "SPX": "Float64",
            "ES_BID": "Float64",
            "ES_ASK": "Float64",
            "VIX": "Float64",
            "session": "category",
        }
    )
    assets = assets.sort_values("timestamp").reset_index(drop=True)
    return assets


# %% [markdown]
# ## Load Assets Across Sessions

# %%
# Load, clean, and combine SPX, ES, and VIX snapshots from ETH and RTH.
assets_df = load_assets()
assets_df.head()

# %%
assets_df.dtypes

# %% [markdown]
# ## Load Call Option Quotes

# %%
# Build the consolidated call option dataframe with bid/ask/mid quotes.
call_options_df = load_option_book("call")
call_options_df.head()

# %% [markdown]
# ## Load Put Option Quotes

# %%
# Build the consolidated put option dataframe with bid/ask/mid quotes.
put_options_df = load_option_book("put")
put_options_df.head()
