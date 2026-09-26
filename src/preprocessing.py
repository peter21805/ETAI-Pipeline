"""
Preprocessing -- same file, same job as week 2 (raw data in, model-ready train/test
split out), now carrying the diagnosis-driven, leak-safe recipe this week's EDA
notebook (01_eda_introduction.ipynb) justifies: category cleanup, domain-rule/placeholder
-> NaN conversion, de-duplication, redundant-column removal, MNAR imputation with a
`_was_missing` indicator, and the train/test split itself.

MCAR columns are NOT imputed yet -- their rows with missing values are still dropped,
as in week 2.

Two things every function here respects, on purpose:
  - leak-safe: `clean_dataset` and `split_features_target` are target- and
    split-independent, so they're safe to run on the whole dataset before splitting.
    `split_train_test` is the boundary line -- everything after it (MNAR imputation and
    encoding, inside `build_preprocessor`'s ColumnTransformer) is fit only on the
    training fold, never on data it's about to be evaluated against.
  - deployable from day one: every function up to (not including) the split is
    target-column-agnostic -- `y` comes back as `None` and nothing else breaks when
    called on label-free inference data, which never gets split at all.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from src.data_diagnostics import flag_invalid_values


def _canonicalize_categories(df: pd.DataFrame, columns_and_maps: dict, placeholder_tokens: set) -> pd.DataFrame:
    out = df.copy()
    for col, mapping in columns_and_maps.items():
        if col not in out.columns:
            continue
        cleaned = out[col].astype(str).str.strip()
        lowered = cleaned.str.lower()
        out[col] = lowered.map(mapping).fillna(cleaned)
        out.loc[out[col].astype(str).str.strip().isin(placeholder_tokens), col] = np.nan
    return out


def clean_dataset(df: pd.DataFrame, diagnostics_config: dict) -> pd.DataFrame:
    """
    Applies this week's diagnosis: category cleanup, domain-rule/placeholder -> NaN
    conversion, de-duplication, and redundant-column removal. Target-agnostic -- safe
    to call on label-free inference data, since none of this depends on a target column.
    """
    out = df.copy()
    placeholder_tokens = set(diagnostics_config.get("placeholder_tokens", []))

    # numeric columns that load as text purely because of a placeholder token
    for col in diagnostics_config.get("numeric_text_columns", []):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col].replace(list(placeholder_tokens), np.nan), errors="coerce")

    flag_invalid_values(out, diagnostics_config.get("validity_rules", {}))

    out = _canonicalize_categories(out, diagnostics_config.get("canonical_categories", {}), placeholder_tokens)

    out = out.drop_duplicates()
    id_column = diagnostics_config.get("id_column")
    if id_column and id_column in out.columns:
        out = out.drop_duplicates(subset=id_column, keep="first")

    columns_to_drop = [c for c in diagnostics_config.get("redundant_columns", []) if c in out.columns]
    out = out.drop(columns=columns_to_drop)

    # MCAR columns: no imputation yet -- drop those rows, as in week 2
    out = out.dropna(subset=[c for c in diagnostics_config.get("mcar_columns", []) if c in out.columns])

    return out


def add_missingness_indicators(df: pd.DataFrame, mnar_indicator_sources: list) -> pd.DataFrame:
    """Adds a `<col>_was_missing` flag for each MNAR-diagnosed column, before that
    column gets imputed -- so a model can still see the pattern even though the fill
    value itself (median/mode) can't carry it. Target-agnostic."""
    out = df.copy()
    for col in mnar_indicator_sources:
        if col in out.columns:
            out[f"{col}_was_missing"] = out[col].isna().astype(int)
    return out


def split_features_target(df: pd.DataFrame, data_config: dict, mnar_indicator_sources: list):
    """
    Returns (X, y, extras). `y` is `None` and `extras` has no target column when called
    on label-free inference data -- nothing downstream requires the target to be present.
    """
    target = data_config["target"]
    sensitive_attr = data_config["sensitive_attr"]
    drop_columns = data_config.get("drop_columns", [])

    df = add_missingness_indicators(df, mnar_indicator_sources)
    y = df[target] if target in df.columns else None

    extras_cols = [c for c in [sensitive_attr, "score_text"] if c in df.columns]
    extras = df[extras_cols].copy() if extras_cols else None

    always_drop = set(drop_columns) | {target, sensitive_attr}
    feature_cols = [c for c in df.columns if c not in always_drop]
    X = df[feature_cols]
    return X, y, extras


def build_preprocessor(preprocessing_config: dict) -> ColumnTransformer:
    """
    Builds a leak-safe ColumnTransformer: only the MNAR columns are imputed (median /
    mode), everything else is passed through or one-hot encoded as-is. Fit happens on
    the training fold only, via the surrounding sklearn Pipeline's own fit/transform
    discipline.
    """
    numeric_features = preprocessing_config["numeric_features"]
    categorical_features = preprocessing_config["categorical_features"]
    mnar_numeric = preprocessing_config.get("mnar_numeric", [])
    mnar_categorical = preprocessing_config.get("mnar_categorical", [])
    mnar_indicator_sources = preprocessing_config.get("mnar_indicator_sources", [])
    imputation = preprocessing_config.get("imputation", {})

    mnar_categorical_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy=imputation.get("categorical_strategy", "most_frequent"))),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    indicator_cols = [f"{c}_was_missing" for c in mnar_indicator_sources]

    return ColumnTransformer([
        ("mnar_numeric", SimpleImputer(strategy=imputation.get("numeric_strategy", "median")), mnar_numeric),
        ("mnar_categorical", mnar_categorical_pipeline, mnar_categorical),
        ("numeric", "passthrough", numeric_features),
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ("indicators", "passthrough", indicator_cols),
    ])


def split_train_test(X, y, extras, test_size: float, random_state: int):
    """
    Stratified split of X, y, and the extras frame (race/score_text, kept aside for the
    fairness report) together, so all three stay row-aligned. This is the leak-safe
    boundary line -- everything downstream (imputation and encoding, inside
    build_preprocessor's ColumnTransformer) may only ever be fit on X_train, never on
    X_test or the full dataset.
    """
    X_train, X_test, y_train, y_test, extras_train, extras_test = train_test_split(
        X, y, extras, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test, extras_train, extras_test
