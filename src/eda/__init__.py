"""
Exploratory Data Analysis (EDA) package.

This package contains reusable functions for exploring the
used phone price dataset.
"""

from .overview import (
    dataset_info,
    summary_statistics,
    missing_values,
    duplicated_rows,
)

from .correlations import (
    resale_price_correlations,
    add_depreciation_column,
)

from .distributions import (
    resale_price_distribution,
    depreciation_distribution,
)

from .time_analysis import (
    describe_time_columns,
    plot_time_histograms,
    time_correlations,
)

from .numerical_analysis import (
    original_vs_resale,
    age_vs_resale,
    age_vs_depreciation,
    battery_health_analysis,
    hardware_correlations,
)

from .condition_analysis import (
    condition_price,
    condition_summary,
    condition_crosstabs,
)

from .binary_analysis import (
    binary_feature_summary,
)

from .categorical_analysis import (
    brand_prices,
    model_prices,
    seller_prices,
    city_prices,
    os_prices,
    brand_os_table,
    brand_os_summary,
    list_models,
    apple_models,
)

__all__ = [
    "dataset_info",
    "summary_statistics",
    "missing_values",
    "duplicated_rows",
    "resale_price_correlations",
    "add_depreciation_column",
    "resale_price_distribution",
    "depreciation_distribution",
    "describe_time_columns",
    "plot_time_histograms",
    "time_correlations",
    "original_vs_resale",
    "age_vs_resale",
    "age_vs_depreciation",
    "battery_health_analysis",
    "hardware_correlations",
    "condition_price",
    "condition_summary",
    "condition_crosstabs",
    "binary_feature_summary",
    "brand_prices",
    "model_prices",
    "seller_prices",
    "city_prices",
    "os_prices",
    "brand_os_table",
    "brand_os_summary",
    "list_models",
    "apple_models",
]