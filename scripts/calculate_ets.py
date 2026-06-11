"""
Estimate EU ETS cost from voyage CO2 emissions.

Unit flow:
  total_co2_ton [t CO2]
    from column total_co2_ton, OR
    fuel_consumed_ton [t fuel] × co2_factor_kg_per_kg_fuel [kg CO2 / kg fuel]

  ets_covered_co2_ton [t CO2] = total_co2_ton × ets_coverage_rate [–]

  estimated_ets_cost_eur [EUR] = ets_covered_co2_ton × eua_price_eur_per_tco2 [EUR / t CO2]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ETS_REQUIRED_COLUMNS = (
    "voyage_id",
    "ship_id",
    "ets_coverage_rate",
    "eua_price_eur_per_tco2",
)


def validate_input(df: pd.DataFrame, input_path: Path) -> None:
    missing = [col for col in ETS_REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in {input_path}: {', '.join(missing)}"
        )

    has_total = "total_co2_ton" in df.columns
    has_fuel = "fuel_consumed_ton" in df.columns and "co2_factor_kg_per_kg_fuel" in df.columns
    if not has_total and not has_fuel:
        raise ValueError(
            f"Input must include total_co2_ton or "
            f"(fuel_consumed_ton + co2_factor_kg_per_kg_fuel): {input_path}"
        )


def resolve_total_co2_ton(df: pd.DataFrame) -> pd.Series:
    if "total_co2_ton" in df.columns:
        total = df["total_co2_ton"].astype(float)
        if "fuel_consumed_ton" in df.columns and "co2_factor_kg_per_kg_fuel" in df.columns:
            fuel_ton = df["fuel_consumed_ton"].astype(float)
            co2_factor = df["co2_factor_kg_per_kg_fuel"].astype(float)
            computed = fuel_ton * co2_factor
            return total.fillna(computed)
        return total

    fuel_ton = df["fuel_consumed_ton"].astype(float)
    co2_factor = df["co2_factor_kg_per_kg_fuel"].astype(float)
    return fuel_ton * co2_factor


def calculate_ets(df: pd.DataFrame) -> pd.DataFrame:
    total_co2_ton = resolve_total_co2_ton(df)
    coverage = df["ets_coverage_rate"].astype(float)
    eua_price = df["eua_price_eur_per_tco2"].astype(float)

    ets_covered_co2_ton = total_co2_ton * coverage
    estimated_ets_cost_eur = ets_covered_co2_ton * eua_price

    return pd.DataFrame(
        {
            "voyage_id": df["voyage_id"],
            "ship_id": df["ship_id"],
            "total_co2_ton": total_co2_ton,
            "ets_covered_co2_ton": ets_covered_co2_ton,
            "eua_price_eur_per_tco2": eua_price,
            "estimated_ets_cost_eur": estimated_ets_cost_eur,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate EU ETS cost from voyage CO2 emissions CSV."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input voyage CSV (raw voyage data or CII result with CO2 columns)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output ETS cost CSV path",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(args.input)
    validate_input(df, args.input)

    result = calculate_ets(df)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False, float_format="%.6f")

    print(f"Processed {len(result)} voyage(s)")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
