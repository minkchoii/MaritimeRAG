"""
Calculate Carbon Intensity Indicator (CII) from voyage fuel consumption data.

Unit flow (IMO-style simplified):
  fuel_consumed_ton [t fuel]
    × co2_factor_kg_per_kg_fuel [kg CO2 / kg fuel]
    → total_co2_ton [t CO2]

  Because 1 t fuel = 1,000 kg fuel:
    CO2 [kg] = fuel_consumed_ton × 1,000 × co2_factor_kg_per_kg_fuel
    CO2 [t]  = CO2 [kg] / 1,000 = fuel_consumed_ton × co2_factor_kg_per_kg_fuel

  total_co2_g [g CO2] = total_co2_ton × 1,000,000

  transport_work [t·nm] = dwt [t] × distance_nm [nm]

  cii_gco2_per_dwt_nm [g CO2 / (t·nm)] = total_co2_g / transport_work
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = (
    "voyage_id",
    "ship_id",
    "fuel_consumed_ton",
    "co2_factor_kg_per_kg_fuel",
    "distance_nm",
    "dwt",
)

GRAMS_PER_TON = 1_000_000


def validate_input(df: pd.DataFrame, input_path: Path) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in {input_path}: {', '.join(missing)}"
        )


def calculate_cii(df: pd.DataFrame) -> pd.DataFrame:
    fuel_ton = df["fuel_consumed_ton"].astype(float)
    co2_factor = df["co2_factor_kg_per_kg_fuel"].astype(float)
    distance_nm = df["distance_nm"].astype(float)
    dwt = df["dwt"].astype(float)

    # kg CO2 = (t fuel × 1000 kg/t) × (kg CO2 / kg fuel) → divide by 1000 → t CO2
    total_co2_ton = fuel_ton * co2_factor
    total_co2_g = total_co2_ton * GRAMS_PER_TON

    transport_work = dwt * distance_nm

    cii = total_co2_g / transport_work
    cii = cii.where(transport_work > 0)

    return pd.DataFrame(
        {
            "voyage_id": df["voyage_id"],
            "ship_id": df["ship_id"],
            "total_co2_ton": total_co2_ton,
            "transport_work": transport_work,
            "cii_gco2_per_dwt_nm": cii,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate CII from voyage fuel consumption CSV."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input voyage CSV path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output CII results CSV path",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(args.input)
    validate_input(df, args.input)

    result = calculate_cii(df)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False, float_format="%.6f")

    print(f"Processed {len(result)} voyage(s)")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
