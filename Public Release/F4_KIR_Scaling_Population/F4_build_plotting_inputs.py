"""Build the Figure 4 plotting inputs from generated KIR population tables."""

import argparse
from pathlib import Path

import pandas as pd


PLOT_COLUMNS = ["control_ir", "fm550_ir", "kir125x_ir", "kir15x_ir"]


def build_input(source_path: Path, output_path: Path) -> None:
    source = pd.read_csv(source_path)
    missing = set(PLOT_COLUMNS).difference(source.columns)
    if missing:
        raise ValueError(f"{source_path} is missing columns: {sorted(missing)}")
    source.loc[:, PLOT_COLUMNS].to_csv(output_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kir-dmsn", type=Path, required=True)
    parser.add_argument("--kir-imsn", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    build_input(args.kir_dmsn, args.output_dir / "Figure4_Input_IR_DMSN.csv")
    build_input(args.kir_imsn, args.output_dir / "Figure4_Input_IR_IMSN.csv")


if __name__ == "__main__":
    main()