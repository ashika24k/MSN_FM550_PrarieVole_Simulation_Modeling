"""Build the four Figure 5-6 plotting inputs from channel population tables."""

import argparse
from pathlib import Path

import pandas as pd


CHANNELS = ("kir", "kaf", "kas", "kdr", "naf", "sk", "bk")
CELL_TYPES = {"DMSN": "dmsn", "IMSN": "imsn"}
SCALES = (("1.25", "125x"), ("1.5", "15x"))


def read_channel_table(source_root: Path, channel: str, cell_type: str) -> pd.DataFrame:
    path = source_root / channel / f"MSN_Population_Analysis_Results_125x15x_{cell_type}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing generated population table: {path}")
    return pd.read_csv(path)


def build_input(source_root: Path, output_dir: Path, cell_label: str, cell_type: str, scale_label: str, scale_suffix: str) -> None:
    tables = {channel: read_channel_table(source_root, channel, cell_label) for channel in CHANNELS}
    reference = tables["kir"]
    required = {"fm550_ir", f"kir{scale_suffix}_ir"}
    if not required.issubset(reference.columns):
        raise ValueError(f"KIR table lacks required columns: {sorted(required.difference(reference.columns))}")

    result = pd.DataFrame({"FM550": reference["fm550_ir"]})
    for channel, table in tables.items():
        column = f"{channel}{scale_suffix}_ir"
        if column not in table.columns:
            raise ValueError(f"{channel} table lacks required column: {column}")
        if len(table) != len(reference):
            raise ValueError(f"{channel} table has {len(table)} rows; expected {len(reference)}")
        result[channel] = table[column]

    output_name = f"Figure5&6_Input_IR_{cell_type}_{scale_label}.csv"
    result.to_csv(output_dir / output_name, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).parent / "generation")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for cell_label, cell_type in CELL_TYPES.items():
        for scale_label, scale_suffix in SCALES:
            build_input(args.source_root, args.output_dir, cell_label, cell_type, scale_label, scale_suffix)


if __name__ == "__main__":
    main()