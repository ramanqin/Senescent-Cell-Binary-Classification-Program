"""改进版002拉曼光谱标注工具的命令行入口。"""

# 【衰老数据改动】增加任意文件夹输入、0/1组独立配额、
# 可复现抽样参数，以及无需打开GUI的数据扫描汇总模式。

import argparse
from pathlib import Path

from core import scan_spectra, summarize_records


def main():
    parser = argparse.ArgumentParser(description="Raman spectrum QC annotation tool with recursive folder sampling")
    parser.add_argument("--data", default=r"D:\raw_data", help="Any folder containing TXT spectra")
    parser.add_argument("--output", default=r"D:\raman_annotation_results\raman_qc_annotations_group01.csv")
    parser.add_argument("--group0-size", type=int, default=50, help="Young/0 quota; -1 means all, 0 means none")
    parser.add_argument("--group1-size", type=int, default=50, help="Aging/1 quota; -1 means all, 0 means none")
    parser.add_argument("--unknown-size", type=int, default=0, help="Unknown-label quota; -1 means all")
    parser.add_argument("--seed", type=int, default=20260813, help="Reproducible sampling seed")
    parser.add_argument("--strategy", choices=["balanced", "random"], default="balanced")
    parser.add_argument("--scan", action="store_true", help="Validate and summarize the dataset without opening the GUI")
    args = parser.parse_args()

    if args.scan:
        records = scan_spectra(
            Path(args.data),
            blind_order=False,
            seed=args.seed,
            strategy=args.strategy,
            group0_size=args.group0_size,
            group1_size=args.group1_size,
            unknown_size=args.unknown_size,
        )
        print(summarize_records(records).to_string(index=False))
        print(f"\nTotal spectra: {len(records)}")
        return

    from gui import run_app

    run_app(args.data, args.output, args.group0_size, args.group1_size, args.unknown_size, args.seed, args.strategy)


if __name__ == "__main__":
    main()
