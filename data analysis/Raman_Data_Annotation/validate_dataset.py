import argparse
from collections import Counter
from pathlib import Path

from core import calculate_qc_metrics, read_spectrum, scan_spectra


def main():
    parser = argparse.ArgumentParser(description="Validate a Raman spectrum directory")
    parser.add_argument("data", type=Path, help="Folder containing Raman TXT spectra")
    parser.add_argument("--expected-files", type=int)
    parser.add_argument("--expected-subjects", type=int)
    parser.add_argument("--expected-spectra-per-subject", type=int)
    args = parser.parse_args()
    records = scan_spectra(args.data, blind_order=False)
    subject_counts = Counter((row["class_original"], row["subject_id"]) for row in records)
    spectrum_uids = {row["spectrum_uid"] for row in records}

    errors = []
    axis_status = Counter()
    shapes = Counter()
    ranges = Counter()
    missing_count = 0

    for record in records:
        try:
            spectrum = read_spectrum(record["file_absolute_path"])
            metrics = calculate_qc_metrics(spectrum)
            shapes[metrics["n_points"]] += 1
            ranges[(round(metrics["x_min"], 2), round(metrics["x_max"], 2))] += 1
            axis_status[metrics["axis_monotonic"]] += 1
            missing_count += metrics["missing_count"]
        except Exception as exc:
            errors.append((record["file_relative_path"], str(exc)))

    if args.expected_files is not None:
        assert len(records) == args.expected_files
        assert len(spectrum_uids) == args.expected_files
    if args.expected_subjects is not None:
        assert len(subject_counts) == args.expected_subjects
    if args.expected_spectra_per_subject is not None:
        assert all(count == args.expected_spectra_per_subject for count in subject_counts.values())
    assert not errors

    print("FULL_VALIDATION_OK")
    print("files", len(records), "subjects", len(subject_counts), "unique_ids", len(spectrum_uids))
    print("shapes", dict(shapes))
    print("ranges", dict(ranges))
    print("axis", dict(axis_status), "nonfinite", missing_count, "errors", len(errors))
    print("tif_linked", sum(bool(row["tif_path"]) for row in records))
    print("png_linked", sum(bool(row["png_path"]) for row in records))


if __name__ == "__main__":
    main()
