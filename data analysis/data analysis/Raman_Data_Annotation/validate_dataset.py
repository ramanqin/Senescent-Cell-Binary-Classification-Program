from collections import Counter

from core import calculate_qc_metrics, read_spectrum, scan_spectra


def main():
    records = scan_spectra(r"D:\raw_data", blind_order=False)
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

    assert len(records) == 1000
    assert len(spectrum_uids) == 1000
    assert all(count == 50 for count in subject_counts.values())
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
