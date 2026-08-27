from __future__ import annotations

import json
import shutil
import traceback
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import pandas as pd

from parameter_finding import Filter_Data_Collect, Index_Comparison


def _passes_minimum(value, minimum):
    """下限<=0表示关闭该条件；启用时缺失值一律不通过。"""
    return minimum <= 0 or (pd.notna(value) and value >= minimum)


def _passes_maximum(value, maximum):
    """峰宽上限<=0或旧版哨兵值10000表示关闭；启用时缺失值不通过。"""
    return maximum <= 0 or maximum >= 10000 or (pd.notna(value) and value <= maximum)


def validate_paths(input_path, output_path):
    """确保输入、输出独立，避免递归扫描到自身结果。"""
    if not str(input_path).strip():
        raise ValueError("输入文件夹为空")
    if not str(output_path).strip():
        raise ValueError("输出文件夹为空")

    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()
    if not input_path.exists():
        raise ValueError("输入文件夹不存在")
    if not input_path.is_dir():
        raise ValueError("输入路径不是文件夹")
    if input_path == output_path:
        raise ValueError("输出文件夹不能和输入文件夹相同")
    if output_path in input_path.parents or input_path in output_path.parents:
        raise ValueError("输入文件夹和输出文件夹不能互为父子目录")


def _create_run_directory(output_root):
    """为每次清洗建立独立目录，绝不覆盖或删除以往结果。"""
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = output_root / f"cleaning_run_{timestamp}"
    suffix = 1
    while candidate.exists():
        candidate = output_root / f"cleaning_run_{timestamp}_{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate


def _evaluate_result(result, params):
    """逐项返回清洗条件，供最终判断和运行清单记录。"""
    return {
        "Finger_SNR": result["Finger_SNR"] >= params.Finger_Min_SNR,
        "Finger_Peak_STD": result["Finger_Peak_STD"] <= params.Finger_Peak_Max_STD,
        "Finger_L_Noise_STD": result["Finger_L_Noise_STD"] <= params.Finger_Noise_Max_STD,
        "Finger_R_Noise_STD": result["Finger_R_Noise_STD"] <= params.Finger_Noise_Max_STD,
        "Finger_Peak_Height": _passes_minimum(result["Finger_3_Height"], params.Finger_Peak_Min_Height),
        "Finger_Peak_Min_Width": _passes_minimum(result["Finger_3_Width"], params.Finger_Peak_Min_Length),
        "Finger_Peak_Max_Width": _passes_maximum(result["Finger_3_Width"], params.Finger_Peak_Max_Length),
        "Silence_SNR": result["Silence_SNR"] >= params.Silence_Min_SNR,
        "Silence_STD": result["Silence_STD"] <= params.Silence_Max_STD,
        "CH_SNR": result["CH_SNR"] >= params.CH_Min_SNR,
        "CH_Peak_STD": result["CH_Peak_STD"] <= params.CH_Peak_Max_STD,
        "CH_L_Noise_STD": result["CH_L_Noise_STD"] <= params.CH_Noise_Max_STD,
        "CH_R_Noise_STD": result["CH_R_Noise_STD"] <= params.CH_Noise_Max_STD,
        "CH_Peak_Height": _passes_minimum(result["CH_Peak_Height"], params.CH_Peak_Min_Height),
        "CH_Peak_Width": _passes_minimum(result["CH_Peak_Width"], params.CH_Peak_Min_Length),
        "Peak_Height_Ratio": _passes_minimum(result["Peak_Height_Ratio_3"], params.Peak_Height_Ratio),
    }


def Back_Filter_Excute(params, log_function, progress_function):
    """执行一次不可覆盖、可追溯的批量清洗。"""
    input_root = Path(params.Input_Folder_Path).resolve()
    output_root = Path(params.Output_Folder_Path).resolve()
    validate_paths(input_root, output_root)

    input_files = sorted(input_root.rglob("*.txt"))
    if not input_files:
        log_function("\n输入文件夹中没有找到 txt 文件。\n")
        return None

    run_dir = _create_run_directory(output_root)
    passed_dir = run_dir / "passed"
    failed_dir = run_dir / "failed"
    run_id = run_dir.name
    log_function(f"\n本次结果目录：{run_dir}\n")

    passed_files = []
    failed_files = []
    records = []
    counters = {"finger": 0, "silence": 0, "ch": 0, "ratio": 0}

    for processed_count, txt_path in enumerate(input_files, start=1):
        relative_path = txt_path.relative_to(input_root)
        record = {
            "run_id": run_id,
            "stage": "cleaning",
            "source_root": str(input_root),
            "source_path": str(txt_path.resolve()),
            "relative_path": relative_path.as_posix(),
            "output_path": "",
            "status": "error",
            "failed_rules": "",
            "error": "",
        }
        try:
            current_df = pd.read_csv(
                txt_path,
                sep="\t",
                header=None,
                encoding="utf-8",
                names=["Raman_Shift", "Raman_Intensity"],
            )
            finger = Index_Comparison(current_df, params.Finger_Start, params.Finger_End)
            silence = Index_Comparison(current_df, params.Silence_Start, params.Silence_End)
            ch = Index_Comparison(current_df, params.CH_Start, params.CH_End)
            result = Filter_Data_Collect(params, txt_path, *finger, *silence, *ch)
            checks = _evaluate_result(result, params)

            finger_pass = all(checks[name] for name in checks if name.startswith("Finger_"))
            silence_pass = checks["Silence_SNR"] and checks["Silence_STD"]
            ch_pass = all(checks[name] for name in checks if name.startswith("CH_"))
            ratio_pass = checks["Peak_Height_Ratio"]
            passed = finger_pass and silence_pass and ch_pass and ratio_pass

            counters["finger"] += int(finger_pass)
            counters["silence"] += int(silence_pass)
            counters["ch"] += int(ch_pass)
            counters["ratio"] += int(ratio_pass)

            record.update(result)
            record.update({f"check_{name}": bool(value) for name, value in checks.items()})
            record.update(
                {
                    "status": "pass" if passed else "fail",
                    "failed_rules": ";".join(name for name, value in checks.items() if not value),
                    "pass_finger": finger_pass,
                    "pass_silence": silence_pass,
                    "pass_ch": ch_pass,
                    "pass_ratio": ratio_pass,
                }
            )
            (passed_files if passed else failed_files).append(txt_path)
        except Exception as exc:
            failed_files.append(txt_path)
            record.update(
                {
                    "status": "error",
                    "failed_rules": "processing_error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            log_function(f"\n文件处理失败，已归入failed：{txt_path}\n")
            log_function(traceback.format_exc())
        records.append(record)

        if processed_count % 10 == 0 or processed_count == len(input_files):
            progress_function(
                "进度：{}/{}，{:.2%}      ".format(
                    processed_count, len(input_files), processed_count / len(input_files)
                )
            )

    record_by_source = {record["source_path"]: record for record in records}
    for status_dir, paths in ((passed_dir, passed_files), (failed_dir, failed_files)):
        for source_path in paths:
            relative_path = source_path.relative_to(input_root)
            export_path = status_dir / relative_path
            export_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, export_path)
            record_by_source[str(source_path.resolve())]["output_path"] = str(export_path.resolve())

    manifest_path = run_dir / "run_manifest.csv"
    pd.DataFrame(records).to_csv(manifest_path, index=False, encoding="utf-8-sig")
    parameters = asdict(params) if hasattr(params, "__dataclass_fields__") else vars(params)
    (run_dir / "run_parameters.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "stage": "cleaning",
                "source_root": str(input_root),
                "output_root": str(run_dir),
                "parameters": parameters,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    total = len(input_files)
    passed_count = len(passed_files)
    log_function(
        "\n原有{}个样本；现有{}个文件达到了标准；通过率{:.2%}".format(
            total, passed_count, passed_count / total
        )
    )
    log_function(
        "\n通过率：\n"
        "指纹区R1通过率{:.2%}；\n"
        "静默区R2通过率{:.2%}；\n"
        "C-H峰R3通过率{:.2%}；\n"
        "峰比R4通过率{:.2%}".format(
            counters["finger"] / total,
            counters["silence"] / total,
            counters["ch"] / total,
            counters["ratio"] / total,
        )
    )
    log_function(f"\n运行清单：{manifest_path}\n")
    log_function(f"通过光谱目录：{passed_dir}\n")
    return run_dir
