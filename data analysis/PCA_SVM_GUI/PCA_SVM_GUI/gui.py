from __future__ import annotations

import os
from pathlib import Path
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from pca_svm_analysis import run_analysis


def split_aliases(text: str) -> list[str]:
    return [value.strip() for value in text.replace("，", ",").split(",") if value.strip()]


class PCA_SVM_App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("通用PCA-SVM光谱分析")
        self.geometry("780x550")
        self.minsize(720, 510)

        self.input_dir = tk.StringVar()
        self.negative_name = tk.StringVar(value="年轻")
        self.negative_aliases = tk.StringVar(value="年轻,年轻细胞,young")
        self.positive_name = tk.StringVar(value="衰老")
        self.positive_aliases = tk.StringVar(value="衰老,衰老细胞,aging")
        self.outer_splits = tk.IntVar(value=5)
        self.inner_splits = tk.IntVar(value=4)
        self.pca_variance = tk.DoubleVar(value=0.95)
        self.auto_align_grid = tk.BooleanVar(value=True)
        self.resample_step = tk.DoubleVar(value=3.0)
        self.status_text = tk.StringVar(value="请选择预处理光谱根目录。")
        self.result_dir: Path | None = None

        self._build_ui()

    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=18)
        root.pack(fill="both", expand=True)
        root.columnconfigure(1, weight=1)

        ttk.Label(root, text="PCA-SVM光谱二分类", font=("Microsoft YaHei", 17, "bold")).grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 18)
        )

        ttk.Label(root, text="光谱根目录").grid(row=1, column=0, sticky="w", pady=6)
        ttk.Entry(root, textvariable=self.input_dir).grid(row=1, column=1, sticky="ew", padx=8)
        ttk.Button(root, text="选择文件夹", command=self._choose_folder).grid(row=1, column=2)

        ttk.Label(root, text="阴性类别名称").grid(row=2, column=0, sticky="w", pady=6)
        ttk.Entry(root, textvariable=self.negative_name, width=14).grid(row=2, column=1, sticky="w", padx=8)
        ttk.Label(root, text="例如：年轻").grid(row=2, column=2, sticky="w")

        ttk.Label(root, text="阴性目录别名").grid(row=3, column=0, sticky="w", pady=6)
        ttk.Entry(root, textvariable=self.negative_aliases).grid(row=3, column=1, columnspan=2, sticky="ew", padx=8)

        ttk.Label(root, text="阳性类别名称").grid(row=4, column=0, sticky="w", pady=6)
        ttk.Entry(root, textvariable=self.positive_name, width=14).grid(row=4, column=1, sticky="w", padx=8)
        ttk.Label(root, text="例如：衰老").grid(row=4, column=2, sticky="w")

        ttk.Label(root, text="阳性目录别名").grid(row=5, column=0, sticky="w", pady=6)
        ttk.Entry(root, textvariable=self.positive_aliases).grid(row=5, column=1, columnspan=2, sticky="ew", padx=8)

        options = ttk.LabelFrame(root, text="分析参数", padding=10)
        options.grid(row=6, column=0, columnspan=3, sticky="ew", pady=(14, 8))
        ttk.Label(options, text="外层折数").grid(row=0, column=0, padx=(0, 5))
        ttk.Spinbox(options, from_=2, to=10, width=6, textvariable=self.outer_splits).grid(row=0, column=1)
        ttk.Label(options, text="内层折数").grid(row=0, column=2, padx=(20, 5))
        ttk.Spinbox(options, from_=2, to=10, width=6, textvariable=self.inner_splits).grid(row=0, column=3)
        ttk.Label(options, text="PCA保留方差").grid(row=0, column=4, padx=(20, 5))
        ttk.Entry(options, width=8, textvariable=self.pca_variance).grid(row=0, column=5)

        grid_options = ttk.Frame(root)
        grid_options.grid(row=7, column=0, columnspan=3, sticky="w", pady=5)
        ttk.Checkbutton(
            grid_options,
            text="波数网格不一致时自动统一",
            variable=self.auto_align_grid,
        ).pack(side="left")
        ttk.Label(grid_options, text="公共网格步长").pack(side="left", padx=(18, 5))
        ttk.Entry(grid_options, width=7, textvariable=self.resample_step).pack(side="left")
        ttk.Label(grid_options, text="cm⁻¹").pack(side="left", padx=4)

        ttk.Label(
            root,
            text="结果固定保存到所选光谱目录下的 result_plot 文件夹。",
            foreground="#555555",
        ).grid(row=8, column=0, columnspan=3, sticky="w", pady=5)

        buttons = ttk.Frame(root)
        buttons.grid(row=9, column=0, columnspan=3, sticky="ew", pady=(12, 8))
        self.run_button = ttk.Button(buttons, text="开始分析", command=self._start)
        self.run_button.pack(side="left")
        self.open_button = ttk.Button(buttons, text="打开结果文件夹", command=self._open_result, state="disabled")
        self.open_button.pack(side="left", padx=10)

        self.progress = ttk.Progressbar(root, mode="indeterminate")
        self.progress.grid(row=10, column=0, columnspan=3, sticky="ew", pady=8)
        ttk.Label(root, textvariable=self.status_text, wraplength=700).grid(
            row=11, column=0, columnspan=3, sticky="w", pady=5
        )

    def _choose_folder(self) -> None:
        selected = filedialog.askdirectory(title="选择预处理光谱根目录")
        if selected:
            self.input_dir.set(selected)
            self.result_dir = Path(selected) / "result_plot"
            self.status_text.set(f"结果将保存到：{self.result_dir}")

    def _make_config(self) -> dict:
        input_path = Path(self.input_dir.get().strip()).resolve()
        if not input_path.is_dir():
            raise FileNotFoundError("请选择有效的光谱根目录")
        negative_aliases = split_aliases(self.negative_aliases.get())
        positive_aliases = split_aliases(self.positive_aliases.get())
        if not negative_aliases or not positive_aliases:
            raise ValueError("两个类别都必须至少填写一个目录别名")
        variance = float(self.pca_variance.get())
        if not 0 < variance < 1:
            raise ValueError("PCA保留方差必须在0与1之间，例如0.95")
        step = float(self.resample_step.get())
        if step <= 0:
            raise ValueError("公共网格步长必须大于0")

        self.result_dir = input_path / "result_plot"
        return {
            "input_dir": str(input_path),
            "output_dir": str(self.result_dir),
            "negative_class": {
                "name": self.negative_name.get().strip(),
                "folders": negative_aliases,
            },
            "positive_class": {
                "name": self.positive_name.get().strip(),
                "folders": positive_aliases,
            },
            "extensions": ["txt", "csv", "dat"],
            "sample_depth_after_class": 1,
            "outer_splits": int(self.outer_splits.get()),
            "inner_splits": int(self.inner_splits.get()),
            "random_seed": 42,
            "pca_variance": variance,
            "auto_align_grid": bool(self.auto_align_grid.get()),
            "resample_step": step,
            "class_weight": "balanced",
            "n_jobs": -1,
            "c_values": [0.01, 0.1, 1.0, 10.0, 100.0],
            "gamma_values": ["scale", 0.01, 0.1, 1.0],
            "roc_filename": "ROC.png",
            "grid_tolerance": 1e-6,
            "min_points": 20,
        }

    def _start(self) -> None:
        try:
            config = self._make_config()
        except Exception as exc:
            messagebox.showerror("参数错误", str(exc))
            return
        self.run_button.configure(state="disabled")
        self.open_button.configure(state="disabled")
        self.progress.start(12)
        self.status_text.set("正在读取光谱并进行嵌套交叉验证，请稍候……")
        threading.Thread(target=self._worker, args=(config,), daemon=True).start()

    def _worker(self, config: dict) -> None:
        try:
            result = run_analysis(config)
            self.after(0, self._finished, result)
        except Exception as exc:
            self.after(0, self._failed, exc)

    def _finished(self, result: dict) -> None:
        self.progress.stop()
        self.run_button.configure(state="normal")
        self.open_button.configure(state="normal")
        metrics = result["metrics"]
        grid = result["wave_number_grid"]
        grid_text = (
            f"；已自动统一为{grid['start']:.1f}–{grid['end']:.1f} cm⁻¹、步长{grid['median_step']:.2f}"
            if grid["auto_aligned"] else "；波数网格原本一致"
        )
        self.status_text.set(
            f"完成：AUC={metrics['roc_auc']:.3f}，平衡准确率={metrics['balanced_accuracy']:.3f}；"
            f"结果：{self.result_dir}{grid_text}"
        )
        messagebox.showinfo("分析完成", f"ROC图和结果已保存到：\n{self.result_dir}")

    def _failed(self, exc: Exception) -> None:
        self.progress.stop()
        self.run_button.configure(state="normal")
        self.status_text.set(f"分析失败：{exc}")
        messagebox.showerror("分析失败", str(exc))

    def _open_result(self) -> None:
        if self.result_dir and self.result_dir.is_dir():
            os.startfile(self.result_dir)
        else:
            messagebox.showwarning("结果不存在", "尚未生成结果文件夹")


if __name__ == "__main__":
    PCA_SVM_App().mainloop()
