from __future__ import annotations

import csv
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import queue
import threading
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib import rcParams

rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

from spectral_preprocessor import (
    PreprocessConfig,
    find_spectra,
    preprocess,
    read_spectrum,
    save_spectrum,
)


class SpectrumPreprocessorApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("拉曼光谱批量预处理软件")
        self.geometry("1320x820")
        self.minsize(1120, 700)
        self.stop_event = threading.Event()
        self.events: queue.Queue = queue.Queue()
        self.worker: threading.Thread | None = None
        self.preview_file: Path | None = None

        self._make_variables()
        self._build_ui()
        self.after(100, self._poll_events)

    def _make_variables(self) -> None:
        d = PreprocessConfig()
        self.input_dir = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.extensions = tk.StringVar(value="txt,csv,dat")
        self.recursive = tk.BooleanVar(value=True)
        self.preserve_tree = tk.BooleanVar(value=True)
        self.overwrite = tk.BooleanVar(value=False)
        self.output_format = tk.StringVar(value="txt")
        self.output_suffix = tk.StringVar(value="_预处理")
        self.precision = tk.IntVar(value=8)

        self.vars: dict[str, tk.Variable] = {
            "crop_enabled": tk.BooleanVar(value=d.crop_enabled),
            "ranges": tk.StringVar(value=d.ranges),
            "cosmic_enabled": tk.BooleanVar(value=d.cosmic_enabled),
            "cosmic_window": tk.IntVar(value=d.cosmic_window),
            "cosmic_threshold": tk.DoubleVar(value=d.cosmic_threshold),
            "cosmic_max_width": tk.IntVar(value=d.cosmic_max_width),
            "cosmic_passes": tk.IntVar(value=d.cosmic_passes),
            "resample_enabled": tk.BooleanVar(value=d.resample_enabled),
            "resample_step": tk.DoubleVar(value=d.resample_step),
            "baseline_method": tk.StringVar(value=d.baseline_method),
            "baseline_lambda": tk.DoubleVar(value=d.baseline_lambda),
            "baseline_p": tk.DoubleVar(value=d.baseline_p),
            "baseline_iterations": tk.IntVar(value=d.baseline_iterations),
            "baseline_poly_order": tk.IntVar(value=d.baseline_poly_order),
            "sg_enabled": tk.BooleanVar(value=d.sg_enabled),
            "sg_window": tk.IntVar(value=d.sg_window),
            "sg_polyorder": tk.IntVar(value=d.sg_polyorder),
            "sg_derivative": tk.IntVar(value=d.sg_derivative),
            "normalization": tk.StringVar(value=d.normalization),
            "min_points": tk.IntVar(value=d.min_points),
        }

    def _build_ui(self) -> None:
        style = ttk.Style(self)
        style.configure("Title.TLabel", font=("Microsoft YaHei UI", 16, "bold"))
        style.configure("Section.TLabel", font=("Microsoft YaHei UI", 10, "bold"))

        top = ttk.Frame(self, padding=(12, 10))
        top.pack(fill="x")
        ttk.Label(top, text="拉曼光谱批量预处理软件", style="Title.TLabel").grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 8))
        ttk.Label(top, text="输入文件夹").grid(row=1, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.input_dir).grid(row=1, column=1, sticky="ew", padx=6)
        ttk.Button(top, text="选择…", command=self._choose_input).grid(row=1, column=2)
        ttk.Label(top, text="输出文件夹").grid(row=2, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(top, textvariable=self.output_dir).grid(row=2, column=1, sticky="ew", padx=6, pady=(6, 0))
        ttk.Button(top, text="选择…", command=self._choose_output).grid(row=2, column=2, pady=(6, 0))
        top.columnconfigure(1, weight=1)

        body = ttk.Panedwindow(self, orient="horizontal")
        body.pack(fill="both", expand=True, padx=12, pady=(0, 8))
        left = ttk.Frame(body, width=450)
        right = ttk.Frame(body)
        body.add(left, weight=0)
        body.add(right, weight=1)

        notebook = ttk.Notebook(left)
        notebook.pack(fill="both", expand=True)
        self._build_data_tab(notebook)
        self._build_processing_tab(notebook)
        self._build_output_tab(notebook)

        command = ttk.Frame(left, padding=(0, 10, 0, 0))
        command.pack(fill="x")
        ttk.Button(command, text="载入参数", command=self._load_config).pack(side="left")
        ttk.Button(command, text="保存参数", command=self._save_config).pack(side="left", padx=5)
        ttk.Button(command, text="预览光谱", command=self._preview).pack(side="left", padx=(12, 5))
        self.run_button = ttk.Button(command, text="开始批处理", command=self._start_batch)
        self.run_button.pack(side="left")
        self.stop_button = ttk.Button(command, text="停止", command=self._stop, state="disabled")
        self.stop_button.pack(side="left", padx=5)
        self.progress = ttk.Progressbar(left, mode="determinate")
        self.progress.pack(fill="x", pady=(0, 8))
        self.status = ttk.Label(left, text="就绪")
        self.status.pack(fill="x")

        self.figure = Figure(figsize=(7.2, 5.2), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_xlabel("Raman shift (cm-1)")
        self.ax.set_ylabel("Intensity")
        self.ax.set_title("预处理预览")
        self.figure.tight_layout()
        self.canvas = FigureCanvasTkAgg(self.figure, master=right)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        toolbar = NavigationToolbar2Tk(self.canvas, right, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(fill="x")
        ttk.Label(right, text="运行日志", style="Section.TLabel").pack(anchor="w", pady=(6, 2))
        log_frame = ttk.Frame(right)
        log_frame.pack(fill="x")
        self.log = tk.Text(log_frame, height=9, wrap="word", state="disabled")
        scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log.yview)
        self.log.configure(yscrollcommand=scrollbar.set)
        self.log.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def _tab(self, notebook: ttk.Notebook, title: str) -> ttk.Frame:
        frame = ttk.Frame(notebook, padding=12)
        notebook.add(frame, text=title)
        frame.columnconfigure(1, weight=1)
        return frame

    def _entry_row(self, parent: ttk.Frame, row: int, label: str, variable: tk.Variable, unit: str = "") -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(parent, textvariable=variable, width=18).grid(row=row, column=1, sticky="ew", padx=6, pady=4)
        ttk.Label(parent, text=unit).grid(row=row, column=2, sticky="w")

    def _build_data_tab(self, notebook: ttk.Notebook) -> None:
        f = self._tab(notebook, "数据与波段")
        self._entry_row(f, 0, "文件扩展名", self.extensions, "逗号分隔")
        ttk.Checkbutton(f, text="递归读取子文件夹", variable=self.recursive).grid(row=1, column=0, columnspan=3, sticky="w", pady=5)
        ttk.Separator(f).grid(row=2, column=0, columnspan=3, sticky="ew", pady=8)
        ttk.Checkbutton(f, text="截取波数范围", variable=self.vars["crop_enabled"]).grid(row=3, column=0, columnspan=3, sticky="w")
        self._entry_row(f, 4, "保留波段", self.vars["ranges"], "例：600-1800,2700-3600")
        ttk.Checkbutton(f, text="按固定步长重采样", variable=self.vars["resample_enabled"]).grid(row=5, column=0, columnspan=3, sticky="w", pady=(10, 2))
        self._entry_row(f, 6, "重采样步长", self.vars["resample_step"], "cm⁻¹")
        self._entry_row(f, 7, "最低有效点数", self.vars["min_points"], "点")
        ttk.Label(f, text="多波段会分别插值，不会跨越1800–2700 cm⁻¹等空白区。", wraplength=390).grid(row=8, column=0, columnspan=3, sticky="w", pady=(12, 0))

    def _build_processing_tab(self, notebook: ttk.Notebook) -> None:
        outer = self._tab(notebook, "预处理参数")
        canvas = tk.Canvas(outer, highlightthickness=0)
        scrollbar = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        inner = ttk.Frame(canvas)
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.grid(row=0, column=0, columnspan=3, sticky="nsew")
        scrollbar.grid(row=0, column=3, sticky="ns")
        outer.rowconfigure(0, weight=1)
        outer.columnconfigure(0, weight=1)
        inner.columnconfigure(1, weight=1)
        r = 0
        ttk.Label(inner, text="宇宙射线", style="Section.TLabel").grid(row=r, column=0, columnspan=3, sticky="w"); r += 1
        ttk.Checkbutton(inner, text="启用局部MAD宇宙射线修复", variable=self.vars["cosmic_enabled"]).grid(row=r, column=0, columnspan=3, sticky="w"); r += 1
        for label, key, unit in [("窗口宽度", "cosmic_window", "奇数"), ("动态因子", "cosmic_threshold", "P7建议12"), ("最大峰宽", "cosmic_max_width", "采样点"), ("修复迭代", "cosmic_passes", "次")]:
            self._entry_row(inner, r, label, self.vars[key], unit); r += 1
        ttk.Separator(inner).grid(row=r, column=0, columnspan=3, sticky="ew", pady=8); r += 1
        ttk.Label(inner, text="基线校正", style="Section.TLabel").grid(row=r, column=0, columnspan=3, sticky="w"); r += 1
        ttk.Label(inner, text="方法").grid(row=r, column=0, sticky="w", pady=4)
        ttk.Combobox(inner, textvariable=self.vars["baseline_method"], values=["无", "ALS", "airPLS", "多项式"], state="readonly").grid(row=r, column=1, sticky="ew", padx=6); r += 1
        for label, key, unit in [("平滑参数 λ", "baseline_lambda", "越大越平滑"), ("ALS参数 p", "baseline_p", "通常0.001–0.05"), ("迭代次数", "baseline_iterations", "次"), ("多项式阶数", "baseline_poly_order", "阶")]:
            self._entry_row(inner, r, label, self.vars[key], unit); r += 1
        ttk.Separator(inner).grid(row=r, column=0, columnspan=3, sticky="ew", pady=8); r += 1
        ttk.Label(inner, text="SG平滑与导数", style="Section.TLabel").grid(row=r, column=0, columnspan=3, sticky="w"); r += 1
        ttk.Checkbutton(inner, text="启用Savitzky–Golay", variable=self.vars["sg_enabled"]).grid(row=r, column=0, columnspan=3, sticky="w"); r += 1
        for label, key, unit in [("SG窗口", "sg_window", "奇数"), ("多项式阶数", "sg_polyorder", "阶"), ("导数阶数", "sg_derivative", "0/1/2")]:
            self._entry_row(inner, r, label, self.vars[key], unit); r += 1
        ttk.Separator(inner).grid(row=r, column=0, columnspan=3, sticky="ew", pady=8); r += 1
        ttk.Label(inner, text="归一化", style="Section.TLabel").grid(row=r, column=0, columnspan=3, sticky="w"); r += 1
        ttk.Label(inner, text="方法").grid(row=r, column=0, sticky="w", pady=4)
        ttk.Combobox(inner, textvariable=self.vars["normalization"], values=["无", "向量归一化", "面积归一化", "SNV", "Min-Max"], state="readonly").grid(row=r, column=1, sticky="ew", padx=6)

    def _build_output_tab(self, notebook: ttk.Notebook) -> None:
        f = self._tab(notebook, "输出设置")
        ttk.Checkbutton(f, text="保留输入目录结构", variable=self.preserve_tree).grid(row=0, column=0, columnspan=3, sticky="w", pady=4)
        ttk.Checkbutton(f, text="覆盖已有输出文件", variable=self.overwrite).grid(row=1, column=0, columnspan=3, sticky="w", pady=4)
        ttk.Label(f, text="输出格式").grid(row=2, column=0, sticky="w", pady=4)
        ttk.Combobox(f, textvariable=self.output_format, values=["txt", "csv"], state="readonly").grid(row=2, column=1, sticky="ew", padx=6)
        self._entry_row(f, 3, "文件名后缀", self.output_suffix, "可留空")
        self._entry_row(f, 4, "有效数字", self.precision, "位")
        ttk.Separator(f).grid(row=5, column=0, columnspan=3, sticky="ew", pady=8)
        ttk.Label(f, text="每次运行会在输出文件夹生成：\n• 预处理后的双列光谱\n• 处理报告.csv\n• 本次参数.json", justify="left").grid(row=6, column=0, columnspan=3, sticky="w")

    def _choose_input(self) -> None:
        path = filedialog.askdirectory(title="选择光谱输入文件夹")
        if path:
            self.input_dir.set(path)
            if not self.output_dir.get():
                self.output_dir.set(str(Path(path).parent / (Path(path).name + "_预处理结果")))

    def _choose_output(self) -> None:
        path = filedialog.askdirectory(title="选择输出文件夹")
        if path:
            self.output_dir.set(path)

    def _config(self) -> PreprocessConfig:
        values = {key: var.get() for key, var in self.vars.items()}
        cfg = PreprocessConfig(**values)
        cfg.validate()
        return cfg

    def _extensions(self) -> list[str]:
        values = [x.strip().lstrip(".") for x in self.extensions.get().replace("，", ",").split(",") if x.strip()]
        if not values:
            raise ValueError("至少需要一个文件扩展名")
        return values

    def _save_config(self) -> None:
        try:
            cfg = self._config()
            path = filedialog.asksaveasfilename(title="保存参数", defaultextension=".json", filetypes=[("JSON", "*.json")])
            if path:
                payload = {"preprocess": asdict(cfg), "extensions": self.extensions.get(), "recursive": self.recursive.get(), "preserve_tree": self.preserve_tree.get(), "overwrite": self.overwrite.get(), "output_format": self.output_format.get(), "output_suffix": self.output_suffix.get(), "precision": self.precision.get()}
                Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                self._write_log(f"参数已保存：{path}")
        except Exception as e:
            messagebox.showerror("参数错误", str(e))

    def _load_config(self) -> None:
        path = filedialog.askopenfilename(title="载入参数", filetypes=[("JSON", "*.json"), ("全部文件", "*.*")])
        if not path:
            return
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            values = payload.get("preprocess", payload)
            cfg = PreprocessConfig(**values)
            cfg.validate()
            for key, value in asdict(cfg).items():
                self.vars[key].set(value)
            for key, variable in [("extensions", self.extensions), ("recursive", self.recursive), ("preserve_tree", self.preserve_tree), ("overwrite", self.overwrite), ("output_format", self.output_format), ("output_suffix", self.output_suffix), ("precision", self.precision)]:
                if key in payload:
                    variable.set(payload[key])
            self._write_log(f"参数已载入：{path}")
        except Exception as e:
            messagebox.showerror("载入失败", str(e))

    def _preview(self) -> None:
        try:
            cfg = self._config()
            chosen = filedialog.askopenfilename(title="选择一条光谱预览", initialdir=self.input_dir.get() or None,
                                                filetypes=[("光谱文件", "*.txt *.csv *.dat"), ("全部文件", "*.*")])
            if not chosen:
                return
            self.preview_file = Path(chosen)
            x, y = read_spectrum(chosen)
            result = preprocess(x, y, cfg)
            self.ax.clear()
            self.ax.plot(x, y, color="#777777", lw=0.9, alpha=0.75, label="原始光谱")
            self.ax.plot(result.x, result.y, color="#1769aa", lw=1.1, label="预处理后")
            self.ax.set_xlabel("Raman shift (cm-1)")
            self.ax.set_ylabel("Intensity")
            self.ax.set_title(self.preview_file.name)
            self.ax.legend()
            self.ax.grid(alpha=0.18)
            self.figure.tight_layout()
            self.canvas.draw_idle()
            self._write_log(f"预览完成：{self.preview_file.name}；修复宇宙射线点数={result.cosmic_points}，输出点数={result.output_points}")
        except Exception as e:
            messagebox.showerror("预览失败", str(e))

    def _start_batch(self) -> None:
        if self.worker and self.worker.is_alive():
            return
        try:
            cfg = self._config()
            input_root = Path(self.input_dir.get()).expanduser().resolve()
            output_root = Path(self.output_dir.get()).expanduser().resolve()
            if not input_root.is_dir():
                raise ValueError("请选择有效的输入文件夹")
            if input_root == output_root:
                raise ValueError("输出文件夹不能与输入文件夹相同")
            extensions = self._extensions()
            files = find_spectra(input_root, extensions, self.recursive.get())
            if output_root.is_relative_to(input_root):
                files = [p for p in files if not p.resolve().is_relative_to(output_root)]
            if not files:
                raise ValueError("输入文件夹中没有找到光谱文件")
        except Exception as e:
            messagebox.showerror("无法开始", str(e))
            return
        self.stop_event.clear()
        self.progress.configure(maximum=len(files), value=0)
        self.run_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.status.configure(text=f"准备处理 {len(files)} 个文件…")
        options = {"format": self.output_format.get(), "suffix": self.output_suffix.get(), "precision": self.precision.get(), "preserve": self.preserve_tree.get(), "overwrite": self.overwrite.get()}
        self.worker = threading.Thread(target=self._batch_worker, args=(input_root, output_root, files, cfg, options), daemon=True)
        self.worker.start()

    def _batch_worker(self, input_root: Path, output_root: Path, files: list[Path], cfg: PreprocessConfig, options: dict) -> None:
        output_root.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report: list[dict] = []
        output_grids: set[tuple[int, float, float, float]] = set()
        success = failed = skipped = 0
        for index, path in enumerate(files, 1):
            if self.stop_event.is_set():
                break
            item = {"input": str(path), "status": "", "output": "", "input_points": "", "output_points": "",
                    "x_start": "", "x_end": "", "median_step": "", "cosmic_points": "", "message": ""}
            try:
                relative_parent = path.relative_to(input_root).parent if options["preserve"] else Path()
                ext = ".csv" if options["format"] == "csv" else ".txt"
                destination = output_root / relative_parent / (path.stem + options["suffix"] + ext)
                if destination.exists() and not options["overwrite"]:
                    skipped += 1
                    item.update(status="跳过", output=str(destination), message="输出文件已存在")
                else:
                    x, y = read_spectrum(path)
                    result = preprocess(x, y, cfg)
                    save_spectrum(destination, result.x, result.y, options["format"], options["precision"])
                    success += 1
                    median_step = float(np.median(np.diff(result.x))) if result.x.size > 1 else float("nan")
                    grid_signature = (result.output_points, round(float(result.x[0]), 6),
                                      round(float(result.x[-1]), 6), round(median_step, 6))
                    output_grids.add(grid_signature)
                    item.update(status="成功", output=str(destination), input_points=result.input_points,
                                output_points=result.output_points, x_start=float(result.x[0]),
                                x_end=float(result.x[-1]), median_step=median_step,
                                cosmic_points=result.cosmic_points)
            except Exception as e:
                failed += 1
                item.update(status="失败", message=str(e))
            report.append(item)
            self.events.put(("progress", index, len(files), path.name, success, failed, skipped))
        report_path = output_root / f"处理报告_{timestamp}.csv"
        with report_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=["input", "status", "output", "input_points", "output_points",
                                                  "x_start", "x_end", "median_step", "cosmic_points", "message"])
            writer.writeheader(); writer.writerows(report)
        parameter_path = output_root / f"本次参数_{timestamp}.json"
        parameter_path.write_text(json.dumps({"preprocess": asdict(cfg), "output": options}, ensure_ascii=False, indent=2), encoding="utf-8")
        grid_warning = ""
        if not cfg.resample_enabled and len(output_grids) > 1:
            grid_warning = f"检测到{len(output_grids)}套输出波数网格；建模前建议启用重采样。"
        self.events.put(("done", success, failed, skipped, len(report), str(report_path),
                         self.stop_event.is_set(), grid_warning))

    def _stop(self) -> None:
        self.stop_event.set()
        self.status.configure(text="正在停止，将在当前文件处理完后结束…")

    def _poll_events(self) -> None:
        try:
            while True:
                event = self.events.get_nowait()
                if event[0] == "progress":
                    _, index, total, name, success, failed, skipped = event
                    self.progress.configure(value=index)
                    self.status.configure(text=f"{index}/{total}  {name}  成功{success} / 失败{failed} / 跳过{skipped}")
                elif event[0] == "done":
                    _, success, failed, skipped, processed, report, stopped, grid_warning = event
                    self.run_button.configure(state="normal")
                    self.stop_button.configure(state="disabled")
                    text = f"{'已停止' if stopped else '处理完成'}：成功{success}，失败{failed}，跳过{skipped}；报告：{report}"
                    if grid_warning:
                        text += f"；注意：{grid_warning}"
                    self.status.configure(text=text)
                    self._write_log(text)
                    if not stopped:
                        warning_text = f"\n\n注意：{grid_warning}" if grid_warning else ""
                        messagebox.showinfo("处理完成", f"成功：{success}\n失败：{failed}\n跳过：{skipped}\n\n报告：{report}{warning_text}")
        except queue.Empty:
            pass
        self.after(100, self._poll_events)

    def _write_log(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", f"[{datetime.now():%H:%M:%S}] {text}\n")
        self.log.see("end")
        self.log.configure(state="disabled")


def main() -> None:
    app = SpectrumPreprocessorApp()
    app.mainloop()


if __name__ == "__main__":
    main()
