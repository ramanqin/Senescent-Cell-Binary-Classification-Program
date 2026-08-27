from __future__ import annotations

# 【衰老数据改动】与原大肠杆菌002工具相比，本界面增加了按组抽样、
# 元数据盲法显示、四区域诊断图、明确的QC控件、自动质量指标，
# 以及可恢复的固定抽样清单。可搜索“【衰老数据改动】”查看主要适配点。

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
from matplotlib import font_manager, rcParams
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from core import (
    ISSUE_FIELDS,
    annotation_for_path,
    build_annotation,
    calculate_qc_metrics,
    load_sample_manifest,
    load_annotations,
    read_spectrum,
    resolve_input_folder,
    save_annotations,
    save_sample_manifest,
    scan_spectra,
    smoothed_intensity,
    upsert_annotation,
)
from qc_assistant import HIGH_CONFIDENCE, predict_qc, train_qc_tree
from state import ProcessState


ISSUE_LABELS = {
    "vague_finger_peak": "2－指纹峰模糊",
    "missing_finger_peak": "3－指纹峰丢失",
    "low_finger_peak": "4－指纹峰强度低",
    "low_snr": "5－信噪比低",
    "missing_ch_peak": "6－碳氢峰丢失",
    "low_ch_peak": "7－碳氢峰强度低",
    "cosmic_rays": "8－存在宇宙射线",
}

# 【衰老数据改动】按照原002工具的数字标注习惯，将异常原因固定映射到
# 数字键2～8；数字1不再使用，PASS和FAIL分别使用P键和F键。
ISSUE_KEY_MAP = {
    "2": "vague_finger_peak",
    "3": "missing_finger_peak",
    "4": "low_finger_peak",
    "5": "low_snr",
    "6": "missing_ch_peak",
    "7": "low_ch_peak",
    "8": "cosmic_rays",
}


def configure_matplotlib_chinese_font() -> str:
    # 【衰老数据改动】原工具没有为Matplotlib配置中文字体，
    # 导致光谱图的中文标题显示为方框；此处自动选择系统已安装的中文字体。
    """为Matplotlib图表标题和坐标轴选择系统已安装的中文字体。"""
    available = {font.name for font in font_manager.fontManager.ttflist}
    candidates = [
        "Microsoft YaHei UI",
        "Microsoft YaHei",
        "SimHei",
        "DengXian",
        "Microsoft JhengHei UI",
        "Microsoft JhengHei",
        "Noto Sans CJK SC",
        "Source Han Sans CN",
    ]
    selected = next((name for name in candidates if name in available), "DejaVu Sans")
    # 中文字体优先，DejaVu Sans作为数学符号的后备字体。
    rcParams["font.family"] = "sans-serif"
    rcParams["font.sans-serif"] = [selected, "DejaVu Sans"]
    rcParams["axes.unicode_minus"] = False
    return selected


class RamanAnnotationApp:
    def __init__(self, root: tk.Tk, state: ProcessState):
        self.root = root
        self.state = state
        self.root.title("Raman Data Annotation")
        self.root.geometry("1480x920")
        self.root.minsize(1180, 760)
        self.plot_font_name = configure_matplotlib_chinese_font()

        # 【衰老数据改动】启动时最大化到Windows可用工作区。
        # Tk的zoomed状态会避开任务栏，防止开启显示缩放后QC按钮被遮挡。
        try:
            self.root.state("zoomed")
        except tk.TclError:
            pass

        self.data_root_var = tk.StringVar(value=str(state.data_root))
        self.save_path_var = tk.StringVar(value=str(state.save_csv_path))
        self.group0_size_var = tk.StringVar(value=str(state.group0_size))
        self.group1_size_var = tk.StringVar(value=str(state.group1_size))
        self.unknown_size_var = tk.StringVar(value=str(state.unknown_size))
        self.sample_seed_var = tk.StringVar(value=str(state.sample_seed))
        self.sampling_strategy_var = tk.StringVar(value=state.sampling_strategy)
        self.show_identity_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="")
        self.progress_var = tk.StringVar(value="尚未加载数据")
        self.identity_var = tk.StringVar(value="")
        self.metrics_var = tk.StringVar(value="")
        self.screening_var = tk.StringVar(value="整体预筛：加载光谱后显示多指标")
        self.assistant_var = tk.StringVar(value="决策树待训练")
        self.issue_vars = {field: tk.BooleanVar(value=False) for field in ISSUE_FIELDS}
        self.qc_tree_model = None
        self.qc_tree_message = "决策树待训练"
        self.current_tree_suggestion = None

        self._configure_style()
        self._build_layout()
        self._bind_keys()

    def _configure_style(self):
        style = ttk.Style()
        style.configure("Meta.TLabel", font=("Microsoft YaHei UI", 10))
        style.configure("Status.TRadiobutton", font=("Microsoft YaHei UI", 11, "bold"))
        style.configure("Action.TButton", font=("Microsoft YaHei UI", 10, "bold"), padding=(10, 8))

    def _build_layout(self):
        top = ttk.Frame(self.root, padding=10)
        top.pack(fill=tk.X)

        # 【衰老数据改动】删除界面顶部的大标题以节省纵向空间，
        # 输入设置控件从第一行开始排列。
        ttk.Label(top, text="待抽样文件夹").grid(row=0, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.data_root_var).grid(row=0, column=1, columnspan=3, sticky="ew", padx=6)
        ttk.Button(top, text="选择任意文件夹", command=self._choose_data_root).grid(row=0, column=4)
        ttk.Button(top, text="加载/继续标注", command=self.load_task).grid(row=0, column=5, padx=(8, 0))

        ttk.Label(top, text="标注CSV（存在则自动继续）").grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(top, textvariable=self.save_path_var).grid(row=1, column=1, columnspan=3, sticky="ew", padx=6, pady=(6, 0))
        ttk.Button(top, text="选择", command=self._choose_save_path).grid(row=1, column=4, pady=(6, 0))
        ttk.Checkbutton(top, text="显示组别与个体（默认盲法）", variable=self.show_identity_var, command=self._refresh_identity).grid(row=1, column=5, padx=(8, 0), pady=(6, 0))

        sample_frame = ttk.Frame(top)
        sample_frame.grid(row=2, column=0, columnspan=6, sticky="ew", pady=(8, 0))
        ttk.Label(sample_frame, text="0组（年轻 young）").pack(side=tk.LEFT)
        ttk.Entry(sample_frame, textvariable=self.group0_size_var, width=7).pack(side=tk.LEFT, padx=(5, 14))
        ttk.Label(sample_frame, text="1组（衰老 aging）").pack(side=tk.LEFT)
        ttk.Entry(sample_frame, textvariable=self.group1_size_var, width=7).pack(side=tk.LEFT, padx=(5, 14))
        ttk.Label(sample_frame, text="未知组").pack(side=tk.LEFT)
        ttk.Entry(sample_frame, textvariable=self.unknown_size_var, width=7).pack(side=tk.LEFT, padx=(5, 14))
        ttk.Label(sample_frame, text="每组-1=全部，0=不抽", foreground="#555555").pack(side=tk.LEFT, padx=(0, 18))
        ttk.Label(sample_frame, text="组内方式").pack(side=tk.LEFT)
        ttk.Combobox(
            sample_frame,
            textvariable=self.sampling_strategy_var,
            values=("balanced", "random"),
            state="readonly",
            width=12,
        ).pack(side=tk.LEFT, padx=(5, 18))
        ttk.Label(sample_frame, text="随机种子").pack(side=tk.LEFT)
        ttk.Entry(sample_frame, textvariable=self.sample_seed_var, width=12).pack(side=tk.LEFT, padx=(5, 18))
        ttk.Label(sample_frame, text="balanced：组内均衡个体；random：组内简单随机", foreground="#555555").pack(side=tk.LEFT)
        top.columnconfigure(1, weight=1)
        top.columnconfigure(2, weight=1)
        top.columnconfigure(3, weight=1)

        info = ttk.Frame(self.root, padding=(10, 0, 10, 6))
        info.pack(fill=tk.X)
        ttk.Label(info, textvariable=self.progress_var, style="Meta.TLabel").pack(side=tk.LEFT)
        ttk.Label(info, textvariable=self.identity_var, style="Meta.TLabel").pack(side=tk.LEFT, padx=24)
        ttk.Label(info, textvariable=self.metrics_var, style="Meta.TLabel").pack(side=tk.RIGHT)

        plot_help = ttk.Label(
            self.root,
            text=(
                "图示：左上=全谱总体质量；右上=指纹区（主要生物分子峰）；"
                "左下=静默区（噪声/尖峰检查）；右下=C–H区（脂质和蛋白相关宽峰）。"
                "蓝线为原始谱，橙线为SG平滑预览。"
            ),
            foreground="#4a4a4a",
            font=("Microsoft YaHei UI", 9),
            padding=(10, 0, 10, 4),
        )
        plot_help.pack(fill=tk.X)
        # 【衰老数据改动】把原本只写入CSV、但界面未显示的整体质量指标展开，
        # 避免人工筛选只盯住强度最大值和最小值。
        ttk.Label(
            self.root,
            textvariable=self.screening_var,
            foreground="#2f4f4f",
            font=("Microsoft YaHei UI", 9),
            padding=(10, 0, 10, 4),
        ).pack(fill=tk.X)

        body = ttk.Panedwindow(self.root, orient=tk.VERTICAL)
        body.pack(fill=tk.BOTH, expand=True, padx=10)

        plot_frame = ttk.Frame(body)
        body.add(plot_frame, weight=5)
        self.figure = Figure(figsize=(13, 6), dpi=100, constrained_layout=True)
        self.axes = self.figure.subplots(2, 2)
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        control = ttk.Frame(body, padding=(4, 8))
        body.add(control, weight=2)

        # 【衰老数据改动】将原来的数字/自由文本输入框改为明确的
        # PASS、REVIEW、FAIL单选项和可多选异常原因，使标注便于追溯和审核。
        status_frame = ttk.LabelFrame(control, text="总体QC结论（必选）", padding=8)
        status_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        for row, (value, label) in enumerate([("pass", "PASS 合格（P键）"), ("review", "REVIEW 复核"), ("fail", "FAIL 不合格（F键）")]):
            ttk.Radiobutton(status_frame, text=label, value=value, variable=self.status_var, style="Status.TRadiobutton").grid(row=row, column=0, sticky="w", pady=2)

        issue_frame = ttk.LabelFrame(control, text="异常原因（数字键2～8，可多选）", padding=8)
        issue_frame.grid(row=0, column=1, sticky="nsew", padx=8)
        for index, field in enumerate(ISSUE_FIELDS):
            ttk.Checkbutton(issue_frame, text=ISSUE_LABELS[field], variable=self.issue_vars[field]).grid(row=index // 4, column=index % 4, sticky="w", padx=7, pady=2)
        ttk.Label(
            issue_frame,
            text="键盘2～8可切换对应异常；数字1不使用",
            foreground="#555555",
        ).grid(row=2, column=0, columnspan=4, sticky="w", padx=7, pady=(7, 0))

        # 【衰老数据改动】删除备注区域，将翻页和保存按钮放到原备注区所在的第三列，
        # 即使屏幕较矮也能让按钮位于任务栏上方。CSV中的other_mark空列仅用于兼容旧结果。
        action_frame = ttk.LabelFrame(control, text="操作", padding=10)
        action_frame.grid(row=0, column=2, sticky="nsew", padx=(8, 0))
        ttk.Button(action_frame, text="← 上一条", command=self.previous, style="Action.TButton").grid(row=0, column=0, sticky="ew", pady=(0, 6))
        ttk.Button(action_frame, text="仅保存（Ctrl+S）", command=self.save_current, style="Action.TButton").grid(row=1, column=0, sticky="ew", pady=6)
        ttk.Button(action_frame, text="保存并下一条 →", command=self.save_and_next, style="Action.TButton").grid(row=2, column=0, sticky="ew", pady=6)
        ttk.Button(action_frame, text="跳到下一条未标注", command=self.next_unannotated, style="Action.TButton").grid(row=3, column=0, sticky="ew", pady=(6, 0))
        ttk.Separator(action_frame, orient=tk.HORIZONTAL).grid(row=4, column=0, sticky="ew", pady=(10, 7))
        ttk.Label(action_frame, textvariable=self.assistant_var, wraplength=250, justify=tk.LEFT).grid(row=5, column=0, sticky="w")
        self.apply_tree_button = ttk.Button(
            action_frame,
            text="采用高置信PASS建议",
            command=self._apply_tree_pass_suggestion,
            state=tk.DISABLED,
        )
        self.apply_tree_button.grid(row=6, column=0, sticky="ew", pady=(7, 0))
        action_frame.columnconfigure(0, weight=1)

        control.columnconfigure(1, weight=1)
        control.columnconfigure(2, weight=1)

    def _bind_keys(self):
        # 【衰老数据改动】普通左右方向键用于切换上下条谱图；当焦点位于
        # 路径、抽样数量、随机种子等输入控件时，方向键仍保留编辑光标功能。
        # Ctrl+左右方向键作为不受输入焦点限制的备用快捷键继续保留。
        self.root.bind("<Control-s>", self._shortcut_save)
        self.root.bind("<Control-Left>", self._shortcut_previous)
        self.root.bind("<Control-Right>", self._shortcut_next)
        self.root.bind("<Left>", self._arrow_previous)
        self.root.bind("<Right>", self._arrow_next)
        self.root.bind("<KeyPress>", self._qc_keyboard_input, add="+")

    def _qc_keyboard_input(self, event):
        """处理P/F状态键和2～8异常原因键；输入控件内不拦截按键。"""
        if self._focus_is_text_input():
            return None

        key = str(event.char or "").lower()
        if key == "p":
            # 设置PASS时同步清除所有异常，保证当前标注始终满足校验规则。
            self.status_var.set("pass")
            for variable in self.issue_vars.values():
                variable.set(False)
            return "break"
        if key == "f":
            self.status_var.set("fail")
            return "break"
        if key in ISSUE_KEY_MAP:
            field = ISSUE_KEY_MAP[key]
            variable = self.issue_vars[field]
            variable.set(not variable.get())
            return "break"
        # 数字1明确不分配功能，其他按键继续交给Tk处理。
        return None

    def _focus_is_text_input(self) -> bool:
        """判断当前焦点是否位于需要使用方向键编辑内容的输入控件。"""
        focused = self.root.focus_get()
        return isinstance(focused, (tk.Entry, ttk.Entry, ttk.Combobox, tk.Text))

    def _arrow_previous(self, _event=None):
        if self._focus_is_text_input():
            return None
        self.previous()
        return "break"

    def _arrow_next(self, _event=None):
        if self._focus_is_text_input():
            return None
        self.save_and_next()
        return "break"

    def _shortcut_save(self, _event=None):
        self.save_current()
        return "break"

    def _shortcut_previous(self, _event=None):
        self.previous()
        return "break"

    def _shortcut_next(self, _event=None):
        self.save_and_next()
        return "break"

    def _choose_data_root(self):
        selected = filedialog.askdirectory(parent=self.root, initialdir=self.data_root_var.get() or "D:\\")
        if selected:
            self.data_root_var.set(selected)

    def _choose_save_path(self):
        selected = filedialog.asksaveasfilename(
            parent=self.root,
            initialfile="raman_qc_annotations.csv",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
        )
        if selected:
            self.save_path_var.set(selected)

    def load_task(self):
        # 递归导入任意文件夹，分别应用年轻组(0)和衰老组(1)
        # 的抽样配额；继续标注时复用已固定的抽样清单。
        try:
            self.state.data_root = resolve_input_folder(self.data_root_var.get())
            self.data_root_var.set(str(self.state.data_root))
            self.state.group0_size = int(self.group0_size_var.get().strip())
            self.state.group1_size = int(self.group1_size_var.get().strip())
            self.state.unknown_size = int(self.unknown_size_var.get().strip())
            self.state.sample_seed = int(self.sample_seed_var.get().strip())
            self.state.sampling_strategy = self.sampling_strategy_var.get().strip()
            if any(size < -1 for size in (self.state.group0_size, self.state.group1_size, self.state.unknown_size)):
                raise ValueError("每组抽样数量必须为-1或非负整数；-1表示全部，0表示不抽。")
            self.state.save_csv_path = Path(self.save_path_var.get()).resolve()
            if self.state.data_root == self.state.save_csv_path.parent or self.state.data_root in self.state.save_csv_path.parents:
                raise ValueError("标注CSV不能保存在原始数据目录内，请使用独立结果目录。")
            existing_sample = load_sample_manifest(self.state.save_csv_path)
            if existing_sample:
                selected_folders = {str(row.get("selected_folder", "")) for row in existing_sample}
                if str(self.state.data_root) not in selected_folders:
                    raise ValueError(
                        "该标注CSV已经关联另一份抽样清单。若要对新文件夹抽样，请选择新的标注CSV文件名。"
                    )
                self.state.records = existing_sample
            else:
                self.state.records = scan_spectra(
                    self.state.data_root,
                    blind_order=True,
                    seed=self.state.sample_seed,
                    strategy=self.state.sampling_strategy,
                    group0_size=self.state.group0_size,
                    group1_size=self.state.group1_size,
                    unknown_size=self.state.unknown_size,
                )
                save_sample_manifest(self.state.records, self.state.save_csv_path)
            self.state.annotations = load_annotations(self.state.save_csv_path)
            self._train_qc_assistant()
            completed = self.state.annotations["qc_status"].astype(str).str.lower().isin({"pass", "review", "fail"})
            annotated_paths = set(self.state.annotations.loc[completed, "file_absolute_path"].astype(str))
            self.state.current_index = next((i for i, row in enumerate(self.state.records) if row["file_absolute_path"] not in annotated_paths), 0)
            self.show_current()
        except Exception as exc:
            messagebox.showerror("加载失败", str(exc), parent=self.root)

    def _current_record(self) -> dict:
        if not self.state.records:
            raise ValueError("请先加载数据")
        return self.state.records[self.state.current_index]

    def show_current(self):
        try:
            record = self._current_record()
            self.state.current_spectrum = read_spectrum(record["file_absolute_path"])
            self.state.current_metrics = calculate_qc_metrics(self.state.current_spectrum)
            existing = annotation_for_path(self.state.annotations, record["file_absolute_path"])
            self._load_annotation_controls(existing)
            self._draw_spectrum(self.state.current_spectrum)
            self._refresh_info()
            self._refresh_qc_assistant()
        except Exception as exc:
            messagebox.showerror("读取失败", str(exc), parent=self.root)

    def _load_annotation_controls(self, row: dict | None):
        self.status_var.set(str(row.get("qc_status", "")) if row else "")
        for field, variable in self.issue_vars.items():
            value = row.get(field, 0) if row else 0
            variable.set(bool(value) and str(value).lower() not in {"0", "nan", "false", ""})

    def _draw_spectrum(self, spectrum):
        # 【衰老数据改动】将原来单一的全谱图改为四区域QC视图：
        # 全波段、指纹区、静默区和C-H伸缩区。
        x = spectrum["Raman_Shift"].to_numpy(float)
        y = spectrum["Raman_Intensity"].to_numpy(float)
        smooth = smoothed_intensity(y)
        panels = [
            (self.axes[0, 0], float(np.nanmin(x)), float(np.nanmax(x)), "全波段：总体质量与基线", "观察基线、饱和、平坦谱及全局异常"),
            (self.axes[0, 1], 500, 1800, "指纹区 500–1800 cm$^{-1}$", "观察主要生物分子峰是否清晰、缺失或过低"),
            (self.axes[1, 0], 1800, 2700, "静默区 1800–2700 cm$^{-1}$", "理论信号较少，用于判断噪声和宇宙射线尖峰"),
            (self.axes[1, 1], 2700, 3200, "C–H伸缩区 2700–3200 cm$^{-1}$", "观察脂质/蛋白相关C–H宽峰的强度与形态"),
        ]
        for axis, xmin, xmax, title, purpose in panels:
            axis.clear()
            mask = (x >= xmin) & (x <= xmax)
            axis.plot(x[mask], y[mask], color="#183153", linewidth=0.7, label="Raw")
            axis.plot(x[mask], smooth[mask], color="#e07a2f", linewidth=0.9, alpha=0.85, label="SG preview")
            axis.set_title(f"{title}\n{purpose}", fontsize=10, fontname=self.plot_font_name)
            axis.set_xlabel("拉曼位移 Raman shift (cm$^{-1}$)", fontname=self.plot_font_name)
            axis.set_ylabel("强度 Intensity (a.u.)", fontname=self.plot_font_name)
            axis.grid(alpha=0.16)
            axis.margins(x=0.01, y=0.08)
        self.axes[0, 0].legend(["原始谱 Raw", "SG平滑预览"], loc="upper right", fontsize=8, prop={"family": self.plot_font_name})
        self.canvas.draw_idle()

    def _refresh_info(self):
        record = self._current_record()
        total = len(self.state.records)
        sample_paths = {row["file_absolute_path"] for row in self.state.records}
        annotated = 0 if self.state.annotations is None else int(
            self.state.annotations["file_absolute_path"].astype(str).isin(sample_paths).sum()
        )
        self.progress_var.set(f"当前 {self.state.current_index + 1}/{total} ｜ 已标注 {annotated}/{total}")
        self._refresh_identity()
        metrics = self.state.current_metrics
        snr = metrics.get("robust_snr_db", float("nan"))
        self.metrics_var.set(
            f"点数 {metrics.get('n_points', '')} ｜ 范围 {metrics.get('x_min', 0):.1f}–{metrics.get('x_max', 0):.1f} ｜ "
            f"稳健SNR {snr:.1f} dB ｜ 自动尖峰 {metrics.get('spike_count_auto', '')}"
        )
        axis_text = "正常" if metrics.get("axis_monotonic") else "异常"
        self.screening_var.set(
            "整体预筛："
            f"P05/中位/P95 {metrics.get('intensity_p05', np.nan):.1f}/"
            f"{metrics.get('intensity_median', np.nan):.1f}/"
            f"{metrics.get('intensity_p95', np.nan):.1f} ｜ "
            f"稳健动态范围 {metrics.get('robust_intensity_range', np.nan):.1f} ｜ "
            f"静默区噪声 {metrics.get('silent_noise_mad', np.nan):.1f} ｜ "
            f"饱和连续点 {metrics.get('saturation_run_max', '')} ｜ "
            f"缺失 {metrics.get('missing_count', '')} ｜ 波数轴 {axis_text}"
        )

    def _train_qc_assistant(self):
        """仅使用已经人工保存的记录训练辅助树，数据不足时保持禁用。"""
        self.qc_tree_model, self.qc_tree_message = train_qc_tree(self.state.annotations)

    def _refresh_qc_assistant(self):
        """显示建议，但不自动修改当前人工标注。"""
        self.current_tree_suggestion = None
        self.apply_tree_button.configure(state=tk.DISABLED)
        if self.qc_tree_model is None:
            self.assistant_var.set(self.qc_tree_message)
            return
        suggestion = predict_qc(self.qc_tree_model, self.state.current_metrics)
        self.current_tree_suggestion = suggestion
        self.assistant_var.set(f"{suggestion['text']}\n{self.qc_tree_message}")
        if suggestion["suggestion"] == "pass" and suggestion["pass_probability"] >= HIGH_CONFIDENCE:
            self.apply_tree_button.configure(state=tk.NORMAL)

    def _apply_tree_pass_suggestion(self):
        """采用建议只填充PASS控件，仍需人工点击保存。"""
        suggestion = self.current_tree_suggestion or {}
        if suggestion.get("suggestion") != "pass" or suggestion.get("pass_probability", 0) < HIGH_CONFIDENCE:
            return
        self.status_var.set("pass")
        for variable in self.issue_vars.values():
            variable.set(False)

    def _refresh_identity(self):
        if not self.state.records:
            self.identity_var.set("")
            return
        record = self._current_record()
        if self.show_identity_var.get():
            self.identity_var.set(
                f"组别 {record['class_original']} ｜ 个体 {record['subject_id']} ｜ 光谱 {record['spectrum_no']} ｜ {record['file_name']}"
            )
        else:
            blinded = hashlib_short(record["spectrum_uid"])
            self.identity_var.set(f"盲法编号 {blinded} ｜ {record['file_name']}")

    def save_current(self) -> bool:
        try:
            record = self._current_record()
            issues = {field: variable.get() for field, variable in self.issue_vars.items()}
            row = build_annotation(record, self.state.current_metrics, self.status_var.get(), issues)
            self.state.annotations = upsert_annotation(self.state.annotations, row)
            save_annotations(self.state.annotations, self.state.save_csv_path)
            # 每次新增人工结论后重训，后续光谱即可逐步获得辅助建议。
            self._train_qc_assistant()
            self._refresh_info()
            self._refresh_qc_assistant()
            return True
        except Exception as exc:
            messagebox.showerror("保存失败", str(exc), parent=self.root)
            return False

    def save_and_next(self):
        if self.save_current():
            self.next_unannotated()

    def next_unannotated(self):
        if not self.state.records:
            return
        if self.state.annotations is None or self.state.annotations.empty:
            annotated_paths = set()
        else:
            completed = self.state.annotations["qc_status"].astype(str).str.lower().isin({"pass", "review", "fail"})
            annotated_paths = set(self.state.annotations.loc[completed, "file_absolute_path"].astype(str))
        total = len(self.state.records)
        for offset in range(1, total + 1):
            candidate = (self.state.current_index + offset) % total
            if self.state.records[candidate]["file_absolute_path"] not in annotated_paths:
                self.state.current_index = candidate
                self.show_current()
                return
        messagebox.showinfo("完成", "全部光谱均已有标注。仍可使用上一条进行复核和覆盖。", parent=self.root)

    def previous(self):
        if self.state.records:
            self.state.current_index = max(0, self.state.current_index - 1)
            self.show_current()


def hashlib_short(value: str) -> str:
    import hashlib

    return "QC-" + hashlib.sha1(value.encode("utf-8")).hexdigest()[:10].upper()


def run_app(
    data_root: str | None = None,
    output_csv: str | None = None,
    group0_size: int = 50,
    group1_size: int = 50,
    unknown_size: int = 0,
    seed: int = 20260813,
    strategy: str = "balanced",
):
    root = tk.Tk()
    state = ProcessState()
    if data_root:
        state.data_root = Path(data_root)
    if output_csv:
        state.save_csv_path = Path(output_csv)
    state.group0_size = group0_size
    state.group1_size = group1_size
    state.unknown_size = unknown_size
    state.sample_seed = seed
    state.sampling_strategy = strategy
    RamanAnnotationApp(root, state)
    root.mainloop()
