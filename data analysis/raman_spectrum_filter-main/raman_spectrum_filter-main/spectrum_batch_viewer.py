"""拉曼光谱分组抽样与键盘翻页查看工具。

运行方式：
    python spectrum_batch_viewer.py

选择一个数据根目录后，程序会把每个直接包含 TXT 文件的文件夹识别为
一个“组”。可以查看单组的随机样本，也可以对全部组分别抽样。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib

matplotlib.use("TkAgg")
matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Arial Unicode MS",
    "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import pandas as pd


ALL_GROUPS_LABEL = "【全部组：每组分别抽样】"


@dataclass(frozen=True)
class SampleItem:
    """一条已抽中的光谱及其所属组。"""

    group: str
    path: Path


def discover_groups(root_path: Path) -> dict[str, list[Path]]:
    """递归查找TXT，并按其直接父目录分组。"""

    root_path = root_path.resolve()
    groups: dict[str, list[Path]] = {}
    for txt_path in sorted(root_path.rglob("*.txt")):
        try:
            group_path = txt_path.parent.relative_to(root_path)
        except ValueError:
            continue
        group_name = "（根目录）" if str(group_path) == "." else str(group_path)
        groups.setdefault(group_name, []).append(txt_path)
    return dict(sorted(groups.items(), key=lambda item: item[0].lower()))


def read_spectrum(txt_path: Path) -> pd.DataFrame:
    """读取常见两列TXT光谱，允许制表符、空格、逗号或分号分隔。"""

    raw = pd.read_csv(
        txt_path,
        sep=r"[\t,; ]+",
        engine="python",
        header=None,
        comment="#",
        usecols=[0, 1],
        names=["Raman_Shift", "Raman_Intensity"],
        encoding="utf-8-sig",
    )
    raw["Raman_Shift"] = pd.to_numeric(raw["Raman_Shift"], errors="coerce")
    raw["Raman_Intensity"] = pd.to_numeric(
        raw["Raman_Intensity"], errors="coerce"
    )
    spectrum = (
        raw.dropna(subset=["Raman_Shift", "Raman_Intensity"])
        .sort_values("Raman_Shift")
        .drop_duplicates("Raman_Shift", keep="first")
        .reset_index(drop=True)
    )
    if len(spectrum) < 2:
        raise ValueError("没有读到至少两个有效的两列数值点")
    return spectrum


def make_samples(
    groups: dict[str, list[Path]],
    selected_group: str,
    sample_size: int,
    seed: int | None,
) -> list[SampleItem]:
    """按指定组抽样；选择全部组时，每个组分别抽取sample_size条。"""

    if sample_size < 1:
        raise ValueError("抽样数量必须大于0")

    rng = random.Random(seed)
    group_names = list(groups) if selected_group == ALL_GROUPS_LABEL else [selected_group]
    samples: list[SampleItem] = []
    for group_name in group_names:
        files = groups.get(group_name, [])
        chosen = rng.sample(files, k=min(sample_size, len(files)))
        samples.extend(SampleItem(group_name, path) for path in sorted(chosen))
    return samples


class SpectrumBatchViewer:
    """Tkinter简易前端。"""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("拉曼光谱分组抽样查看器")
        self.root.geometry("1180x820")
        self.root.minsize(900, 640)

        self.folder_var = tk.StringVar()
        self.group_var = tk.StringVar()
        self.sample_size_var = tk.StringVar(value="10")
        self.seed_var = tk.StringVar(value="42")
        self.show_regions_var = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(value="请选择数据文件夹。")
        self.file_info_var = tk.StringVar(value="尚未抽样")

        self.groups: dict[str, list[Path]] = {}
        self.samples: list[SampleItem] = []
        self.current_index = -1
        self.spectrum_cache: dict[Path, pd.DataFrame] = {}

        self._build_ui()
        self.root.bind("<Left>", self._on_left_key)
        self.root.bind("<Right>", self._on_right_key)

    def _build_ui(self) -> None:
        outer = ttk.Frame(self.root, padding=10)
        outer.pack(fill=tk.BOTH, expand=True)

        settings = ttk.LabelFrame(outer, text="抽样设置", padding=8)
        settings.pack(fill=tk.X)
        settings.columnconfigure(1, weight=1)

        ttk.Label(settings, text="数据根目录：").grid(row=0, column=0, sticky="w")
        ttk.Entry(settings, textvariable=self.folder_var).grid(
            row=0, column=1, columnspan=5, sticky="ew", padx=5
        )
        ttk.Button(settings, text="选择文件夹", command=self.choose_folder).grid(
            row=0, column=6, padx=(5, 0)
        )

        ttk.Label(settings, text="选择组：").grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.group_combo = ttk.Combobox(
            settings, textvariable=self.group_var, state="readonly", width=36
        )
        self.group_combo.grid(row=1, column=1, sticky="ew", padx=5, pady=(8, 0))
        self.group_combo.bind("<<ComboboxSelected>>", self._update_group_status)

        ttk.Label(settings, text="抽样数：").grid(row=1, column=2, padx=(12, 0), pady=(8, 0))
        ttk.Spinbox(
            settings,
            from_=1,
            to=9999,
            textvariable=self.sample_size_var,
            width=8,
        ).grid(row=1, column=3, pady=(8, 0))

        ttk.Label(settings, text="随机种子：").grid(
            row=1, column=4, padx=(12, 0), pady=(8, 0)
        )
        ttk.Entry(settings, textvariable=self.seed_var, width=10).grid(
            row=1, column=5, pady=(8, 0)
        )
        ttk.Button(settings, text="抽样并绘图", command=self.sample_and_show).grid(
            row=1, column=6, padx=(10, 0), pady=(8, 0)
        )

        options = ttk.Frame(settings)
        options.grid(row=2, column=0, columnspan=7, sticky="ew", pady=(8, 0))
        ttk.Checkbutton(
            options,
            text="显示指纹区、静默区和C–H区背景",
            variable=self.show_regions_var,
            command=self.refresh_plot,
        ).pack(side=tk.LEFT)
        ttk.Label(
            options,
            text="提示：随机种子留空表示每次随机；选择“全部组”时，抽样数表示每组数量。",
            foreground="#555555",
        ).pack(side=tk.RIGHT)

        self.figure = Figure(figsize=(9, 5.8), dpi=100, constrained_layout=True)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=outer)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        self.toolbar = NavigationToolbar2Tk(self.canvas, outer, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.pack(fill=tk.X)

        navigation = ttk.Frame(outer)
        navigation.pack(fill=tk.X, pady=(6, 0))
        self.previous_button = ttk.Button(
            navigation, text="← 上一张", command=self.show_previous, state=tk.DISABLED
        )
        self.previous_button.pack(side=tk.LEFT)
        ttk.Label(navigation, textvariable=self.file_info_var, anchor="center").pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=10
        )
        self.next_button = ttk.Button(
            navigation, text="下一张 →", command=self.show_next, state=tk.DISABLED
        )
        self.next_button.pack(side=tk.RIGHT)

        ttk.Separator(outer).pack(fill=tk.X, pady=(6, 3))
        ttk.Label(outer, textvariable=self.status_var, anchor="w").pack(fill=tk.X)
        self._draw_empty("请选择包含TXT光谱的数据文件夹")

    def choose_folder(self) -> None:
        initial = self.folder_var.get().strip() or str(Path.home())
        selected = filedialog.askdirectory(title="选择光谱数据根目录", initialdir=initial)
        if not selected:
            return
        self.folder_var.set(selected)
        self.scan_folder()

    def scan_folder(self) -> None:
        folder_text = self.folder_var.get().strip()
        if not folder_text:
            messagebox.showwarning("缺少路径", "请先选择数据根目录。")
            return
        root_path = Path(folder_text)
        if not root_path.is_dir():
            messagebox.showerror("路径无效", "所选数据根目录不存在或不是文件夹。")
            return

        self.groups = discover_groups(root_path)
        self.samples = []
        self.current_index = -1
        self.spectrum_cache.clear()
        if not self.groups:
            self.group_combo["values"] = []
            self.group_var.set("")
            self._draw_empty("该目录及其子目录中没有找到TXT光谱")
            self.status_var.set("未找到TXT文件。")
            return

        values = [ALL_GROUPS_LABEL, *self.groups.keys()]
        self.group_combo["values"] = values
        self.group_var.set(values[0])
        total = sum(len(files) for files in self.groups.values())
        self.status_var.set(f"已识别 {len(self.groups)} 个组，共 {total} 个TXT光谱。")
        self.file_info_var.set("请设置抽样数，然后点击“抽样并绘图”")
        self._set_navigation_state()

    def sample_and_show(self) -> None:
        if not self.groups:
            self.scan_folder()
            if not self.groups:
                return
        try:
            sample_size = int(self.sample_size_var.get().strip())
            seed_text = self.seed_var.get().strip()
            seed = None if seed_text == "" else int(seed_text)
            selected_group = self.group_var.get()
            self.samples = make_samples(
                self.groups, selected_group, sample_size, seed
            )
        except ValueError as exc:
            messagebox.showerror("抽样参数错误", str(exc))
            return

        if not self.samples:
            messagebox.showwarning("没有样本", "当前设置没有抽取到光谱。")
            return
        self.current_index = 0
        self.spectrum_cache.clear()
        self.show_current()
        self._set_navigation_state()
        selected = self.group_var.get()
        self.status_var.set(
            f"已完成抽样：{selected}，共 {len(self.samples)} 条；可按键盘 ← / → 切换。"
        )

    def show_current(self) -> None:
        if not (0 <= self.current_index < len(self.samples)):
            return
        item = self.samples[self.current_index]
        try:
            spectrum = self.spectrum_cache.get(item.path)
            if spectrum is None:
                spectrum = read_spectrum(item.path)
                self.spectrum_cache[item.path] = spectrum
        except Exception as exc:
            self._draw_empty(f"读取失败\n{item.path.name}\n{exc}")
            self.file_info_var.set(
                f"{self.current_index + 1}/{len(self.samples)}｜{item.group}｜读取失败"
            )
            return

        self.ax.clear()
        if self.show_regions_var.get():
            self.ax.axvspan(600, 1800, color="#66bb6a", alpha=0.10, label="指纹区")
            self.ax.axvspan(1800, 2700, color="#9e9e9e", alpha=0.08, label="静默区")
            self.ax.axvspan(2700, 3200, color="#ef5350", alpha=0.08, label="C–H区")
        self.ax.plot(
            spectrum["Raman_Shift"],
            spectrum["Raman_Intensity"],
            color="#1565c0",
            linewidth=1.0,
        )
        self.ax.set_title(item.path.name)
        self.ax.set_xlabel("Raman shift (cm$^{-1}$)")
        self.ax.set_ylabel("Intensity (a.u.)")
        self.ax.grid(alpha=0.18)
        if self.show_regions_var.get():
            self.ax.legend(loc="best", fontsize=8)
        self.canvas.draw_idle()

        self.file_info_var.set(
            f"{self.current_index + 1}/{len(self.samples)}｜组：{item.group}｜"
            f"点数：{len(spectrum)}｜{item.path.name}"
        )
        self.status_var.set(str(item.path))
        self._set_navigation_state()

    def show_previous(self) -> None:
        if self.current_index > 0:
            self.current_index -= 1
            self.show_current()

    def show_next(self) -> None:
        if self.current_index + 1 < len(self.samples):
            self.current_index += 1
            self.show_current()

    def refresh_plot(self) -> None:
        if self.samples:
            self.show_current()

    def _update_group_status(self, _event: tk.Event | None = None) -> None:
        selected = self.group_var.get()
        if selected == ALL_GROUPS_LABEL:
            total = sum(len(files) for files in self.groups.values())
            self.status_var.set(
                f"全部 {len(self.groups)} 个组，共 {total} 条；抽样数将应用到每个组。"
            )
        elif selected in self.groups:
            self.status_var.set(f"组 {selected}：{len(self.groups[selected])} 条光谱。")

    def _set_navigation_state(self) -> None:
        has_previous = bool(self.samples) and self.current_index > 0
        has_next = bool(self.samples) and self.current_index + 1 < len(self.samples)
        self.previous_button.configure(
            state=tk.NORMAL if has_previous else tk.DISABLED
        )
        self.next_button.configure(state=tk.NORMAL if has_next else tk.DISABLED)

    def _draw_empty(self, text: str) -> None:
        self.ax.clear()
        self.ax.text(
            0.5,
            0.5,
            text,
            transform=self.ax.transAxes,
            ha="center",
            va="center",
            fontsize=13,
            color="#666666",
        )
        self.ax.set_axis_off()
        self.canvas.draw_idle()

    @staticmethod
    def _is_text_input(event: tk.Event) -> bool:
        widget_class = event.widget.winfo_class()
        return widget_class in {"Entry", "TEntry", "Spinbox", "TSpinbox"}

    def _on_left_key(self, event: tk.Event) -> None:
        if not self._is_text_input(event):
            self.show_previous()

    def _on_right_key(self, event: tk.Event) -> None:
        if not self._is_text_input(event):
            self.show_next()


def main() -> None:
    root = tk.Tk()
    try:
        root.tk.call("tk", "scaling", 1.2)
    except tk.TclError:
        pass
    SpectrumBatchViewer(root)
    root.mainloop()


if __name__ == "__main__":
    main()
