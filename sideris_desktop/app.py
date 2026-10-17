"""The native Sideris window (Tkinter, no web view).

A calm, light interface in the spirit of a modern writing tool: a stone
sidebar, serif display type, warm copper accents and one clear column for
the conversation.  Chat talks to the loopback HTTP service, so the window
is only a client - the model, memory and tools all live in the local
process.

The application is intentionally standard-library only.  No Electron, no
bundled browser, no telemetry.
"""

from __future__ import annotations

import json
import os
import platform
import queue
import re
import sys
import threading
import time
import tkinter as tk
import uuid
from pathlib import Path
from tkinter import filedialog, font as tkfont, messagebox, ttk
from typing import Any, Callable

from . import __app_name__, __release_name__, __version__
from .config import (
    Paths,
    Settings,
    SettingsStore,
    asset_dir,
    bundle_dir,
    open_path,
    paths,
    platform_label,
    read_json,
    total_ram_gb,
    write_json,
)
from .core import LEVELS, level_info
from .downloads import human_bytes
from .installer import Setup, activate_runtime, inference_ready
from .models import DEFAULT_MODEL, ModelManager, catalogue, get_spec, recommend_key
from .server import LocalRuntime, create_server, probe_server
from .service import (
    ServiceError,
    install_login_item,
    install_service,
    login_item_status,
    remove_login_item,
    service_status,
    uninstall_service,
)
from .training import (
    TrainingError,
    TrainingManager,
    TrainingNotReadyError,
    TrainingPermissionError,
    example_dataset,
    latest_checkpoint,
    prepare_dataset,
)


# --------------------------------------------------------------------------
# palette and type
# --------------------------------------------------------------------------
BG = "#FAF9F5"
SIDEBAR = "#F0EEE6"
SIDEBAR_HOVER = "#E8E4D8"
SIDEBAR_ACTIVE = "#E3DED0"
CARD = "#FFFFFF"
CARD_ALT = "#FBFAF6"
BORDER = "#E5E1D6"
BORDER_STRONG = "#D9D3C4"
TEXT = "#262521"
MUTED = "#6F6B62"
FAINT = "#9A958A"
ACCENT = "#C2693F"
ACCENT_HOVER = "#A9552F"
ACCENT_SOFT = "#F6E7DC"
USER_BG = "#EFECE3"
CODE_BG = "#F4F1E9"
GREEN = "#4F7A5A"
RED = "#A5483D"
WHITE = "#FFFFFF"

UI_FAMILIES = [
    "Inter", "SF Pro Text", "Segoe UI Variable Text", "Segoe UI", "Avenir Next",
    "Helvetica Neue", "DejaVu Sans", "Arial",
]
DISPLAY_FAMILIES = [
    "Iowan Old Style", "Charter", "Georgia", "Palatino Linotype", "Palatino",
    "Times New Roman", "DejaVu Serif", "Liberation Serif",
]
MONO_FAMILIES = [
    "JetBrains Mono", "SF Mono", "Cascadia Code", "Consolas", "Menlo",
    "DejaVu Sans Mono", "Liberation Mono", "Courier New",
]

SUGGESTIONS = [
    ("Explain a hard idea", "Explain how a transformer model actually works, in plain language."),
    ("Work with my code", "Review my latest Python file and point out the real bugs, not style nits."),
    ("Think it through", "Help me plan a focused week: I have three projects and too little time."),
]


def _pick_family(candidates: list[str], available: set[str], fallback: str) -> str:
    for name in candidates:
        if name in available:
            return name
    return fallback


# --------------------------------------------------------------------------
# chat persistence
# --------------------------------------------------------------------------
class ChatStore:
    """One JSON file per conversation, in the user's data directory."""

    def __init__(self, paths_obj: Paths) -> None:
        self.paths = paths_obj
        self.paths.chats.mkdir(parents=True, exist_ok=True)

    def _file(self, chat_id: str) -> Path:
        safe = re.sub(r"[^a-zA-Z0-9_-]", "", chat_id) or "chat"
        return self.paths.chats / f"{safe}.json"

    def list(self, limit: int = 40) -> list[dict]:
        items = []
        try:
            files = sorted(
                self.paths.chats.glob("*.json"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
        except OSError:
            return []
        for file in files[:limit]:
            data = read_json(file, None)
            if not isinstance(data, dict) or not isinstance(data.get("messages"), list):
                continue
            items.append(
                {
                    "id": data.get("id") or file.stem,
                    "title": data.get("title") or "Untitled",
                    "updated_at": data.get("updated_at"),
                    "messages": data["messages"],
                }
            )
        return items

    def load(self, chat_id: str) -> dict | None:
        data = read_json(self._file(chat_id), None)
        if isinstance(data, dict) and isinstance(data.get("messages"), list):
            return data
        return None

    def save(self, chat: dict) -> dict:
        chat["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        write_json(self._file(chat["id"]), chat)
        return chat

    def new(self) -> dict:
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        return {
            "id": uuid.uuid4().hex[:12],
            "title": "New conversation",
            "created_at": now,
            "updated_at": now,
            "messages": [],
        }


# --------------------------------------------------------------------------
# small widgets
# --------------------------------------------------------------------------
class ScrollFrame(tk.Frame):
    """A vertically scrollable frame for long settings pages."""

    def __init__(self, master: tk.Misc, bg: str = BG, **kwargs) -> None:
        super().__init__(master, bg=bg, **kwargs)
        self.canvas = tk.Canvas(self, bg=bg, highlightthickness=0, bd=0)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = tk.Frame(self.canvas, bg=bg)
        self._window = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        self.inner.bind("<Configure>", self._on_inner)
        self.canvas.bind("<Configure>", self._on_canvas)
        self.canvas.bind("<Enter>", lambda _e: self._bind_wheel(True))
        self.canvas.bind("<Leave>", lambda _e: self._bind_wheel(False))
        self.inner.bind("<Enter>", lambda _e: self._bind_wheel(True))

    def _on_inner(self, _event=None) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas(self, event) -> None:
        self.canvas.itemconfigure(self._window, width=event.width)

    def _bind_wheel(self, active: bool) -> None:
        if active:
            self.canvas.bind_all("<MouseWheel>", self._wheel)
            self.canvas.bind_all("<Button-4>", self._wheel_linux)
            self.canvas.bind_all("<Button-5>", self._wheel_linux)
        else:
            self.canvas.unbind_all("<MouseWheel>")
            self.canvas.unbind_all("<Button-4>")
            self.canvas.unbind_all("<Button-5>")

    def _wheel(self, event) -> None:
        delta = event.delta
        if platform.system() == "Darwin":
            self.canvas.yview_scroll(-int(delta), "units")
        else:
            self.canvas.yview_scroll(int(-delta / 120), "units")

    def _wheel_linux(self, event) -> None:
        self.canvas.yview_scroll(-1 if event.num == 4 else 1, "units")


class HoverLabel(tk.Label):
    """Flat label that behaves like a quiet text button."""

    def __init__(self, master, *, bg: str, hover_bg: str, fg: str, hover_fg: str | None = None,
                 command: Callable[[], None], **kwargs) -> None:
        super().__init__(master, bg=bg, fg=fg, cursor="hand2", **kwargs)
        self._bg = bg
        self._hover_bg = hover_bg
        self._fg = fg
        self._hover_fg = hover_fg or fg
        self._command = command
        self.bind("<Enter>", self._enter)
        self.bind("<Leave>", self._leave)
        self.bind("<Button-1>", self._click)

    def _enter(self, _event=None) -> None:
        self.configure(bg=self._hover_bg, fg=self._hover_fg)

    def _leave(self, _event=None) -> None:
        self.configure(bg=self._bg, fg=self._fg)

    def _click(self, _event=None) -> None:
        self._command()

    def set_bg(self, bg: str, hover_bg: str) -> None:
        self._bg = bg
        self._hover_bg = hover_bg
        self.configure(bg=bg)


class Card(tk.Frame):
    def __init__(self, master, **kwargs) -> None:
        super().__init__(
            master,
            bg=CARD,
            highlightbackground=BORDER,
            highlightthickness=1,
            bd=0,
            **kwargs,
        )


# --------------------------------------------------------------------------
# main application
# --------------------------------------------------------------------------
class SiderisApp:
    def __init__(self, root: tk.Tk, *, page: str = "chat") -> None:
        self.root = root
        self.closing = False
        self.store = SettingsStore()
        self.settings = self.store.load()
        self.paths = self.store.paths
        self.paths.ensure()
        self.models = ModelManager(self.paths.root)
        self.chats = ChatStore(self.paths)
        self.training = TrainingManager(self.paths.root, self.store)
        self.chat_record: dict = self._initial_chat()
        self.chat_messages: list[dict] = list(self.chat_record.get("messages", []))
        self._streaming = False
        self._stream_cancel = False
        self._response = None
        self._pending_chat: dict | None = None
        self._queue: queue.Queue = queue.Queue()
        self._setup_queue: queue.Queue = queue.Queue()
        self._setup_running = False
        self._setup_process = None
        self._action_frames: list[tk.Frame] = []
        self._images: list[tk.PhotoImage] = []
        self._toast_after = None
        self._training_after = None
        self._page = "chat"

        self.root.title(f"{__app_name__} {__version__}")
        self.root.configure(bg=BG)
        self.root.geometry("1020x700")
        self.root.minsize(900, 620)
        self._center_window()
        self._set_icon()

        self._pick_fonts()
        self._build_styles()
        self._start_local_service()
        self._build_shell()
        self._build_chat_page()
        self._build_setup_page()
        self._build_training_page()
        self._build_settings_page()
        self._show_page(page)
        self._connect_server_poll()
        self._poll_chat()
        self._poll_setup()
        self._refresh_training()
        self._refresh_recents()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(120, self._focus_composer)

    # -- startup helpers -------------------------------------------------
    def _center_window(self) -> None:
        self.root.update_idletasks()
        width, height = 1020, 700
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        x = max((screen_w - width) // 2, 0)
        y = max((screen_h - height) // 3, 0)
        self.root.geometry(f"{width}x{height}+{x}+{y}")

    def _asset(self, name: str) -> Path:
        return asset_dir() / name

    def _set_icon(self) -> None:
        try:
            for name in ("logo-256.png", "logo.png"):
                file = self._asset(name)
                if file.is_file():
                    image = tk.PhotoImage(file=str(file))
                    self._images.append(image)
            if self._images:
                self.root.iconphoto(True, *self._images)
        except Exception:
            pass

    def _pick_fonts(self) -> None:
        try:
            available = set(tkfont.families(self.root))
        except Exception:
            available = set()
        ui = _pick_family(UI_FAMILIES, available, "TkDefaultFont")
        display = _pick_family(DISPLAY_FAMILIES, available, ui)
        mono = _pick_family(MONO_FAMILIES, available, "TkFixedFont")
        self.font_ui = tkfont.Font(family=ui, size=10)
        self.font_ui_small = tkfont.Font(family=ui, size=9)
        self.font_ui_tiny = tkfont.Font(family=ui, size=8)
        self.font_ui_bold = tkfont.Font(family=ui, size=10, weight="bold")
        self.font_composer = tkfont.Font(family=ui, size=11)
        self.font_display = tkfont.Font(family=display, size=26)
        self.font_display_small = tkfont.Font(family=display, size=19)
        self.font_display_tiny = tkfont.Font(family=display, size=15)
        self.font_display_bold = tkfont.Font(family=display, size=16, weight="bold")
        self.font_mono = tkfont.Font(family=mono, size=9)
        self.font_mono_small = tkfont.Font(family=mono, size=8)
        self.font_icon = tkfont.Font(family=ui, size=15)
        self.font_send = tkfont.Font(family=ui, size=14, weight="bold")

    def _build_styles(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        def configure(name, **options):
            try:
                style.configure(name, **options)
            except tk.TclError:
                pass

        def mapping(name, **options):
            try:
                style.map(name, **options)
            except tk.TclError:
                pass

        configure(
            "Sideris.TCombobox",
            fieldbackground=WHITE,
            background=WHITE,
            foreground=TEXT,
            arrowcolor=MUTED,
            bordercolor=BORDER,
            lightcolor=WHITE,
            darkcolor=WHITE,
            padding=(8, 5),
            font=self.font_ui,
        )
        mapping(
            "Sideris.TCombobox",
            fieldbackground=[("readonly", WHITE)],
            foreground=[("readonly", TEXT)],
            bordercolor=[("focus", ACCENT)],
        )
        configure(
            "Sideris.TCheckbutton",
            background=CARD,
            foreground=TEXT,
            font=self.font_ui,
            focuscolor=CARD,
        )
        mapping(
            "Sideris.TCheckbutton",
            background=[("active", CARD), ("!active", CARD)],
            foreground=[("disabled", FAINT)],
        )
        configure(
            "Sideris.Horizontal.TProgressbar",
            troughcolor="#EDE8DC",
            background=ACCENT,
            bordercolor="#EDE8DC",
            lightcolor=ACCENT,
            darkcolor=ACCENT,
            thickness=6,
        )
        configure(
            "Sideris.TSpinbox",
            fieldbackground=WHITE,
            background=WHITE,
            foreground=TEXT,
            bordercolor=BORDER,
            arrowcolor=MUTED,
            padding=(6, 4),
            font=self.font_ui_small,
        )
        configure(
            "Sideris.Vertical.TScrollbar",
            background=BORDER_STRONG,
            troughcolor=BG,
            bordercolor=BG,
            arrowcolor=MUTED,
        )
        for option, value in (
            ("*TCombobox*Listbox.background", WHITE),
            ("*TCombobox*Listbox.foreground", TEXT),
            ("*TCombobox*Listbox.selectBackground", ACCENT_SOFT),
            ("*TCombobox*Listbox.selectForeground", TEXT),
            ("*TCombobox*Listbox.font", self.font_ui),
        ):
            try:
                self.root.option_add(option, value)
            except tk.TclError:
                pass

    def _start_local_service(self) -> None:
        self.runtime = LocalRuntime(self.paths.root)
        self.server_port = self.settings.port
        self._owns_server = False
        self._server = None
        self._server_ready = False
        existing = probe_server(self.settings.port)
        if existing:
            self.server_port = self.settings.port
            self._server_ready = True
            return
        port = self.settings.port
        try:
            self._server = create_server(self.runtime, port=port)
        except OSError:
            try:
                self._server = create_server(self.runtime, port=0)
            except OSError:
                self._server = None
        if self._server is not None:
            self.server_port = self._server.server_address[1]
            self._owns_server = True
            self._server_ready = True
            thread = threading.Thread(
                target=self._server.serve_forever,
                kwargs={"poll_interval": 0.3},
                daemon=True,
            )
            thread.start()

    def _initial_chat(self) -> dict:
        chat_id = self.settings.last_chat_id
        if chat_id:
            loaded = self.chats.load(chat_id)
            if loaded and loaded.get("messages"):
                return loaded
        return self.chats.new()

    # -- shell -----------------------------------------------------------
    def _build_shell(self) -> None:
        self.sidebar = tk.Frame(self.root, bg=SIDEBAR, width=238)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)
        self.main = tk.Frame(self.root, bg=BG)
        self.main.pack(side="left", fill="both", expand=True)
        self._build_sidebar()
        self.page_host = tk.Frame(self.main, bg=BG)
        self.page_host.pack(fill="both", expand=True)

    def _build_sidebar(self) -> None:
        brand = tk.Frame(self.sidebar, bg=SIDEBAR)
        brand.pack(fill="x", padx=18, pady=(20, 16))
        logo = None
        try:
            logo_file = self._asset("logo-256.png")
            if logo_file.is_file():
                image = tk.PhotoImage(file=str(logo_file))
                image = image.subsample(4, 4)
                self._images.append(image)
                logo = tk.Label(brand, image=image, bg=SIDEBAR, bd=0)
        except Exception:
            logo = None
        if logo is not None:
            logo.pack(side="left", padx=(0, 8))
        else:
            tk.Label(brand, text="✦", bg=SIDEBAR, fg=ACCENT, font=self.font_display_bold).pack(side="left")
        text = tk.Frame(brand, bg=SIDEBAR)
        text.pack(side="left")
        tk.Label(text, text=__app_name__, bg=SIDEBAR, fg=TEXT, font=self.font_display_bold, anchor="w").pack(fill="x")
        tk.Label(
            text, text="P R I V A T E   A I", bg=SIDEBAR, fg=FAINT,
            font=self.font_ui_tiny, anchor="w",
        ).pack(fill="x", pady=(1, 0))

        self.new_chat_button = HoverLabel(
            self.sidebar,
            text="+   New conversation",
            bg=ACCENT,
            hover_bg=ACCENT_HOVER,
            fg=WHITE,
            hover_fg=WHITE,
            command=self._new_chat,
            font=self.font_ui_bold,
            anchor="w",
            padx=14,
            pady=9,
            bd=0,
        )
        self.new_chat_button.pack(fill="x", padx=14, pady=(2, 14))

        self.nav_widgets: dict[str, tuple[tk.Frame, tk.Frame, tk.Label]] = {}
        for key, label in (("chat", "Chat"), ("setup", "Setup"), ("training", "Training"), ("settings", "Settings")):
            row = tk.Frame(self.sidebar, bg=SIDEBAR)
            row.pack(fill="x", padx=10, pady=1)
            accent_bar = tk.Frame(row, bg=SIDEBAR, width=3)
            accent_bar.pack(side="left", fill="y")
            label_widget = tk.Label(
                row,
                text=label,
                bg=SIDEBAR,
                fg=MUTED,
                font=self.font_ui,
                anchor="w",
                padx=12,
                pady=7,
                cursor="hand2",
            )
            label_widget.pack(side="left", fill="x", expand=True)
            row.bind("<Enter>", lambda _e, r=row, l=label_widget: self._nav_hover(r, l, True))
            row.bind("<Leave>", lambda _e, r=row, l=label_widget: self._nav_hover(r, l, False))
            for widget in (row, accent_bar, label_widget):
                widget.bind("<Button-1>", lambda _e, k=key: self._show_page(k))
            self.nav_widgets[key] = (row, accent_bar, label_widget)

        recents_header = tk.Frame(self.sidebar, bg=SIDEBAR)
        recents_header.pack(fill="x", padx=22, pady=(20, 4))
        tk.Label(
            recents_header, text="R E C E N T", bg=SIDEBAR, fg=FAINT,
            font=self.font_ui_tiny, anchor="w",
        ).pack(fill="x")

        self.recents_frame = tk.Frame(self.sidebar, bg=SIDEBAR)
        self.recents_frame.pack(fill="both", expand=True, padx=10, pady=(0, 6))

        footer = tk.Frame(self.sidebar, bg=SIDEBAR)
        footer.pack(side="bottom", fill="x", padx=18, pady=(6, 14))
        status_row = tk.Frame(footer, bg=SIDEBAR)
        status_row.pack(fill="x")
        self.status_dot = tk.Label(status_row, text="●", bg=SIDEBAR, fg=FAINT, font=self.font_ui_tiny)
        self.status_dot.pack(side="left", padx=(0, 5))
        self.status_text = tk.Label(
            status_row, text="Checking…", bg=SIDEBAR, fg=MUTED,
            font=self.font_ui_tiny, anchor="w",
        )
        self.status_text.pack(side="left", fill="x")
        self.toast_label = tk.Label(
            footer, text="", bg=SIDEBAR, fg=ACCENT, font=self.font_ui_tiny,
            anchor="w", wraplength=200, justify="left",
        )
        self.toast_label.pack(fill="x", pady=(4, 0))
        tk.Label(
            footer, text=f"{__release_name__} {__version__}", bg=SIDEBAR, fg=FAINT,
            font=self.font_ui_tiny, anchor="w",
        ).pack(fill="x", pady=(3, 0))

    def _nav_hover(self, row: tk.Frame, label: tk.Label, entering: bool) -> None:
        if self._page == self._nav_key_for(row):
            return
        bg = SIDEBAR_HOVER if entering else SIDEBAR
        row.configure(bg=bg)
        label.configure(bg=bg)

    def _nav_key_for(self, row: tk.Frame) -> str | None:
        for key, (candidate, _bar, _label) in self.nav_widgets.items():
            if candidate is row:
                return key
        return None

    def _update_nav(self) -> None:
        for key, (row, bar, label) in getattr(self, "nav_widgets", {}).items():
            active = key == self._page
            row.configure(bg=SIDEBAR_ACTIVE if active else SIDEBAR)
            label.configure(
                bg=SIDEBAR_ACTIVE if active else SIDEBAR,
                fg=TEXT if active else MUTED,
                font=self.font_ui_bold if active else self.font_ui,
            )
            bar.configure(bg=ACCENT if active else SIDEBAR)

    def _show_page(self, key: str) -> None:
        self._page = key
        for frame in self.pages.values():
            frame.pack_forget()
        self.pages[key].pack(fill="both", expand=True)
        self._update_nav()
        if key == "setup":
            self._refresh_setup_status()
        elif key == "training":
            self._refresh_training()
        elif key == "settings":
            self._refresh_settings()
        elif key == "chat":
            self._focus_composer()

    def _toast(self, message: str, seconds: int = 5) -> None:
        self.toast_label.configure(text=message)
        if self._toast_after:
            try:
                self.root.after_cancel(self._toast_after)
            except Exception:
                pass
        self._toast_after = self.root.after(seconds * 1000, lambda: self.toast_label.configure(text=""))

    def _focus_composer(self) -> None:
        if hasattr(self, "composer_text") and not self._streaming:
            self.composer_text.focus_set()

    # -- shared page header ----------------------------------------------
    def _page_header(self, parent: tk.Misc, title: str, subtitle: str) -> tk.Frame:
        header = tk.Frame(parent, bg=BG)
        header.pack(fill="x", padx=34, pady=(26, 8))
        tk.Label(
            header, text=title, bg=BG, fg=TEXT, font=self.font_display_small,
            anchor="w",
        ).pack(anchor="w")
        tk.Label(
            header, text=subtitle, bg=BG, fg=MUTED, font=self.font_ui,
            anchor="w", justify="left",
        ).pack(anchor="w", pady=(3, 0))
        return header

    def _card_title(self, card: tk.Frame, title: str, description: str = "") -> tk.Frame:
        head = tk.Frame(card, bg=CARD)
        head.pack(fill="x", padx=18, pady=(16, 4))
        tk.Label(head, text=title, bg=CARD, fg=TEXT, font=self.font_ui_bold, anchor="w").pack(anchor="w")
        if description:
            tk.Label(
                head, text=description, bg=CARD, fg=MUTED, font=self.font_ui_small,
                anchor="w", justify="left", wraplength=620,
            ).pack(anchor="w", pady=(2, 0))
        return head

    # ==================================================================
    # chat page
    # ==================================================================
    def _build_chat_page(self) -> None:
        self.pages = getattr(self, "pages", {})
        page = tk.Frame(self.page_host, bg=BG)
        self.pages["chat"] = page

        topbar = tk.Frame(page, bg=BG)
        topbar.pack(fill="x", padx=32, pady=(20, 6))
        left = tk.Frame(topbar, bg=BG)
        left.pack(side="left", fill="x", expand=True)
        self.chat_title_label = tk.Label(
            left, text=self.chat_record.get("title") or "Chat", bg=BG, fg=TEXT,
            font=self.font_display_tiny, anchor="w",
        )
        self.chat_title_label.pack(anchor="w")
        self.chat_sub_label = tk.Label(
            left, text="On this computer. Nothing leaves it.", bg=BG, fg=FAINT,
            font=self.font_ui_tiny, anchor="w",
        )
        self.chat_sub_label.pack(anchor="w")
        right = tk.Frame(topbar, bg=BG)
        right.pack(side="right")
        self.model_chip = tk.Label(
            right, text="No model", bg=SIDEBAR, fg=MUTED, font=self.font_ui_tiny,
            padx=10, pady=5,
        )
        self.model_chip.pack(side="left", padx=(0, 10))
        self.health_dot = tk.Label(right, text="●", bg=BG, fg=FAINT, font=self.font_ui_tiny)
        self.health_dot.pack(side="left")

        body = tk.Frame(page, bg=BG)
        body.pack(fill="both", expand=True)
        self.empty_frame = tk.Frame(body, bg=BG)
        transcript_holder = tk.Frame(body, bg=BG)
        self.transcript_holder = transcript_holder

        # empty state
        empty_inner = tk.Frame(self.empty_frame, bg=BG)
        empty_inner.pack(expand=True)
        try:
            logo_file = self._asset("logo-256.png")
            if logo_file.is_file():
                image = tk.PhotoImage(file=str(logo_file))
                image = image.subsample(3, 3)
                self._images.append(image)
                tk.Label(empty_inner, image=image, bg=BG, bd=0).pack(pady=(0, 10))
        except Exception:
            pass
        hour = time.localtime().tm_hour
        greeting = "Good morning" if hour < 12 else ("Good afternoon" if hour < 18 else "Good evening")
        tk.Label(
            empty_inner, text=greeting, bg=BG, fg=TEXT, font=self.font_display,
        ).pack()
        tk.Label(
            empty_inner, text="Where should we start?", bg=BG, fg=MUTED,
            font=self.font_ui,
        ).pack(pady=(4, 22))

        cards = tk.Frame(empty_inner, bg=BG)
        cards.pack()
        for index, (title, prompt) in enumerate(SUGGESTIONS):
            card = tk.Frame(
                cards, bg=CARD, highlightbackground=BORDER, highlightthickness=1,
                width=190, height=86, cursor="hand2",
            )
            card.grid(row=0, column=index, padx=5, sticky="nsew")
            card.grid_propagate(False)
            inner = tk.Frame(card, bg=CARD)
            inner.pack(fill="both", expand=True, padx=12, pady=10)
            heading = tk.Label(inner, text=title, bg=CARD, fg=TEXT, font=self.font_ui_bold, anchor="w")
            heading.pack(anchor="w")
            body_label = tk.Label(
                inner, text=prompt, bg=CARD, fg=MUTED, font=self.font_ui_tiny,
                anchor="w", justify="left", wraplength=162,
            )
            body_label.pack(anchor="w", pady=(4, 0))
            for widget in (card, inner, heading, body_label):
                widget.bind("<Button-1>", lambda _e, p=prompt: self._use_suggestion(p))
                widget.bind("<Enter>", lambda _e, c=card: c.configure(highlightbackground=ACCENT))
                widget.bind("<Leave>", lambda _e, c=card: c.configure(highlightbackground=BORDER))

        # transcript
        self.transcript = tk.Text(
            transcript_holder,
            bg=BG,
            fg=TEXT,
            bd=0,
            highlightthickness=0,
            wrap="word",
            padx=30,
            pady=10,
            cursor="arrow",
            font=self.font_ui,
            spacing1=1,
            spacing3=2,
            insertwidth=1,
            selectbackground=ACCENT_SOFT,
            selectforeground=TEXT,
        )
        self.transcript.pack(fill="both", expand=True)
        self.transcript.configure(state="disabled")
        self._configure_transcript_tags()
        self.transcript.bind("<MouseWheel>", self._transcript_wheel)
        self.transcript.bind("<Button-4>", self._transcript_wheel)
        self.transcript.bind("<Button-5>", self._transcript_wheel)

        # composer
        composer_outer = tk.Frame(page, bg=BG)
        composer_outer.pack(fill="x", padx=32, pady=(4, 18))
        self.composer = tk.Frame(
            composer_outer, bg=WHITE, highlightbackground=BORDER_STRONG, highlightthickness=1,
        )
        self.composer.pack(fill="x")
        self.composer_text = tk.Text(
            self.composer,
            height=2,
            bg=WHITE,
            fg=TEXT,
            bd=0,
            highlightthickness=0,
            wrap="word",
            font=self.font_composer,
            padx=14,
            pady=11,
            insertbackground=TEXT,
            spacing1=1,
            spacing3=2,
            undo=True,
            maxundo=50,
        )
        self.composer_text.pack(fill="x", padx=2, pady=(2, 0))
        self.composer_text.tag_configure("placeholder", foreground=FAINT)
        controls = tk.Frame(self.composer, bg=WHITE)
        controls.pack(fill="x", padx=10, pady=(0, 8))
        self.level_var = tk.StringVar(value=self._level_label(self.settings.thinking_level))
        self.level_box = ttk.Combobox(
            controls,
            textvariable=self.level_var,
            values=[f"{key}  ·  {LEVELS[key]['max_tokens']} tokens" for key in LEVELS],
            state="readonly",
            width=17,
            style="Sideris.TCombobox",
        )
        self.level_box.pack(side="left")
        self.level_box.bind("<<ComboboxSelected>>", self._on_level_change)
        self.level_hint = tk.Label(
            controls, text="", bg=WHITE, fg=FAINT, font=self.font_ui_tiny,
        )
        self.level_hint.pack(side="left", padx=10)
        self.send_button = HoverLabel(
            controls,
            text="↑",
            bg=ACCENT,
            hover_bg=ACCENT_HOVER,
            fg=WHITE,
            command=self._send_or_stop,
            font=self.font_send,
            padx=11,
            pady=2,
            bd=0,
        )
        self.send_button.pack(side="right")
        tk.Label(
            controls, text="Enter to send · Shift+Enter for a new line",
            bg=WHITE, fg=FAINT, font=self.font_ui_tiny,
        ).pack(side="right", padx=10)
        self._update_level_hint()

        self.composer_text.bind("<Return>", self._on_composer_return)
        self.composer_text.bind("<KeyRelease>", self._on_composer_key)
        self.composer_text.bind("<FocusIn>", self._clear_placeholder)
        self.composer_text.bind("<FocusOut>", self._restore_placeholder)
        self._placeholder_active = False
        self._set_placeholder()

        self._render_history()
        if self.chat_messages:
            self.empty_frame.pack_forget()
            transcript_holder.pack(fill="both", expand=True)
        else:
            transcript_holder.pack_forget()
            self.empty_frame.pack(fill="both", expand=True)

    def _configure_transcript_tags(self) -> None:
        t = self.transcript
        t.tag_configure("user_caption", font=self.font_ui_tiny, foreground=FAINT,
                        justify="right", spacing1=18, spacing3=2, rmargin=62)
        t.tag_configure(
            "user_body", font=self.font_ui, foreground=TEXT, background=USER_BG,
            justify="right", lmargin1=150, lmargin2=150, rmargin=62, spacing3=12,
        )
        t.tag_configure("assistant_caption", font=self.font_ui_bold, foreground=ACCENT,
                        spacing1=16, spacing3=3, lmargin1=62, lmargin2=62)
        t.tag_configure(
            "msg_assistant", font=self.font_ui, foreground=TEXT,
            lmargin1=62, lmargin2=62, rmargin=62, spacing1=2, spacing3=6,
        )
        t.tag_configure("md_h1", font=self.font_display_small, foreground=TEXT,
                        lmargin1=62, lmargin2=62, rmargin=62, spacing1=16, spacing3=6)
        t.tag_configure("md_h2", font=self.font_display_tiny, foreground=TEXT,
                        lmargin1=62, lmargin2=62, rmargin=62, spacing1=12, spacing3=4)
        t.tag_configure("md_h3", font=self.font_ui_bold, foreground=TEXT,
                        lmargin1=62, lmargin2=62, rmargin=62, spacing1=10, spacing3=3)
        t.tag_configure("md_bold", font=self.font_ui_bold)
        t.tag_configure("md_italic", font=(self.font_ui.actual("family"), 10, "italic"))
        t.tag_configure("md_inline_code", font=self.font_mono, background=CODE_BG,
                        foreground="#7A3E22")
        t.tag_configure("md_code", font=self.font_mono, background=CODE_BG,
                        lmargin1=74, lmargin2=74, rmargin=74, spacing1=2, spacing3=2)
        t.tag_configure("md_quote", font=(self.font_ui.actual("family"), 10, "italic"),
                        foreground=MUTED, lmargin1=82, lmargin2=82, rmargin=62, spacing1=2)
        t.tag_configure("md_bullet", lmargin1=78, lmargin2=98, rmargin=62, spacing3=2)
        t.tag_configure("stream_status", font=(self.font_ui.actual("family"), 10, "italic"),
                        foreground=FAINT, lmargin1=62, lmargin2=62, spacing1=8)
        t.tag_configure("error_text", foreground=RED, lmargin1=62, lmargin2=62,
                        rmargin=62, spacing1=6, spacing3=10)
        t.tag_configure("placeholder", foreground=FAINT)

    def _transcript_wheel(self, event) -> None:
        if event.num == 4:
            self.transcript.yview_scroll(-2, "units")
        elif event.num == 5:
            self.transcript.yview_scroll(2, "units")
        else:
            delta = event.delta
            if platform.system() == "Darwin":
                self.transcript.yview_scroll(-int(delta), "units")
            else:
                self.transcript.yview_scroll(int(-delta / 120) * 2, "units")
        return "break"

    def _append(self, text: str, tags=()) -> None:
        self.transcript.configure(state="normal")
        self.transcript.insert("end", text, tags)
        self.transcript.configure(state="disabled")
        self.transcript.see("end")

    def _clear_transcript(self) -> None:
        self.transcript.configure(state="normal")
        self.transcript.delete("1.0", "end")
        self.transcript.configure(state="disabled")
        for frame in self._action_frames:
            try:
                frame.destroy()
            except Exception:
                pass
        self._action_frames.clear()

    def _render_history(self) -> None:
        self._clear_transcript()
        for message in self.chat_messages:
            role = message.get("role")
            content = str(message.get("content") or "")
            if role == "user":
                self._insert_user(content)
            elif role == "assistant":
                self._insert_assistant(content, actions=False)
        if self.chat_messages:
            self.empty_frame.pack_forget()
            self.transcript_holder.pack(fill="both", expand=True)
        else:
            self.transcript_holder.pack_forget()
            self.empty_frame.pack(fill="both", expand=True)

    def _insert_user(self, text: str) -> None:
        self._append("\n", ("user_caption",))
        self._append("You\n", ("user_caption",))
        self._append(text.rstrip("\n") + "\n", ("user_body",))

    def _insert_assistant(self, text: str, *, actions: bool = True) -> None:
        self._append("\n", ("assistant_caption",))
        self._append("Sideris\n", ("assistant_caption",))
        self._insert_markdown(text)
        if actions:
            self._add_action_row(text)

    def _add_action_row(self, text: str) -> None:
        frame = tk.Frame(self.transcript, bg=BG)
        copy_label = tk.Label(
            frame, text="Copy", bg=BG, fg=FAINT, font=self.font_ui_tiny,
            cursor="hand2", padx=0, pady=2,
        )
        copy_label.pack(side="left", padx=(62, 14))
        copy_label.bind("<Button-1>", lambda _e: self._copy(text))
        copy_label.bind("<Enter>", lambda _e: copy_label.configure(fg=MUTED))
        copy_label.bind("<Leave>", lambda _e: copy_label.configure(fg=FAINT))
        regen = tk.Label(
            frame, text="Regenerate", bg=BG, fg=FAINT, font=self.font_ui_tiny,
            cursor="hand2", padx=0, pady=2,
        )
        regen.pack(side="left")
        regen.bind("<Button-1>", lambda _e: self._regenerate())
        regen.bind("<Enter>", lambda _e: regen.configure(fg=MUTED))
        regen.bind("<Leave>", lambda _e: regen.configure(fg=FAINT))
        self.transcript.configure(state="normal")
        self.transcript.window_create("end", window=frame, padx=0, pady=4)
        self.transcript.insert("end", "\n")
        self.transcript.configure(state="disabled")
        self._action_frames.append(frame)

    def _copy(self, text: str) -> None:
        try:
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            self._toast("Copied to clipboard")
        except tk.TclError:
            self._toast("Could not access the clipboard")

    # -- markdown --------------------------------------------------------
    INLINE_RE = re.compile(r"(\*\*[^*\n]+\*\*|`[^`\n]+`|\*[^*\n]+\*|\[[^\]\n]+\]\([^)\s]+\))")

    def _insert_inline(self, text: str, base: str = "msg_assistant") -> None:
        t = self.transcript
        position = 0
        for match in self.INLINE_RE.finditer(text):
            if match.start() > position:
                t.insert("end", text[position : match.start()], (base,))
            token = match.group(0)
            if token.startswith("**"):
                t.insert("end", token[2:-2], (base, "md_bold"))
            elif token.startswith("`"):
                t.insert("end", token[1:-1], (base, "md_inline_code"))
            elif token.startswith("["):
                label, _, url = token[1:].partition("](")
                t.insert("end", label, (base,))
            elif token.startswith("*"):
                t.insert("end", token[1:-1], (base, "md_italic"))
            else:
                t.insert("end", token, (base,))
            position = match.end()
        if position < len(text):
            t.insert("end", text[position:], (base,))

    def _insert_markdown(self, text: str) -> None:
        self.transcript.configure(state="normal")
        try:
            lines = (text or "").replace("\r\n", "\n").split("\n")
            index = 0
            while index < len(lines):
                line = lines[index]
                stripped = line.strip()
                if stripped.startswith("```"):
                    language = stripped[3:].strip()
                    code_lines = []
                    index += 1
                    while index < len(lines) and not lines[index].strip().startswith("```"):
                        code_lines.append(lines[index])
                        index += 1
                    index += 1
                    if language:
                        self.transcript.insert("end", f"{language}\n", ("md_code",))
                    body = "\n".join(code_lines)
                    self.transcript.insert("end", body + "\n", ("md_code",))
                    continue
                if stripped.startswith("#"):
                    level = len(stripped) - len(stripped.lstrip("#"))
                    heading = stripped[level:].strip()
                    tag = "md_h1" if level == 1 else ("md_h2" if level == 2 else "md_h3")
                    self._insert_inline(heading, tag)
                    self.transcript.insert("end", "\n", (tag,))
                    index += 1
                    continue
                bullet = re.match(r"^\s*[-*+]\s+(.*)$", line)
                numbered = re.match(r"^\s*(\d+)[.)]\s+(.*)$", line)
                if bullet:
                    self.transcript.insert("end", "•  ", ("md_bullet",))
                    self._insert_inline(bullet.group(1), "md_bullet")
                    self.transcript.insert("end", "\n", ("md_bullet",))
                    index += 1
                    continue
                if numbered:
                    self.transcript.insert("end", f"{numbered.group(1)}.  ", ("md_bullet",))
                    self._insert_inline(numbered.group(2), "md_bullet")
                    self.transcript.insert("end", "\n", ("md_bullet",))
                    index += 1
                    continue
                if stripped.startswith(">"):
                    quote = stripped.lstrip(">").strip()
                    self._insert_inline(quote, "md_quote")
                    self.transcript.insert("end", "\n", ("md_quote",))
                    index += 1
                    continue
                if stripped in {"---", "***", "___"}:
                    self.transcript.insert("end", "—" * 18 + "\n", ("msg_assistant",))
                    index += 1
                    continue
                if not stripped:
                    self.transcript.insert("end", "\n", ("msg_assistant",))
                    index += 1
                    continue
                self._insert_inline(line, "msg_assistant")
                self.transcript.insert("end", "\n", ("msg_assistant",))
                index += 1
            self.transcript.insert("end", "\n", ("msg_assistant",))
        finally:
            self.transcript.configure(state="disabled")
            self.transcript.see("end")

    # -- composer --------------------------------------------------------
    def _set_placeholder(self) -> None:
        if self.composer_text.get("1.0", "end-1c"):
            return
        self.composer_text.insert("1.0", "Message Sideris…", ("placeholder",))
        self.composer_text.mark_set("insert", "1.0")
        self._placeholder_active = True

    def _clear_placeholder(self, _event=None) -> None:
        if self._placeholder_active:
            self.composer_text.delete("1.0", "end")
            self._placeholder_active = False

    def _restore_placeholder(self, _event=None) -> None:
        if not self.composer_text.get("1.0", "end-1c").strip():
            self.composer_text.delete("1.0", "end")
            self._set_placeholder()

    def _on_composer_key(self, _event=None) -> None:
        if self._placeholder_active:
            return
        content = self.composer_text.get("1.0", "end-1c")
        lines = max(1, content.count("\n") + 1)
        self.composer_text.configure(height=max(2, min(lines + 1, 8)))

    def _on_composer_return(self, event):
        if event.state & 0x0001:  # Shift
            return None
        self._send_or_stop()
        return "break"

    def _level_label(self, key: str) -> str:
        key = key if key in LEVELS else "medium"
        return f"{key}  ·  {LEVELS[key]['max_tokens']} tokens"

    def _update_level_hint(self) -> None:
        key = self.level_var.get().split()[0] if self.level_var.get() else "medium"
        info = level_info(key)
        self.level_hint.configure(text=f"{info['max_tokens']} tokens · {info['passes']} pass" + (
            "es" if info["passes"] > 1 else ""
        ))

    def _on_level_change(self, _event=None) -> None:
        key = self.level_var.get().split()[0]
        self.settings.thinking_level = key if key in LEVELS else "medium"
        self.store.save(self.settings)
        self._update_level_hint()

    def _use_suggestion(self, prompt: str) -> None:
        self._clear_placeholder()
        self.composer_text.delete("1.0", "end")
        self.composer_text.insert("1.0", prompt)
        self.composer_text.focus_set()
        self.composer_text.mark_set("insert", "end")
        self._on_composer_key()

    # -- sending ---------------------------------------------------------
    def _send_or_stop(self) -> None:
        if self._streaming:
            self._cancel_stream()
        else:
            self._send()

    def _send(self) -> None:
        if self._streaming:
            return
        text = self.composer_text.get("1.0", "end-1c").strip()
        if not text or self._placeholder_active:
            return
        if not self._server_ready or not self.server_port:
            self._append("\nSideris's local service is not running. Restart the app, or run 'sideris serve'.\n",
                         ("error_text",))
            return
        self._clear_placeholder()
        self.composer_text.delete("1.0", "end")
        self.composer_text.configure(height=2)
        self._start_generation(text)

    def _start_generation(self, text: str) -> None:
        self.empty_frame.pack_forget()
        self.transcript_holder.pack(fill="both", expand=True)
        if not self.chat_messages:
            title = " ".join(text.split())
            if len(title) > 52:
                title = title[:52].rsplit(" ", 1)[0] + "…"
            self.chat_record["title"] = title
            self.chat_title_label.configure(text=title)
        self.chat_messages.append({"role": "user", "content": text})
        self._insert_user(text)
        self.chat_record["messages"] = self.chat_messages
        self.chats.save(self.chat_record)
        self.settings.last_chat_id = self.chat_record["id"]
        self.store.save(self.settings)
        self._refresh_recents()

        self._streaming = True
        self._stream_cancel = False
        self._got_token = False
        self.send_button.configure(text="Stop", font=self.font_ui_small, padx=9)
        self.composer_text.configure(state="disabled")
        self._append("\n", ("assistant_caption",))
        self._append("Sideris\n", ("assistant_caption",))
        self.transcript.configure(state="normal")
        self.transcript.insert("end", "Thinking…", ("stream_status",))
        self.transcript.mark_set("stream_status_start", "end-1c")
        self.transcript.configure(state="disabled")
        self.transcript.see("end")
        self.transcript.mark_set("assistant_start", "end-1c")
        self.transcript.mark_gravity("assistant_start", "left")

        level = self.level_var.get().split()[0] if self.level_var.get() else "medium"
        payload = {
            "messages": [
                {"role": m["role"], "content": m["content"]}
                for m in self.chat_messages
                if m.get("role") in ("user", "assistant")
            ],
            "level": level,
        }
        self._pending_chat = self.chat_record
        thread = threading.Thread(target=self._chat_worker, args=(payload,), daemon=True)
        thread.start()

    def _chat_worker(self, payload: dict) -> None:
        import urllib.error
        import urllib.request

        url = f"http://127.0.0.1:{self.server_port}/v1/chat/stream"
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
            method="POST",
        )
        try:
            response = urllib.request.urlopen(request, timeout=3600)
        except urllib.error.HTTPError as exc:
            try:
                detail = json.loads(exc.read().decode("utf-8"))
                message = detail.get("error", {}).get("message", str(exc))
            except Exception:
                message = f"HTTP {exc.code}"
            self._queue.put(("error", message))
            return
        except Exception as exc:
            self._queue.put(("error", f"Could not reach the local service: {exc}"))
            return
        self._response = response
        try:
            with response:
                for raw in response:
                    if self._stream_cancel:
                        break
                    line = raw.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    try:
                        event = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    kind = event.get("type")
                    if kind == "status":
                        self._queue.put(("status", event.get("text", "")))
                    elif kind == "token":
                        self._queue.put(("token", event.get("text", "")))
                    elif kind == "done":
                        self._queue.put(("done", event.get("reply", {})))
                    elif kind == "error":
                        self._queue.put(("error", event.get("message", "Unknown error")))
                if self._stream_cancel:
                    self._queue.put(("cancelled", None))
        except Exception as exc:
            if self._stream_cancel:
                self._queue.put(("cancelled", None))
            else:
                self._queue.put(("error", f"The connection to the local service dropped: {exc}"))
        finally:
            self._response = None

    def _cancel_stream(self) -> None:
        self._stream_cancel = True
        response = self._response
        if response is not None:
            try:
                response.close()
            except Exception:
                pass
        if not self.closing:
            self._finish_cancelled()

    def _poll_chat(self) -> None:
        if self.closing:
            return
        try:
            while True:
                kind, payload = self._queue.get_nowait()
                if kind == "status":
                    self._set_stream_status(str(payload))
                elif kind == "token":
                    self._append_token(str(payload))
                elif kind == "done":
                    self._finish_reply(payload if isinstance(payload, dict) else {})
                elif kind == "error":
                    self._finish_error(str(payload))
                elif kind == "cancelled":
                    self._finish_cancelled()
        except queue.Empty:
            pass
        self.root.after(30, self._poll_chat)

    def _set_stream_status(self, text: str) -> None:
        if not text or not self._streaming or self._got_token:
            return
        try:
            self.transcript.configure(state="normal")
            self.transcript.delete("stream_status_start", "end-1c")
            self.transcript.insert(
                "end", ("…  " + text) if not text.endswith("…") else text, ("stream_status",)
            )
            self.transcript.configure(state="disabled")
            self.transcript.see("end")
        except tk.TclError:
            pass

    def _append_token(self, token: str) -> None:
        if not self._streaming or not token:
            return
        try:
            if not self._got_token:
                self._got_token = True
                if self.transcript.tag_ranges("stream_status"):
                    self.transcript.configure(state="normal")
                    self.transcript.delete("stream_status_start", "end-1c")
                    self.transcript.configure(state="disabled")
            self.transcript.configure(state="normal")
            self.transcript.insert("end", token, ("msg_assistant",))
            self.transcript.configure(state="disabled")
            self.transcript.see("end")
        except tk.TclError:
            pass

    def _finish_reply(self, reply: dict) -> None:
        if not self._streaming:
            return
        text = str(reply.get("text") or reply.get("reply") or "")
        self._end_streaming_state()
        try:
            self.transcript.configure(state="normal")
            self.transcript.delete("assistant_start", "end-1c")
            self.transcript.configure(state="disabled")
        except tk.TclError:
            pass
        if not text:
            text = "The local model returned an empty answer. Try again, or choose a lower thinking level."
        self._insert_markdown(text)
        self._add_action_row(text)
        self.chat_messages.append({"role": "assistant", "content": text})
        self.chat_record["messages"] = self.chat_messages
        self.chats.save(self.chat_record)
        self._refresh_recents()
        meta = []
        if reply.get("tools"):
            meta.append("tools: " + ", ".join(str(t) for t in reply["tools"]))
        if reply.get("model"):
            meta.append(str(reply["model"]))
        if reply.get("elapsed"):
            meta.append(f"{float(reply['elapsed']):.1f}s")
        if meta:
            self.chat_sub_label.configure(text="  ·  ".join(meta))
        self._focus_composer()

    def _finish_error(self, message: str) -> None:
        if not self._streaming:
            return
        self._end_streaming_state()
        try:
            self.transcript.configure(state="normal")
            self.transcript.delete("assistant_start", "end-1c")
            self.transcript.insert("end", "\n" + message + "\n", ("error_text",))
            self.transcript.configure(state="disabled")
            self.transcript.see("end")
        except tk.TclError:
            pass
        if self.chat_messages and self.chat_messages[-1].get("role") == "user":
            pass  # keep the user's message so they can retry
        self._toast(message[:120])

    def _finish_cancelled(self) -> None:
        if not self._streaming:
            return
        self._end_streaming_state()
        try:
            if not self._got_token:
                self.transcript.configure(state="normal")
                self.transcript.delete("stream_status_start", "end-1c")
                self.transcript.configure(state="disabled")
            self._append("\nStopped.\n", ("error_text",))
        except tk.TclError:
            pass
        if self.chat_messages and self.chat_messages[-1].get("role") == "user":
            # Remove the unfinished user turn so a retry is clean.
            self.chat_messages.pop()
            self.chat_record["messages"] = self.chat_messages
            self.chats.save(self.chat_record)

    def _end_streaming_state(self) -> None:
        self._streaming = False
        self._stream_cancel = False
        self.send_button.configure(text="↑", font=self.font_send, padx=11)
        self.composer_text.configure(state="normal")
        self._set_placeholder()
        try:
            self.transcript.configure(state="disabled")
        except tk.TclError:
            pass

    def _regenerate(self) -> None:
        if self._streaming:
            return
        if self.chat_messages and self.chat_messages[-1].get("role") == "assistant":
            self.chat_messages.pop()
        if not self.chat_messages or self.chat_messages[-1].get("role") != "user":
            return
        last = self.chat_messages.pop()
        self._render_history()
        self._start_generation(last["content"])

    def _new_chat(self) -> None:
        if self._streaming:
            self._cancel_stream()
        self.chat_record = self.chats.new()
        self.chat_messages = []
        self._render_history()
        self.chat_title_label.configure(text="New conversation")
        self.chat_sub_label.configure(text="On this computer. Nothing leaves it.")
        self._show_page("chat")
        self._refresh_recents()
        self._focus_composer()

    def _refresh_recents(self) -> None:
        for child in self.recents_frame.winfo_children():
            child.destroy()
        chats = self.chats.list(limit=8)
        if not chats:
            tk.Label(
                self.recents_frame, text="No conversations yet", bg=SIDEBAR, fg=FAINT,
                font=self.font_ui_tiny, anchor="w",
            ).pack(fill="x", padx=12, pady=4)
            return
        for item in chats:
            active = item["id"] == self.chat_record.get("id")
            row = tk.Frame(self.recents_frame, bg=SIDEBAR_ACTIVE if active else SIDEBAR)
            row.pack(fill="x")
            label = tk.Label(
                row,
                text=item["title"],
                bg=SIDEBAR_ACTIVE if active else SIDEBAR,
                fg=TEXT if active else MUTED,
                font=self.font_ui_tiny,
                anchor="w",
                padx=12,
                pady=5,
                cursor="hand2",
            )
            label.pack(fill="x")
            for widget in (row, label):
                widget.bind("<Button-1>", lambda _e, cid=item["id"]: self._load_chat(cid))
            row.bind("<Enter>", lambda _e, r=row, l=label: self._recent_hover(r, l, True))
            row.bind("<Leave>", lambda _e, r=row, l=label: self._recent_hover(r, l, False))

    def _recent_hover(self, row: tk.Frame, label: tk.Label, entering: bool) -> None:
        bg = SIDEBAR_HOVER if entering else row.cget("bg")
        label.configure(bg=bg)
        row.configure(bg=bg)

    def _load_chat(self, chat_id: str) -> None:
        if self._streaming:
            self._cancel_stream()
        data = self.chats.load(chat_id)
        if not data:
            return
        self.chat_record = data
        self.chat_messages = list(data.get("messages", []))
        self.chat_title_label.configure(text=data.get("title") or "Chat")
        self.settings.last_chat_id = chat_id
        self.store.save(self.settings)
        self._render_history()
        self._show_page("chat")
        self._refresh_recents()

    # -- health polling ---------------------------------------------------
    def _connect_server_poll(self) -> None:
        self._update_model_chip()
        if self._server is not None:
            self._server_ready = True
        elif not self._server_ready:
            self._server_ready = bool(probe_server(self.server_port))
        if self._server_ready:
            self.health_dot.configure(fg=GREEN)
            self.status_dot.configure(fg=GREEN if self._model_ready() else ACCENT)
            self.status_text.configure(text="Ready · local" if self._model_ready() else "Setup needed")
        else:
            self.health_dot.configure(fg=RED)
            self.status_text.configure(text="Service offline")
        self.root.after(4000, self._connect_server_poll)

    def _model_ready(self) -> bool:
        return bool(self.models.select(self.settings.model))

    def _update_model_chip(self) -> None:
        selected = self.models.select(self.settings.model)
        if selected:
            key = selected[0]
            try:
                name = get_spec(key).display_name
            except KeyError:
                name = Path(selected[1]).stem
            if len(name) > 22:
                name = name[:22] + "…"
            self.model_chip.configure(text=f"{name}  ·  local", fg=MUTED)
        else:
            self.model_chip.configure(text="No model installed", fg=FAINT)

    # ==================================================================
    # setup page
    # ==================================================================
    def _build_setup_page(self) -> None:
        page = tk.Frame(self.page_host, bg=BG)
        self.pages["setup"] = page
        scroll = ScrollFrame(page, bg=BG)
        scroll.pack(fill="both", expand=True)
        content = scroll.inner

        header = tk.Frame(content, bg=BG)
        header.pack(fill="x", padx=34, pady=(26, 10))
        tk.Label(header, text="Set up Sideris", bg=BG, fg=TEXT,
                 font=self.font_display_small, anchor="w").pack(anchor="w")
        tk.Label(
            header,
            text="Your computer is the data centre. Choose a model, grant only the "
                 "permissions you want, and Sideris will do the rest - resumably.",
            bg=BG, fg=MUTED, font=self.font_ui, anchor="w", justify="left",
            wraplength=640,
        ).pack(anchor="w", pady=(4, 0))

        model_card = Card(content)
        model_card.pack(fill="x", padx=34, pady=8)
        self._card_title(
            model_card, "1 · The model",
            "Downloaded once from Hugging Face, then runs entirely offline. Larger "
            "models are meaningfully smarter and need more RAM.",
        )
        model_body = tk.Frame(model_card, bg=CARD)
        model_body.pack(fill="x", padx=18, pady=(6, 16))
        self.model_choice = tk.StringVar(value=self._model_label(self.settings.model))
        self.model_combo = ttk.Combobox(
            model_body,
            textvariable=self.model_choice,
            values=[self._model_label(spec.key) for spec in catalogue()],
            state="readonly",
            style="Sideris.TCombobox",
        )
        self.model_combo.pack(fill="x")
        self.model_combo.bind("<<ComboboxSelected>>", self._on_model_choice)
        self.model_detail = tk.Label(
            model_body, text="", bg=CARD, fg=MUTED, font=self.font_ui_small,
            anchor="w", justify="left", wraplength=620,
        )
        self.model_detail.pack(fill="x", pady=(8, 0))
        self.disk_label = tk.Label(
            model_body, text="", bg=CARD, fg=FAINT, font=self.font_ui_tiny, anchor="w",
        )
        self.disk_label.pack(fill="x", pady=(4, 0))
        self._on_model_choice()

        perm_card = Card(content)
        perm_card.pack(fill="x", padx=34, pady=8)
        self._card_title(
            perm_card, "2 · Permissions",
            "Everything below is off by default. Nothing downloads, installs, trains "
            "or starts with your computer unless you allow it here.",
        )
        perm_body = tk.Frame(perm_card, bg=CARD)
        perm_body.pack(fill="x", padx=18, pady=(8, 16))
        self.perm_vars: dict[str, tk.BooleanVar] = {
            name: tk.BooleanVar(value=getattr(self.settings.permissions, name))
            for name in ("model_download", "background_service", "training", "start_at_login")
        }
        perm_rows = [
            ("model_download", "Download models", "Large files from Hugging Face. Resumable; kept on this PC."),
            ("background_service", "Run in the background", "A local service so the API and chats stay available after the window closes."),
            ("training", "Fine-tune on my data", "Installs a separate PyTorch runtime only if you start training."),
            ("start_at_login", "Open at login", "Launch the Sideris window after you sign in."),
        ]
        for name, title, description in perm_rows:
            row = tk.Frame(perm_body, bg=CARD)
            row.pack(fill="x", pady=4)
            check = ttk.Checkbutton(
                row, text=title, variable=self.perm_vars[name],
                style="Sideris.TCheckbutton",
                command=self._save_permissions,
            )
            check.pack(anchor="w")
            tk.Label(
                row, text=description, bg=CARD, fg=MUTED, font=self.font_ui_tiny,
                anchor="w", justify="left",
            ).pack(anchor="w", padx=24)

        install_card = Card(content)
        install_card.pack(fill="x", padx=34, pady=8)
        self._card_title(
            install_card, "3 · Install or resume",
            "Each step is recorded. If setup is interrupted, run it again and it "
            "continues from the same place - including partial model downloads.",
        )
        install_body = tk.Frame(install_card, bg=CARD)
        install_body.pack(fill="x", padx=18, pady=(8, 16))
        button_row = tk.Frame(install_body, bg=CARD)
        button_row.pack(fill="x")
        self.install_button = HoverLabel(
            button_row, text="Install / resume setup", bg=ACCENT, hover_bg=ACCENT_HOVER,
            fg=WHITE, command=self._run_setup, font=self.font_ui_bold, padx=16, pady=8,
        )
        self.install_button.pack(side="left")
        self.setup_state_label = tk.Label(
            button_row, text="", bg=CARD, fg=MUTED, font=self.font_ui_small, anchor="w",
        )
        self.setup_state_label.pack(side="left", padx=12)
        self.setup_progress = ttk.Progressbar(
            install_body, orient="horizontal", mode="determinate",
            style="Sideris.Horizontal.TProgressbar", maximum=100,
        )
        self.setup_progress.pack(fill="x", pady=(14, 4))
        self.setup_progress_label = tk.Label(
            install_body, text="", bg=CARD, fg=FAINT, font=self.font_ui_tiny, anchor="w",
        )
        self.setup_progress_label.pack(fill="x")
        log_frame = tk.Frame(install_body, bg=CARD)
        log_frame.pack(fill="both", expand=True, pady=(10, 0))
        self.setup_log = tk.Text(
            log_frame, height=9, bg=CARD_ALT, fg=MUTED, bd=0, relief="flat",
            highlightthickness=1, highlightbackground=BORDER, font=self.font_mono_small,
            wrap="word", padx=10, pady=8, state="disabled",
        )
        self.setup_log.pack(fill="both", expand=True)
        self.setup_log.tag_configure("ok", foreground=GREEN)
        self.setup_log.tag_configure("fail", foreground=RED)
        self.setup_log.tag_configure("run", foreground=TEXT)
        self.setup_log.tag_configure("skip", foreground=FAINT)

        note = tk.Frame(content, bg=BG)
        note.pack(fill="x", padx=34, pady=(6, 26))
        tk.Label(
            note,
            text="Sideris never uploads your chats, files or prompts. The only network "
                 "requests are the model download and the optional training runtime - "
                 "both only after permission.",
            bg=BG, fg=FAINT, font=self.font_ui_tiny, justify="left", wraplength=660,
            anchor="w",
        ).pack(anchor="w")

        self._refresh_setup_status()

    def _model_label(self, key: str) -> str:
        try:
            spec = get_spec(key)
        except KeyError:
            return key
        installed = "  ✓ installed" if self.models.is_complete(key) else ""
        return f"{spec.display_name}   ·   {spec.size_label}   ·   {spec.recommended_for}{installed}"

    def _selected_model_key(self) -> str:
        label = self.model_choice.get()
        for spec in catalogue():
            if label.startswith(spec.display_name):
                return spec.key
        return self.settings.model

    def _on_model_choice(self, _event=None) -> None:
        key = self._selected_model_key()
        try:
            spec = get_spec(key)
        except KeyError:
            return
        installed = self.models.is_complete(key)
        state = "Installed and ready." if installed else "Not downloaded yet."
        self.model_detail.configure(
            text=f"{spec.display_name} · {spec.size_label} · needs about {spec.ram_label} · "
                 f"{spec.recommended_for}. {state}\n{spec.blurb}"
        )
        free = self.models.disk_free_gb()
        ram = total_ram_gb()
        self.disk_label.configure(
            text=f"Disk free: {free} GB    ·    RAM: {ram} GB    ·    "
                 f"Licence: {spec.license}    ·    Runs offline after download."
        )

    def _save_permissions(self) -> None:
        before = self.settings.permissions.to_dict()
        for name, var in self.perm_vars.items():
            setattr(self.settings.permissions, name, bool(var.get()))
        self.store.save(self.settings)
        if before.get("start_at_login") != self.settings.permissions.start_at_login:
            try:
                if self.settings.permissions.start_at_login:
                    install_login_item(permission=True)
                    self._toast("Sideris will open at login")
                else:
                    remove_login_item()
                    self._toast("Start at login disabled")
            except Exception as exc:
                self.perm_vars["start_at_login"].set(False)
                self.settings.permissions.start_at_login = False
                self.store.save(self.settings)
                self._toast(f"Could not change start at login: {exc}")
        if before.get("background_service") != self.settings.permissions.background_service:
            if not self.settings.permissions.background_service:
                self._toast("Background permission off (uninstall the service in Settings)")

    def _refresh_setup_status(self) -> None:
        status = Setup(self.paths.root, self.store).status()
        if status["ready"]:
            self.setup_state_label.configure(text="Ready to chat", fg=GREEN)
        else:
            done = sum(1 for step in status["steps"] if step["done"])
            self.setup_state_label.configure(text=f"{done} of {len(status['steps'])} steps complete", fg=MUTED)
        self._update_model_chip()

    def _setup_log_line(self, line: str) -> None:
        try:
            self.setup_log.configure(state="normal")
            tag = "run"
            if line.startswith("[ ok ]"):
                tag = "ok"
            elif line.startswith("[fail]"):
                tag = "fail"
            elif line.startswith("[skip]"):
                tag = "skip"
            self.setup_log.insert("end", line + "\n", (tag,))
            self.setup_log.see("end")
            self.setup_log.configure(state="disabled")
        except tk.TclError:
            pass

    def _poll_setup(self) -> None:
        if self.closing:
            return
        try:
            while True:
                kind, payload = self._setup_queue.get_nowait()
                if kind == "log":
                    self._setup_log_line(str(payload))
                elif kind == "progress":
                    percent, message = payload
                    self.setup_progress.configure(value=percent)
                    self.setup_progress_label.configure(text=f"{percent:.0f}%  ·  {message}")
                elif kind == "done":
                    self._setup_finished(payload)
        except queue.Empty:
            pass
        self.root.after(120, self._poll_setup)

    def _run_setup(self) -> None:
        if self._setup_running:
            self._toast("Setup is already running")
            return
        key = self._selected_model_key()
        self.settings.model = key
        for name, var in self.perm_vars.items():
            setattr(self.settings.permissions, name, bool(var.get()))
        self.store.save(self.settings)
        if not self.settings.permissions.model_download:
            self._toast("Tick 'Download models' first - nothing is fetched without it.")
            return
        self._setup_running = True
        self.install_button.configure(text="Working…")
        self.setup_log.configure(state="normal")
        self.setup_log.delete("1.0", "end")
        self.setup_log.configure(state="disabled")
        self.setup_progress.configure(value=0)
        self.setup_progress_label.configure(text="Starting…")
        self._setup_log_line(f"Model: {self._model_label(key)}")
        self._setup_log_line(f"Data:  {self.paths.root}")

        def worker() -> None:
            def log(line: str) -> None:
                self._setup_queue.put(("log", line))

            def progress(percent: float, message: str) -> None:
                self._setup_queue.put(("progress", (percent, message)))

            try:
                setup = Setup(self.paths.root, self.store, on_log=log, on_progress=progress)
                result = setup.run(model=key, settings=self.settings)
                self._setup_queue.put(("done", result))
            except Exception as exc:
                log(f"[fail] {type(exc).__name__}: {exc}")
                self._setup_queue.put(("done", {"ok": False, "results": [], "error": str(exc)}))

        threading.Thread(target=worker, daemon=True).start()

    def _setup_finished(self, result: dict) -> None:
        self._setup_running = False
        self.install_button.configure(text="Install / resume setup")
        if result.get("ok"):
            self._toast("Setup complete. Sideris is ready.")
            self.setup_progress.configure(value=100)
            self.setup_progress_label.configure(text="100%  ·  Ready")
        else:
            self._toast("Setup finished with problems - see the log.")
            self.setup_progress_label.configure(text="Setup needs attention - run again to resume")
        self.models = ModelManager(self.paths.root)
        self.runtime.refresh_models()
        self._refresh_setup_status()
        self._update_model_chip()
        self._refresh_settings()
        self._refresh_training()

    # ==================================================================
    # training page
    # ==================================================================
    def _build_training_page(self) -> None:
        page = tk.Frame(self.page_host, bg=BG)
        self.pages["training"] = page
        scroll = ScrollFrame(page, bg=BG)
        scroll.pack(fill="both", expand=True)
        content = scroll.inner

        header = tk.Frame(content, bg=BG)
        header.pack(fill="x", padx=34, pady=(26, 10))
        tk.Label(header, text="Training", bg=BG, fg=TEXT,
                 font=self.font_display_small, anchor="w").pack(anchor="w")
        tk.Label(
            header,
            text="Teach a small local model your style, vocabulary and tasks with real "
                 "LoRA fine-tuning. Runs in its own process and resumes from checkpoints.",
            bg=BG, fg=MUTED, font=self.font_ui, anchor="w", justify="left", wraplength=650,
        ).pack(anchor="w", pady=(4, 0))

        honest = tk.Frame(content, bg=ACCENT_SOFT, highlightbackground="#E9D3C1", highlightthickness=1)
        honest.pack(fill="x", padx=34, pady=(4, 8))
        tk.Label(
            honest,
            text="An honest note",
            bg=ACCENT_SOFT, fg=ACCENT_HOVER, font=self.font_ui_bold, anchor="w",
        ).pack(fill="x", padx=14, pady=(10, 2))
        tk.Label(
            honest,
            text="This is real fine-tuning, and it genuinely helps a small local model "
                 "follow your formats and tone. It is not frontier-model training. "
                 "Models like ChatGPT are built with data-centre pretraining on trillions "
                 "of tokens; no laptop can reproduce that. Use this to specialise a local "
                 "model, not to match a cloud system.",
            bg=ACCENT_SOFT, fg="#6B4A36", font=self.font_ui_small, anchor="w",
            justify="left", wraplength=640,
        ).pack(fill="x", padx=14, pady=(0, 11))

        data_card = Card(content)
        data_card.pack(fill="x", padx=34, pady=8)
        self._card_title(
            data_card, "Dataset",
            "A .jsonl file, one object per line: {\"text\": ...} or "
            "{\"prompt\": ..., \"response\": ...}. Small, high-quality sets work best.",
        )
        data_body = tk.Frame(data_card, bg=CARD)
        data_body.pack(fill="x", padx=18, pady=(8, 16))
        entry_row = tk.Frame(data_body, bg=CARD)
        entry_row.pack(fill="x")
        self.dataset_var = tk.StringVar(value=self._default_dataset())
        entry = tk.Entry(
            entry_row, textvariable=self.dataset_var, bg=WHITE, fg=TEXT,
            relief="flat", highlightthickness=1, highlightbackground=BORDER,
            highlightcolor=ACCENT, insertbackground=TEXT, font=self.font_ui_small,
        )
        entry.pack(side="left", fill="x", expand=True, ipady=6, padx=(0, 8))
        HoverLabel(
            entry_row, text="Browse…", bg=CARD_ALT, hover_bg=SIDEBAR_HOVER, fg=TEXT,
            command=self._browse_dataset, font=self.font_ui_small, padx=12, pady=6,
        ).pack(side="left", padx=(0, 6))
        HoverLabel(
            entry_row, text="Create example", bg=CARD_ALT, hover_bg=SIDEBAR_HOVER, fg=TEXT,
            command=self._make_example_dataset, font=self.font_ui_small, padx=12, pady=6,
        ).pack(side="left")

        options_row = tk.Frame(data_body, bg=CARD)
        options_row.pack(fill="x", pady=(12, 0))
        tk.Label(options_row, text="Time budget (hours)", bg=CARD, fg=MUTED,
                 font=self.font_ui_tiny).pack(side="left")
        self.hours_var = tk.StringVar(value="2")
        ttk.Spinbox(
            options_row, from_=0.25, to=24, increment=0.25, width=6,
            textvariable=self.hours_var, style="Sideris.TSpinbox",
        ).pack(side="left", padx=(6, 16))
        tk.Label(options_row, text="Epochs", bg=CARD, fg=MUTED,
                 font=self.font_ui_tiny).pack(side="left")
        self.epochs_var = tk.StringVar(value="1")
        ttk.Spinbox(
            options_row, from_=0.1, to=10, increment=0.1, width=6,
            textvariable=self.epochs_var, style="Sideris.TSpinbox",
        ).pack(side="left", padx=(6, 16))
        tk.Label(
            options_row, text="Checkpoint every 50 steps · keep the last 3",
            bg=CARD, fg=FAINT, font=self.font_ui_tiny,
        ).pack(side="left")

        actions = tk.Frame(data_body, bg=CARD)
        actions.pack(fill="x", pady=(14, 0))
        self.train_start_button = HoverLabel(
            actions, text="Start training", bg=ACCENT, hover_bg=ACCENT_HOVER,
            fg=WHITE, command=self._start_training, font=self.font_ui_bold,
            padx=16, pady=8,
        )
        self.train_start_button.pack(side="left")
        self.train_stop_button = HoverLabel(
            actions, text="Stop", bg=CARD_ALT, hover_bg=SIDEBAR_HOVER, fg=TEXT,
            command=self._stop_training, font=self.font_ui_small, padx=14, pady=8,
        )
        self.train_stop_button.pack(side="left", padx=8)
        self.train_error_label = tk.Label(
            actions, text="", bg=CARD, fg=RED, font=self.font_ui_tiny, anchor="w",
            justify="left", wraplength=520,
        )
        self.train_error_label.pack(side="left", padx=8)

        status_card = Card(content)
        status_card.pack(fill="x", padx=34, pady=8)
        self._card_title(status_card, "This run", "Live progress, read from the trainer every 5 seconds.")
        status_body = tk.Frame(status_card, bg=CARD)
        status_body.pack(fill="x", padx=18, pady=(8, 16))
        top = tk.Frame(status_body, bg=CARD)
        top.pack(fill="x")
        self.training_pill = tk.Label(
            top, text="idle", bg=CARD_ALT, fg=MUTED, font=self.font_ui_tiny,
            padx=10, pady=4,
        )
        self.training_pill.pack(side="left")
        self.training_run_label = tk.Label(
            top, text="", bg=CARD, fg=FAINT, font=self.font_ui_tiny, anchor="w",
        )
        self.training_run_label.pack(side="left", padx=10)
        self.training_bar = ttk.Progressbar(
            status_body, orient="horizontal", mode="determinate",
            style="Sideris.Horizontal.TProgressbar", maximum=100,
        )
        self.training_bar.pack(fill="x", pady=(12, 6))
        self.training_detail = tk.Label(
            status_body, text="No training run yet.", bg=CARD, fg=MUTED,
            font=self.font_ui_small, anchor="w", justify="left", wraplength=640,
        )
        self.training_detail.pack(fill="x")
        self.training_checkpoint = tk.Label(
            status_body, text="", bg=CARD, fg=FAINT, font=self.font_mono_small,
            anchor="w", justify="left", wraplength=640,
        )
        self.training_checkpoint.pack(fill="x", pady=(6, 0))
        self.training_log = tk.Text(
            status_body, height=8, bg=CARD_ALT, fg=MUTED, bd=0, relief="flat",
            highlightthickness=1, highlightbackground=BORDER, font=self.font_mono_small,
            wrap="none", padx=10, pady=8, state="disabled",
        )
        self.training_log.pack(fill="both", expand=True, pady=(10, 0))

        hint = tk.Frame(content, bg=BG)
        hint.pack(fill="x", padx=34, pady=(4, 26))
        tk.Label(
            hint,
            text="Training downloads the base model's Hugging Face weights on first run "
                 "(a few GB) and writes a LoRA adapter under your data folder. Close the "
                 "window if you like - the trainer keeps running and resumes after a "
                 "shutdown.",
            bg=BG, fg=FAINT, font=self.font_ui_tiny, justify="left", wraplength=660,
            anchor="w",
        ).pack(anchor="w")

    def _default_dataset(self) -> str:
        example = self.paths.datasets / "example.jsonl"
        return str(example) if example.exists() else ""

    def _browse_dataset(self) -> None:
        file = filedialog.askopenfilename(
            title="Choose a JSONL training dataset",
            initialdir=str(self.paths.datasets),
            filetypes=[("JSON Lines", "*.jsonl"), ("All files", "*.*")],
        )
        if file:
            self.dataset_var.set(file)

    def _make_example_dataset(self) -> None:
        target = self.paths.datasets / "example.jsonl"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(example_dataset(), encoding="utf-8")
        self.dataset_var.set(str(target))
        self._toast(f"Example dataset written to {target.name}")

    def _start_training(self) -> None:
        self.train_error_label.configure(text="")
        if not self.settings.permissions.training:
            self.train_error_label.configure(
                text="Training permission is off. Enable it in Setup or Settings first."
            )
            return
        dataset = self.dataset_var.get().strip()
        if not dataset:
            self.train_error_label.configure(text="Choose a .jsonl dataset first.")
            return
        try:
            prepare_dataset(dataset)
            hours = float(self.hours_var.get())
            epochs = float(self.epochs_var.get())
        except (TrainingError, ValueError) as exc:
            self.train_error_label.configure(text=str(exc))
            return
        try:
            state = self.training.start(dataset, hours=hours, epochs=epochs)
        except TrainingPermissionError as exc:
            self.train_error_label.configure(text=str(exc))
            return
        except TrainingNotReadyError as exc:
            self.train_error_label.configure(
                text=str(exc) + " Use Setup → permissions → Install."
            )
            return
        except TrainingError as exc:
            self.train_error_label.configure(text=str(exc))
            return
        self._toast("Training started")
        if state.resumed_from:
            self._toast("Resumed from the latest checkpoint")
        self._refresh_training()

    def _stop_training(self) -> None:
        try:
            self.training.stop()
            self._toast("Stop requested - a checkpoint will be saved")
        except TrainingError as exc:
            self.train_error_label.configure(text=str(exc))
        self._refresh_training()

    def _refresh_training(self) -> None:
        if self.closing:
            return
        if self._training_after:
            try:
                self.root.after_cancel(self._training_after)
            except Exception:
                pass
            self._training_after = None
        try:
            info = self.training.status()
        except Exception:
            info = None
        if info:
            state = info.get("state", {})
            status = str(state.get("status") or "idle")
            colors = {
                "running": (ACCENT_SOFT, ACCENT_HOVER),
                "starting": (ACCENT_SOFT, ACCENT_HOVER),
                "stopping": (ACCENT_SOFT, ACCENT_HOVER),
                "completed": ("#E8F0E8", GREEN),
                "stopped": (CARD_ALT, MUTED),
                "interrupted": ("#FBF0E8", ACCENT_HOVER),
                "failed": ("#F8E8E4", RED),
                "idle": (CARD_ALT, MUTED),
            }.get(status, (CARD_ALT, MUTED))
            self.training_pill.configure(text=status, bg=colors[0], fg=colors[1])
            run_bits = []
            if state.get("run_id"):
                run_bits.append(f"run {state['run_id']}")
            if state.get("pid"):
                run_bits.append(f"pid {state['pid']}")
            if state.get("resumed_from"):
                run_bits.append("resumed")
            self.training_run_label.configure(text="  ·  ".join(run_bits))

            step = int(state.get("step") or 0)
            max_steps = state.get("max_steps") or 0
            if max_steps:
                self.training_bar.configure(value=min(100.0, step * 100.0 / max_steps))
            elif status in {"running", "starting", "stopping"}:
                self.training_bar.configure(value=0)
            self.training_bar.configure(mode="determinate")

            bits = []
            if max_steps:
                bits.append(f"step {step} of {max_steps}")
            elif step:
                bits.append(f"step {step}")
            if state.get("epoch"):
                bits.append(f"epoch {float(state['epoch']):.2f}")
            if state.get("loss") is not None:
                try:
                    bits.append(f"loss {float(state['loss']):.4f}")
                except (TypeError, ValueError):
                    pass
            if state.get("max_hours"):
                bits.append(f"budget {float(state['max_hours']):g}h")
            detail = "  ·  ".join(bits) or "No training run yet."
            if state.get("error"):
                detail += "\n" + str(state["error"])
            self.training_detail.configure(text=detail)
            checkpoint = info.get("latest_checkpoint") or state.get("checkpoint")
            self.training_checkpoint.configure(
                text=(f"Latest checkpoint: {checkpoint}" if checkpoint else
                      "Checkpoints appear here every 50 steps.")
            )
            try:
                tail = self.training.logs_tail(40)
                if tail:
                    self.training_log.configure(state="normal")
                    self.training_log.delete("1.0", "end")
                    self.training_log.insert("end", tail)
                    self.training_log.see("end")
                    self.training_log.configure(state="disabled")
            except Exception:
                pass
            if status in {"running", "starting", "stopping"}:
                self.training_detail.configure(text=detail + "\nTraining is running in its own process.")
        self._training_after = self.root.after(5000, self._refresh_training)

    # ==================================================================
    # settings page
    # ==================================================================
    def _build_settings_page(self) -> None:
        page = tk.Frame(self.page_host, bg=BG)
        self.pages["settings"] = page
        scroll = ScrollFrame(page, bg=BG)
        scroll.pack(fill="both", expand=True)
        content = scroll.inner

        header = tk.Frame(content, bg=BG)
        header.pack(fill="x", padx=34, pady=(26, 10))
        tk.Label(header, text="Settings", bg=BG, fg=TEXT,
                 font=self.font_display_small, anchor="w").pack(anchor="w")
        tk.Label(
            header,
            text="Permissions, the background service and your local data.",
            bg=BG, fg=MUTED, font=self.font_ui, anchor="w",
        ).pack(anchor="w", pady=(4, 0))

        perms = Card(content)
        perms.pack(fill="x", padx=34, pady=8)
        self._card_title(perms, "Permissions", "Change these at any time. Off means off.")
        perm_body = tk.Frame(perms, bg=CARD)
        perm_body.pack(fill="x", padx=18, pady=(6, 14))
        labels = {
            "model_download": ("Download models", "Needed to fetch model weights from Hugging Face."),
            "background_service": ("Run in the background", "Allow the local service to be installed."),
            "training": ("Fine-tune on my data", "Allow LoRA training runs on this computer."),
            "start_at_login": ("Open at login", "Launch Sideris after you sign in."),
        }
        for name in ("model_download", "background_service", "training", "start_at_login"):
            row = tk.Frame(perm_body, bg=CARD)
            row.pack(fill="x", pady=3)
            title, description = labels[name]
            ttk.Checkbutton(
                row, text=title, variable=self.perm_vars[name], style="Sideris.TCheckbutton",
                command=self._save_permissions,
            ).pack(anchor="w")
            tk.Label(
                row, text=description, bg=CARD, fg=MUTED, font=self.font_ui_tiny, anchor="w",
            ).pack(anchor="w", padx=24)

        service_card = Card(content)
        service_card.pack(fill="x", padx=34, pady=8)
        self._card_title(
            service_card, "Background service",
            "Runs the local API after the window closes. Nothing is installed without "
            "the permission above.",
        )
        service_body = tk.Frame(service_card, bg=CARD)
        service_body.pack(fill="x", padx=18, pady=(6, 16))
        self.service_status_label = tk.Label(
            service_body, text="", bg=CARD, fg=MUTED, font=self.font_ui_small,
            anchor="w", justify="left",
        )
        self.service_status_label.pack(fill="x")
        row = tk.Frame(service_body, bg=CARD)
        row.pack(fill="x", pady=(10, 0))
        HoverLabel(
            row, text="Install service", bg=ACCENT, hover_bg=ACCENT_HOVER, fg=WHITE,
            command=self._install_service, font=self.font_ui_small, padx=13, pady=7,
        ).pack(side="left")
        HoverLabel(
            row, text="Remove service", bg=CARD_ALT, hover_bg=SIDEBAR_HOVER, fg=TEXT,
            command=self._remove_service, font=self.font_ui_small, padx=13, pady=7,
        ).pack(side="left", padx=8)
        self.service_hint = tk.Label(
            service_body, text="", bg=CARD, fg=FAINT, font=self.font_ui_tiny, anchor="w",
        )
        self.service_hint.pack(fill="x", pady=(8, 0))

        data_card = Card(content)
        data_card.pack(fill="x", padx=34, pady=8)
        self._card_title(data_card, "Your data", "One folder. Easy to inspect, back up or delete.")
        data_body = tk.Frame(data_card, bg=CARD)
        data_body.pack(fill="x", padx=18, pady=(6, 16))
        self.data_path_label = tk.Label(
            data_body, text=str(self.paths.root), bg=CARD, fg=TEXT,
            font=self.font_mono_small, anchor="w", justify="left", wraplength=640,
        )
        self.data_path_label.pack(fill="x")
        buttons = tk.Frame(data_body, bg=CARD)
        buttons.pack(fill="x", pady=(10, 0))
        for label, what in (
            ("Open data folder", "data"),
            ("Open workspace", "workspace"),
            ("Open models", "models"),
            ("Open logs", "logs"),
        ):
            HoverLabel(
                buttons, text=label, bg=CARD_ALT, hover_bg=SIDEBAR_HOVER, fg=TEXT,
                command=lambda w=what: self._open_folder(w),
                font=self.font_ui_tiny, padx=10, pady=6,
            ).pack(side="left", padx=(0, 6))
        self.memory_label = tk.Label(
            data_body, text="", bg=CARD, fg=FAINT, font=self.font_ui_tiny, anchor="w",
        )
        self.memory_label.pack(fill="x", pady=(10, 0))
        HoverLabel(
            data_body, text="Clear remembered facts", bg=CARD, hover_bg=ACCENT_SOFT,
            fg=FAINT, hover_fg=ACCENT_HOVER, command=self._clear_memory,
            font=self.font_ui_tiny, pady=4,
        ).pack(anchor="w", pady=(4, 0))

        about = Card(content)
        about.pack(fill="x", padx=34, pady=(8, 26))
        self._card_title(about, "About", "")
        about_body = tk.Frame(about, bg=CARD)
        about_body.pack(fill="x", padx=18, pady=(4, 16))
        tk.Label(
            about_body,
            text=f"{__release_name__} {__version__}  ·  {platform_label()}  ·  "
                 f"{total_ram_gb()} GB RAM",
            bg=CARD, fg=TEXT, font=self.font_ui_small, anchor="w",
        ).pack(fill="x")
        self.about_backend = tk.Label(
            about_body, text="", bg=CARD, fg=MUTED, font=self.font_ui_tiny, anchor="w",
        )
        self.about_backend.pack(fill="x", pady=(4, 0))
        tk.Label(
            about_body,
            text="Sideris runs open-weight models locally. It does not claim to match "
                 "ChatGPT or Claude: a 1-14B local model is useful, private and yours, "
                 "but frontier systems are trained in data centres at a different scale.",
            bg=CARD, fg=FAINT, font=self.font_ui_tiny, anchor="w", justify="left",
            wraplength=640,
        ).pack(fill="x", pady=(10, 0))

        self._refresh_settings()

    def _refresh_settings(self) -> None:
        try:
            info = service_status(self.paths)
            installed = "installed" if info["installed"] else "not installed"
            running = "running" if info["running"] else "not running"
            self.service_status_label.configure(
                text=f"Service: {installed}  ·  {running}\n"
                     f"Login item: {'on' if info['start_at_login'] else 'off'}  ·  "
                     f"Heartbeat: {info.get('last_heartbeat') or 'never'}"
            )
            self.service_hint.configure(
                text=f"Mode: {info.get('mode') or '-'}   ·   Log: {self.paths.logs / 'service.log'}"
            )
        except Exception:
            self.service_status_label.configure(text="Service status unavailable.")
        try:
            from .core import backend_report

            report = backend_report()
            self.about_backend.configure(
                text=f"Inference: {report['preferred'] or 'not installed'}   ·   "
                     f"llama.cpp: {'yes' if report['llama_cpp'] else 'no'}   ·   "
                     f"transformers: {'yes' if report['transformers'] else 'no'}"
            )
        except Exception:
            pass
        try:
            from .core import MemoryStore

            facts = len(MemoryStore(self.paths.memory).all())
            self.memory_label.configure(
                text=f"{facts} remembered fact{'s' if facts != 1 else ''}  ·  "
                     f"Chats and memory never leave this computer."
            )
        except Exception:
            pass

    def _open_folder(self, what: str) -> None:
        target = {
            "data": self.paths.root,
            "workspace": self.paths.workspace,
            "models": self.paths.models,
            "logs": self.paths.logs,
        }.get(what, self.paths.root)
        target.mkdir(parents=True, exist_ok=True)
        try:
            open_path(target)
        except Exception as exc:
            self._toast(f"Could not open the folder: {exc}")

    def _clear_memory(self) -> None:
        if not messagebox.askyesno(
            "Clear remembered facts",
            "Delete every fact Sideris remembered about you? Conversations stay.",
            parent=self.root,
        ):
            return
        try:
            if self.paths.memory.exists():
                self.paths.memory.write_text("", encoding="utf-8")
            self._toast("Memory cleared")
            self._refresh_settings()
        except OSError as exc:
            self._toast(f"Could not clear memory: {exc}")

    def _install_service(self) -> None:
        if not self.settings.permissions.background_service:
            self._toast("Enable 'Run in the background' first")
            return
        try:
            info = install_service(permission=True)
            self._toast(f"Service installed ({info.get('mode')})")
        except Exception as exc:
            messagebox.showerror(
                "Background service",
                f"Could not install the background service:\n\n{exc}",
                parent=self.root,
            )
        self._refresh_settings()

    def _remove_service(self) -> None:
        try:
            result = uninstall_service()
            if result.get("errors"):
                self._toast("Service removed with warnings - see logs")
            else:
                self._toast("Background service removed")
        except Exception as exc:
            messagebox.showerror(
                "Background service", f"Could not remove the service:\n\n{exc}", parent=self.root
            )
        self._refresh_settings()

    # -- lifecycle -------------------------------------------------------
    def _on_close(self) -> None:
        self.closing = True
        self._stream_cancel = True
        try:
            if self._response is not None:
                self._response.close()
        except Exception:
            pass
        try:
            if self._server is not None:
                self._server.shutdown()
                self._server.server_close()
        except Exception:
            pass
        try:
            self.root.destroy()
        except tk.TclError:
            pass


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------
def run_gui(page: str = "chat", *, smoke: bool = False) -> int:
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        message = (
            "Sideris could not open a window. This usually means Tk is not available "
            f"or there is no display ({exc}).\n"
            "On Linux install the Tk package (for example python3-tk), or use "
            "'sideris chat' in a terminal."
        )
        stream = sys.stderr if sys.stderr is not None else sys.stdout
        if stream is not None:
            try:
                stream.write(message + "\n")
                stream.flush()
            except Exception:
                pass
        return 2
    try:
        activate_runtime()
    except Exception:
        pass
    app = SiderisApp(root, page=page if page in {"chat", "setup", "training", "settings"} else "chat")
    if smoke:
        try:
            root.update_idletasks()
            root.update()
        finally:
            app._on_close()
        return 0
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(run_gui())
