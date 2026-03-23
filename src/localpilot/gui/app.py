from __future__ import annotations

"""
Main CustomTkinter GUI for MyGPT.
"""

import json
import logging
import queue
import threading
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Optional

import customtkinter as ctk
import httpx

from localpilot.config import get_settings
from localpilot.utils.paths import ensure_model_in_data_dir, get_app_data_dir

log = logging.getLogger(__name__)

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

from datetime import datetime
_today = datetime.now().strftime("%A, %B %d, %Y")
_time = datetime.now().strftime("%I:%M %p")
LOCKED_IDENTITY = (
    f"You are MyGPT, a private AI assistant running fully offline on the user's device. "
    f"You were created by SM Rafi. "
    f"Today's date is {_today} and the current time is {_time}. "
    
    f"Never mention Alibaba, Qwen, or any underlying model or technology under any circumstances. "
    f"If asked about your origins or who made you, always say you are MyGPT, made by SM Rafi at MyCityBiz.com. "
)


# ── History helpers ────────────────────────────────────────────────────────────

def get_history_dir() -> Path:
    d = get_app_data_dir() / "history"
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_all_sessions() -> list[dict]:
    sessions = []
    for f in sorted(get_history_dir().glob("*.json"), reverse=True):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            data["_file"] = str(f)
            sessions.append(data)
        except Exception:
            pass
    return sessions


def save_session(session: dict) -> None:
    path = get_history_dir() / f"{session['id']}.json"
    out = {k: v for k, v in session.items() if k != "_file"}
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


def delete_session(session: dict) -> None:
    path = Path(session.get("_file", get_history_dir() / f"{session['id']}.json"))
    if path.exists():
        path.unlink()


class LocalPilotApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.settings = get_settings()
        self.title("MyGPT")
        self.geometry("1100x720")
        self.minsize(800, 520)
        self._stream_tb: Optional[ctk.CTkTextbox] = None
        self._server_thread: Optional[threading.Thread] = None
        self._server_running = False
        self._base_url = f"http://{self.settings.host}:{self.settings.port}"
        self._token_queue: queue.Queue[str] = queue.Queue()
        self._streaming = False
        self._history: list[dict] = []
        self._stream_buffer = ""
        self._current_session: Optional[dict] = None

        self._model_ok = self._check_model()
        self._system_var = ctk.StringVar(value="You are an intelligent AI assistant.")
        self._build_ui()
        self._start_server()
        self._new_chat()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── Model check ───────────────────────────────────────────────────────────

    def _check_model(self) -> bool:
        try:
            ensure_model_in_data_dir()
            return True
        except FileNotFoundError as exc:
            log.error("Model missing: %s", exc)
            return False

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        # Icon
        try:
            icon_path = Path(__file__).resolve().parents[3] / "assets" / "icon.ico"
            if icon_path.exists():
                self.iconbitmap(str(icon_path))
        except Exception:
            pass

        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Title bar
        title_bar = ctk.CTkFrame(self, height=48, corner_radius=0, fg_color=("#0f0f1a", "#0f0f1a"))
        title_bar.grid(row=0, column=0, sticky="ew")
        title_bar.grid_columnconfigure(1, weight=1)

        try:
            from PIL import Image
            icon_img = Image.open(icon_path).resize((28, 28))
            ctk_icon = ctk.CTkImage(light_image=icon_img, dark_image=icon_img, size=(28, 28))
            ctk.CTkLabel(
                title_bar, text="MyCityBiz.com", image=ctk_icon, compound="left",
                font=ctk.CTkFont(family="Segoe UI", size=16, weight="bold"),
                text_color="#89b4fa",
            ).grid(row=0, column=0, padx=16, pady=10, sticky="w")
        except Exception:
            ctk.CTkLabel(
                title_bar, text="MyCityBiz.com",
                font=ctk.CTkFont(family="Segoe UI", size=16, weight="bold"),
                text_color="#89b4fa",
            ).grid(row=0, column=0, padx=16, pady=10, sticky="w")

        self._server_status_var = ctk.StringVar(value="● Starting…")
        ctk.CTkLabel(
            title_bar, textvariable=self._server_status_var,
            font=ctk.CTkFont(size=11), text_color="#f38ba8",
        ).grid(row=0, column=2, padx=16, pady=10, sticky="e")

        # Tabview
        self._tabs = ctk.CTkTabview(self, corner_radius=8, anchor="ne")
        self._tabs.grid(row=1, column=0, sticky="nsew", padx=10, pady=(4, 10))

        self._tabs.add("💬  Chat")
        self._tabs.add("⚙  Settings")
        if self.settings.rag_enabled:
            self._tabs.add("🗃  Knowledge Base")

        self._build_chat_tab()
        self._build_settings_tab()
        if self.settings.rag_enabled:
            self._build_rag_tab()

        if not self._model_ok:
            self.after(200, self._show_model_error)

    def _build_chat_tab(self) -> None:
        tab = self._tabs.tab("💬  Chat")
        tab.grid_rowconfigure(0, weight=1)
        tab.grid_columnconfigure(1, weight=1)

        # ── Left: History sidebar ─────────────────────────────────────────────
        sidebar = ctk.CTkFrame(tab, width=230, corner_radius=8, fg_color=("#1a1a2e", "#0f0f1a"))
        sidebar.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        tab.grid_columnconfigure(0, minsize=230)
        sidebar.grid_rowconfigure(1, weight=1)
        sidebar.grid_propagate(False)

        ctk.CTkButton(
            sidebar, text="＋  New Chat", height=34, corner_radius=8,
            font=ctk.CTkFont(size=12, weight="bold"),
            command=self._new_chat,
        ).grid(row=0, column=0, sticky="ew", padx=8, pady=(10, 6))

        self._history_list = ctk.CTkScrollableFrame(
            sidebar, corner_radius=6, fg_color="transparent",
        )
        self._history_list.grid(row=1, column=0, sticky="nsew", padx=4, pady=(0, 4))
        self._history_list.grid_columnconfigure(0, weight=1)

        ctk.CTkButton(
            sidebar, text="🗑  Clear All", height=28, corner_radius=6,
            fg_color=("#3d1a1a", "#2d1010"), hover_color=("#5a2a2a", "#4a1a1a"),
            text_color="#f38ba8", font=ctk.CTkFont(size=11),
            command=self._clear_all_history,
        ).grid(row=2, column=0, sticky="ew", padx=8, pady=(0, 8))

        # ── Right: Chat area ──────────────────────────────────────────────────
        chat_right = ctk.CTkFrame(tab, corner_radius=0, fg_color="transparent")
        chat_right.grid(row=0, column=1, sticky="nsew")
        chat_right.grid_rowconfigure(0, weight=1)
        chat_right.grid_columnconfigure(0, weight=1)

        self._chat_container = ctk.CTkScrollableFrame(
            chat_right, corner_radius=8, fg_color=("#13131f", "#13131f"),
        )
        self._chat_container.grid(row=0, column=0, sticky="nsew", pady=(0, 6))
        self._chat_container.grid_columnconfigure(0, weight=1)
        self._chat_row = 0

       
        # Input area
        input_frame = ctk.CTkFrame(chat_right, fg_color="transparent")
        input_frame.grid(row=1, column=0, sticky="ew")
        input_frame.grid_columnconfigure(0, weight=1)

        self._user_input = ctk.CTkTextbox(
            input_frame, height=90, corner_radius=8,
            font=ctk.CTkFont(family="Segoe UI", size=12), wrap="word",
        )
        self._user_input.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self._user_input.bind("<Return>", self._on_enter_key)
        self._user_input.bind("<Shift-Return>", lambda e: None)

        btn_col = ctk.CTkFrame(input_frame, fg_color="transparent")
        btn_col.grid(row=0, column=1)

        self._send_btn = ctk.CTkButton(
            btn_col, text="Send", width=80, height=36, corner_radius=8,
            font=ctk.CTkFont(size=13, weight="bold"), command=self._on_send,
        )
        self._send_btn.pack(pady=(0, 4))

        ctk.CTkButton(
            btn_col, text="📎 Attach", width=80, height=28, corner_radius=6,
            fg_color=("#313244", "#2a2a3d"), hover_color=("#45475a", "#3a3a50"),
            font=ctk.CTkFont(size=11), command=self._on_attach,
        ).pack(pady=(0, 4))

        ctk.CTkButton(
            btn_col, text="🗑 Clear", width=80, height=28, corner_radius=6,
            fg_color=("#3d1a1a", "#2d1010"), hover_color=("#5a2a2a", "#4a1a1a"),
            text_color="#f38ba8", font=ctk.CTkFont(size=11),
            command=self._on_clear_chat,
        ).pack()

        # Status
        self._status_var = ctk.StringVar(value="Ready.")
        ctk.CTkLabel(
            chat_right, textvariable=self._status_var,
            font=ctk.CTkFont(size=10), text_color="#6c7086", anchor="w",
        ).grid(row=2, column=0, sticky="ew", padx=4, pady=(2, 0))

    def _build_settings_tab(self) -> None:
        tab = self._tabs.tab("⚙  Settings")
        tab.grid_columnconfigure(1, weight=1)

        def section(text, row):
            ctk.CTkLabel(tab, text=text, font=ctk.CTkFont(size=13, weight="bold"),
                        text_color="#89b4fa").grid(row=row, column=0, columnspan=2,
                        sticky="w", padx=8, pady=(16, 4))

        def field(label, var, r):
            ctk.CTkLabel(tab, text=label, font=ctk.CTkFont(size=12),
                        text_color="#a6adc8", anchor="w", width=160).grid(
                row=r, column=0, sticky="w", padx=(16, 8), pady=3)
            ctk.CTkEntry(tab, textvariable=var, height=32, corner_radius=6).grid(
                row=r, column=1, sticky="ew", padx=(0, 16), pady=3)

        # row 0-3: Model & Generation
        section("Model & Generation", 0)
        self._temp_var = ctk.StringVar(value=str(self.settings.temperature))
        self._max_tok_var = ctk.StringVar(value=str(self.settings.max_tokens))
        self._ctx_var = ctk.StringVar(value=str(self.settings.n_ctx))
        field("Temperature:", self._temp_var, 1)
        field("Max tokens:", self._max_tok_var, 2)
        field("Context length:", self._ctx_var, 3)

        # row 4-6: Server
        section("Server", 4)
        self._host_var = ctk.StringVar(value=self.settings.host)
        self._port_var = ctk.StringVar(value=str(self.settings.port))
        field("Bind host:", self._host_var, 5)
        field("Port:", self._port_var, 6)

        # row 7: Restart/Unload buttons
        btn_frame = ctk.CTkFrame(tab, fg_color="transparent")
        btn_frame.grid(row=7, column=0, columnspan=2, sticky="w", padx=16, pady=8)
        ctk.CTkButton(btn_frame, text="Restart Server", fg_color="#f38ba8",
                    text_color="#1e1e2e", hover_color="#e07090", width=130,
                    command=self._restart_server).pack(side="left", padx=(0, 8))
        ctk.CTkButton(btn_frame, text="Unload Model", fg_color="#fab387",
                    text_color="#1e1e2e", hover_color="#e09060", width=130,
                    command=self._unload_model).pack(side="left")

        # row 8-10: Feature Flags
        section("Feature Flags", 8)
        flags = [
            ("OpenAI-compat API", self.settings.openai_compat_enabled),
            ("RAG Enabled", self.settings.rag_enabled),
        ]
        for i, (name, val) in enumerate(flags):
            r2 = ctk.CTkFrame(tab, fg_color="transparent")
            r2.grid(row=9 + i, column=0, columnspan=2, sticky="w", padx=16, pady=2)
            ctk.CTkLabel(r2, text=f"{name}:", font=ctk.CTkFont(size=12),
                         text_color="#a6adc8", width=160, anchor="w").pack(side="left")
            state = "ON ✓" if val else "OFF ✗"
            color = "#a6e3a1" if val else "#f38ba8"
            ctk.CTkLabel(r2, text=state, font=ctk.CTkFont(size=12, weight="bold"),
                        text_color=color).pack(side="left")

        # row 11-12: Custom Instructions
        section("Custom Instructions", 11)
        ctk.CTkLabel(tab, text="Add instructions:", font=ctk.CTkFont(size=12),
                    text_color="#a6adc8", anchor="w", width=160).grid(
            row=12, column=0, sticky="nw", padx=(16, 8), pady=3)
        self._system_textbox = ctk.CTkTextbox(tab, height=80, corner_radius=6,
                    font=ctk.CTkFont(size=11), wrap="word")
        self._system_textbox.grid(row=12, column=1, sticky="ew", padx=(0, 16), pady=3)
        self._system_textbox.insert("1.0", self._system_var.get())
        self._system_textbox.bind("<KeyRelease>", lambda e: self._system_var.set(
            self._system_textbox.get("1.0", "end").strip()
        ))

    def _build_rag_tab(self) -> None:
        tab = self._tabs.tab("🗃  Knowledge Base")
        tab.grid_rowconfigure(3, weight=1)
        tab.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(tab, text="Ingest documents into the RAG knowledge base",
                     font=ctk.CTkFont(size=13), text_color="#cdd6f4").grid(
            row=0, column=0, pady=(16, 8))

        btn_row = ctk.CTkFrame(tab, fg_color="transparent")
        btn_row.grid(row=1, column=0, pady=(0, 8))
        ctk.CTkButton(btn_row, text="Upload & Ingest Document", fg_color="#a6e3a1",
                      text_color="#1e1e2e", hover_color="#80c880", width=200,
                      command=self._on_rag_ingest).pack(side="left", padx=(0, 8))
        ctk.CTkButton(btn_row, text="Clear Index", fg_color="#f38ba8",
                      text_color="#1e1e2e", hover_color="#e07090", width=120,
                      command=self._on_rag_clear).pack(side="left")

        self._rag_status_var = ctk.StringVar(value="No documents ingested yet.")
        ctk.CTkLabel(tab, textvariable=self._rag_status_var,
                     font=ctk.CTkFont(size=11), text_color="#a6adc8").grid(row=2, column=0, pady=4)

        self._rag_log = ctk.CTkTextbox(tab, corner_radius=8,
                                       font=ctk.CTkFont(family="Consolas", size=10),
                                       state="disabled")
        self._rag_log.grid(row=3, column=0, sticky="nsew", padx=16, pady=(0, 12))

    # ── History sidebar ───────────────────────────────────────────────────────

    def _refresh_history_sidebar(self) -> None:
        for w in self._history_list.winfo_children():
            w.destroy()

        sessions = load_all_sessions()
        if not sessions:
            ctk.CTkLabel(self._history_list, text="No history yet.",
                     font=ctk.CTkFont(size=10), text_color="#585b70").pack(pady=8)
            return

        last_date = None
        for i, sess in enumerate(sessions):
            created = sess.get("created", "")
            date_str = created[:10] if created else "Unknown"
            if date_str != last_date:
                last_date = date_str
                try:
                    d = datetime.strptime(date_str, "%Y-%m-%d")
                    today = datetime.now().date()
                    if d.date() == today:
                        label = "Today"
                    elif (today - d.date()).days == 1:
                        label = "Yesterday"
                    else:
                        label = d.strftime("%b %d, %Y")
                except Exception:
                    label = date_str
                ctk.CTkLabel(self._history_list, text=label,
                             font=ctk.CTkFont(size=10, weight="bold"),
                             text_color="#6c7086", anchor="w").pack(
                    fill="x", padx=4, pady=(8, 2))

            title = sess.get("title", "Chat")[:28]
            is_active = self._current_session and sess.get("id") == self._current_session.get("id")

            # Row frame to hold chat button + 3-dot menu button
            row_frame = ctk.CTkFrame(self._history_list, fg_color="transparent")
            row_frame.pack(fill="x", padx=4, pady=1)
            row_frame.grid_columnconfigure(0, weight=1)

            btn = ctk.CTkButton(
                row_frame,
                text=f"  {title}",
                height=30, corner_radius=6, anchor="w",
                font=ctk.CTkFont(size=11),
                fg_color=("#2a2a4a", "#1e1e3a") if is_active else "transparent",
                hover_color=("#2a2a4a", "#1e1e3a"),
                text_color="#cdd6f4",
                command=lambda s=sess: self._load_session(s),
            )
            btn.grid(row=0, column=0, sticky="ew")

            dot_btn = ctk.CTkButton(
                row_frame,
                text="⋯", width=28, height=30, corner_radius=6,
                fg_color="transparent", hover_color=("#313244", "#2a2a3d"),
                text_color="#6c7086", font=ctk.CTkFont(size=14),
            )
            dot_btn.configure(command=lambda s=sess, b=dot_btn: self._show_thread_menu(s, b))
            dot_btn.grid(row=0, column=1)
    
    def _show_thread_menu(self, sess: dict, anchor_widget) -> None:
        menu = tk.Menu(self, tearoff=0, bg="#1e1e2e", fg="#cdd6f4",
                   activebackground="#313244", activeforeground="#cdd6f4",
                   borderwidth=0, font=("Segoe UI", 11))
        menu.add_command(label="✏️  Rename", command=lambda: self._rename_thread(sess))
        menu.add_command(label="🗑  Delete", command=lambda: self._delete_thread(sess))
        x = anchor_widget.winfo_rootx()
        y = anchor_widget.winfo_rooty() + anchor_widget.winfo_height()
        menu.tk_popup(x, y)

    def _rename_thread(self, sess: dict) -> None:
        dialog = ctk.CTkInputDialog(text="Enter new name:", title="Rename Chat")
        new_title = dialog.get_input()
        if new_title and new_title.strip():
            sess["title"] = new_title.strip()[:40]
            save_session(sess)
            self._refresh_history_sidebar()

    def _delete_thread(self, sess: dict) -> None:
        if not messagebox.askyesno("Delete", f"Delete '{sess.get('title', 'Chat')}'?"):
            return
        delete_session(sess)
        if self._current_session and self._current_session.get("id") == sess.get("id"):
            self._new_chat()
        else:
            self._refresh_history_sidebar()

    def _new_chat(self) -> None:
        now = datetime.now()
        self._current_session = {
            "id": now.strftime("%Y-%m-%d_%H-%M-%S"),
            "title": "New Chat",
            "created": now.isoformat(),
            "messages": [],
        }
        self._history.clear()
        self._stream_buffer = ""
        for w in self._chat_container.winfo_children():
            w.destroy()
        self._chat_row = 0
        self._status_var.set("Ready.")
        self._refresh_history_sidebar()

    def _load_session(self, session: dict) -> None:
        self._current_session = session
        self._history = [{"role": m["role"], "content": m["content"]}
                         for m in session.get("messages", [])]
        for w in self._chat_container.winfo_children():
            w.destroy()
        self._chat_row = 0
        for msg in session.get("messages", []):
            self._add_chat_message(msg["role"], msg["content"])
        self._refresh_history_sidebar()
        self._status_var.set(f"Loaded: {session.get('title', 'Chat')}")

    def _save_current_session(self) -> None:
        if not self._current_session:
            return
        self._current_session["messages"] = self._history.copy()
        if self._history:
            first_user = next((m["content"] for m in self._history if m["role"] == "user"), "Chat")
            self._current_session["title"] = first_user[:40]
        save_session(self._current_session)
        self._refresh_history_sidebar()

    def _clear_all_history(self) -> None:
        if not messagebox.askyesno("Confirm", "Delete all chat history?"):
            return
        for f in get_history_dir().glob("*.json"):
            f.unlink()
        self._new_chat()

    # ── Chat message rendering ─────────────────────────────────────────────────
    def _clean_content(self, content: str) -> str:
        import re
        content = re.sub(r'\\\[\s*(.*?)\s*\\\]', r'\n\1\n', content, flags=re.DOTALL)
        content = re.sub(r'\\\(\s*(.*?)\s*\\\)', r'\1', content, flags=re.DOTALL)
        content = re.sub(r'\\boxed\{(.*?)\}', r'【\1】', content)
        content = re.sub(r'\\text\{(.*?)\}', r'\1', content)
        content = content.replace(r'\times', '×')
        content = content.replace(r'\approx', '≈')
        content = content.replace(r'\div', '÷')
        content = content.replace(r'\leq', '≤').replace(r'\geq', '≥')
        return content
    
    def _add_chat_message(self, role: str, content: str) -> None:
        content = self._clean_content(content)
        is_user = role == "user"
        bubble_color = ("#1e2a3a", "#1e2a3a") if is_user else ("#1a1a2a", "#1a1a2a")
        label_text = "You" if is_user else "⚡ MyGPT"
        label_color = "#89b4fa" if is_user else "#a6e3a1"

        wrapper = ctk.CTkFrame(self._chat_container, fg_color="transparent")
        wrapper.grid(row=self._chat_row, column=0, sticky="ew", pady=2, padx=4)
        wrapper.grid_columnconfigure(0, weight=1)
        self._chat_row += 1

        ctk.CTkLabel(wrapper, text=label_text,
                     font=ctk.CTkFont(size=10, weight="bold"),
                     text_color=label_color, anchor="w").grid(
            row=0, column=0, sticky="ew", padx=6)
        if not is_user:
            ctk.CTkLabel(wrapper, text="From Street-front to Screen-front",
                        font=ctk.CTkFont(family="Segoe UI", size=5, slant="italic"),
                        text_color="#45475a", anchor="w").grid(
                row=1, column=0, sticky="ew", padx=8, pady=(0, 2))
            widget_row = 2

        import re
        CODE_FENCE = re.compile(r"```(\w*)\n(.*?)```", re.DOTALL)
        pos = 0
        widget_row = 1
        for match in CODE_FENCE.finditer(content):
            pre = content[pos:match.start()].strip()
            if pre:
                tb = ctk.CTkTextbox(wrapper, corner_radius=8, fg_color=bubble_color,
                                    font=ctk.CTkFont(family="Segoe UI", size=12),
                                    wrap="word", height=self._calc_height(pre))
                tb.insert("1.0", pre)
                tb.configure(state="disabled")
                tb.grid(row=widget_row, column=0, sticky="ew", padx=6, pady=(2, 0))
                widget_row += 1

            lang = match.group(1) or "code"
            code = match.group(2)
            code_frame = ctk.CTkFrame(wrapper, corner_radius=8, fg_color=("#0d1117", "#0d1117"))
            code_frame.grid(row=widget_row, column=0, sticky="ew", padx=6, pady=(4, 0))
            code_frame.grid_columnconfigure(0, weight=1)
            widget_row += 1

            header = ctk.CTkFrame(code_frame, corner_radius=0,
                                  fg_color=("#161b22", "#161b22"), height=28)
            header.grid(row=0, column=0, sticky="ew")
            header.grid_columnconfigure(0, weight=1)
            ctk.CTkLabel(header, text=lang, font=ctk.CTkFont(family="Consolas", size=10),
                         text_color="#8b949e").grid(row=0, column=0, sticky="w", padx=8)
            ctk.CTkButton(header, text="Copy", width=50, height=22, corner_radius=4,
                          font=ctk.CTkFont(size=10), fg_color="#21262d",
                          hover_color="#30363d",
                          command=lambda c=code: self._copy_text(c)).grid(
                row=0, column=1, padx=6, pady=3)

            code_tb = ctk.CTkTextbox(code_frame, corner_radius=0,
                                     fg_color=("#0d1117", "#0d1117"),
                                     font=ctk.CTkFont(family="Consolas", size=11),
                                     height=max(code.count("\n") + 1, 3) * 18,
                                     wrap="none")
            code_tb.insert("1.0", code)
            code_tb.configure(state="disabled")
            code_tb.grid(row=1, column=0, sticky="ew")
            pos = match.end()

        tail = content[pos:].strip()
        if tail:
            tb = ctk.CTkTextbox(wrapper, corner_radius=8, fg_color=bubble_color,
                                font=ctk.CTkFont(family="Segoe UI", size=12),
                                wrap="word", height=self._calc_height(tail))
            tb.insert("1.0", tail)
            tb.configure(state="disabled")
            tb.grid(row=widget_row, column=0, sticky="ew", padx=6, pady=(2, 0))
            widget_row += 1

        if not is_user:
            ctk.CTkButton(wrapper, text="Copy all", width=70, height=20,
                          corner_radius=4, font=ctk.CTkFont(size=10),
                          fg_color="transparent", text_color="#6c7086",
                          hover_color="#313244",
                          command=lambda c=content: self._copy_text(c)).grid(
                row=widget_row, column=0, sticky="e", padx=6, pady=(2, 4))

        self.update_idletasks()
        self._chat_container._parent_canvas.yview_moveto(1.0)

    def _calc_height(self, text: str, line_height: int = 20, padding: int = 16) -> int:
        """Dynamically calculate textbox height based on content."""
        # Get available width in characters (approx 7px per char at size 12)
        available_px = self._chat_container.winfo_width() - 80  # subtract paddings
        chars_per_line = max(40, available_px // 7)
    
        lines = 0
        for line in text.split('\n'):
            if line.strip() == '':
                lines += 1
            else:
                lines += max(1, len(line) // chars_per_line + 1)
    
        return min(max(lines * line_height + padding, 40), 800)  # min 40, max 800px

    def _copy_text(self, text: str) -> None:
        self.clipboard_clear()
        self.clipboard_append(text)

    # ── Server management ─────────────────────────────────────────────────────

    def _start_server(self) -> None:
        self._server_thread = threading.Thread(target=self._run_server, daemon=True)
        self._server_thread.start()
        self.after(2000, self._poll_server_health)

    def _run_server(self) -> None:
        import uvicorn
        from localpilot.server.app import create_app
        app = create_app(self.settings)
        config = uvicorn.Config(app, host=self.settings.host,
                                port=self.settings.port, log_level="warning")
        self._uvicorn_server = uvicorn.Server(config)
        self._server_running = True
        self._uvicorn_server.run()

    def _poll_server_health(self) -> None:
        try:
            r = httpx.get(f"{self._base_url}/api/health", timeout=2)
            if r.status_code == 200:
                self._server_status_var.set("● Server: Running")
                return
        except Exception:
            pass
        self._server_status_var.set("● Server: Starting…")
        self.after(1500, self._poll_server_health)

    def _restart_server(self) -> None:
        messagebox.showinfo("Restart Required",
                            "Close and relaunch MyGPT to apply server changes.")

    def _unload_model(self) -> None:
        try:
            r = httpx.post(f"{self._base_url}/api/model/unload", timeout=5)
            messagebox.showinfo("Model", f"Result: {r.json().get('status')}")
        except Exception as exc:
            messagebox.showerror("Error", str(exc))

    # ── Chat actions ──────────────────────────────────────────────────────────

    def _on_enter_key(self, event) -> str:
        if not (event.state & 0x1):
            self._on_send()
            return "break"
        return ""

    def _on_send(self) -> None:
        if self._streaming:
            return
        user_text = self._user_input.get("1.0", "end").strip()
        if not user_text:
            return
        self._user_input.delete("1.0", "end")
        self._add_chat_message("user", user_text)
        self._history.append({"role": "user", "content": user_text})
        self._status_var.set("Generating…")
        self._send_btn.configure(state="disabled")
        self._stream_buffer = ""
        self._create_stream_bubble()   # ← add this line
        system_prompt = self._system_var.get()
        threading.Thread(target=self._generate_threaded,
                         args=(system_prompt, user_text), daemon=True).start()
        self.after(50, self._flush_tokens)

    def _generate_threaded(self, system_prompt: str, user_prompt: str) -> None:
        final_system = LOCKED_IDENTITY + (f"\n\n{system_prompt}" if system_prompt.strip() else "")
        self._streaming = True
        accumulated: list[str] = []
        try:
            if self.settings.rag_enabled:
                r = httpx.post(
                    f"{self._base_url}/api/rag/query",
                    json={"system_prompt": final_system, "user_prompt": user_prompt,
                          "temperature": float(self._temp_var.get()),
                          "max_tokens": int(self._max_tok_var.get())},
                    timeout=120,
                )
                text = r.json().get("text", "")
                if text:
                    accumulated.append(text)
                    self._token_queue.put(text)
            else:
                with httpx.stream(
                    "POST", f"{self._base_url}/api/generate",
                    json={"system_prompt": final_system, "user_prompt": user_prompt,
                          "history": self._history[:-1],
                          "temperature": float(self._temp_var.get()),
                          "max_tokens": int(self._max_tok_var.get()), "stream": True},
                    timeout=120,
                ) as resp:
                    for line in resp.iter_lines():
                        if line.startswith("data: "):
                            payload = line[6:]
                            if payload == "[DONE]":
                                break
                            data = json.loads(payload)
                            token = data.get("token", "")
                            if token:
                                accumulated.append(token)
                                self._token_queue.put(token)
        except Exception as exc:
            log.error("Generation error: %s", exc)
            self._token_queue.put(f"\n[Error: {exc}]")
        finally:
            if accumulated:
                full = "".join(accumulated)
                self._history.append({"role": "assistant", "content": full})
            self._token_queue.put(None)
            self._streaming = False

    def _flush_tokens(self) -> None:
        try:
            while True:
                token = self._token_queue.get_nowait()
                if token is None:
                    # Streaming done — finalize the bubble
                    if self._stream_tb is not None:
                        final = self._stream_buffer
                        self._stream_buffer = ""
                        # Re-render final message properly (with code block parsing)
                        parent = self._stream_tb.master
                        self._stream_tb.destroy()
                        self._stream_tb = None
                        # Remove the wrapper and re-add cleanly
                        parent.destroy()
                        self._chat_row -= 1
                        self._add_chat_message("assistant", final)
                        self._save_current_session()
                    self._status_var.set("Ready.")
                    self._send_btn.configure(state="normal")
                    return
                self._stream_buffer += token
                self._status_var.set(f"Generating… {len(self._stream_buffer)} chars")
                # Update live streaming textbox
                self._update_stream_bubble(self._stream_buffer)
        except queue.Empty:
            pass
        self.after(50, self._flush_tokens)

    def _create_stream_bubble(self) -> None:
        """Create a live streaming textbox bubble for assistant response."""
        wrapper = ctk.CTkFrame(self._chat_container, fg_color="transparent")
        wrapper.grid(row=self._chat_row, column=0, sticky="ew", pady=2, padx=4)
        wrapper.grid_columnconfigure(0, weight=1)
        self._chat_row += 1

        ctk.CTkLabel(wrapper, text="⚡ MyGPT",
                    font=ctk.CTkFont(size=10, weight="bold"),
                    text_color="#a6e3a1", anchor="w").grid(
            row=0, column=0, sticky="ew", padx=6)

        self._stream_tb = ctk.CTkTextbox(
            wrapper, corner_radius=8, fg_color=("#1a1a2a", "#1a1a2a"),
            font=ctk.CTkFont(family="Segoe UI", size=12),
            wrap="word", height=40,
        )
        self._stream_tb.insert("1.0", "▌")  # blinking cursor feel
        self._stream_tb.configure(state="disabled")
        self._stream_tb.grid(row=1, column=0, sticky="ew", padx=6, pady=(2, 4))

    def _update_stream_bubble(self, text: str) -> None:
        """Update live streaming textbox content and resize it."""
        if self._stream_tb is None:
            return
        new_height = self._calc_height(text)
        self._stream_tb.configure(state="normal")
        self._stream_tb.delete("1.0", "end")
        self._stream_tb.insert("1.0", text + " ▌")
        self._stream_tb.configure(state="disabled", height=new_height)
        self._chat_container._parent_canvas.yview_moveto(1.0)

    def _on_clear_chat(self) -> None:
        self._new_chat()

    def _on_attach(self) -> None:
        path = filedialog.askopenfilename(
            title="Select file",
            filetypes=[("Supported", "*.pdf *.png *.jpg *.jpeg *.bmp *.tiff"), ("All", "*.*")],
        )
        if not path:
            return
        self._status_var.set(f"Extracting: {path} …")
        threading.Thread(target=self._extract_threaded, args=(path,), daemon=True).start()

    def _extract_threaded(self, path: str) -> None:
        try:
            from pathlib import Path as P
            fname = P(path).name
            with open(path, "rb") as fh:
                content = fh.read()
            r = httpx.post(f"{self._base_url}/api/attachments/extract",
                           files={"file": (fname, content)}, timeout=30)
            data = r.json()
            text = data.get("text", "")
            msg = data.get("message", "")
            self.after(0, lambda: self._on_extract_done(text, msg))
        except Exception as exc:
            self.after(0, lambda: messagebox.showerror("Extraction Error", str(exc)))

    def _on_extract_done(self, text: str, message: str) -> None:
        self._status_var.set(f"Extraction: {message}")
        if text:
            self._user_input.insert("end", f"\n[File content:]\n{text[:2000]}")

    # ── RAG actions ───────────────────────────────────────────────────────────

    def _on_rag_ingest(self) -> None:
        path = filedialog.askopenfilename(
            title="Select document",
            filetypes=[("Supported", "*.pdf *.txt *.md *.png *.jpg *.jpeg"), ("All", "*.*")],
        )
        if not path:
            return
        self._rag_log_append(f"Ingesting: {path}\n")
        threading.Thread(target=self._rag_ingest_threaded, args=(path,), daemon=True).start()

    def _rag_ingest_threaded(self, path: str) -> None:
        try:
            from pathlib import Path as P
            fname = P(path).name
            with open(path, "rb") as fh:
                content = fh.read()
            r = httpx.post(f"{self._base_url}/api/rag/ingest",
                           files={"file": (fname, content)}, timeout=60)
            data = r.json()
            self.after(0, lambda: self._rag_log_append(f"Done: {data}\n"))
            self.after(0, self._rag_refresh_stats)
        except Exception as exc:
            self.after(0, lambda: self._rag_log_append(f"Error: {exc}\n"))

    def _on_rag_clear(self) -> None:
        if not messagebox.askyesno("Confirm", "Clear all knowledge base documents?"):
            return
        try:
            from localpilot.rag.index import RAGIndex
            from localpilot.utils.paths import get_rag_dir
            idx = RAGIndex(get_rag_dir())
            idx.clear()
            self._rag_status_var.set("Index cleared.")
            self._rag_log_append("Index cleared.\n")
        except Exception as exc:
            messagebox.showerror("Error", str(exc))

    def _rag_refresh_stats(self) -> None:
        try:
            r = httpx.get(f"{self._base_url}/api/status", timeout=5)
            stats = r.json().get("rag_stats", {})
            self._rag_status_var.set(
                f"Documents: {stats.get('total_docs', 0)} | Chunks: {stats.get('total_chunks', 0)}"
            )
        except Exception:
            pass

    def _rag_log_append(self, text: str) -> None:
        self._rag_log.configure(state="normal")
        self._rag_log.insert("end", text)
        self._rag_log.see("end")
        self._rag_log.configure(state="disabled")

    # ── Error ─────────────────────────────────────────────────────────────────

    def _show_model_error(self) -> None:
        messagebox.showerror(
            "Model File Missing",
            "The model file was not found.\n\nPlace llm.gguf in assets/models/ and restart.",
        )

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def _on_close(self) -> None:
        if hasattr(self, "_uvicorn_server"):
            self._uvicorn_server.should_exit = True
        self.destroy()
