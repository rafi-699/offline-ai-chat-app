from __future__ import annotations

import multiprocessing
multiprocessing.freeze_support()

"""Entry point for RafiGPT."""
import sys
import os

# Redirect stdout/stderr when frozen (no console) so uvicorn gets valid handles
if getattr(sys, 'frozen', False):
    log_dir = os.path.join(os.getenv('LOCALAPPDATA', ''), 'RafiGPT', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    _log_file = open(os.path.join(log_dir, 'rafigpt.log'), 'w', buffering=1)
    sys.stdout = _log_file
    sys.stderr = _log_file
    
import logging
import sys
import threading
import tkinter as tk
from pathlib import Path

log = logging.getLogger(__name__)


def _setup_logging() -> None:
    try:
        from localpilot.utils.paths import get_log_path
        log_path = get_log_path()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers: list[logging.Handler] = [
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, encoding="utf-8"),
        ]
    except Exception:
        handlers = [logging.StreamHandler(sys.stdout)]

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )


def _show_download_window() -> bool:
    """
    Show a CustomTkinter download progress window.
    Downloads llm.gguf and returns True on success, False on failure.
    """
    import customtkinter as ctk
    from localpilot.utils.paths import download_model

    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")

    win = ctk.CTk()
    win.title("RafiGPT — First Launch Setup")
    win.geometry("460x260")
    win.resizable(False, False)
    win.eval("tk::PlaceWindow . center")

    ctk.CTkLabel(
        win, text="⚡  RafiGPT",
        font=ctk.CTkFont(size=20, weight="bold"),
        text_color="#89b4fa",
    ).pack(pady=(24, 4))

    ctk.CTkLabel(
        win, text="Downloading AI model — one-time setup",
        font=ctk.CTkFont(size=12),
        text_color="#a6adc8",
    ).pack(pady=(0, 16))

    progress_bar = ctk.CTkProgressBar(win, width=380, height=16, corner_radius=8)
    progress_bar.pack(pady=(0, 8))
    progress_bar.set(0)

    size_label_var = ctk.StringVar(value="Starting download…")
    ctk.CTkLabel(win, textvariable=size_label_var,
                 font=ctk.CTkFont(size=11), text_color="#6c7086").pack()

    status_var = ctk.StringVar(value="")
    ctk.CTkLabel(win, textvariable=status_var,
                 font=ctk.CTkFont(size=11), text_color="#f38ba8").pack(pady=4)

    result = {"success": False}

    def _fmt(b: int) -> str:
        if b >= 1024 ** 3:
            return f"{b / 1024**3:.1f} GB"
        return f"{b / 1024**2:.1f} MB"

    def _on_progress(downloaded: int, total: int) -> None:
        if total > 0:
            pct = downloaded / total
            win.after(0, lambda: progress_bar.set(pct))
            win.after(0, lambda: size_label_var.set(
                f"{_fmt(downloaded)} / {_fmt(total)}  ({pct*100:.0f}%)"
            ))
        else:
            win.after(0, lambda: size_label_var.set(f"Downloaded: {_fmt(downloaded)}"))

    def _do_download() -> None:
        try:
            download_model(progress_callback=_on_progress)
            result["success"] = True
            win.after(0, lambda: status_var.set("✓ Download complete! Launching…"))
            win.after(1200, win.destroy)
        except RuntimeError as exc:
            win.after(0, lambda: status_var.set(f"✗ {exc}"))
            win.after(0, lambda: _add_retry_btn())

    def _add_retry_btn() -> None:
        ctk.CTkButton(
            win, text="Retry", width=100,
            fg_color="#f38ba8", text_color="#1e1e2e",
            command=lambda: [_retry_btn.pack_forget(), threading.Thread(target=_do_download, daemon=True).start()]
        ).pack(pady=8)

    _retry_btn = None

    threading.Thread(target=_do_download, daemon=True).start()
    win.mainloop()
    return result["success"]


def main() -> None:
    _setup_logging()
    log.info("RafiGPT starting…")

    try:
        from localpilot.utils.paths import ensure_model_in_data_dir
        try:
            ensure_model_in_data_dir()
        except FileNotFoundError:
            # Model not found locally — show download window
            log.info("Model not found. Showing download window.")
            success = _show_download_window()
            if not success:
                log.error("Model download failed or was cancelled.")
                sys.exit(1)

        from localpilot.gui.app import LocalPilotApp
        app = LocalPilotApp()
        app.mainloop()

    except Exception as exc:
        log.error("Fatal error: %s", exc)
        try:
            import tkinter.messagebox as mb
            mb.showerror("RafiGPT — Fatal Error", str(exc))
        except Exception:
            pass
        sys.exit(1)


if __name__ == "__main__":
    main()
