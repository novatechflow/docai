"""Simple PDF viewer/editor UI using Tkinter with OCR + RAG entry points."""

from __future__ import annotations

import io
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog

from pypdf import PdfReader

from docai_toolkit.config import AppConfig
from docai_toolkit.ocr import RemoteOcrClient, TesseractOcrClient, run_ocr_to_markdown
from docai_toolkit.rag import build_index_from_markdown, chat_over_corpus


class PDFViewerApp:
    def __init__(self, root: tk.Tk, config: AppConfig | None = None) -> None:
        self.root = root
        self.root.title("PDF Viewer")
        self.config = config or AppConfig()

        self.text_area = tk.Text(
            self.root,
            height=24,
            width=90,
            wrap=tk.WORD,
        )
        scrollbar = tk.Scrollbar(self.root, orient=tk.VERTICAL, command=self.text_area.yview)
        self.text_area.configure(yscrollcommand=scrollbar.set)

        self._build_menu()
        self._build_buttons()

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.text_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.status_bar = tk.Label(self.root, text="", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        self._bind_keys()

    def _build_menu(self) -> None:
        menubar = tk.Menu(self.root)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open", command=self.open_pdf)
        file_menu.add_command(label="Save As...", command=self.save_as_pdf)
        file_menu.add_separator()
        file_menu.add_command(label="OCR Import to Markdown", command=self.ocr_import)
        file_menu.add_command(label="Chat with Docs", command=self.chat_with_docs)
        file_menu.add_separator()
        file_menu.add_command(label="Settings", command=self.open_settings)
        menubar.add_cascade(label="File", menu=file_menu)
        self.root.config(menu=menubar)

    def _build_buttons(self) -> None:
        button_frame = tk.Frame(self.root)
        button_frame.pack(side=tk.TOP, pady=6)

        open_button = tk.Button(button_frame, text="Open PDF", command=self.open_pdf)
        open_button.pack(side=tk.LEFT, padx=4)

        save_button = tk.Button(button_frame, text="Save As PDF", command=self.save_as_pdf)
        save_button.pack(side=tk.LEFT, padx=4)

        ocr_button = tk.Button(button_frame, text="OCR → Markdown", command=self.ocr_import)
        ocr_button.pack(side=tk.LEFT, padx=4)

        chat_button = tk.Button(button_frame, text="Chat", command=self.chat_with_docs)
        chat_button.pack(side=tk.LEFT, padx=4)

    def _bind_keys(self) -> None:
        self.text_area.bind("<Control-s>", lambda _: self.save_as_pdf())
        self.text_area.bind("<Control-o>", lambda _: self.open_pdf())
        self.text_area.bind("<Control-m>", lambda _: self.ocr_import())
        self.text_area.bind("<Control-j>", lambda _: self.chat_with_docs())

    def open_pdf(self) -> None:
        file_path = filedialog.askopenfilename(filetypes=[("PDF Files", "*.pdf")])
        if not file_path:
            return

        try:
            with open(file_path, "rb") as file_handle:
                reader = PdfReader(file_handle)
                pages_text = []
                for page in reader.pages:
                    text = page.extract_text() or ""
                    pages_text.append(text)
        except Exception as exc:  # broad catch to surface errors to the UI
            messagebox.showerror("Error opening PDF", str(exc))
            self._set_status("Failed to open PDF.")
            return

        self.text_area.delete("1.0", tk.END)
        self.text_area.insert(tk.END, "\n\n".join(pages_text).strip())
        self._set_status(f"Opened: {file_path}")

    def save_as_pdf(self) -> None:
        file_path = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF Files", "*.pdf")],
        )
        if not file_path:
            return

        content = self.text_area.get("1.0", tk.END).rstrip()
        if not content:
            messagebox.showinfo("Nothing to save", "The document is empty.")
            return

        try:
            # Deferred import so the app still starts without reportlab installed.
            from reportlab.lib.pagesizes import letter
            from reportlab.pdfgen import canvas
        except ImportError:
            messagebox.showerror(
                "Missing dependency",
                "Saving requires reportlab. Install with:\n\npip install reportlab",
            )
            return

        try:
            self._write_pdf(file_path, content, canvas, letter)
        except Exception as exc:  # broad catch to keep UI responsive
            messagebox.showerror("Error saving PDF", str(exc))
            self._set_status("Failed to save PDF.")
            return

        self._set_status(f"Saved: {file_path}")

    def _write_pdf(self, file_path: str, content: str, canvas, page_size) -> None:
        """Render plain text into a simple PDF file."""
        buffer = io.BytesIO()
        pdf_canvas = canvas.Canvas(buffer, pagesize=page_size)
        width, height = page_size

        left_margin = 64
        top_margin = 64
        line_height = 14
        x = left_margin
        y = height - top_margin

        for line in content.splitlines() or [""]:
            if y <= top_margin:
                pdf_canvas.showPage()
                y = height - top_margin
            pdf_canvas.drawString(x, y, line)
            y -= line_height

        pdf_canvas.save()
        buffer.seek(0)
        with open(file_path, "wb") as output:
            output.write(buffer.getvalue())

    def _set_status(self, message: str = "") -> None:
        self.status_bar.config(text=message)

    def _report_error(self, title: str, exc: Exception, status: str) -> None:
        detail = str(exc)
        self.root.after(0, lambda: (messagebox.showerror(title, detail), self._set_status(status)))

    def open_settings(self) -> None:
        """Simple settings panel for HF token/endpoints and output dir."""
        win = tk.Toplevel(self.root)
        win.title("Settings")
        win.grab_set()

        row = 0
        labels = [
            ("HF Token", "llm_api_key", self.config.llm.api_key or ""),
            ("OCR Endpoint", "ocr_endpoint", self.config.ocr.endpoint or ""),
            ("Embedding Endpoint", "emb_endpoint", self.config.embeddings.endpoint or ""),
            ("Embedding Model", "emb_model", self.config.embeddings.model),
            ("LLM Endpoint", "llm_endpoint", self.config.llm.endpoint or ""),
            ("LLM Model", "llm_model", self.config.llm.model),
            ("Output Dir", "output_dir", str(self.config.output_dir)),
        ]
        entries: dict[str, tk.Entry] = {}
        for label_text, key, value in labels:
            tk.Label(win, text=label_text).grid(row=row, column=0, sticky="w", padx=6, pady=4)
            ent = tk.Entry(win, width=50, show="*" if key == "llm_api_key" else "")
            ent.insert(0, value)
            ent.grid(row=row, column=1, sticky="w", padx=6, pady=4)
            entries[key] = ent
            row += 1

        def choose_output_dir() -> None:
            chosen = filedialog.askdirectory()
            if chosen:
                entries["output_dir"].delete(0, tk.END)
                entries["output_dir"].insert(0, chosen)

        output_row_index = len(labels) - 1
        choose_button = tk.Button(win, text="Browse", command=choose_output_dir)
        choose_button.grid(row=output_row_index, column=2, padx=4, pady=4)

        def save():
            self.config.llm.api_key = entries["llm_api_key"].get().strip() or None
            self.config.llm.endpoint = entries["llm_endpoint"].get().strip() or None
            self.config.llm.model = entries["llm_model"].get().strip() or self.config.llm.model

            self.config.ocr.endpoint = entries["ocr_endpoint"].get().strip() or None

            self.config.embeddings.endpoint = entries["emb_endpoint"].get().strip() or None
            self.config.embeddings.model = entries["emb_model"].get().strip() or self.config.embeddings.model

            output_dir = Path(entries["output_dir"].get().strip() or self.config.output_dir)
            self.config.output_dir = output_dir

            try:
                self.config.save()
            except Exception as exc:
                messagebox.showerror("Settings not saved", str(exc))
                return

            win.destroy()
            self._set_status("Settings saved.")

        save_button = tk.Button(win, text="Save", command=save)
        save_button.grid(row=row, column=1, sticky="e", padx=6, pady=8)

    def ocr_import(self) -> None:
        pdf_path_str = filedialog.askopenfilename(filetypes=[("PDF Files", "*.pdf")])
        if not pdf_path_str:
            return
        pdf_path = Path(pdf_path_str)

        try:
            client = self._get_ocr_client()
        except RuntimeError as exc:
            messagebox.showerror("OCR unavailable", str(exc))
            return
        except NotImplementedError as exc:
            messagebox.showinfo("OCR not configured", str(exc))
            return

        def worker():
            try:
                md_path = run_ocr_to_markdown(pdf_path, self.config.output_dir, client)
            except Exception as exc:  # broad to keep UI responsive
                self._report_error("OCR failed", exc, "OCR failed.")
                return
            self.root.after(
                0,
                lambda: (
                    messagebox.showinfo("OCR complete", f"OCR output saved to:\n{md_path}"),
                    self._set_status(f"OCR saved: {md_path}"),
                ),
            )

        self._set_status("Running OCR...")
        threading.Thread(target=worker, daemon=True).start()

    def _get_ocr_client(self):
        if self.config.ocr.endpoint:
            return RemoteOcrClient(
                api_key=self.config.ocr.api_key or self.config.llm.api_key,
                endpoint=self.config.ocr.endpoint,
            )
        try:
            return TesseractOcrClient()
        except RuntimeError as exc:
            raise RuntimeError("Install pytesseract and pdf2image for local OCR.") from exc

    def chat_with_docs(self) -> None:
        md_path_str = filedialog.askopenfilename(
            title="Select Markdown file",
            filetypes=[("Markdown Files", "*.md")],
            initialdir=self.config.output_dir,
        )
        if not md_path_str:
            return
        md_path = Path(md_path_str)

        query = simpledialog.askstring("Chat", "Ask a question about the document:")
        if not query:
            return

        def worker():
            try:
                db = build_index_from_markdown(
                    [md_path],
                    embedding_model=self.config.embeddings.model,
                    embedding_endpoint=self.config.embeddings.endpoint,
                    embedding_api_key=self.config.embeddings.api_key or self.config.llm.api_key,
                )
            except Exception as exc:  # broad to keep UI responsive
                self._report_error("Indexing failed", exc, "Indexing failed.")
                return

            try:
                answer = chat_over_corpus(
                    db,
                    query,
                    model_id=self.config.llm.model,
                    endpoint=self.config.llm.endpoint,
                    api_key=self.config.llm.api_key,
                    max_new_tokens=self.config.llm.max_new_tokens,
                )
            except Exception as exc:  # broad to keep UI responsive
                self._report_error("Chat failed", exc, "Chat failed.")
                return

            self.root.after(
                0,
                lambda: (
                    messagebox.showinfo("Answer", answer),
                    self._set_status("Chat complete."),
                ),
            )

        self._set_status("Indexing and chatting...")
        threading.Thread(target=worker, daemon=True).start()


def main() -> None:
    root = tk.Tk()
    config = AppConfig.from_env()
    PDFViewerApp(root, config=config)
    root.mainloop()


if __name__ == "__main__":
    main()
