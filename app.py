import os
import re
import json
import threading
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn

import customtkinter as ctk
from tkinter import messagebox

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from gensim.models.fasttext import load_facebook_vectors


MODEL_PT  = "recipe_fasttext_mlp.pt"
LABELS_JS = "recipe_fasttext_mlp_labels.json"
FT_BIN    = "cc.en.300.bin"


class MLPBN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


_tok_re = re.compile(r"[a-zA-Z][a-zA-Z0-9_'-]*")

_STOP = {
    "a","an","the","and","or","but","of","to","in","on","for","with","without","from","by","at","as",
    "is","are","was","were","be","been","being","into","over","under","up","down","off","out","about",
    "this","that","these","those","it","its","your","my","our","their"
}

def tokenize(text: str) -> List[str]:
    toks = _tok_re.findall((text or "").lower())
    toks = [t for t in toks if len(t) > 1 and t not in _STOP]
    if len(toks) >= 2:
        toks += [f"{toks[i]}_{toks[i+1]}" for i in range(len(toks) - 1)]
    return toks


def _hex_rgba(h: str, a: float):
    h = h.lstrip("#")
    return (int(h[0:2],16)/255, int(h[2:4],16)/255, int(h[4:6],16)/255, float(a))

def _ease_out(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return 1 - (1 - t) ** 3


class Predictor:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedding_dim: int = 300
        self.model: Optional[nn.Module] = None
        self.labels: List[str] = []
        self.ft = None

    def load(self) -> None:
        base = os.path.dirname(os.path.abspath(__file__))
        pt = os.path.join(base, MODEL_PT)
        js = os.path.join(base, LABELS_JS)
        fb = os.path.join(base, FT_BIN)

        for p in (pt, js, fb):
            if not os.path.exists(p):
                raise FileNotFoundError(f"Missing {os.path.basename(p)} next to app.py")

        ckpt = torch.load(pt, map_location="cpu")
        sd = ckpt["state_dict"]

        labels = ckpt.get("label_cols")
        if not labels:
            with open(js, "r", encoding="utf-8") as f:
                labels = json.load(f)
        self.labels = list(labels)

        self.embedding_dim = int(ckpt.get("embedding_dim", 300))

        hidden_dim = int(sd["net.0.weight"].shape[0])
        out_dim = int(sd["net.4.weight"].shape[0])

        if len(self.labels) != out_dim:
            raise ValueError(f"Labels length ({len(self.labels)}) != model out_dim ({out_dim})")

        self.model = MLPBN(self.embedding_dim, hidden_dim, out_dim, dropout=0.0).to(self.device)
        self.model.load_state_dict(sd)
        self.model.eval()

        self.ft = load_facebook_vectors(fb)

    def vec(self, title: str) -> np.ndarray:
        toks = tokenize(title)
        if not toks:
            return np.zeros(self.embedding_dim, dtype=np.float32)
        vecs = np.stack([self.ft.get_vector(t) for t in toks]).astype(np.float32)
        return vecs.mean(axis=0)

    @torch.no_grad()
    def predict_topk(self, title: str, topk: int) -> List[Tuple[str, float]]:
        if self.model is None or self.ft is None:
            raise RuntimeError("Model not loaded")

        x = torch.tensor(self.vec(title), dtype=torch.float32).unsqueeze(0).to(self.device)
        logits = self.model(x).detach().cpu().squeeze(0)
        idx = torch.topk(logits, k=min(topk, logits.numel())).indices.numpy().tolist()
        probs = torch.sigmoid(logits[idx]).numpy().tolist()
        return [(self.labels[i], float(p)) for i, p in zip(idx, probs)]


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("green")

        self.title("Recipe Ingredient Predictor")
        self.geometry("1000x640")
        self.minsize(960, 620)

        self.pred = Predictor()
        self.status = ctk.StringVar(value="Loading model…")
        self.title_var = ctk.StringVar(value="")
        self.topk_var = ctk.StringVar(value="10")

        self._anim_job = None
        self._step_i = 0
        self._steps = 34
        self._targets: List[float] = []
        self._pcts: List[str] = []

        self._bar = "#22C55E"
        self._edge = "#0B3D1D"
        self._txt  = "#EDEDED"
        self._grid = "#2B2B2B"
        self._bg   = "#111111"

        self._build()
        self._autoload()

    def _build(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        root = ctk.CTkFrame(self, corner_radius=16)
        root.grid(row=0, column=0, sticky="nsew", padx=14, pady=14)
        root.grid_columnconfigure(0, weight=0)
        root.grid_columnconfigure(1, weight=1)
        root.grid_rowconfigure(0, weight=1)

        left = ctk.CTkFrame(root, corner_radius=16)
        left.grid(row=0, column=0, sticky="nsw", padx=(10, 10), pady=10)
        left.grid_columnconfigure(0, weight=1)

        right = ctk.CTkFrame(root, corner_radius=16)
        right.grid(row=0, column=1, sticky="nsew", padx=(0, 10), pady=10)
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(2, weight=1)

        ctk.CTkLabel(left, text="Predict", font=ctk.CTkFont(size=18, weight="bold")).grid(
            row=0, column=0, sticky="w", padx=14, pady=(14, 10)
        )

        e = ctk.CTkEntry(left, textvariable=self.title_var, height=38, corner_radius=12, placeholder_text="e.g. apple pie")
        e.grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 10))
        e.bind("<Return>", lambda ev: self._submit())

        ctrl = ctk.CTkFrame(left, corner_radius=12)
        ctrl.grid(row=2, column=0, sticky="ew", padx=14, pady=(0, 10))
        ctrl.grid_columnconfigure((0, 1), weight=1)

        ctk.CTkLabel(ctrl, text="Top-K").grid(row=0, column=0, sticky="w", padx=12, pady=(10, 0))
        ctk.CTkEntry(ctrl, textvariable=self.topk_var, width=80).grid(row=1, column=0, sticky="w", padx=12, pady=(6, 12))
        ctk.CTkButton(ctrl, text="Submit", command=self._submit, height=36).grid(row=1, column=1, sticky="e", padx=12, pady=(6, 12))

        ctk.CTkLabel(left, textvariable=self.status, wraplength=280, justify="left").grid(
            row=3, column=0, sticky="ew", padx=14, pady=(0, 14)
        )

        ctk.CTkLabel(right, text="Predicted ingredients", font=ctk.CTkFont(size=18, weight="bold")).grid(
            row=0, column=0, sticky="w", padx=14, pady=(14, 6)
        )
        ctk.CTkLabel(right, text="Horizontal bar chart (probabilities in %)", font=ctk.CTkFont(size=12)).grid(
            row=1, column=0, sticky="w", padx=14, pady=(0, 10)
        )

        self.fig = plt.Figure(figsize=(7.2, 5.0), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor(self._bg)
        self.fig.patch.set_facecolor(self._bg)

        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().grid(row=2, column=0, sticky="nsew", padx=14, pady=14)
        self._empty()

    def _autoload(self):
        def worker():
            try:
                self.pred.load()
                self.after(0, lambda: self.status.set(f"Loaded. Labels: {len(self.pred.labels)} | device: {self.pred.device}"))
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Load error", str(e)))
                self.after(0, lambda: self.status.set("Load failed. Check files next to app.py."))
        threading.Thread(target=worker, daemon=True).start()

    def _submit(self):
        title = self.title_var.get().strip()
        if not title:
            messagebox.showwarning("Empty input", "Type a recipe title first.")
            return
        if self.pred.model is None or self.pred.ft is None:
            messagebox.showwarning("Not ready", "Model is still loading.")
            return
        try:
            k = int(self.topk_var.get().strip())
            k = max(1, min(k, 50))
        except Exception:
            k = 10

        self.status.set("Predicting…")
        self.update_idletasks()

        def worker():
            try:
                preds = self.pred.predict_topk(title, topk=k)
                self.after(0, lambda: self._animate(preds))
                self.after(0, lambda: self.status.set("Done."))
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Prediction error", str(e)))
                self.after(0, lambda: self.status.set("Prediction failed."))
        threading.Thread(target=worker, daemon=True).start()

    def _empty(self):
        self.ax.clear()
        self.ax.set_facecolor(self._bg)
        self.ax.set_title("No predictions yet", color=self._txt, fontsize=14, pad=14)
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(-0.5, 9.5)
        self.ax.tick_params(colors=self._txt)
        self.ax.grid(axis="x", alpha=0.18, color=self._grid)
        for s in self.ax.spines.values():
            s.set_color("#333333")
        self.canvas.draw_idle()

    def _animate(self, preds: List[Tuple[str, float]]):
        if self._anim_job is not None:
            try:
                self.after_cancel(self._anim_job)
            except Exception:
                pass
            self._anim_job = None

        labels = [p[0] for p in preds][::-1]
        vals = [float(p[1]) for p in preds][::-1]
        self._targets = vals
        self._pcts = [f"{v*100:.1f}%" for v in vals]
        self._step_i = 0

        self.ax.clear()
        self.ax.set_facecolor(self._bg)
        self.ax.set_title("Top-K predicted labels", color=self._txt, fontsize=14, pad=14)

        x_max = max(0.35, max(vals) * 1.15)
        self.ax.set_xlim(0, x_max)
        self.ax.set_ylim(-0.5, len(labels) - 0.5)
        self.ax.tick_params(colors=self._txt)
        self.ax.set_yticks(range(len(labels)))
        self.ax.set_yticklabels(labels, color=self._txt, fontsize=10)
        self.ax.grid(axis="x", alpha=0.18, color=self._grid)
        for s in self.ax.spines.values():
            s.set_color("#333333")

        self.bars = self.ax.barh(
            range(len(labels)),
            [0.0] * len(labels),
            color=_hex_rgba(self._bar, 0.20),
            edgecolor=_hex_rgba(self._edge, 0.85),
            linewidth=1.0
        )
        self.texts = [
            self.ax.text(0.0, i, "", va="center", ha="left", color=_hex_rgba(self._txt, 0.0), fontsize=10)
            for i in range(len(labels))
        ]
        self.canvas.draw_idle()
        self._tick()

    def _tick(self):
        self._step_i += 1
        e = _ease_out(self._step_i / max(1, self._steps))

        bar_rgba = _hex_rgba(self._bar, 0.25 + 0.65 * e)
        txt_rgba = _hex_rgba(self._txt, 0.05 + 0.95 * e)

        for bar, target in zip(self.bars, self._targets):
            bar.set_width(target * e)
            bar.set_facecolor(bar_rgba)

        x_lim = self.ax.get_xlim()[1]
        for i, (target, pct) in enumerate(zip(self._targets, self._pcts)):
            w = target * e
            self.texts[i].set_text(pct)
            self.texts[i].set_x(min(w + 0.01, x_lim * 0.98))
            self.texts[i].set_color(txt_rgba)

        self.canvas.draw_idle()
        if self._step_i < self._steps:
            self._anim_job = self.after(16, self._tick)
        else:
            self._anim_job = None


if __name__ == "__main__":
    App().mainloop()