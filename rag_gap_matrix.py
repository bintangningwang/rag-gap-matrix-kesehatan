"""
Pipeline RAG untuk men-generate Literature GAP Matrix
=====================================================
MK Introduction to Generative AI — Magister AI UDINUS
Domain: Kesehatan (RAG untuk informasi kesehatan, gizi, dan olahraga)

Enam tahap RAG (sesuai materi Bab 5–6):
  1. Document processing  : PDF -> teks per halaman (PyMuPDF), buang bagian References
  2. Chunking             : potongan ~1000 karakter, overlap 200, simpan nomor halaman
  3. Embedding            : sentence-transformers paraphrase-multilingual-mpnet-base-v2 (lokal, 768 dim)
  4. Penyimpanan vektor   : satu indeks FAISS (inner product = cosine) per paper
  5. Retrieval            : untuk tiap kolom matriks, query bilingual -> top-k chunk dari paper itu
  6. Augmented generation : Groq (llama-3.3-70b-versatile), temperature 0, output JSON + nomor halaman

Tambahan:
  - Evaluasi otomatis: skor dukungan (faithfulness proxy) tiap sel, latency, token
  - Pembanding dengan matriks acuan manual (reference/gap_matrix_manual.json)
  - Analisis gap (paragraf) + usulan paper
  - Output: Excel, Markdown, dan DOCX siap unggah ke Kulino

Cara pakai:
  export GROQ_API_KEY=...            # wajib
  export GITHUB_URL=https://github.com/<user>/rag-gap-matrix-kesehatan
  python rag_gap_matrix.py
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Konfigurasi
# ----------------------------------------------------------------------------
BASE = Path(__file__).resolve().parent
PAPERS_CSV = BASE / "papers.csv"
PDF_DIR = BASE / "papers"
OUT_DIR = BASE / "outputs"
REFERENCE_JSON = BASE / "reference" / "gap_matrix_manual.json"

EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = 200
TOP_K = int(os.getenv("TOP_K", "3"))
MIN_SUPPORT = 0.45          # sel dengan skor dukungan di bawah ini ditandai untuk dicek manual

GITHUB_URL = os.getenv("GITHUB_URL", "https://github.com/bintangningwang/rag-gap-matrix-kesehatan")
NAMA = os.getenv("NAMA_MAHASISWA", "Bintang Nugraha Kasaluri Satrianingwang")
TOPIK_USULAN = os.getenv(
    "TOPIK_USULAN",
    "Asisten hidrasi personal berbasis RAG untuk pelari maraton/trail running "
    "(rekomendasi cairan & elektrolit, edukasi risiko hiponatremia dan dehidrasi) "
    "yang memakai pedoman resmi (NATA, ACSM/AND/DC, ISSN, Kemenkes RI) sebagai basis pengetahuan, "
    "dengan pengguna berbahasa Indonesia.",
)

# Kolom sesuai format PPT dosen: Paper | DOI | Problem | Metode | Dataset | Temuan | Keterbatasan | Future work
COLUMNS: dict[str, list[str]] = {
    "Problem": [
        "research problem, background and objective of this study",
        "the aim of this study is to address the challenge",
        "masalah penelitian dan tujuan studi",
    ],
    "Metode": [
        "proposed method: retrieval-augmented generation pipeline, retriever, embedding model, vector database, large language model",
        "system architecture and implementation details, prompt, chunking, temperature",
        "metode dan arsitektur sistem RAG",
    ],
    "Dataset": [
        "dataset, knowledge base, guidelines and documents used as retrieval source",
        "evaluation questions, test set, participants, patients, number of samples",
        "dataset dan sumber data penelitian",
    ],
    "Temuan": [
        "main results: accuracy, performance compared with baseline, statistically significant improvement",
        "key findings of the study",
        "hasil dan temuan utama",
    ],
    "Keterbatasan": [
        "limitations of this study",
        "this study has several limitations, weaknesses, constraints",
        "keterbatasan penelitian",
    ],
    "Future work": [
        "future work and future research directions",
        "further studies should, future research should explore",
        "penelitian selanjutnya",
    ],
}

REF_HEADING = re.compile(r"^\s*(references|bibliography|daftar pustaka|literature cited)\s*$", re.I | re.M)


# ----------------------------------------------------------------------------
# Struktur data
# ----------------------------------------------------------------------------
@dataclass
class Chunk:
    paper_id: str
    page: int
    text: str


@dataclass
class PaperIndex:
    paper_id: str
    chunks: list[Chunk]
    index: object  # faiss.IndexFlatIP
    vectors: np.ndarray


@dataclass
class RunLog:
    rows: list[dict] = field(default_factory=list)

    def add(self, **kw):
        self.rows.append(kw)


# ----------------------------------------------------------------------------
# 0. Unduh PDF (opsional) — semua paper di papers.csv berlisensi open access
# ----------------------------------------------------------------------------
def download_pdfs(papers: pd.DataFrame, pdf_dir: Path = PDF_DIR) -> None:
    import requests

    pdf_dir.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "Mozilla/5.0 (academic RAG coursework)"}
    for _, p in papers.iterrows():
        dest = pdf_dir / f"{p['id']}.pdf"
        if dest.exists() and dest.stat().st_size > 10_000:
            continue
        try:
            r = requests.get(p["pdf_url"], headers=headers, timeout=60, allow_redirects=True)
            if r.ok and r.content[:4] == b"%PDF":
                dest.write_bytes(r.content)
                print(f"[unduh] {p['id']} OK ({len(r.content)//1024} KB)")
            else:
                print(f"[unduh] {p['id']} GAGAL (status {r.status_code}). Unduh manual dari "
                      f"https://doi.org/{p['doi']} lalu simpan sebagai papers/{p['id']}.pdf")
        except Exception as e:  # noqa: BLE001
            print(f"[unduh] {p['id']} GAGAL ({e}). Unduh manual: https://doi.org/{p['doi']}")


# ----------------------------------------------------------------------------
# 1. Document processing
# ----------------------------------------------------------------------------
def extract_pages(pdf_path: Path) -> list[tuple[int, str]]:
    """Kembalikan [(nomor_halaman, teks)], berhenti di heading References."""
    import pymupdf

    pages: list[tuple[int, str]] = []
    with pymupdf.open(pdf_path) as doc:
        for i, page in enumerate(doc, start=1):
            text = page.get_text("text")
            text = re.sub(r"-\n(?=[a-z])", "", text)          # gabung kata terpotong
            text = re.sub(r"[ \t]+", " ", text)
            m = REF_HEADING.search(text)
            if m and i > 2:
                before = text[: m.start()].strip()
                if before:
                    pages.append((i, before))
                break
            pages.append((i, text.strip()))
    return pages


# ----------------------------------------------------------------------------
# 2. Chunking
# ----------------------------------------------------------------------------
def chunk_pages(paper_id: str, pages: list[tuple[int, str]],
                size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page_no, text in pages:
        text = re.sub(r"\s*\n\s*", " ", text).strip()
        if len(text) < 80:
            continue
        start = 0
        while start < len(text):
            end = min(start + size, len(text))
            # usahakan berhenti di akhir kalimat
            if end < len(text):
                dot = text.rfind(". ", start + size // 2, end)
                if dot != -1:
                    end = dot + 1
            chunks.append(Chunk(paper_id, page_no, text[start:end].strip()))
            if end >= len(text):
                break
            start = max(end - overlap, start + 1)
    return chunks


# ----------------------------------------------------------------------------
# 3. Embedding (lokal)
# ----------------------------------------------------------------------------
class LocalEmbedder:
    def __init__(self, model_name: str = EMBED_MODEL):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        v = self.model.encode(texts, batch_size=32, show_progress_bar=False,
                              normalize_embeddings=True, convert_to_numpy=True)
        return v.astype("float32")


# ----------------------------------------------------------------------------
# 4. Penyimpanan vektor (FAISS)
# ----------------------------------------------------------------------------
def build_index(paper_id: str, chunks: list[Chunk], embedder) -> PaperIndex:
    import faiss

    vecs = embedder.encode([c.text for c in chunks])
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    return PaperIndex(paper_id, chunks, index, vecs)


# ----------------------------------------------------------------------------
# 5. Retrieval
# ----------------------------------------------------------------------------
def retrieve(pidx: PaperIndex, queries: list[str], embedder, k: int | None = None) -> list[tuple[Chunk, float]]:
    k = k or TOP_K
    qv = embedder.encode(queries)
    q = qv.mean(axis=0, keepdims=True)
    q = q / (np.linalg.norm(q) + 1e-9)
    scores, ids = pidx.index.search(q.astype("float32"), min(k, len(pidx.chunks)))
    return [(pidx.chunks[i], float(s)) for s, i in zip(scores[0], ids[0]) if i >= 0]


# ----------------------------------------------------------------------------
# 6. Augmented generation (Groq)
# ----------------------------------------------------------------------------
class GroqLLM:
    def __init__(self, model: str = LLM_MODEL):
        from groq import Groq

        self.client = Groq(api_key=os.environ["GROQ_API_KEY"])
        self.model = model

    def chat_json(self, system: str, user: str, max_tokens: int = 2000) -> tuple[dict, dict]:
        for attempt in range(6):
            try:
                t0 = time.time()
                r = self.client.chat.completions.create(
                    model=self.model,
                    temperature=0,
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"},
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": user}],
                )
                meta = {
                    "latency_s": round(time.time() - t0, 2),
                    "prompt_tokens": r.usage.prompt_tokens,
                    "completion_tokens": r.usage.completion_tokens,
                }
                return json.loads(r.choices[0].message.content), meta
            except Exception as e:  # noqa: BLE001 — rate limit free tier / JSON rusak
                if "413" in str(e) or "too large" in str(e).lower():
                    raise RuntimeError("Prompt terlalu besar untuk batas token Groq. Turunkan TOP_K "
                                       "(mis. os.environ['TOP_K']='2') lalu restart & jalankan ulang.") from e
                wait = 15 * (attempt + 1)
                print(f"   [groq] {type(e).__name__}: {str(e)[:120]} -> coba lagi {wait}s")
                time.sleep(wait)
        raise RuntimeError("Groq gagal setelah 6 percobaan")


SYSTEM_EXTRACT = """Anda asisten riset yang menyusun literature gap matrix.
Aturan:
- Gunakan HANYA informasi dalam KONTEKS. Jangan menambah pengetahuan dari luar.
- Tulis dalam Bahasa Indonesia akademik, ringkas (1–3 kalimat per kolom), pertahankan angka penting.
- Nama model, dataset, metrik, dan istilah teknis boleh tetap dalam bahasa Inggris.
- Jika informasi tidak ada di konteks, tulis persis: "Tidak disebutkan".
- Sertakan nomor halaman sumber untuk setiap kolom.
Keluarkan JSON dengan kunci: {cols}. Tiap kunci berisi objek {{"isi": str, "halaman": [int]}}."""


def build_context(retrieved: dict[str, list[tuple[Chunk, float]]]) -> str:
    """Gabungkan chunk tiap kolom; chunk yang sama hanya dikirim sekali (hemat token free tier)."""
    seen: dict[int, str] = {}
    parts = []
    for col, hits in retrieved.items():
        parts.append(f"### Potongan relevan untuk kolom '{col}'")
        for ch, _ in hits:
            key = id(ch)
            if key in seen:
                parts.append(f"(lihat potongan {seen[key]} di atas)")
                continue
            seen[key] = f"#{len(seen) + 1}"
            parts.append(f"{seen[key]} [hlm. {ch.page}] {ch.text}")
    return "\n".join(parts)


def extract_row(paper: pd.Series, pidx: PaperIndex, embedder, llm, log: RunLog) -> dict:
    retrieved = {col: retrieve(pidx, qs, embedder) for col, qs in COLUMNS.items()}
    system = SYSTEM_EXTRACT.format(cols=", ".join(f'"{c}"' for c in COLUMNS))
    user = (f"Judul paper: {paper['judul']}\n\nKONTEKS:\n{build_context(retrieved)}\n\n"
            "Isi setiap kolom gap matrix untuk paper ini.")
    data, meta = llm.chat_json(system, user)
    log.add(tahap="ekstraksi", paper=paper["id"], **meta)

    row = {"ID": paper["id"], "Paper": f"{paper['penulis_tahun']} — {paper['judul']}",
           "DOI": paper["doi"]}
    evidence = []
    for col in COLUMNS:
        cell = data.get(col) or {}
        isi = (cell.get("isi") if isinstance(cell, dict) else str(cell)) or "Tidak disebutkan"
        hal = cell.get("halaman", []) if isinstance(cell, dict) else []
        row[col] = isi
        row[f"{col} (hlm.)"] = ", ".join(str(h) for h in hal)
        # faithfulness proxy: kemiripan isi sel dengan chunk yang di-retrieve
        hits = retrieved[col]
        if isi.strip().lower().startswith("tidak disebutkan") or not hits:
            support = None
        else:
            cv = embedder.encode([isi])[0]
            support = float(max(np.dot(pidx.vectors[pidx.chunks.index(h[0])], cv) for h in hits))
        row[f"{col} (dukungan)"] = None if support is None else round(support, 3)
        for ch, s in hits:
            evidence.append({"ID": paper["id"], "Kolom": col, "Halaman": ch.page,
                             "Skor retrieval": round(s, 3), "Potongan": ch.text[:600]})
    row["_evidence"] = evidence
    return row


# ----------------------------------------------------------------------------
# Analisis gap & usulan paper
# ----------------------------------------------------------------------------
SYSTEM_GAP = """Anda pembimbing riset magister AI. Berdasarkan literature gap matrix,
tulis analisis gap dalam Bahasa Indonesia akademik. Keluarkan JSON dengan kunci:
"pola_umum" (paragraf 4–6 kalimat tentang kecenderungan metode, dataset, dan evaluasi),
"gaps" (daftar 4–6 objek {"gap": str, "bukti": str berisi ID paper, mis. "P02, P06"}),
"analisis_usulan" (2 paragraf: posisi usulan paper terhadap gap dan kontribusinya),
"judul_usulan" (str), "rumusan_masalah" (daftar 3 pertanyaan penelitian),
"metode_usulan" (paragraf singkat rancangan metode dan evaluasi).
Hanya rujuk paper yang ada di matriks; jangan mengarang referensi baru."""


def analyze_gap(matrix: pd.DataFrame, llm, log: RunLog, topik: str = TOPIK_USULAN) -> dict:
    cols = ["ID", "Paper"] + list(COLUMNS)
    compact = matrix[cols].to_dict(orient="records")
    user = (f"Rencana topik paper kelompok: {topik}\n\nGAP MATRIX (JSON):\n"
            f"{json.dumps(compact, ensure_ascii=False)}")
    data, meta = llm.chat_json(SYSTEM_GAP, user, max_tokens=3000)
    log.add(tahap="analisis_gap", paper="ALL", **meta)
    return data


# ----------------------------------------------------------------------------
# Evaluasi terhadap matriks acuan manual
# ----------------------------------------------------------------------------
def compare_with_reference(matrix: pd.DataFrame, embedder, ref_path: Path = REFERENCE_JSON) -> pd.DataFrame:
    if not ref_path.exists():
        return pd.DataFrame()
    ref = json.loads(ref_path.read_text(encoding="utf-8"))
    rows = []
    for _, r in matrix.iterrows():
        if r["ID"] not in ref:
            continue
        out = {"ID": r["ID"]}
        for col in COLUMNS:
            a, b = str(r[col]), ref[r["ID"]].get(col, "")
            va, vb = embedder.encode([a, b])
            out[col] = round(float(np.dot(va, vb)), 3)
        rows.append(out)
    df = pd.DataFrame(rows)
    if not df.empty:
        mean = {"ID": "Rata-rata", **{c: round(df[c].mean(), 3) for c in COLUMNS}}
        df = pd.concat([df, pd.DataFrame([mean])], ignore_index=True)
    return df


# ----------------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------------
def save_excel(matrix, evidence, gap, log_df, ref_cmp, path: Path):
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        info = pd.DataFrame({
            "Item": ["Nama", "Mata kuliah", "Tugas", "GitHub", "Embedding", "LLM", "Chunk", "Top-k"],
            "Nilai": [NAMA, "Introduction to Generative AI (P36104)", "Literature GAP Matrix via RAG",
                      GITHUB_URL, EMBED_MODEL, f"Groq {LLM_MODEL} (temperature 0)",
                      f"{CHUNK_SIZE} karakter, overlap {CHUNK_OVERLAP}", TOP_K],
        })
        info.to_excel(xw, sheet_name="Info", index=False)
        main_cols = ["Paper", "DOI"] + list(COLUMNS)
        matrix[main_cols].to_excel(xw, sheet_name="Gap Matrix", index=False)
        detail_cols = ["ID"] + [c for col in COLUMNS for c in (f"{col} (hlm.)", f"{col} (dukungan)")]
        matrix[detail_cols].to_excel(xw, sheet_name="Halaman & Dukungan", index=False)
        evidence.to_excel(xw, sheet_name="Bukti Retrieval", index=False)
        gap_rows = [("Pola umum", gap.get("pola_umum", ""))]
        gap_rows += [(f"Gap {i+1} ({g.get('bukti', '')})", g.get("gap", "")) for i, g in enumerate(gap.get("gaps", []))]
        gap_rows += [("Analisis usulan", "\n\n".join(_as_list(gap.get("analisis_usulan", "")))),
                     ("Judul usulan", gap.get("judul_usulan", "")),
                     ("Rumusan masalah", "\n".join(_as_list(gap.get("rumusan_masalah", [])))),
                     ("Metode usulan", gap.get("metode_usulan", ""))]
        pd.DataFrame(gap_rows, columns=["Bagian", "Isi"]).to_excel(xw, sheet_name="Analisis Gap", index=False)
        log_df.to_excel(xw, sheet_name="Log Latency & Token", index=False)
        if not ref_cmp.empty:
            ref_cmp.to_excel(xw, sheet_name="Evaluasi vs Manual", index=False)
    # rapikan lebar kolom
    from openpyxl import load_workbook
    from openpyxl.styles import Alignment, Font

    wb = load_workbook(path)
    for ws in wb.worksheets:
        for cell in ws[1]:
            cell.font = Font(bold=True)
        for col in ws.columns:
            letter = col[0].column_letter
            longest = max(len(str(c.value or "")) for c in col)
            ws.column_dimensions[letter].width = min(max(12, longest * 0.9), 60)
            for c in col:
                c.alignment = Alignment(wrap_text=True, vertical="top")
    wb.save(path)


def _as_list(x):
    if isinstance(x, list):
        return [str(i) for i in x]
    return [str(x)] if x else []


def save_markdown(matrix, gap, path: Path):
    cols = ["Paper", "DOI"] + list(COLUMNS)
    lines = ["# Literature GAP Matrix — RAG di Domain Kesehatan", "",
             f"Nama: {NAMA}  ", f"GitHub: {GITHUB_URL}", "",
             "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in matrix.iterrows():
        lines.append("| " + " | ".join(str(r[c]).replace("|", "/").replace("\n", " ") for c in cols) + " |")
    lines += ["", "## Analisis Gap", "", gap.get("pola_umum", ""), ""]
    for g in gap.get("gaps", []):
        lines.append(f"- **{g.get('gap', '')}** ({g.get('bukti', '')})")
    lines += ["", "## Usulan Paper", "", f"**Judul:** {gap.get('judul_usulan', '')}", ""]
    lines += _as_list(gap.get("analisis_usulan", ""))
    lines += ["", "**Rumusan masalah:**"] + [f"{i+1}. {q}" for i, q in enumerate(_as_list(gap.get("rumusan_masalah", [])))]
    lines += ["", "**Metode usulan:** " + str(gap.get("metode_usulan", ""))]
    path.write_text("\n".join(lines), encoding="utf-8")


def save_docx(matrix, gap, papers, path: Path, ref_cmp=None, log_df=None, note: str | None = None):
    """Dokumen siap unggah ke Kulino (A4 landscape)."""
    from docx import Document
    from docx.enum.section import WD_ORIENT
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Cm, Pt, RGBColor

    doc = Document()
    sec = doc.sections[0]
    sec.orientation = WD_ORIENT.LANDSCAPE
    sec.page_width, sec.page_height = Cm(29.7), Cm(21.0)
    for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
        setattr(sec, side, Cm(1.6))

    st = doc.styles["Normal"]
    st.font.name = "Calibri"
    st.font.size = Pt(10.5)
    for lvl, size in ((1, 14), (2, 12)):
        h = doc.styles[f"Heading {lvl}"]
        h.font.name, h.font.size, h.font.color.rgb = "Calibri", Pt(size), RGBColor(0x1F, 0x3A, 0x5F)

    t = doc.add_paragraph()
    r = t.add_run("Literature GAP Matrix: Retrieval-Augmented Generation (RAG) di Domain Kesehatan")
    r.bold, r.font.size = True, Pt(16)
    meta = [("Nama", NAMA), ("Mata kuliah", "Introduction to Generative AI (P36104), Magister AI UDINUS"),
            ("Tugas", "Tugas tambahan awal RTM 2: Literature GAP Matrix via pipeline RAG"),
            ("Repositori GitHub", GITHUB_URL),
            ("Pipeline", f"PyMuPDF → chunk {CHUNK_SIZE}/{CHUNK_OVERLAP} → {EMBED_MODEL.split('/')[-1]} (lokal) "
                         f"→ FAISS → top-{TOP_K} per kolom → Groq {LLM_MODEL}")]
    for k, v in meta:
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(0)
        rk = p.add_run(f"{k}: ")
        rk.bold = True
        p.add_run(v)
    if note:
        p = doc.add_paragraph()
        rn = p.add_run(note)
        rn.italic, rn.font.color.rgb = True, RGBColor(0xA0, 0x30, 0x30)

    doc.add_heading("1. Gap Matrix", level=1)
    cols = ["No", "Paper", "DOI"] + list(COLUMNS)
    widths = [0.8, 3.4, 2.4, 3.1, 3.6, 3.1, 3.4, 3.2, 3.5]  # total 26.5 cm
    table = doc.add_table(rows=1, cols=len(cols))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False

    def shade(cell, hex_color):
        tcPr = cell._tc.get_or_add_tcPr()
        shd = OxmlElement("w:shd")
        shd.set(qn("w:val"), "clear")
        shd.set(qn("w:color"), "auto")
        shd.set(qn("w:fill"), hex_color)
        tcPr.append(shd)

    def fill(cell, text, bold=False, size=8):
        cell.text = ""
        par = cell.paragraphs[0]
        run = par.add_run(str(text))
        run.bold, run.font.size = bold, Pt(size)

    for i, c in enumerate(cols):
        fill(table.rows[0].cells[i], c, bold=True, size=8.5)
        shade(table.rows[0].cells[i], "D9E2F3")
    # ulangi header di setiap halaman
    trPr = table.rows[0]._tr.get_or_add_trPr()
    th = OxmlElement("w:tblHeader")
    th.set(qn("w:val"), "true")
    trPr.append(th)

    for n, (_, r) in enumerate(matrix.iterrows(), start=1):
        cells = table.add_row().cells
        vals = [n, r["Paper"], r["DOI"]] + [r[c] for c in COLUMNS]
        for i, v in enumerate(vals):
            fill(cells[i], v)
    for row in table.rows:
        for i, w in enumerate(widths):
            row.cells[i].width = Cm(w)

    doc.add_heading("2. Analisis Gap dan Usulan Paper", level=1)
    doc.add_heading("2.1 Pola umum penelitian", level=2)
    doc.add_paragraph(gap.get("pola_umum", ""))
    doc.add_heading("2.2 Gap penelitian yang teridentifikasi", level=2)
    for i, g in enumerate(gap.get("gaps", []), start=1):
        p = doc.add_paragraph(style="List Number")
        rb = p.add_run(f"{g.get('gap', '')} ")
        p.add_run(f"(bukti: {g.get('bukti', '')})").italic = True
    doc.add_heading("2.3 Usulan paper", level=2)
    p = doc.add_paragraph()
    p.add_run("Judul usulan: ").bold = True
    p.add_run(gap.get("judul_usulan", ""))
    for para in _as_list(gap.get("analisis_usulan", "")):
        doc.add_paragraph(para)
    p = doc.add_paragraph()
    p.add_run("Rumusan masalah:").bold = True
    for q in _as_list(gap.get("rumusan_masalah", [])):
        doc.add_paragraph(q, style="List Number 2")
    p = doc.add_paragraph()
    p.add_run("Rancangan metode: ").bold = True
    p.add_run(str(gap.get("metode_usulan", "")))

    if ref_cmp is not None and not ref_cmp.empty:
        doc.add_heading("3. Evaluasi Pipeline", level=1)
        doc.add_paragraph(
            "Kemiripan semantik (cosine, model embedding yang sama) antara isi sel hasil pipeline dan matriks "
            "acuan yang disusun manual. Nilai mendekati 1 berarti isi sel sejalan dengan pembacaan manual.")
        mean = ref_cmp[ref_cmp["ID"] == "Rata-rata"]
        if not mean.empty:
            doc.add_paragraph("; ".join(f"{c}: {mean.iloc[0][c]}" for c in COLUMNS))
        if log_df is not None and not log_df.empty:
            doc.add_paragraph(
                f"Total panggilan LLM: {len(log_df)}; rata-rata latency {log_df['latency_s'].mean():.2f} s; "
                f"total token {int(log_df['prompt_tokens'].sum() + log_df['completion_tokens'].sum())}.")

    doc.add_heading("Daftar Pustaka", level=1)
    for _, p_ in papers.iterrows():
        doc.add_paragraph(f"{p_['penulis_tahun']}. {p_['judul']}. {p_['jurnal']}. "
                          f"https://doi.org/{p_['doi']}")
    doc.save(path)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def run(embedder=None, llm=None, download: bool = True, only: list[str] | None = None) -> dict:
    OUT_DIR.mkdir(exist_ok=True)
    papers = pd.read_csv(PAPERS_CSV)
    if only:
        papers = papers[papers["id"].isin(only)]
    if download:
        download_pdfs(papers)

    embedder = embedder or LocalEmbedder()
    llm = llm or GroqLLM()
    log = RunLog()

    rows, evidence = [], []
    for _, p in papers.iterrows():
        pdf = PDF_DIR / f"{p['id']}.pdf"
        if not pdf.exists():
            print(f"[lewati] {p['id']}: PDF belum ada di {pdf}")
            continue
        pages = extract_pages(pdf)
        chunks = chunk_pages(p["id"], pages)
        pidx = build_index(p["id"], chunks, embedder)
        print(f"[{p['id']}] {len(pages)} halaman, {len(chunks)} chunk -> ekstraksi ...")
        row = extract_row(p, pidx, embedder, llm, log)
        evidence += row.pop("_evidence")
        rows.append(row)

    matrix = pd.DataFrame(rows)
    evidence_df = pd.DataFrame(evidence)
    log_df = pd.DataFrame(log.rows)

    print("[analisis] menyusun analisis gap ...")
    gap = analyze_gap(matrix, llm, log)
    log_df = pd.DataFrame(log.rows)
    ref_cmp = compare_with_reference(matrix, embedder)

    # sel yang perlu dicek manual
    flags = []
    for _, r in matrix.iterrows():
        for col in COLUMNS:
            s = r.get(f"{col} (dukungan)")
            if s is not None and not pd.isna(s) and s < MIN_SUPPORT:
                flags.append(f"{r['ID']} / {col} (dukungan {s})")
    if flags:
        print("[cek manual] sel dengan dukungan rendah:\n  - " + "\n  - ".join(flags))

    matrix.to_csv(OUT_DIR / "gap_matrix.csv", index=False)
    (OUT_DIR / "analisis_gap.json").write_text(json.dumps(gap, ensure_ascii=False, indent=2), encoding="utf-8")
    save_excel(matrix, evidence_df, gap, log_df, ref_cmp, OUT_DIR / "gap_matrix.xlsx")
    save_markdown(matrix, gap, OUT_DIR / "gap_matrix.md")
    save_docx(matrix, gap, papers[papers["id"].isin(matrix["ID"])], OUT_DIR / "Literature_GAP_Matrix_RAG_Kesehatan.docx", ref_cmp, log_df)
    print(f"[selesai] output di {OUT_DIR}")
    return {"matrix": matrix, "gap": gap, "log": log_df, "ref_cmp": ref_cmp, "flags": flags}


if __name__ == "__main__":
    run()
