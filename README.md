# rag-gap-matrix-kesehatan

Pipeline **Retrieval-Augmented Generation (RAG)** untuk men-generate *Literature GAP Matrix* dari 10 artikel jurnal tentang RAG di domain kesehatan.
Tugas tambahan awal RTM 2, MK Introduction to Generative AI (P36104), Magister AI UDINUS.

## Alur pipeline

```
papers.csv ──► unduh PDF (open access)
   │
   ▼
1. Document processing   PyMuPDF, teks per halaman, bagian References dibuang
2. Chunking              ±1000 karakter, overlap 200, potong di akhir kalimat, simpan no. halaman
3. Embedding             sentence-transformers/paraphrase-multilingual-mpnet-base-v2 (lokal, 768 dim)
4. Vector store          FAISS IndexFlatIP (cosine), satu indeks per paper
5. Retrieval             per kolom matriks: rata-rata embedding 3 query (EN+ID) → top-3 chunk (atur lewat env TOP_K)
6. Generation            Groq llama-3.3-70b-versatile, temperature 0, output JSON + halaman sumber
   │
   ▼
Analisis gap (LLM)  →  Evaluasi  →  outputs/  (DOCX untuk Kulino, XLSX, MD, CSV)
```

Kolom matriks mengikuti format dosen: **Paper | DOI | Problem | Metode | Dataset | Temuan | Keterbatasan | Future work**.

## Evaluasi yang dihasilkan otomatis

| Dimensi | Cara ukur |
|---|---|
| Faithfulness (proxy) | Skor dukungan: cosine antara isi sel dan chunk yang di-retrieve. Sel < 0,45 ditandai untuk dicek manual |
| Relevansi | Kemiripan isi sel pipeline dengan matriks acuan manual (`reference/gap_matrix_manual.json`) |
| Ketelusuran | Nomor halaman per sel + sheet *Bukti Retrieval* (potongan teks yang dipakai) |
| Biaya & latency | Token prompt/completion dan waktu tiap panggilan Groq (sheet *Log Latency & Token*) |

## Korpus (10 artikel, semua open access)

| ID | Artikel | Publisher | DOI |
|---|---|---|---|
| P01 | Kresevic et al. (2024), npj Digital Medicine | Springer Nature | 10.1038/s41746-024-01091-y |
| P02 | Chen et al. (2024), J Med Internet Res | JMIR | 10.2196/58158 |
| P03 | Alkhalaf et al. (2024), J Biomed Inform | Elsevier | 10.1016/j.jbi.2024.104662 |
| P04 | Lee et al. (2024), J Pers Med | MDPI | 10.3390/jpm14121131 |
| P05 | Hou et al. (2025), J Med Internet Res | JMIR | 10.2196/67677 |
| P06 | Ke et al. (2025), npj Digital Medicine | Springer Nature | 10.1038/s41746-025-01519-z |
| P07 | Farías et al. (2025), Technologies | MDPI | 10.3390/technologies13040140 |
| P08 | Gavai & van Hillegersberg (2025), PLOS Digital Health | PLOS | 10.1371/journal.pdig.0000758 |
| P09 | Xu et al. (2025), Frontiers in Public Health | Frontiers | 10.3389/fpubh.2025.1635381 |
| P10 | Jeon et al. (2025), JMIR Formative Research | JMIR | 10.2196/72553 |

PDF tidak disimpan di repo. Skrip mengunduhnya otomatis dari `pdf_url`. Bila ada yang gagal (mis. diblokir publisher), unduh manual lewat DOI dan simpan sebagai `papers/P0x.pdf`.

## Cara menjalankan

**Google Colab (disarankan):** buka `gap_matrix_colab.ipynb`, isi `GROQ_API_KEY` di Colab Secrets, lalu Run all.

**Lokal:**
```bash
pip install -r requirements.txt
export GROQ_API_KEY=gsk_...
export GITHUB_URL=https://github.com/<username>/rag-gap-matrix-kesehatan
python rag_gap_matrix.py
```

Output di `outputs/`:
- `Literature_GAP_Matrix_RAG_Kesehatan.docx`: matriks + analisis gap + usulan paper (siap unggah ke Kulino)
- `gap_matrix.xlsx`: sheet Info, Gap Matrix, Halaman & Dukungan, Bukti Retrieval, Analisis Gap, Log Latency & Token, Evaluasi vs Manual
- `gap_matrix.md`, `gap_matrix.csv`, `analisis_gap.json`

## Struktur repo

```
rag_gap_matrix.py           pipeline lengkap (6 tahap RAG + analisis + evaluasi + output)
gap_matrix_colab.ipynb      notebook Colab langkah demi langkah
papers.csv                  metadata 10 artikel (judul, jurnal, DOI, URL PDF)
reference/                  matriks & analisis acuan manual (pembanding evaluasi)
papers/  outputs/           PDF dan hasil (diabaikan git)
```

## Catatan
- API key tidak pernah ditulis di kode; pakai environment variable / Colab Secrets.
- Output LLM tetap harus diverifikasi manual. Pakai sheet *Bukti Retrieval* dan nomor halaman.
