# Document Meaning Comparison

A small Python project for comparing the semantic similarity of two documents using Sentence-BERT embeddings and cosine similarity.

The project includes both:

- a command-line interface (`compare_documents.py`) for quick comparisons and batch processing
- a Streamlit web app (`streamlit_app.py`) for interactive file-based comparison and visual output

It supports plain text, Markdown, PDF, and DOCX inputs, and can optionally surface sentence-level alignment matches along with the overall document score.

---

## What the project does

This project uses Sentence-BERT embeddings and cosine similarity to compare documents:

- document similarity score: a coverage-adjusted cosine score that measures shared meaning and penalizes content found in only one document
- sentence alignment matches: which sentence in document 1 best matches which sentence in document 2
- interpretation: labels the score as low, medium, or high

The default model is `sentence-transformers/all-MiniLM-L6-v2`, and the app also includes `sentence-transformers/all-mpnet-base-v2` as a higher-quality alternative.

---

## Project structure

- `compare_documents.py` — CLI driver for single-document and batch comparison
- `streamlit_app.py` — interactive Streamlit interface
- `semantic_compare/nlp_processor.py` — core embedding and similarity logic
- `input/` — example inputs and CSV templates
- `tests/` — unit tests for the processor and CLI behaviors
- `requirements.txt` — Python dependencies

---

## Installation

Create a Python environment and install dependencies:

```bash
pip install -r requirements.txt
```

If you want to use the app in a virtual environment or Conda environment, activate that environment first and then run the install step.

---

## CLI usage

### Compare two raw strings

```bash
python compare_documents.py "This is the first document." "This is the second document."
```

### Compare two files

```bash
python compare_documents.py sample1.txt sample2.txt
```

### Compare PDF or DOCX files

```bash
python compare_documents.py report.pdf proposal.docx
```

### Show sentence alignment details

```bash
python compare_documents.py doc1.txt doc2.txt --sentence-align --top-k 5
```

### Include interpretation label

```bash
python compare_documents.py doc1.txt doc2.txt --with-label
```

### Output JSON

```bash
python compare_documents.py doc1.txt doc2.txt --json
```

### Batch comparison from CSV

The batch CSV must include `doc1` and `doc2` columns. A `pair_id` column is optional.

```bash
python compare_documents.py --batch-csv input/pairs.example.csv --out results.csv
```

You can also write JSON Lines output:

```bash
python compare_documents.py --batch-csv input/pairs.example.csv --out results.jsonl --batch-format jsonl
```

### Using a different SBERT model

```bash
python compare_documents.py doc1.txt doc2.txt --model sentence-transformers/all-mpnet-base-v2
```

---

## Streamlit interface

Start the app with:

```bash
streamlit run streamlit_app.py
```

### What the UI includes

- file upload for both documents
- model selection for the SBERT encoder
- sentence match count slider
- coverage-adjusted document similarity score
- raw whole-document embedding cosine score
- interpretation label (low / medium / high)
- visual similarity spectrum
- preview panes for both uploaded texts
- strongest and weakest sentence match tables
- summary cards for best, average, and lowest sentence similarity

### UI workflow

1. Upload Document 1 and Document 2.
2. Choose a model from the sidebar.
3. Adjust the sentence match count slider.
4. Review the semantic similarity result and sentence alignment lists.
5. Use the preview panes to inspect the source text around the comparison.

### Building a Windows EXE with PyInstaller

If you want to package the Streamlit app as a standalone Windows executable, the project includes a working launcher and spec file so the packaged app launches through Streamlit correctly.

1. Install dependencies including PyInstaller:

```bash
pip install -r requirements.txt
```

2. Use the bundled launcher in the project root:

`run_streamlit.py`

```python
import sys
from pathlib import Path

from streamlit.web import bootstrap


def resolve_app_path() -> str:
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        base_dir = Path(sys._MEIPASS)
    else:
        base_dir = Path(__file__).resolve().parent

    app_path = base_dir / "streamlit_app.py"
    if not app_path.exists():
        raise FileNotFoundError(f"Could not find Streamlit app: {app_path}")
    return str(app_path)


if __name__ == "__main__":
    app_path = resolve_app_path()
    bootstrap.run(
        app_path,
        "",
        [],
        {
            "server.headless": False,
            "server.port": 8501,
        },
    )
```

3. Build the executable using the project spec file:

```bash
pyinstaller run_streamlit.spec
```

This produces a Windows EXE in the `dist/` folder that launches the Streamlit UI instead of exiting silently.

Notes:

- The `run_streamlit.spec` file already includes the needed Streamlit hooks and the app files.
- Use `console=False` in the spec for a GUI app; this is already configured for the packaged executable.
- The launcher is required because Streamlit apps should be started through the Streamlit bootstrap path rather than by direct script execution.

---

## Underlying comparison method

### 1. Text normalization

The processor applies minimal normalization before comparing text:

- whitespace is collapsed
- leading/trailing spaces are removed
- the text is otherwise preserved so sentence and word meaning remain intact

This keeps the semantic comparison stable without aggressively rewriting the content.

### 2. Embedding generation

Both documents are encoded with a Sentence-BERT model into dense vectors. These vectors capture semantic meaning rather than exact word overlap.

The raw document embedding similarity is computed as the cosine similarity between the two embeddings. The reported `cosine_similarity` is then adjusted using bidirectional sentence coverage:

- each sentence is matched to its strongest counterpart
- weak matches below 0.5 are treated as misses
- sentence matches are weighted by sentence length
- coverage is calculated in both directions and combined with a geometric mean

This prevents a document from receiving a high score merely because it contains one matching document plus several unrelated paragraphs. The unadjusted embedding result is also available as `embedding_cosine_similarity` in programmatic results.

- values near `1.0` imply very similar meaning
- values near `0.0` imply weak similarity
- values in between are moderate semantic similarity

### 3. Interpretation thresholds

The project interprets score ranges as:

- below `0.4` = low
- between `0.4` and `0.7` = medium
- above `0.7` = high

These thresholds are intentionally simple and are useful for quick human-readable feedback.

### 4. Sentence alignment

For sentence-level analysis, the text is split into sentence-like units and each sentence in document 1 is paired with the sentence in document 2 that has the highest cosine similarity to it.

This is a greedy best-match strategy, not a full global optimization across all sentence pairs. In other words, the method does not search for the globally least-similar sentence pair across the entire matrix. Instead, for each sentence in document 1, it finds the single best partner in document 2, then ranks those best matches by their cosine similarity score.

So the lowest score in the alignment list is the weakest among the selected best local matches, not the least-similar pairing possible across every sentence combination. It answers: “Which sentence in document 2 is closest to each sentence in document 1?” rather than “Which two sentences are the most different overall?”

This generates a list of best sentence matches and allows the app to show:

- strongest sentence matches
- weakest sentence matches
- average sentence-level similarity
- best sentence similarity

This is useful because the overall document score and the strongest sentence match are not always identical; they measure similarity at different levels of granularity.

The lowest scoring pair is not the least similar possible pairing but the least similar best pairing.

---

## Available SBERT models and trade-offs

The project exposes two models in the UI and CLI:

| Model | Strengths | Trade-offs | Best use case |
| --- | --- | --- | --- |
| `sentence-transformers/all-MiniLM-L6-v2` | Fast, lightweight, lower memory usage, quick to run, good general semantic matching | May miss finer semantic nuance compared with larger models; less powerful on subtle paraphrases or domain-heavy text | Everyday comparison, demos, quick local processing, low-resource environments |
| `sentence-transformers/all-mpnet-base-v2` | Higher semantic quality, better on nuanced paraphrases and difficult similarity tasks, often stronger overall match quality | Slower inference, larger model and memory footprint, can take longer to load and compare | Higher-quality comparisons, more precise document similarity, stronger sentence alignment |

### Recommendation

- Use `all-MiniLM-L6-v2` when you want speed and responsiveness.
- Use `all-mpnet-base-v2` when accuracy matters more than runtime.

In practice, the MiniLM model is usually the better default for local experimentation and interactive tools, while MPNet is the better choice when you want the strongest semantic matching.

---

## Example output

```text
Cosine similarity: 0.869421
Interpretation: high

Top sentence alignments:
1. score=0.9234
   doc1: The project improved system reliability.
   doc2: This project increased the reliability of the system.
2. score=0.8812
   doc1: Users reported fewer errors after deployment.
   doc2: After deployment, users saw fewer errors.
```

---

## Notes

- The project is designed to compare meaning, not exact wording.
- It handles plain text and common document formats.
- The app is useful for comparing academic writing, reports, proposals, drafts, and other text-heavy documents.
- For large documents, the similarity and sentence alignment can still work, but performance depends on the selected model and the amount of text being processed.

---

## License

This project is provided under the repository license. See the license file for details.
