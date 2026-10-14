import tempfile
from pathlib import Path

import streamlit as st

from compare_documents import read_docx, read_pdf
from semantic_compare.nlp_processor import DocumentSemanticProcessor


def similarity_spectrum(score: float):
    """Return a normalized color-coded spectrum position for the similarity score."""
    if score <= 0.0:
        return 0.0
    if score >= 1.0:
        return 1.0
    return score


def render_similarity_spectrum(score: float):
    """Render a visual spectrum using the full 0.0 to 1.0 range with exact score placement."""
    pos = similarity_spectrum(score)
    marker_percent = max(0.0, min(100.0, pos * 100.0))

    spectrum_html = f"""
    <div style="width: 100%; max-width: 700px; margin-top: 12px;">
      <div style="position: relative; height: 28px; border-radius: 10px; overflow: hidden; background: linear-gradient(to right, #d32f2f 0%, #f9a825 40%, #fdd835 70%, #2e7d32 100%); border: 1px solid rgba(0,0,0,0.15);">
        <div style="position: absolute; left: {marker_percent}%; top: -4px; width: 0; height: 0; border-left: 9px solid transparent; border-right: 9px solid transparent; border-bottom: 16px solid #111827; transform: translateX(-50%);"></div>
      </div>
      <div style="display: flex; justify-content: space-between; font-size: 12px; color: #374151; margin-top: 6px;">
        <span>Low 0.0</span>
        <span>Medium 0.5</span>
        <span>High 1.0</span>
      </div>
    </div>
    """
    st.markdown(spectrum_html, unsafe_allow_html=True)
    st.caption(f"Document score: {score:.4f} | marker at {marker_percent:.1f}% of the spectrum")


def sentence_match_table(matches, *, label):
    """Render a sentence match table with the label for strongest or weakest matches."""
    if not matches:
        st.info(f"No {label.lower()} matches were found.")
        return

    frame = [
        {
            "score": round(item["score"], 4),
            "doc1_sentence": item["doc1_sentence"],
            "doc2_sentence": item["doc2_sentence"],
        }
        for item in matches
    ]
    st.dataframe(frame, use_container_width=True)

MODEL_OPTIONS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
]


@st.cache_resource
def get_processor(model_name: str) -> DocumentSemanticProcessor:
    return DocumentSemanticProcessor(model_name=model_name)


def read_uploaded_document(uploaded_file) -> str:
    if uploaded_file is None:
        return ""

    suffix = Path(uploaded_file.name).suffix.lower()
    temp_dir = tempfile.mkdtemp(prefix="doc_compare_")
    temp_path = Path(temp_dir) / uploaded_file.name
    temp_path.write_bytes(uploaded_file.getvalue())

    if suffix == ".pdf":
        return read_pdf(temp_path)
    if suffix == ".docx":
        return read_docx(temp_path)
    return temp_path.read_text(encoding="utf-8", errors="replace")


st.set_page_config(page_title="Document Semantic Compare", page_icon="📄")
st.title("Document Semantic Comparison")
st.caption("Compare two files semantically using SBERT embeddings and cosine similarity.")

with st.sidebar:
    st.header("Settings")
    model_name = st.selectbox("SBERT model", MODEL_OPTIONS, index=0)
    top_k = st.slider("Sentence match count", min_value=1, max_value=25, value=5)
    st.caption("This slider controls both the strongest and weakest match lists.")

col1, col2 = st.columns(2)
with col1:
    doc1_file = st.file_uploader("Select Document 1", type=["txt", "md", "pdf", "docx"])
with col2:
    doc2_file = st.file_uploader("Select Document 2", type=["txt", "md", "pdf", "docx"])

if doc1_file is None or doc2_file is None:
    st.warning("Please upload both documents before comparing.")
else:
    processor = get_processor(model_name)
    text1 = read_uploaded_document(doc1_file)
    text2 = read_uploaded_document(doc2_file)

    if not text1.strip() or not text2.strip():
        st.warning("Both uploaded documents must contain readable text.")
    else:
        result = processor.compare(
            text1=text1,
            text2=text2,
            normalize=True,
            include_alignments=True,
            top_k=max(top_k, 25),
        )

        st.subheader("Comparison Result")
        st.metric(
            label="Coverage-adjusted similarity",
            value=f"{result['cosine_similarity']:.4f}",
        )
        st.caption(
            "This score combines bidirectional sentence coverage with semantic similarity "
            "to penalize unrelated content."
        )
        raw_embedding_score = result.get("embedding_cosine_similarity")
        if raw_embedding_score is not None:
            st.metric(
                label="Raw embedding cosine",
                value=f"{raw_embedding_score:.4f}",
            )
        else:
            st.info(
                "Raw embedding cosine is unavailable from the active processor. "
                "Restart Streamlit after updating the comparison processor."
            )
        st.write(f"Interpretation: {result['interpretation'].upper()}")

        with st.expander("Document 1 preview"):
            st.text_area("Document 1 preview", text1[:4000], height=220, label_visibility="collapsed", key="doc1_preview")
        with st.expander("Document 2 preview"):
            st.text_area("Document 2 preview", text2[:4000], height=220, label_visibility="collapsed", key="doc2_preview")

        alignments = result.get(
            "sentence_coverage",
            result.get("sentence_alignments", []),
        )
        if alignments:
            strongest = sorted(alignments, key=lambda item: item["score"], reverse=True)
            weakest = sorted(alignments, key=lambda item: item["score"])
            average_sentence_score = sum(item["score"] for item in alignments) / len(alignments)

            st.subheader("Sentence-level summary")
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Best sentence match", f"{strongest[0]['score']:.4f}")
            col_b.metric("Average sentence match", f"{average_sentence_score:.4f}")
            col_c.metric("Lowest sentence match", f"{weakest[0]['score']:.4f}")

            st.subheader("Similarity Spectrum")
            render_similarity_spectrum(result["cosine_similarity"])

            strongest_slice = strongest[:top_k]
            weakest_slice = weakest[:top_k]

            st.subheader(f"Strongest {len(strongest_slice)} sentence matches")
            sentence_match_table(strongest_slice, label="strongest")

            st.subheader(f"Lowest {len(weakest_slice)} sentence matches")
            sentence_match_table(weakest_slice, label="lowest")
        else:
            st.info("No sentence-level matches were found for these documents.")
