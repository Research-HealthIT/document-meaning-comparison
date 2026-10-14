import unittest

from semantic_compare.nlp_processor import DocumentSemanticProcessor


class FakeScalar:
    def __init__(self, value: float) -> None:
        self.value = value

    def item(self) -> float:
        return self.value


class FakeEmbedder:
    def encode(self, texts, convert_to_tensor=True):
        return texts


def fake_cos_for_compare(a, b):
    if isinstance(a, list) and isinstance(b, list):
        return [
            [0.95, 0.30],
            [0.20, 0.80],
        ]
    return FakeScalar(0.72)


class DocumentSemanticProcessorTests(unittest.TestCase):
    def setUp(self):
        self.processor = DocumentSemanticProcessor(
            model_name="fake-model",
            embedder=FakeEmbedder(),
            cos_sim_fn=fake_cos_for_compare,
        )

    def test_compare_returns_similarity_and_interpretation(self):
        result = self.processor.compare(
            text1="  alpha   text  ",
            text2="alpha text",
            normalize=True,
            include_alignments=False,
            top_k=5,
        )

        self.assertAlmostEqual(result["cosine_similarity"], (0.95 * 0.80) ** 0.5)
        self.assertAlmostEqual(result["embedding_cosine_similarity"], 0.72)
        self.assertEqual(result["interpretation"], "high")
        self.assertNotIn("sentence_alignments", result)

    def test_coverage_score_penalizes_unrelated_extra_sentences(self):
        processor = DocumentSemanticProcessor(
            model_name="fake-model",
            embedder=FakeEmbedder(),
            cos_sim_fn=lambda a, b: (
                [[0.9, 0.1], [0.1, 0.1]]
                if isinstance(a, list) and isinstance(b, list)
                else FakeScalar(0.9)
            ),
        )

        result = processor.compare(
            text1="Shared sentence.",
            text2="Shared sentence. Unrelated sentence.",
            normalize=False,
            include_alignments=False,
            top_k=5,
        )

        self.assertAlmostEqual(result["cosine_similarity"], (0.9 * 0.45) ** 0.5)
        self.assertLess(result["cosine_similarity"], 0.7)

    def test_compare_raises_on_empty_input_after_normalization(self):
        with self.assertRaises(ValueError):
            self.processor.compare(
                text1="   ",
                text2="valid",
                normalize=True,
                include_alignments=False,
                top_k=5,
            )

    def test_compare_includes_top_k_sentence_alignments(self):
        result = self.processor.compare(
            text1="First sentence. Second sentence.",
            text2="One match here. Another match.",
            normalize=False,
            include_alignments=True,
            top_k=1,
        )

        self.assertIn("sentence_alignments", result)
        alignments = result["sentence_alignments"]
        self.assertEqual(len(alignments), 1)
        self.assertAlmostEqual(alignments[0]["score"], 0.95)
        self.assertIn("doc1_sentence", alignments[0])
        self.assertIn("doc2_sentence", alignments[0])

    def test_sentence_alignment_respects_requested_count_above_five(self):
        processor = DocumentSemanticProcessor(
            model_name="fake-model",
            embedder=FakeEmbedder(),
            cos_sim_fn=lambda a, b: [
                [1.0 if i == j else 0.1 for j in range(len(b))]
                for i in range(len(a))
            ],
        )

        text1 = "Sentence 1. Sentence 2. Sentence 3. Sentence 4. Sentence 5. Sentence 6."
        text2 = "Alpha 1. Alpha 2. Alpha 3. Alpha 4. Alpha 5. Alpha 6."

        result = processor.compare(
            text1=text1,
            text2=text2,
            normalize=False,
            include_alignments=True,
            top_k=6,
        )

        self.assertEqual(len(result["sentence_alignments"]), 6)
        self.assertAlmostEqual(result["sentence_alignments"][0]["score"], 1.0)

    def test_sentence_coverage_includes_sentences_from_both_documents(self):
        result = self.processor.compare(
            text1="Shared sentence.",
            text2="Shared sentence. Unrelated sentence.",
            normalize=False,
            include_alignments=True,
            top_k=5,
        )

        coverage = result["sentence_coverage"]
        self.assertEqual(len(coverage), 3)
        self.assertEqual(
            sum(item["direction"] == "doc2_to_doc1" for item in coverage),
            2,
        )


if __name__ == "__main__":
    unittest.main()
