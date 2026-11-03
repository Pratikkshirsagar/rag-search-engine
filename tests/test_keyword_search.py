import io
import math
import pickle
import runpy
import tempfile
import unittest
from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from cli import lib
from cli.lib import keyword_search


class InvertedIndexTests(unittest.TestCase):
    def setUp(self) -> None:
        self.movies = [
            {"id": 10, "title": "The Matrix", "description": "Hackers are running."},
            {
                "id": 3,
                "title": "Runner",
                "description": "A matrix, matrix and reality.",
            },
            {"id": 8, "title": "Reality", "description": "A runner meets a hacker."},
        ]
        stopwords = patch.object(
            keyword_search, "load_stopwords", return_value=["the", "are", "a", "and"]
        )
        movies = patch.object(keyword_search, "load_movies", return_value=self.movies)
        self.mock_stopwords = stopwords.start()
        self.mock_movies = movies.start()
        self.addCleanup(stopwords.stop)
        self.addCleanup(movies.stop)
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.cache_dir = Path(self.tempdir.name) / "cache"
        cache = patch.object(keyword_search, "CACHE_DIR", str(self.cache_dir))
        cache.start()
        self.addCleanup(cache.stop)

    def build_index(self) -> keyword_search.InvertedIndex:
        index = keyword_search.InvertedIndex()
        index.build()
        return index

    def run_cli(self, *args: str) -> None:
        cli_path = Path(__file__).resolve().parents[1] / "cli" / "keyword_search_cli.py"
        with (
            patch.dict(
                "sys.modules", {"lib": lib, "lib.keyword_search": keyword_search}
            ),
            patch("sys.argv", [str(cli_path), *args]),
        ):
            runpy.run_path(str(cli_path), run_name="__main__")

    def test_build_indexes_titles_and_descriptions_with_existing_ids(self) -> None:
        index = self.build_index()

        self.assertEqual(index.index["matrix"], {3, 10})
        self.assertEqual(index.index["hacker"], {8, 10})
        self.assertEqual(index.index["run"], {10})
        self.assertEqual(index.docmap, {movie["id"]: movie for movie in self.movies})
        self.assertNotIn("the", index.index)
        self.assertNotIn("and", index.index)
        self.assertNotIn("running", index.index)

    def test_build_counts_each_preprocessed_token_per_document(self) -> None:
        index = self.build_index()

        self.assertIsInstance(index.term_frequencies[3], Counter)
        self.assertEqual(
            index.term_frequencies[3], Counter({"matrix": 2, "runner": 1, "realiti": 1})
        )
        self.assertEqual(index.get_tf(10, "matrix"), 1)
        self.assertEqual(index.get_tf(10, "run"), 1)
        self.assertNotIn("and", index.term_frequencies[3])

    def test_tf_combines_stemmed_occurrences_in_title_and_description(self) -> None:
        self.mock_movies.return_value = [
            {"id": 1, "title": "Running", "description": "Runs running run."}
        ]

        index = self.build_index()

        self.assertEqual(index.get_tf(1, "run"), 4)

    def test_get_tf_returns_zero_for_missing_terms_and_documents(self) -> None:
        index = self.build_index()
        original_frequencies = index.term_frequencies.copy()

        self.assertEqual(index.get_tf(3, "missing"), 0)
        self.assertEqual(index.get_tf(999, "matrix"), 0)
        self.assertNotIn("missing", index.term_frequencies[3])
        self.assertNotIn(999, index.term_frequencies)
        self.assertEqual(index.term_frequencies, original_frequencies)

    def test_tokenize_term_normalizes_and_stems_a_single_token(self) -> None:
        stopwords = keyword_search.InvertedIndex().stopwords

        for term, expected in [
            ("MATRIX!", "matrix"),
            ("Hackers", "hacker"),
            ("running", "run"),
            ("matrix and", "matrix"),
        ]:
            with self.subTest(term=term):
                self.assertEqual(keyword_search.tokenize_term(term, stopwords), expected)

    def test_tokenize_term_rejects_zero_or_multiple_tokens(self) -> None:
        stopwords = keyword_search.InvertedIndex().stopwords

        for term in ["", "!!!", "the", "matrix hacker"]:
            with self.subTest(term=term):
                with self.assertRaisesRegex(ValueError, "exactly one token"):
                    keyword_search.tokenize_term(term, stopwords)

    def test_get_documents_returns_sorted_unique_ids_without_mutating_index(
        self,
    ) -> None:
        index = self.build_index()

        self.assertEqual(index.get_documents("matrix"), [3, 10])
        self.assertEqual(index.get_documents("missing"), [])
        self.assertNotIn("missing", index.index)
        self.assertEqual(index.get_documents("Matrix"), [])

    def test_build_clears_stale_documents(self) -> None:
        index = self.build_index()
        self.mock_movies.return_value = [self.movies[0]]

        index.build()

        self.assertEqual(index.get_documents("matrix"), [10])
        self.assertNotIn("realiti", index.index)
        self.assertEqual(list(index.docmap), [10])
        self.assertEqual(list(index.term_frequencies), [10])
        self.assertEqual(index.get_tf(10, "matrix"), 1)

    def test_save_creates_cache_directory_and_pickles_both_attributes(self) -> None:
        index = self.build_index()
        self.assertFalse(self.cache_dir.exists())

        index.save()

        with (self.cache_dir / "index.pkl").open("rb") as file:
            self.assertEqual(pickle.load(file), index.index)
        with (self.cache_dir / "docmap.pkl").open("rb") as file:
            self.assertEqual(pickle.load(file), index.docmap)
        with (self.cache_dir / "term_frequencies.pkl").open("rb") as file:
            self.assertEqual(pickle.load(file), index.term_frequencies)

    def test_load_restores_saved_index(self) -> None:
        original = self.build_index()
        original.save()
        restored = keyword_search.InvertedIndex()

        restored.load()

        self.assertEqual(restored.index, original.index)
        self.assertEqual(restored.docmap, original.docmap)
        self.assertEqual(restored.term_frequencies, original.term_frequencies)
        self.assertIsInstance(restored.term_frequencies[3], Counter)

    def test_load_raises_when_cache_is_missing(self) -> None:
        index = keyword_search.InvertedIndex()

        with self.assertRaises(FileNotFoundError):
            index.load()

    def test_load_raises_if_any_cache_file_is_missing(self) -> None:
        index = self.build_index()
        for filename in ["index.pkl", "docmap.pkl", "term_frequencies.pkl"]:
            with self.subTest(filename=filename):
                index.save()
                (self.cache_dir / filename).unlink()

                with self.assertRaises(FileNotFoundError):
                    keyword_search.InvertedIndex().load()

    def test_build_command_saves_index_without_hardcoded_lookup(self) -> None:
        output = io.StringIO()

        with redirect_stdout(output):
            keyword_search.build_command()

        self.assertEqual(output.getvalue(), "Index built and saved.\n")
        self.assertTrue((self.cache_dir / "index.pkl").is_file())
        self.assertTrue((self.cache_dir / "docmap.pkl").is_file())
        self.assertTrue((self.cache_dir / "term_frequencies.pkl").is_file())

    def test_search_uses_saved_index_without_loading_movies(self) -> None:
        self.build_index().save()
        self.mock_movies.reset_mock()

        results = keyword_search.search_command("MATRIX! hackers")

        self.assertEqual([movie["id"] for movie in results], [3, 10, 8])
        self.mock_movies.assert_not_called()

    def test_search_raises_without_rebuilding_if_cache_is_missing(self) -> None:
        with self.assertRaises(FileNotFoundError):
            keyword_search.search_command("running")

        self.mock_movies.assert_not_called()
        self.assertFalse(self.cache_dir.exists())

    def test_search_raises_without_rebuilding_if_docmap_is_missing(self) -> None:
        self.build_index().save()
        self.mock_movies.reset_mock()
        (self.cache_dir / "docmap.pkl").unlink()

        with self.assertRaises(FileNotFoundError):
            keyword_search.search_command("matrix")

        self.mock_movies.assert_not_called()
        self.assertFalse((self.cache_dir / "docmap.pkl").exists())

    def test_search_matches_whole_tokens_not_substrings(self) -> None:
        self.build_index().save()

        self.assertEqual(keyword_search.search_command("hack"), [])

    def test_search_deduplicates_results_and_respects_limit(self) -> None:
        self.build_index().save()

        results = keyword_search.search_command("matrix matrix reality", limit=2)

        self.assertEqual([movie["id"] for movie in results], [3, 10])

    def test_search_stops_before_next_token_once_five_results_are_found(self) -> None:
        self.mock_movies.return_value = [
            {"id": doc_id, "title": f"Movie {doc_id}", "description": "Matrix"}
            for doc_id in range(8, 0, -1)
        ]
        self.build_index().save()
        original_get_documents = keyword_search.InvertedIndex.get_documents

        with patch.object(
            keyword_search.InvertedIndex,
            "get_documents",
            autospec=True,
            side_effect=original_get_documents,
        ) as get_documents:
            results = keyword_search.search_command("matrix hacker")

        self.assertEqual([movie["id"] for movie in results], [1, 2, 3, 4, 5])
        get_documents.assert_called_once()
        self.assertEqual(get_documents.call_args.args[1], "matrix")

    def test_search_does_not_return_duplicate_documents_across_tokens(self) -> None:
        self.build_index().save()

        results = keyword_search.search_command("matrix matrix reality")

        self.assertEqual([movie["id"] for movie in results], [3, 10, 8])

    def test_search_empty_unknown_and_stopword_only_queries(self) -> None:
        self.build_index().save()

        for query in ["", "!!!", "the and", "unknown"]:
            with self.subTest(query=query):
                self.assertEqual(keyword_search.search_command(query), [])

    def test_search_nonpositive_limits_do_not_build_index(self) -> None:
        for limit in [0, -1]:
            with self.subTest(limit=limit):
                self.assertEqual(keyword_search.search_command("matrix", limit), [])
        self.mock_movies.assert_not_called()

    def test_tf_command_uses_saved_counts_and_tokenizes_the_term(self) -> None:
        self.build_index().save()
        self.mock_movies.reset_mock()

        self.assertEqual(keyword_search.tf_command(3, "MATRIX!"), 2)
        self.assertEqual(keyword_search.tf_command(10, "RUNNING"), 1)
        self.assertEqual(keyword_search.tf_command(3, "unknown"), 0)
        self.assertEqual(keyword_search.tf_command(999, "matrix"), 0)
        self.mock_movies.assert_not_called()

    def test_idf_command_counts_documents_not_repeated_occurrences(self) -> None:
        self.build_index().save()
        self.mock_movies.reset_mock()

        self.assertAlmostEqual(keyword_search.idf_command("matrix"), math.log(4 / 3))
        self.mock_movies.assert_not_called()

    def test_idf_command_normalizes_and_stems_the_term(self) -> None:
        self.build_index().save()

        self.assertAlmostEqual(keyword_search.idf_command("HACKERS!"), math.log(4 / 3))
        self.assertAlmostEqual(keyword_search.idf_command("RUNNING"), math.log(4 / 2))

    def test_idf_command_returns_zero_when_every_document_matches(self) -> None:
        for movie in self.movies:
            movie["description"] += " Galaxy."
        self.build_index().save()

        self.assertEqual(keyword_search.idf_command("galaxy"), 0.0)

    def test_idf_command_smooths_terms_missing_from_all_documents(self) -> None:
        self.build_index().save()

        self.assertAlmostEqual(keyword_search.idf_command("xyzzy"), math.log(4))

    def test_idf_command_handles_an_empty_corpus(self) -> None:
        self.mock_movies.return_value = []
        self.build_index().save()

        self.assertEqual(keyword_search.idf_command("matrix"), 0.0)

    def test_tfidf_command_multiplies_tf_and_idf_and_loads_cache_once(self) -> None:
        self.build_index().save()
        self.mock_movies.reset_mock()
        original_load = keyword_search.InvertedIndex.load

        with patch.object(
            keyword_search.InvertedIndex,
            "load",
            autospec=True,
            side_effect=original_load,
        ) as load:
            score = keyword_search.tfidf_command(3, "matrix")

        self.assertAlmostEqual(score, 2 * math.log(4 / 3))
        load.assert_called_once()
        self.mock_movies.assert_not_called()

    def test_tfidf_command_normalizes_and_stems_the_term(self) -> None:
        self.build_index().save()

        self.assertAlmostEqual(
            keyword_search.tfidf_command(3, "MATRIX!"), 2 * math.log(4 / 3)
        )
        self.assertAlmostEqual(keyword_search.tfidf_command(10, "RUNNING"), math.log(2))

    def test_tfidf_command_returns_zero_for_missing_terms_and_documents(self) -> None:
        self.build_index().save()

        for doc_id, term in [(3, "hacker"), (3, "xyzzy"), (999, "matrix")]:
            with self.subTest(doc_id=doc_id, term=term):
                self.assertEqual(keyword_search.tfidf_command(doc_id, term), 0.0)

    def test_tfidf_command_returns_zero_for_a_universal_term(self) -> None:
        for movie in self.movies:
            movie["description"] += " Galaxy galaxy."
        self.build_index().save()

        self.assertEqual(keyword_search.tfidf_command(3, "galaxy"), 0.0)

    def test_tfidf_command_handles_an_empty_corpus(self) -> None:
        self.mock_movies.return_value = []
        self.build_index().save()

        self.assertEqual(keyword_search.tfidf_command(3, "matrix"), 0.0)

    def test_cli_tfidf_prints_original_arguments_and_two_decimal_places(self) -> None:
        self.build_index().save()

        for doc_id, term, expected in [
            ("3", "MATRIX!", "0.58"),
            ("10", "running", "0.69"),
            ("3", "hacker", "0.00"),
            ("3", "xyzzy", "0.00"),
            ("999", "matrix", "0.00"),
        ]:
            with self.subTest(doc_id=doc_id, term=term):
                output = io.StringIO()
                with redirect_stdout(output):
                    self.run_cli("tfidf", doc_id, term)
                self.assertEqual(
                    output.getvalue(),
                    f"TF-IDF score of '{term}' in document '{doc_id}': {expected}\n",
                )

    def test_cli_tfidf_rejects_terms_that_do_not_produce_one_token(self) -> None:
        self.build_index().save()

        for term in ["", "!!!", "the", "matrix hacker"]:
            with self.subTest(term=term):
                error_output = io.StringIO()
                with redirect_stderr(error_output):
                    with self.assertRaises(SystemExit) as error:
                        self.run_cli("tfidf", "3", term)
                self.assertEqual(error.exception.code, 2)
                self.assertIn("exactly one token", error_output.getvalue())

    def test_cli_tfidf_reports_missing_cache_without_rebuilding(self) -> None:
        error_output = io.StringIO()

        with redirect_stderr(error_output):
            with self.assertRaises(SystemExit) as error:
                self.run_cli("tfidf", "3", "matrix")

        self.assertEqual(error.exception.code, 1)
        self.assertIn("uv run cli/keyword_search_cli.py build", error_output.getvalue())
        self.mock_movies.assert_not_called()
        self.assertFalse(self.cache_dir.exists())

    def test_cli_idf_prints_original_term_and_two_decimal_places(self) -> None:
        self.build_index().save()

        for term, expected in [
            ("MATRIX!", "0.29"),
            ("running", "0.69"),
            ("xyzzy", "1.39"),
        ]:
            with self.subTest(term=term):
                output = io.StringIO()
                with redirect_stdout(output):
                    self.run_cli("idf", term)
                self.assertEqual(
                    output.getvalue(),
                    f"Inverse document frequency of '{term}': {expected}\n",
                )

    def test_cli_idf_rejects_terms_that_do_not_produce_one_token(self) -> None:
        self.build_index().save()

        for term in ["", "!!!", "the", "matrix hacker"]:
            with self.subTest(term=term):
                error_output = io.StringIO()
                with redirect_stderr(error_output):
                    with self.assertRaises(SystemExit) as error:
                        self.run_cli("idf", term)
                self.assertEqual(error.exception.code, 2)
                self.assertIn("exactly one token", error_output.getvalue())

    def test_cli_idf_reports_missing_cache_without_rebuilding(self) -> None:
        error_output = io.StringIO()

        with redirect_stderr(error_output):
            with self.assertRaises(SystemExit) as error:
                self.run_cli("idf", "matrix")

        self.assertEqual(error.exception.code, 1)
        self.assertIn("uv run cli/keyword_search_cli.py build", error_output.getvalue())
        self.mock_movies.assert_not_called()
        self.assertFalse(self.cache_dir.exists())

    def test_cli_tf_prints_only_the_frequency(self) -> None:
        self.build_index().save()

        for doc_id, term, expected in [("3", "MATRIX!", "2\n"), ("3", "unknown", "0\n")]:
            with self.subTest(term=term):
                output = io.StringIO()
                with redirect_stdout(output):
                    self.run_cli("tf", doc_id, term)
                self.assertEqual(output.getvalue(), expected)

    def test_cli_tf_rejects_terms_that_do_not_produce_one_token(self) -> None:
        self.build_index().save()

        for term in ["!!!", "the", "matrix hacker"]:
            with self.subTest(term=term):
                error_output = io.StringIO()
                with redirect_stderr(error_output):
                    with self.assertRaises(SystemExit) as error:
                        self.run_cli("tf", "3", term)
                self.assertEqual(error.exception.code, 2)
                self.assertIn("exactly one token", error_output.getvalue())

    def test_cli_tf_reports_missing_term_frequency_cache(self) -> None:
        self.build_index().save()
        self.mock_movies.reset_mock()
        (self.cache_dir / "term_frequencies.pkl").unlink()
        error_output = io.StringIO()

        with redirect_stderr(error_output):
            with self.assertRaises(SystemExit) as error:
                self.run_cli("tf", "3", "matrix")

        self.assertEqual(error.exception.code, 1)
        self.assertIn("uv run cli/keyword_search_cli.py build", error_output.getvalue())
        self.mock_movies.assert_not_called()

    def test_cli_prints_document_title_and_id(self) -> None:
        self.build_index().save()
        output = io.StringIO()

        with redirect_stdout(output):
            self.run_cli("search", "running")

        self.assertIn("1. The Matrix (ID: 10)", output.getvalue())

    def test_cli_reports_missing_cache_and_exits_without_rebuilding(self) -> None:
        error_output = io.StringIO()

        with redirect_stdout(io.StringIO()), redirect_stderr(error_output):
            with self.assertRaises(SystemExit) as error:
                self.run_cli("search", "matrix")

        self.assertEqual(error.exception.code, 1)
        self.assertIn("Error: Search cache is missing.", error_output.getvalue())
        self.assertIn("uv run cli/keyword_search_cli.py build", error_output.getvalue())
        self.mock_movies.assert_not_called()
        self.assertFalse(self.cache_dir.exists())


if __name__ == "__main__":
    unittest.main()
