import io
import runpy
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from cli import lib
from cli.lib import semantic_search


class SemanticSearchTests(unittest.TestCase):
    def setUp(self) -> None:
        model = patch.object(semantic_search, "SentenceTransformer")
        self.mock_model_class = model.start()
        self.addCleanup(model.stop)
        self.model = self.mock_model_class.return_value
        self.model.__str__.return_value = "Mock MiniLM model"
        self.model.max_seq_length = 256

    def run_cli(self, *args: str) -> None:
        cli_path = Path(__file__).resolve().parents[1] / "cli" / "semantic_search_cli.py"
        with (
            patch.dict(
                "sys.modules", {"lib": lib, "lib.semantic_search": semantic_search}
            ),
            patch("sys.argv", [str(cli_path), *args]),
        ):
            runpy.run_path(str(cli_path), run_name="__main__")

    def test_constructor_loads_minilm_and_stores_model(self) -> None:
        search = semantic_search.SemanticSearch()

        self.mock_model_class.assert_called_once_with("all-MiniLM-L6-v2")
        self.assertIs(search.model, self.model)

    def test_verify_model_prints_model_and_max_sequence_length(self) -> None:
        output = io.StringIO()

        with redirect_stdout(output):
            semantic_search.verify_model()

        self.assertEqual(
            output.getvalue(),
            "Model loaded: Mock MiniLM model\nMax sequence length: 256\n",
        )
        self.mock_model_class.assert_called_once_with("all-MiniLM-L6-v2")

    def test_cli_verify_loads_model_and_prints_information(self) -> None:
        output = io.StringIO()

        with redirect_stdout(output):
            self.run_cli("verify")

        self.assertEqual(
            output.getvalue(),
            "Model loaded: Mock MiniLM model\nMax sequence length: 256\n",
        )
        self.mock_model_class.assert_called_once_with("all-MiniLM-L6-v2")

    def test_cli_without_a_command_prints_help_without_loading_model(self) -> None:
        output = io.StringIO()

        with redirect_stdout(output):
            self.run_cli()

        self.assertIn("Semantic Search CLI", output.getvalue())
        self.assertIn("verify", output.getvalue())
        self.mock_model_class.assert_not_called()

    def test_cli_help_does_not_load_model(self) -> None:
        output = io.StringIO()

        with redirect_stdout(output):
            with self.assertRaises(SystemExit) as error:
                self.run_cli("--help")

        self.assertEqual(error.exception.code, 0)
        self.assertIn("verify", output.getvalue())
        self.mock_model_class.assert_not_called()


if __name__ == "__main__":
    unittest.main()
