import argparse

from lib.keyword_search import (
    InvertedIndex,
    bm25_idf_command,
    bm25_tf_command,
    build_command,
    idf_command,
    search_command,
    tf_command,
    tfidf_command,
)
from lib.search_utils import BM25_B, BM25_K1, DEFAULT_SEARCH_LIMIT


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser(
        "search", help="Search movies using the inverted index"
    )
    search_parser.add_argument("query", type=str, help="Search query")
    bm25search_parser = subparsers.add_parser(
        "bm25search", help="Search movies using full BM25 scoring"
    )
    bm25search_parser.add_argument("query", type=str, help="Search query")
    bm25search_parser.add_argument(
        "--limit", type=int, default=DEFAULT_SEARCH_LIMIT, help="Maximum number of results"
    )
    subparsers.add_parser("build", help="Build and save the inverted index")
    tf_parser = subparsers.add_parser("tf", help="Get a term's frequency in a movie")
    tf_parser.add_argument("doc_id", type=int, help="Movie document ID")
    tf_parser.add_argument("term", type=str, help="Single search term")
    idf_parser = subparsers.add_parser(
        "idf", help="Get a term's inverse document frequency"
    )
    idf_parser.add_argument("term", type=str, help="Single search term")
    bm25_idf_parser = subparsers.add_parser(
        "bm25idf", help="Get BM25 IDF score for a given term"
    )
    bm25_idf_parser.add_argument(
        "term", type=str, help="Term to get BM25 IDF score for"
    )
    bm25_tf_parser = subparsers.add_parser(
        "bm25tf", help="Get BM25 TF score for a given document ID and term"
    )
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument(
        "k1", type=float, nargs="?", default=BM25_K1, help="Tunable BM25 K1 parameter"
    )
    bm25_tf_parser.add_argument(
        "b", type=float, nargs="?", default=BM25_B, help="Tunable BM25 b parameter"
    )
    tfidf_parser = subparsers.add_parser(
        "tfidf", help="Get a term's TF-IDF score in a movie"
    )
    tfidf_parser.add_argument("doc_id", type=int, help="Movie document ID")
    tfidf_parser.add_argument("term", type=str, help="Single search term")

    args = parser.parse_args()

    match args.command:
        case "build":
            build_command()
        case "search":
            print("Searching for:", args.query)
            try:
                results = search_command(args.query)
            except FileNotFoundError:
                parser.exit(
                    1,
                    "Error: Search cache is missing. "
                    "Run 'uv run cli/keyword_search_cli.py build' first.\n",
                )
            for i, res in enumerate(results, 1):
                print(f"{i}. {res['title']} (ID: {res['id']})")
        case "bm25search":
            try:
                inverted_index = InvertedIndex()
                inverted_index.load()
                results = inverted_index.bm25_search(args.query, args.limit)
            except FileNotFoundError:
                parser.exit(
                    1,
                    "Error: Search cache is missing. "
                    "Run 'uv run cli/keyword_search_cli.py build' first.\n",
                )
            for i, (doc_id, score) in enumerate(results, 1):
                title = inverted_index.docmap[doc_id]["title"]
                print(f"{i}. ({doc_id}) {title} - Score: {score:.2f}")
        case "tf":
            try:
                frequency = tf_command(args.doc_id, args.term)
            except FileNotFoundError:
                parser.exit(
                    1,
                    "Error: Search cache is missing. "
                    "Run 'uv run cli/keyword_search_cli.py build' first.\n",
                )
            except ValueError as error:
                parser.error(str(error))
            print(frequency)
        case "idf":
            try:
                idf = idf_command(args.term)
            except FileNotFoundError:
                parser.exit(
                    1,
                    "Error: Search cache is missing. "
                    "Run 'uv run cli/keyword_search_cli.py build' first.\n",
                )
            except ValueError as error:
                parser.error(str(error))
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        case "bm25idf":
            try:
                bm25idf = bm25_idf_command(args.term)
            except FileNotFoundError:
                parser.exit(
                    1,
                    "Error: Search cache is missing. "
                    "Run 'uv run cli/keyword_search_cli.py build' first.\n",
                )
            except ValueError as error:
                parser.error(str(error))
            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")
        case "bm25tf":
            try:
                bm25tf = bm25_tf_command(args.doc_id, args.term, args.k1, args.b)
            except FileNotFoundError:
                parser.exit(
                    1,
                    "Error: Search cache is missing. "
                    "Run 'uv run cli/keyword_search_cli.py build' first.\n",
                )
            except ValueError as error:
                parser.error(str(error))
            print(
                f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}"
            )
        case "tfidf":
            try:
                tf_idf = tfidf_command(args.doc_id, args.term)
            except FileNotFoundError:
                parser.exit(
                    1,
                    "Error: Search cache is missing. "
                    "Run 'uv run cli/keyword_search_cli.py build' first.\n",
                )
            except ValueError as error:
                parser.error(str(error))
            print(
                f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}"
            )
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
