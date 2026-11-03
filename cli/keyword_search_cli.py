import argparse

from lib.keyword_search import (
    build_command,
    idf_command,
    search_command,
    tf_command,
    tfidf_command,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser(
        "search", help="Search movies using the inverted index"
    )
    search_parser.add_argument("query", type=str, help="Search query")
    subparsers.add_parser("build", help="Build and save the inverted index")
    tf_parser = subparsers.add_parser("tf", help="Get a term's frequency in a movie")
    tf_parser.add_argument("doc_id", type=int, help="Movie document ID")
    tf_parser.add_argument("term", type=str, help="Single search term")
    idf_parser = subparsers.add_parser(
        "idf", help="Get a term's inverse document frequency"
    )
    idf_parser.add_argument("term", type=str, help="Single search term")
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
