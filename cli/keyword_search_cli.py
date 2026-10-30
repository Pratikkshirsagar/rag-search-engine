import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            results = []
            print("Searching for:", args.query)
            with open("data/movies.json", "r") as file:
                data = json.load(file)
                movies = data["movies"]

                for movie in movies:
                    title = movie["title"]
                    if args.query in title:
                        results.append(movie)

            for i, movie in enumerate(results[:5], start=1):
                print(f"{i}. {movie['title']}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
