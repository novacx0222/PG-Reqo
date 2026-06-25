import argparse
from postgresql_database_statistic_generator import (
    generate_postgresql_database_statistic,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate PostgreSQL database statistics."
    )
    parser.add_argument(
        "--dbname",
        default="postgres",
        help="PostgreSQL database name. Default: postgres.",
    )
    parser.add_argument(
        "--host",
        default="localhost",
        help="PostgreSQL host. Default: localhost.",
    )
    parser.add_argument(
        "--port",
        default="5432",
        help="PostgreSQL port. Default: 5432.",
    )
    parser.add_argument(
        "--user",
        default="novacx0222",
        help="PostgreSQL username. Default: novacx0222.",
    )
    parser.add_argument(
        "--password",
        default="",
        help="PostgreSQL password. Default: empty string.",
    )
    return parser.parse_args()


def build_db_params(args: argparse.Namespace) -> dict[str, str]:
    return {
        "dbname": args.dbname,
        "user": args.user,
        "password": args.password,
        "host": args.host,
        "port": str(args.port),
    }


if __name__ == "__main__":
    db_params = build_db_params(parse_args())
    generate_postgresql_database_statistic(db_params)
