from Utils.postgresql_database_statistic_generator import generate_postgresql_database_statistic

db_params = {
    "dbname": "postgres",
    "user": "novacx0222",
    "password": "",
    "host": "localhost",
    "port": "5432",
}

if __name__ == "__main__":
    generate_postgresql_database_statistic(db_params)
