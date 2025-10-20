import ray.data as d

# Database libraries
import psycopg2 as pg

# Tables libraries
import pandas as pd

class Reader:

    """
    Base class for a data reader reyling on the Ray Data library.
    """

    def __init__(self):
        """
        Initialize the reader.
        """
        pass

    def read(self, *args, **kwargs) -> d.Dataset:
        """
        Read data and return a Ray Dataset.

        Args:
            *args: Positional arguments for the read method.
            **kwargs: Keyword arguments for the read method.

        Returns:
            d.Dataset: A Ray Dataset containing the read data.

        """

        raise NotImplementedError("Subclasses must implement this method.")

class CSVReader(Reader):

    """
    Reader for CSV files using Ray Data.
    """

    def read(self, file_path: str, **kwargs) -> d.Dataset:

        """
        Read a CSV file and return a Ray Dataset.

        Args:
            file_path (str): Path to the CSV file.
            **kwargs: Additional keyword arguments for the Ray Data read_csv method.
        """

        return d.read_csv(file_path, **kwargs)

class PandasReader(Reader):

    """
    Reader for Pandas DataFrames using Ray Data.
    """

    def read(self, df: pd.DataFrame, **kwargs) -> d.Dataset:

        """
        Read a Pandas DataFrame and return a Ray Dataset.

        Args:
            df (pd.DataFrame): A Pandas DataFrame.
            **kwargs: Additional keyword arguments for the Ray Data from_pandas method.
        """

        return d.from_pandas(df, **kwargs)

class PostgresReader(Reader):

    """
    Reader for PostgreSQL databases using Ray Data.
    """

    def __init__(
        self,
        db_name: str,
        db_user: str,
        db_password: str,
        db_host: str,
        db_port:int,
    ):

        """
        Initialize the PostgresReader with database connection parameters.

        Args:
            db_name (str): Name of the PostgreSQL database.
            db_user (str): Username for the PostgreSQL database.
            db_password (str): Password for the PostgreSQL database.
            db_host (str): Host address of the PostgreSQL database.
            db_port (int): Port number of the PostgreSQL database.
            async (bool): Whether to use asynchronous connection. Default is False.
        """

        self.db_name = db_name
        self.db_user = db_user
        self.db_password = db_password
        self.db_host = db_host
        self.db_port = db_port

    def create_connection(self):
        return pg.connect(
            dbname=self.db_name,
            user=self.db_user,
            password=self.db_password,
            host=self.db_host,
            port=self.db_port,
            )

    def read(self, query: str, **kwargs) -> d.Dataset:
        """
        Reads an SQL query from the PostgreSQL database and returns a Ray Dataset.

        Args:
            query (str): SQL query to execute.
            **kwargs: Additional keyword arguments for the Ray Data read_sql method.

        Returns:
            A Ray Dataset containing the results of the SQL query.
        """
        return d.read_sql(query, self.create_connection, **kwargs)




