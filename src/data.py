# ======================================================================== #
# >>> Module of functions and classes for loading and saving data.         #                                        
# ======================================================================== #

# ======================================================== #
# Imports:                                                 #
# ======================================================== #
# PySpark.SQL
from pyspark.sql import functions as F
# Pandas
import pandas as pd

# ======================================================== #
# DataSpark - Class                                        #
# ======================================================== #
class DataSpark:
    def __init__(
        self, 
        spark,
        dataframe = None, 
        file_location: str = None,
    ):
        """
        Class for manipulating PySpark DataFrames.

        Args:
        spark(SparkSession): Active Spark session.
        dataframe(DataFrame, optional): Initial PySpark DataFrame.
        file_location(str, optional): File path to save/load data.
        """
        try:
            if spark is None:
                raise ValueError("The 'spark' parameter cannot be None.")
            self.spark = spark

            self.file_location = file_location
            self.dataframe = dataframe

        except Exception as e:
            print(f'[Error] Failed to initialize DataSpark: {e}.')

    # ======================================================== #
    # Save Data - Function                                     #
    # ======================================================== #
    def save_data(
        self,
        file_type: str = 'parquet',
        mode: str = 'overwrite',
        delimiter: str = ',',
        header: bool = True
    ):
        try:
            if self.dataframe is None:
                raise ValueError('No DataFrame loaded to save.')

            writer = self.dataframe.write.mode(mode)

            if file_type == 'parquet':
                writer = writer.format('parquet')
            elif file_type == 'csv':
                writer = (writer.format('csv')
                    .option('delimiter', delimiter)
                    .option('header', str(header))
                    .option('encoding', 'UTF-8')
                    .option('escape', '"')
                    .option('multiline', 'true'))
            else:
                raise ValueError("Format '{file_type}' not supported. Please use 'csv' or 'parquet'.")

            writer.save(self.file_location)
            print(f'✅ Data saved in: {self.file_location}.')

        except Exception as e:
            print(f'[ERROR] Failed to save data: {e}.')

    # ======================================================== #
    # Load Data - Function                                     #
    # ======================================================== #
    def load_data(
        self,
        file_type: str = 'csv',
        infer_schema: bool = True,
        header: bool = True,
        delimiter: str = ',',
        encoding: str = 'UTF-8',
        multiline: bool = True,
        escape: str = '"'
    ):
        try:
            if self.spark is None:
                raise ValueError('SparkSession is not initialized.')

            if file_type == 'csv':
                df = (self.spark.read.format('csv')
                    .option('inferSchema', str(infer_schema))
                    .option('header', str(header))
                    .option('delimiter', delimiter)
                    .option('encoding', encoding)
                    .option('multiline', str(multiline))
                    .option('escape', escape)
                    .load(self.file_location))
            elif file_type == 'parquet':
                df = self.spark.read.format('parquet').load(self.file_location)
            else:
                raise ValueError(f"Format '{file_type}' not supported. Please use 'csv' or 'parquet'.")

            self.dataframe = df
            print(f'✅ File loaded from: {self.file_location}.')
            return self.dataframe

        except FileNotFoundError:
            print(f'[ERROR] File not found at: {self.file_location}.')
        except Exception as e:
            print(f"[ERROR] Error loading file '{self.file_location}': {e}.")
    
    # ======================================================== #
    # Load Data Pandas - Function                              #
    # ======================================================== #
    def load_data_pandas(
        self,
        file_type: str = 'parquet',
    )-> pd.DataFrame:
        """
        Loads a dataset from a specified file location using Spark, converts it to a Pandas DataFrame, 
        and stores it in the class instance.

        Args:
            file_type (str, optional): Type of the file to load. Supported values are:
                - "parquet" (default)
                - "csv"

        Returns:
            pd.DataFrame: The loaded dataset as a Pandas DataFrame.

        Raises:
            ValueError: If the specified file type is not supported.
            Exception: If there is an error while loading or converting the dataset.
        """
        try:

            if file_type == 'parquet':
                    df_spark = self.spark.read.parquet(self.file_location)

            elif file_type == 'csv':
                    df_spark = self.spark.read.csv(self.file_location, header=True, inferSchema=True)

            else:
                    raise ValueError(f'Unsupported file type {file_type}. Use csv or parquet.')

            print(f'✅ File loaded successfully from: {self.file_location}')

            self.dataframe = df_spark.toPandas()

            return self.dataframe

        except Exception as e:
            print(f'[ERROR] Error loading file {self.file_location}: {str(e)}.')
