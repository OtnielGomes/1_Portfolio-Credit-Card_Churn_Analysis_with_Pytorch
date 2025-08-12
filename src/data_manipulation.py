from pyspark.sql import functions as F
### DataSpark ###
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


### Find Outliers ###
def find_outliers(
    spark,
    df_num
):
    """
    Calculates and displays the percentage of outliers in each column of a PySpark DataFrame.

    This function identifies outliers using the IQR method (Q3 - Q1).
    It displays the percentage of outliers per column.

    Parameters:
    -----------
    df_num : pyspark.sql.DataFrame
        PySpark DataFrame with numeric columns.

    Returns:
    --------
    None
    """
    try:
        
        # List to save data
        out_col, num_outliers = [], []

        # Total size of the dataframe
        size_df = df_num.count()
        if size_df == 0:
            raise ValueError('The DataFrame has no rows.')
        
        for column in df_num.columns:
            try:
                # Calculation of quartiles (may fail if not numeric)
                quantiles = df_num.approxQuantile(column, [0.25, 0.75], 0)
                if not quantiles or len(quantiles) < 2:
                    print(f'[Warning] Could not calculate quantiles for column: {column}.')
                    continue
                
                Q1, Q3 = quantiles # Lower quartile and Upper quartile
                IQR = Q3 - Q1 # Difference between the third quartile and the first quartile
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                # Filter nulls and create temporary column for outliers
                df_filtered = df_num.filter(F.col(column).isNotNull())
                df_filtered = df_filtered.withColumn(
                    f'{column}_out',
                    F.when((F.col(column) < lower_bound) | (F.col(column) > upper_bound), True).otherwise(False)
                )

                # Count outliers
                n_outliers = df_filtered.filter(F.col(f'{column}_out') == True).count()
                percentage_out = round((n_outliers / size_df) * 100, 2)

                # # Stores the data
                out_col.append(column)
                num_outliers.append(percentage_out)
            
            except Exception as inner_e:
                print(f"[Warning] Failed to process column: '{column}': {inner_e}.")

        # Show Results
        if out_col:
            print('\n✅ Percentage of Outliers by Column:')
            percentage_out_data = spark.createDataFrame([tuple(num_outliers)], out_col)
            percentage_out_data.display()
        else:
            print('⚠️ No outliers could be computed.')

    except Exception as e:
        print(f'[Error] Failed to compute outliers: {e}.')