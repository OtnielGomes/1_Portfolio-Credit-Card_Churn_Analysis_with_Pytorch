############################################################################################################
### >>> Module of functions and classes for data manipulation and verification.                          ###
############################################################################################################

# Imports:
# T-Test
from scipy.stats import ttest_ind
# Pandas
import pandas as pd
# Pyspark.SQL
from pyspark.sql import functions as F

############################################################################################################

### Find Outliers Function ###
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

############################################################################################################

### Ttest Function ###
def ttest_between_groups(
    data: pd.DataFrame, 
    numerical_col: str, 
    group_col: str, 
    group1_val:int  = 1, 
    group2_val:int = 0, 
    alpha: float = 0.05,
    print_summary: bool = True
):
    """
    Performs a t-test comparing a numerical variable between two specified groups.

    This function conducts an independent two-sample t-test (Welch’s t-test) 
    on the specified numerical column between two groups defined by values in a grouping column.

    Args:
        data (pd.DataFrame): The dataset containing the variables.
        numerical_col (str): Name of the numeric column to compare.
        group_col (str): Name of the column representing groups.
        group1_val (any): Value in `group_col` representing the first group.
        group2_val (any): Value in `group_col` representing the second group.
        alpha (float, optional): Significance level for hypothesis testing. Default is 0.05.
        print_summary (bool, optional): Whether to print the test summary. Default is True.

    Returns:
        tuple:
            t_stat (float): The computed t-statistic.
            p_value (float): The p-value of the test.

    Raises:
        Exception: If an error occurs during the test execution.
    """
    try:
        group1 = data[data[group_col] == group1_val][numerical_col]
        group2 = data[data[group_col] == group2_val][numerical_col]

        t_stat, p_value = ttest_ind(group1, group2, equal_var = False)

        if print_summary:
            print(f'\n🟢 t-statistic: {t_stat:.5f}')
            print(f'🔵 p-value: {p_value:.5f}')
            print('-------' * 10)
            if p_value <= alpha:
                print(f'\n✅ Null Hypothesis (H0) Rejected!')
                print(f"There is a significant difference in '{numerical_col}'") 
                print(f'between the two groups ({group1_val} vs {group2_val}).')
            else:
                print(f'\n⛔ Null Hypothesis (H0) Accepted!') 
                print(f"There is no significant difference in '{numerical_col}'") 
                print(f'between the two groups ({group1_val} vs {group2_val})')

    except Exception as e:
        print(f'[ERROR] Failed to perform t-test: {str(e)}')
