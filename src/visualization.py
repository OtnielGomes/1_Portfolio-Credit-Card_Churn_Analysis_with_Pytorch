# Imports:
# Matplotlib
import matplotlib.pyplot as plt
# Seaborn
import seaborn as sns
# Pandas
import pandas as pd

class GraphicsData:

    # Init Attributes
    def __init__(
        self, 
        data: pd.DataFrame,
        ):

        try:
            # Entry checks
            if data.empty:
                raise ValueError('The provided DataFrame is empty.')

            self.data = data

        except Exception  as e:
            print(f'[Error] Failed to load Dataframe : {str(e)}')
    

    ### _initializer_subplot_grid Function ###
    def _initializer_subplot_grid(
        self, 
        num_columns, 
        figsize_per_row
    ):
        """
        Initializes and returns a standardized matplotlib subplot grid layout.

        This utility method calculates the required number of rows based on 
        the number of variables in the dataset and the desired number of 
        columns per row. It then creates a grid of subplots accordingly and 
        applies a consistent styling.

        Args:
            num_columns (int): Number of subplots per row.
            figsize_per_row (int): Vertical size (height) per row in the final figure.

        Returns:
            tuple:
                - fig (matplotlib.figure.Figure): The full matplotlib figure object.
                - ax (np.ndarray of matplotlib.axes._subplots.AxesSubplot): Flattened array of subplot axes.
        """
        num_vars = len(self.data.columns)
        num_rows = (num_vars + num_columns - 1) // num_columns

        plt.rc('font', size = 12)
        fig, ax = plt.subplots(num_rows, num_columns, figsize = (30, num_rows * figsize_per_row))
        ax = ax.flatten()
        sns.set(style = 'whitegrid')

        return fig, ax

    ###_finalize_subplot_layout Function ###
    def _finalize_subplot_layout(
        self,
        fig,
        ax,
        i: int,
        title: str = None,
        fontsize: int = 30,
    ):
        """
        Finalizes and displays a matplotlib figure by adjusting layout and removing unused subplots.

        This method is used after plotting multiple subplots to:
        - Remove any unused axes in the grid.
        - Set a central title for the entire figure.
        - Automatically adjust spacing and layout for better readability.
        - Display the resulting plot.

        Args:
            fig (matplotlib.figure.Figure): The matplotlib figure object containing the subplots.
            ax (np.ndarray of matplotlib.axes.Axes): Array of axes (flattened) for all subplots.
            i (int): Index of the last used subplot (all subplots after this will be removed).
            title (str, optional): Title to be displayed at the top of the entire figure.
            fontsize (int, optional): Font size of the overall title. Default is 30.
        """
        for j in range(i + 1, len(ax)):
                fig.delaxes(ax[j])
        
        plt.suptitle(title, fontsize = fontsize, fontweight = 'bold')
        plt.tight_layout(rect = [0, 0, 1, 0.97])
        plt.show()
    
    ### _format_single_ax Function ###
    def _format_single_ax(
        self, 
        ax,
        title: str = None,
        fontsize: int = 20,
        linewidth: float = 0.9
    ):

        """
        Applies standard formatting to a single subplot axis.

        This method configures a single axis by:
        - Setting the title with specified font size and bold style.
        - Hiding the x and y axis labels.
        - Adding dashed grid lines for both axes with configurable line width.

        Args:
            ax (matplotlib.axes.Axes): The axis to be formatted.
            title (str, optional): Title text for the axis. Defaults to None.
            fontsize (int, optional): Font size for the title. Defaults to 20.
            linewidth (float, optional): Width of the dashed grid lines. Defaults to 0.9.
        """
        ax.set_title(title, fontsize = fontsize, fontweight = 'bold')
        ax.set_xlabel(None)
        ax.set_ylabel(None)
        ax.grid(axis = 'y', which = 'major', linestyle = '--', linewidth = linewidth)
        ax.grid(axis = 'x', which = 'major', linestyle = '--', linewidth = linewidth)

    ### Plot Variable Type Function ###
    def plot_variable_type(
        self,
        count_col: str,
        label_col: str, 
        title = 'Distribution of Variable Types'
    ):
        
        """
        Plots a pie chart to display the proportion of each variable type in the dataset.

        This method uses a pie chart to visualize the distribution of different types of variables
        (e.g., categorical, numerical) based on the values provided in `count_col` and `label_col`.

        Args:
            count_col (str): Name of the column containing the counts for each variable type.
            label_col (str): Name of the column containing the labels/categories of variable types.
            title (str, optional): Title of the pie chart. Defaults to 'Distribution of Variable Types'.

        Raises:
            ValueError: If `count_col` or `label_col` is not found in the DataFrame.
            Exception: For any other error that occurs during plotting.
        """

        try:
            # Entry checks
            if count_col not in self.data.columns:
                raise ValueError(f"Column '{count_col}' does not exist in the DataFrame.")

            if label_col not in self.data.columns:
                raise ValueError(f"Column '{label_col}' does not exist in the DataFrame.")

            # Define AX and Fig
            plt.rc('font', size = 14)
            fig, ax = plt.subplots(figsize = (7, 7))

            ax.pie(
                self.data[count_col],
                labels = self.data[label_col],
                colors = sns.color_palette('Set3', len(self.data)),
                autopct = '%1.1f%%',
                startangle = 120,
                explode=[0.05 if i >= len(self.data) - 2 else 0 for i in range(len(self.data))],
                shadow = False,
            )
            # Config Ax's and Show Graphics
            ax.set_title(title, fontsize = 15, fontweight='bold')
            plt.tight_layout()
            plt.show()
        except Exception  as e:
            print(f'[Error] Failed to generate variable distribution plot: {str(e)}.')

    ### Numerical histograms Function ###
    def numerical_histograms(
        self, 
        num_columns: int = 3,
        figsize_per_row: int = 6,
        color: str = '#a2bffe',
        hue: str = None,
        palette: list = ['#b0ff9d', '#db5856'],
        title: str = 'Histograms of Numerical Variables',
    ):
        """
        Plots histograms with KDE (Kernel Density Estimation) for all numerical columns in the dataset.

        Optionally groups the histograms by a categorical target variable using different colors (hue).
        Useful for visualizing the distribution of numerical features and how they differ between groups.

        Args:
            num_columns (int): Number of plots per row in the subplot grid.
            figsize_per_row (int): Height of each row in inches (controls vertical spacing).
            color (str): Default color for histograms when `hue` is not specified.
            hue (str, optional): Name of the column used for grouping (e.g., 'churn_target'). Must be categorical.
            palette (list): List of colors for hue levels. Only used if `hue` is provided.
            title (str): Title of the entire figure layout.

        Raises:
            Exception: If plotting fails due to missing columns, incorrect types, or rendering errors.
        """
        try:
            # Entry checks
            numeric_cols = self.data.select_dtypes(include = 'number').columns.tolist()
            if hue and hue in numeric_cols:
                numeric_cols.remove(hue)

            # Define AX and Fig
            fig, ax = self._initializer_subplot_grid(num_columns, figsize_per_row)

            for i, column in enumerate(numeric_cols):
                sns.histplot(
                    data = self.data,
                    x = column,
                    kde = True,
                    hue = hue,
                    palette = palette if hue else None,
                    edgecolor = 'black',
                    alpha = 0.4 if hue else 0.7,
                    color = None if hue else color,
                    ax = ax[i],
                )
                # Config Ax's
                self._format_single_ax(ax[i], title = f'Histogram of variable: {column}.')
                
            # Show Graphics
            self._finalize_subplot_layout(fig, ax, i, title = title)
        except Exception as e:
            print(f'[Error] Failed to generate numeric histograms: {str(e)}.')

    ### Numerical Boxplots Function ###
    def numerical_boxplots(
        self, 
        hue: str = None, 
        num_columns: int = 3,
        figsize_per_row: int = 6,
        palette: list = ['#b0ff9d', '#db5856'],
        color: str = '#a2bffe',
        showfliers: bool = False,
        title: str = 'Boxplots of Numerical Variables',
        legend: list = []
    ):
        """
        Plots boxplots for each numerical variable in the dataset.

        Optionally groups the boxplots by a categorical hue variable (e.g., churn target), 
        allowing for comparison of distributions between groups. Helps identify outliers, 
        skewness, and variability in each feature.

        Args:
            hue (str, optional): Column name to group the boxplots (e.g., 'churn_target').
                                If None, individual boxplots are created without grouping.
            num_columns (int): Number of plots per row in the subplot grid.
            figsize_per_row (int): Height (in inches) of each row of plots.
            palette (list): Color palette to use when `hue` is provided.
            color (str): Single color to use when `hue` is not specified.
            showfliers (bool): Whether to display outlier points in the boxplots (default: False).
            title (str): Overall title for the subplot grid.
            legend (list): Custom legend labels to replace default tick labels when `hue` is present.

        Raises:
            ValueError: If the hue column is not found in the DataFrame.
            Exception: If plotting fails due to unexpected issues.
        """
        try:
            # Entry checks
            if hue and hue not in self.data.columns:
                raise ValueError(f"Column '{hue}' not in the DataFrame.")

            numeric_cols = self.data.select_dtypes(include = 'number').columns.tolist()
            if hue and hue in numeric_cols:
                numeric_cols.remove(hue)

            # Define AX and Fig
            fig, ax = self._initializer_subplot_grid(num_columns, figsize_per_row)

            for i, column in enumerate(numeric_cols):
                    sns.boxplot(
                        data = self.data,
                        x = hue if hue else column,
                        y = column if hue else None,
                        hue = hue if hue else None,
                        palette = palette if hue else None,
                        color = None if hue else color,
                        showfliers = showfliers,
                        #legend = False,
                        ax = ax[i]
                    )

                    # Config Ax's
                    if len(legend) > 0:
                        ax[i].set_xticks([l for l in range(0, len(legend))])
                        ax[i].set_xticklabels(legend, fontsize = 16, fontweight = 'bold')
                    
                    if ax[i].get_legend():
                        ax[i].legend_.remove()

                    self._format_single_ax(ax[i], f'Box plot of variable: {column}')
                    ax[i].set_yticklabels([])
                    sns.despine(ax = ax[i], top = True, right = True, left = True, bottom = True)
            
            # Show Graphics
            self._finalize_subplot_layout(fig, ax, i, title = title)
        except Exception as e: 
            print(f'[ERROR] Failed to generate numerical boxplots: {str(e)}.')

    ### Categorical Countplots Function ###
    def categorical_countplots(
        self,
        hue: str = None,
        num_columns: int = 2,
        figsize_per_row: int = 7,
        palette: list = ['#b0ff9d', '#db5856'],
        color: str = '#a2bffe',
        title: str = 'Countplots of Categorical Variables '
    ):
        """
        Plots countplots for all categorical variables in the dataset.

        Optionally groups the bars using a hue column (e.g., 'churn_target'), allowing 
        visual comparison of class distributions between different categories. Annotates
        each bar with its percentage frequency.

        Args:
            hue (str, optional): Name of the column used to group bars (e.g., target variable).
                                If None, no grouping is applied.
            num_columns (int): Number of plots per row in the subplot grid.
            figsize_per_row (int): Height (in inches) of each subplot row.
            palette (list): List of colors to use when `hue` is specified.
            color (str): Default color to use when `hue` is not provided.
            title (str): General title for the entire plot grid.

        Raises:
            ValueError: If the hue column is not found in the DataFrame.
            Exception: If the plot generation fails for unexpected reasons.
        """
        try:
            # Entry checks
            if hue and hue not in self.data.columns:
                raise ValueError(f"Column '{hue}' not found in the DataFrame.")

            categorical_cols = self.data.select_dtypes(include = ['object', 'category']).columns.tolist()
            if hue and hue in categorical_cols:
                categorical_cols.remove(hue)
            
            # Config Ax's
            fig, ax = self._initializer_subplot_grid(num_columns, figsize_per_row)

            for i, column in enumerate(categorical_cols):
                sns.countplot(
                    data = self.data,
                    x = column,
                    hue = hue if hue else None,
                    palette = palette if hue else None,
                    color = None if hue else color,
                    edgecolor = 'white' if hue else 'black',
                    saturation = 1,
                    legend = False,
                    ax = ax[i]
                )
                
                total = len(self.data[column])
                for p in ax[i].patches:
                    height = p.get_height()
                    if height == 0:
                        continue
                    percentage = f'{100 * height / total:.1f}%'
                    x = p.get_x() + p.get_width() / 1.95
                    y = height
                    ax[i].annotate(
                        percentage,
                        (x, y),
                        ha = 'center',
                        va = 'bottom',
                        fontsize = 16,
                        color = 'black'
                    )

                # Config Ax's
                self._format_single_ax(ax[i], f'Countplot of variable: {column}')
                ax[i].set_xticks(range(len(ax[i].get_xticklabels())))
                ax[i].set_xticklabels(ax[i].get_xticklabels(), fontsize = 16)
                
            # Show Graphics
            self._finalize_subplot_layout(fig, ax, i, title = title)
        except Exception as e:
            print(f'[ERROR] Failed to generate categorical countplots: {str(e)}')

    ### Numerical Barplots Function ###
    def numerical_barplots(
        self,
        hue: str = None,
        num_columns: int = 3,
        figsize_per_row: int = 6,
        palette: list = ['#b0ff9d', '#db5856'],
        errorbar = ('ci', 90),
        title: str = 'Barplots of Numerical Variables',
        legend: list = []
    ):
        """
        Plots barplots for each numerical variable, optionally grouped by a hue variable.

        This method creates barplots to visualize the mean (or other estimator) of numerical
        variables in the dataset. It supports grouping by a categorical variable (`hue`)
        and displays error bars (e.g., confidence intervals).

        Args:
            hue (str, optional): Column name to group the barplots (e.g., 'churn_target').
                If None, no grouping is applied. Defaults to None.
            num_columns (int): Number of subplots per row in the grid.
            figsize_per_row (int): Height (in inches) allocated per row of subplots.
            palette (list, optional): List of colors to use when `hue` is specified.
                Defaults to ['#b0ff9d', '#db5856'].
            errorbar (tuple or str, optional): Error bar representation passed to seaborn.barplot.
                Defaults to ('ci', 90) for 90% confidence intervals.
            title (str): Overall title for the figure.
            legend (list, optional): Custom labels to replace default hue legend labels.
                Defaults to an empty list.

        Raises:
            ValueError: If the `hue` column is specified but not found in the DataFrame.
            Exception: For other errors during plotting.
        """
        try:
            # Entry checks
            if hue and hue not in self.data.columns:
                raise ValueError(f"Column '{hue}' not found in the DataFrame.")

            numeric_cols = self.data.select_dtypes(include = 'number').columns.tolist()
            if hue and hue in numeric_cols:
                numeric_cols.remove(hue)
            
            # Define AX and Fig
            fig, ax = self._initializer_subplot_grid(num_columns, figsize_per_row)

            for i, column in enumerate(numeric_cols):
                sns.barplot(
                    data = self.data,
                    x = hue,
                    y = column,
                    hue = hue,
                    errorbar = errorbar,
                    dodge = False,
                    palette = palette,
                    edgecolor = 'white',
                    legend = False,
                    ax = ax[i]
                )

                # Config Ax's
                if len(legend) > 1:
                    ax[i].set_xticks(list(range(len(legend))))
                    ax[i].set_xticklabels(legend, fontsize = 16, fontweight = 'bold')

                self._format_single_ax(ax[i], f'Barplot of variable: {column}')
                ax[i].set_yticklabels([])
                sns.despine(ax = ax[i], top = True, right = True, left = True, bottom = True)
            
            # Show Graphics
            self._finalize_subplot_layout(fig, ax, i, title = title)
        except Exception as e:
            print(f'[ERROR] Failed to generate numerical barplots: {str(e)}.')

    ### Barplot Target Function ###
    def barplot_target(
        self,
        target_col: str,
        percentage_col: str,
        title: str,
        palette: list = ['#b0ff9d', '#db5856'],
    ):
        """
        Plots a bar chart showing the churn rate from a pre-aggregated DataFrame.

        This method visualizes the percentage distribution of the churn target classes,
        using bars colored by the target class and annotated with percentage values.

        Args:
            target_col (str): Name of the column representing the churn target classes
                (e.g., 0 = non-churner, 1 = churner).
            percentage_col (str): Name of the column containing percentage values for each class.
            title (str): Title of the plot.
            palette (list, optional): List of colors for the bars. Defaults to ['#b0ff9d', '#db5856'].

        Raises:
            ValueError: If `target_col` or `percentage_col` are not found in the DataFrame.
            Exception: For any other error occurring during plotting.
        """
        try:
            # Entry checks
            if target_col not in self.data.columns:
                raise ValueError(f"Column '{target_col}' not found in the DataFrame.")

            if percentage_col not in self.data.columns:
                raise ValueError(f"Column '{percentage_col}' not found in the DataFrame.")
            
            # Define AX and Fig
            plt.rc('font', size = 20, weight = 'bold')
            fig, ax = plt.subplots(figsize = (8, 6))

            barplot = sns.barplot(
                data = self.data,
                x = target_col,
                y = percentage_col,
                hue = target_col,
                dodge = False,
                palette = palette,
                edgecolor = 'black',
                saturation = 1,
                legend = False,
                ax = ax
            )

            # Annotate bars
            for v in barplot.patches:
                barplot.annotate(
                    f'{v.get_height():.2f}%',
                    (v.get_x() + v.get_width() / 2., v.get_height() / 1.06),
                    ha = 'center',
                    va = 'top',
                    fontsize = 16,
                    fontweight = 'bold',
                    color = 'black'
                )

            # Config Ax's and Show Graphics
            ax.set_yticklabels([])
            sns.despine(ax = ax, top = True, right = True, left = True, bottom = False)
            self._format_single_ax(ax, title = title, linewidth = 0.5)
            
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f'[Error] Failed to generate Barplot target: {str(e)}.')

    ### Correlation Heatmap Function ###
    def correlation_heatmap(
        self,
        title: str = None,
        cmap: str = 'coolwarm'
    ):
        """
        Plots a heatmap showing the correlation matrix among the numerical columns.

        This method computes the correlation matrix of the dataset and displays it as a heatmap,
        with annotations showing the correlation coefficients.

        Args:
            title (str, optional): Title for the heatmap plot. Defaults to None.
            cmap (str, optional): Colormap to use for the heatmap. Defaults to 'coolwarm'.

        Raises:
            Exception: If the heatmap generation or plotting fails.
        """
        try:
            # Select only the desired columns
            corr_data = self.data.corr()

            # Define AX and Fig
            plt.rc('font', size = 15)
            fig, ax = plt.subplots(figsize = (20, 15))

            sns.heatmap(
                corr_data,
                annot = True,
                cmap = cmap,
                fmt = '.2f',
                linewidths = 0.5,
                ax = ax
            )
            # Config Ax's and Show Graphics
            ax.set_title(title, fontsize = 20, fontweight = 'bold')
            plt.tight_layout(rect = [0, 0, 1, 0.97])
            plt.show()
        except Exception as e:
            print(f'[Error] Failed to generate correlation heatmap: {str(e)}.')

    ### Scatterplots vs Reference Function
    def scatterplots_vs_reference(
        self, 
        x_reference: str,
        hue: str = None,
        exclude_cols: list = [],
        num_columns: int = 3,
        figsize_per_row: int = 6,
        palette: list = ['#b0ff9d', '#db5856'],
        title: str = 'Scatterplot of Numerical Variables vs Reference'
    ):
        """
        Plots scatterplots comparing numerical variables against a reference variable,
        optionally grouped by a hue variable.

        This method creates scatterplots of all numerical columns (excluding specified ones)
        against a single reference numerical column on the X-axis. Points can be colored by
        a categorical hue variable.

        Args:
            x_reference (str): Column name to be used as X-axis in all scatterplots.
            hue (str, optional): Column name used for grouping/coloring points. Defaults to None.
            exclude_cols (list, optional): List of columns to exclude from Y-axis candidates,
                in addition to `x_reference` and `hue`. Defaults to empty list.
            num_columns (int): Number of plots per row in the subplot grid.
            figsize_per_row (int): Height (in inches) allocated per subplot row.
            palette (list, optional): List of colors for the hue categories. Defaults to ['#b0ff9d', '#db5856'].
            title (str): Overall title for the figure.

        Raises:
            ValueError: If `x_reference` or `hue` (when specified) are not found in the DataFrame.
            Exception: For any other errors during plotting.

        """
        try:
            # Entry checks
            if x_reference not in self.data.columns:
                raise ValueError(f"Column '{x_reference}' not found in the DataFrame.")
        
            if hue and hue not in self.data.columns:
                raise ValueError(f"Column '{hue}' not found in the DataFrame.")

            numeric_cols = self.data.select_dtypes(include = 'number').columns.tolist()
            for col in [x_reference, hue] + exclude_cols:
                if col in numeric_cols:
                    numeric_cols.remove(col)

            # Define AX and Fig
            fig, ax = self._initializer_subplot_grid(num_columns, figsize_per_row)

            for i, column in enumerate(numeric_cols):
                sns.scatterplot(
                    data = self.data,
                    x = x_reference,
                    y = column,
                    hue = hue,
                    palette = palette if hue else None,
                    ax = ax[i]
                )

                # Config Ax's
                self._format_single_ax(ax[i], f'{column} x {x_reference}')
                ax[i].set_xticklabels([])
                ax[i].set_yticklabels([])
                sns.despine(ax = ax[i], top = True, right = True, left = True, bottom = True)

            # Show Graphics
            self._finalize_subplot_layout(fig, ax, i, title = title)
        except Exception as e:
            print(f'[ERROR] Failed to generate scatterplots vs reference: {str(e)}.')  

    # Categorical Bar Percentages Function
    def categorical_bar_percentages(
        self,
        hue: str ,
        palette: list = ['#b0ff9d', '#db5856'],
        num_columns: int = 2,
        figsize_per_row: int = 8,
        title: str = 'Barplots Of The Individual Rate Percentages Of Each Column Class'
    ):
        """
        Plots barplots of churn percentages per class of each categorical variable.

        This method calculates the percentage distribution of a binary target (`hue`)
        within each category of all categorical columns in the dataset, and visualizes
        these percentages as barplots.

        Args:
            hue (str): Name of the binary target column (e.g., 'churn_target').
            palette (list, optional): List of colors for the hue classes.
                Defaults to ['#b0ff9d', '#db5856'].
            num_columns (int): Number of subplots per row in the grid.
            figsize_per_row (int): Height (in inches) allocated per subplot row.
            title (str): Overall title for the figure.

        Raises:
            ValueError: If `hue` is not found in the DataFrame.
            Exception: For other errors during computation or plotting.

        Returns:
            None: Displays the plot directly.
        """
        try:
            # Entry checks
            if hue and hue not in self.data.columns:
                raise ValueError(f"Column '{hue}' not found in the DataFrame.")
            categorical_cols = self.data.select_dtypes(include = ['object', 'category']).columns.tolist()
            if hue and hue in categorical_cols:
                categorical_cols.remove(hue)

            # Define AX and Fig
            fig, ax = self._initializer_subplot_grid(num_columns, figsize_per_row)

            for i, column in enumerate(categorical_cols):
                
                total_churn_per_class = self.data.groupby(column)[hue].count().reset_index(name = f'total_count_class')

                result = (
                    self.data.groupby([column, hue])[hue]
                    .count()
                    .reset_index(name = 'frequency')
                    .merge(total_churn_per_class, on = column)
                )
                result['percentage_per_class'] = round((result['frequency'] / result['total_count_class']) * 100, 2)

                sns.barplot(
                    data=result,
                    x = column,
                    y = 'percentage_per_class',
                    hue = hue,
                    palette = palette,
                    edgecolor = 'white',
                    saturation = 1,
                    legend = False,
                    ax = ax[i]
                )

                # Annotate bars
                for p in ax[i].patches:
                    height = p.get_height()
                    percentage = f'{height:.1f}%'
                    x = p.get_x() + p.get_width() / 2
                    ax[i].annotate(
                        percentage,
                        (x, height),
                        ha='center',
                        va='bottom',
                        fontsize=14,
                        color='black'
                    )

                # Config Ax's
                self._format_single_ax(ax[i], f'Barplot of variable: {column}')
                ax[i].set_xticks(range(len(ax[i].get_xticklabels())))
                ax[i].set_xticklabels(ax[i].get_xticklabels(), fontsize = 16)
            
            # Show Graphics
            self._finalize_subplot_layout(fig, ax, i, title = title)
        except Exception as e:
            print(f'[ERROR] Failed to generate percentage barplots: {str(e)}.')
