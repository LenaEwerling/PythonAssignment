import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.sql import text
import matplotlib.pyplot as plt
import unittest
import math

class DatabaseSaveError(Exception):
    pass

class DataHandler:
    """Base class for handling data loading and database operations."""
    
    def __init__(self):
        """Initialize the DataHandler with a SQLite database engine."""
        self._engine = create_engine('sqlite:///assignment.db')
        self._train_df = None
        self._test_df = None
        self._ideal_df = None

    def load_data(self, train_file: str, test_file: str, ideal_file: str) -> None:
        """
        Load CSV files into DataFrames.

        Args:
            train_file (str): Path to the training data CSV file.
            test_file (str): Path to the test data CSV file.
            ideal_file (str): Path to the ideal functions CSV file.

        Raises:
            FileNotFoundError: If any of the CSV files cannot be found.
        """
        try:
            self._train_df = pd.read_csv(train_file)
            self._test_df = pd.read_csv(test_file)
            self._ideal_df = pd.read_csv(ideal_file)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Error: File not found - {e}")

    def save_to_database(self) -> None:
        """
        Save training and ideal data to SQLite database.

        Raises:
            Exception: If database operations fail.
        """
        try:
            self._train_df.to_sql('training_data', self._engine, if_exists='replace', index=False)
            self._ideal_df.to_sql('ideal_functions', self._engine, if_exists='replace', index=False)
        except DatabaseSaveError as e:
            raise DatabaseSaveError(f"Error saving to database: {e}")

    def get_train_data(self) -> pd.DataFrame:
        """Get the training DataFrame."""
        return self._train_df

    def get_test_data(self) -> pd.DataFrame:
        """Get the test DataFrame."""
        return self._test_df

    def get_ideal_data(self) -> pd.DataFrame:
        """Get the ideal functions DataFrame."""
        return self._ideal_df

    def get_engine(self):
        """Get the SQLAlchemy engine."""
        return self._engine


class FunctionSelector(DataHandler):
    """Class for selecting the best ideal functions based on least-square criterion."""
    
    def __init__(self):
        """Initialize the FunctionSelector."""
        super().__init__()
        self._best_functions = {}
        self._max_deviations = {}

    def calculate_mse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate the Mean Squared Error (MSE) between two arrays.

        Args:
            y_true (np.ndarray): True values.
            y_pred (np.ndarray): Predicted values.

        Returns:
            float: Mean Squared Error.
        """
        return np.mean((y_true - y_pred) ** 2)

    def select_best_functions(self) -> tuple[dict, dict]:
        """
        Select the best ideal functions for each training dataset column.

        Returns:
            tuple: Dictionary of best functions and dictionary of max deviations.
        """
        for y_col in ['y1', 'y2', 'y3', 'y4']:
            min_mse = float('inf')
            best_func = None
            y_true = self._train_df[y_col].values
            for ideal_col in [f'y{i}' for i in range(1, 51)]:
                y_pred = self._ideal_df[ideal_col].values
                mse = self.calculate_mse(y_true, y_pred)
                if mse < min_mse:
                    min_mse = mse
                    best_func = ideal_col
            self._best_functions[y_col] = best_func
            self._max_deviations[y_col] = np.max(np.abs(y_true - self._ideal_df[best_func].values))
        return self._best_functions, self._max_deviations


class TestDataMapper(DataHandler):
    """Class for mapping test data to the selected ideal functions."""
    
    def __init__(self, best_functions: dict, max_deviations: dict):
        """
        Initialize the TestDataMapper.

        Args:
            best_functions (dict): Mapping of training columns to ideal functions.
            max_deviations (dict): Maximum deviations for each training column.
        """
        super().__init__()
        self._best_functions = best_functions
        self._max_deviations = max_deviations
        self._results_df = None

    def map_test_data(self) -> pd.DataFrame:
        """
        Map test data to the best ideal functions based on deviation criterion.

        Returns:
            pd.DataFrame: DataFrame with mapped test data.
        """
        results = []
        sqrt_2 = math.sqrt(2)
        # Reverse mapping to find training column for ideal function
        reverse_best_functions = {v: k for k, v in self._best_functions.items()}

        for _, row in self._test_df.iterrows():
            x, y = row['x'], row['y']
            best_func = None
            min_deviation = float('inf')

            for ideal_func in self._best_functions.values():
                if x in self._ideal_df['x'].values:
                    ideal_y = self._ideal_df[self._ideal_df['x'] == x][ideal_func].values[0]
                else:
                    ideal_y = np.interp(x, self._ideal_df['x'], self._ideal_df[ideal_func])
                deviation = abs(y - ideal_y)
                if deviation < min_deviation:
                    min_deviation = deviation
                    best_func = ideal_func

            # Get the corresponding training column for the best function
            train_col = reverse_best_functions[best_func]
            max_allowed_deviation = self._max_deviations[train_col] * sqrt_2
            chosen_function = best_func if min_deviation <= max_allowed_deviation else f'({best_func})'

            results.append({
                'x': x,
                'y': y,
                'chosen_function': chosen_function,
                'deviation': min_deviation
            })

        self._results_df = pd.DataFrame(results)
        try:
            self._results_df.to_sql('test_results', self._engine, if_exists='replace', index=False)
        except DatabaseSaveError as e:
            raise DatabaseSaveError(f"Error saving test results to database: {e}")
        return self._results_df

    def get_results(self) -> pd.DataFrame:
        """Get the results DataFrame."""
        return self._results_df


class Visualizer:
    """Class for visualizing training, test, and ideal function data."""
    
    def __init__(self, train_df: pd.DataFrame, test_df: pd.DataFrame, ideal_df: pd.DataFrame, 
                 best_functions: dict, results_df: pd.DataFrame):
        """
        Initialize the Visualizer.

        Args:
            train_df (pd.DataFrame): Training data.
            test_df (pd.DataFrame): Test data.
            ideal_df (pd.DataFrame): Ideal functions data.
            best_functions (dict): Mapping of training columns to ideal functions.
            results_df (pd.DataFrame): Test data mapping results.
        """
        self._train_df = train_df
        self._test_df = test_df
        self._ideal_df = ideal_df
        self._best_functions = best_functions
        self._results_df = results_df

    def visualize(self, output_file: str = 'visualization.png') -> None:
        """
        Create and save a plot of training, test, and ideal function data with colored test points.

        Args:
            output_file (str): Path to save the visualization.
        """
        plt.figure(figsize=(12, 8))
        
        # Plot training data
        for y_col in ['y1', 'y2', 'y3', 'y4']:
            plt.plot(self._train_df['x'], self._train_df[y_col], label=f'Training {y_col}', linestyle='--')
        
        # Plot ideal functions
        colors = {'y41': 'blue', 'y42': 'green', 'y11': 'orange', 'y48': 'purple'}
        for y_col, ideal_func in self._best_functions.items():
            plt.plot(self._ideal_df['x'], self._ideal_df[ideal_func], label=f'Ideal {ideal_func}', 
                     color=colors.get(ideal_func, 'black'), alpha=0.7)

        # Plot valid test points
        valid_points = self._results_df[~self._results_df['chosen_function'].str.startswith('(')]
        for func in valid_points['chosen_function'].unique():
            subset = valid_points[valid_points['chosen_function'] == func]
            plt.scatter(subset['x'], subset['y'], label=f'Test Data ({func})', 
                       color=colors.get(func, 'red'), s=50, marker='o', zorder=5)

        # Plot all invalid test points together
        invalid_points = self._results_df[self._results_df['chosen_function'].str.startswith('(')]
        if not invalid_points.empty:
            plt.scatter(invalid_points['x'], invalid_points['y'], label='Test Data (Invalid)', 
                       color='red', s=50, marker='x', zorder=5)

        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Training Data, Test Data and Selected Ideal Functions')
        plt.legend(loc='upper center')
        plt.grid(True)
        plt.savefig(output_file)
        plt.close()
        print(f"Visualization saved as '{output_file}'")


class UnitTestRunner(unittest.TestCase):
    """Class for running unit tests on the assignment logic."""
    
    def __init__(self, methodName: str, data_handler: DataHandler, test_data_mapper: TestDataMapper):
        """
        Initialize the UnitTestRunner.

        Args:
            methodName (str): Name of the test method.
            data_handler (DataHandler): Instance of DataHandler.
            test_data_mapper (TestDataMapper): Instance of TestDataMapper.
        """
        super().__init__(methodName)
        self._data_handler = data_handler
        self._test_data_mapper = test_data_mapper

    def test_mse_calculation(self):
        """Test the Mean Squared Error (MSE) calculation."""
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1.1, 2.1, 3.1])
        mse = FunctionSelector.calculate_mse(self, y_true, y_pred)
        self.assertAlmostEqual(mse, 0.01, places=4)

    def test_database_tables(self):
        """Test if the required database tables exist."""
        with self._data_handler.get_engine().connect() as conn:
            tables = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table';")).fetchall()
            table_names = [t[0] for t in tables]
            self.assertIn('training_data', table_names)
            self.assertIn('ideal_functions', table_names)
            self.assertIn('test_results', table_names)

    def test_function_assignment(self):
        """Test if test data points are assigned correctly."""
        results_df = self._test_data_mapper.get_results()
        self.assertTrue(len(results_df) > 0, "No test points were processed")
        valid_df = results_df[~results_df['chosen_function'].str.startswith('(')]
        self.assertTrue(all(valid_df['deviation'] >= 0), "Deviations should not be negative for valid assignments")


def main():
    """Main function to execute the assignment logic."""
    # Initialize data handler and load data
    data_handler = DataHandler()
    data_handler.load_data('train.csv', 'test.csv', 'ideal.csv')
    data_handler.save_to_database()

    # Select best functions
    function_selector = FunctionSelector()
    function_selector._train_df = data_handler.get_train_data()
    function_selector._ideal_df = data_handler.get_ideal_data()
    best_functions, max_deviations = function_selector.select_best_functions()
    print("Selected Functions:", best_functions)
    print("Maximum Deviations:", max_deviations)

    # Map test data
    test_data_mapper = TestDataMapper(best_functions, max_deviations)
    test_data_mapper._test_df = data_handler.get_test_data()
    test_data_mapper._ideal_df = data_handler.get_ideal_data()
    test_data_mapper._engine = data_handler.get_engine()
    results_df = test_data_mapper.map_test_data()

    # Analyze test data mappings
    print("\nTest Points per Ideal Function:")
    print(results_df['chosen_function'].value_counts())

    # Print database contents
    with data_handler.get_engine().connect() as conn:
        print("\nTraining Data (first 5 rows):")
        print(pd.read_sql("SELECT * FROM training_data LIMIT 5", conn))
        print("\nIdeal Functions (first 5 rows):")
        print(pd.read_sql("SELECT * FROM ideal_functions LIMIT 5", conn))
        print("\nTest Results (first 5 rows):")
        print(pd.read_sql("SELECT * FROM test_results LIMIT 5", conn))

    # Visualize data
    visualizer = Visualizer(data_handler.get_train_data(), data_handler.get_test_data(),
                           data_handler.get_ideal_data(), best_functions, results_df)
    visualizer.visualize()

    # Run unit tests
    print("\nStarting Unit Tests...")
    suite = unittest.TestSuite()
    test_runner = UnitTestRunner('test_mse_calculation', data_handler, test_data_mapper)
    suite.addTest(test_runner)
    test_runner = UnitTestRunner('test_database_tables', data_handler, test_data_mapper)
    suite.addTest(test_runner)
    test_runner = UnitTestRunner('test_function_assignment', data_handler, test_data_mapper)
    suite.addTest(test_runner)
    unittest.TextTestRunner(verbosity=2).run(suite)


if __name__ == '__main__':
    main()