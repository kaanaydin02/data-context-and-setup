from pathlib import Path
import pandas as pd


class Olist:
    """
    The Olist class provides methods to interact with Olist's e-commerce data.
    """

    def get_data(self):
        """
        This function returns a Python dict.
        Its keys are 'sellers', 'orders', 'order_items' etc...
        Its values are pandas.DataFrames loaded from csv files
        """
        csv_path = Path.home() / ".workintech" / "olist" / "data" / "csv"

        file_names = [f.name for f in csv_path.glob("*.csv")]

        key_names = [
            f.replace("olist_", "").replace("_dataset", "").replace(".csv", "")
            for f in file_names
        ]

        data = {}
        for key, file_name in zip(key_names, file_names):
            data[key] = pd.read_csv(csv_path / file_name)

        return data

    def ping(self):
        """
        You call ping I print pong.
        """
        print("pong")
