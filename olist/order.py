import pandas as pd
import numpy as np
from olist.data import Olist


class Order:
    """
    DataFrames containing all orders as index,
    and various properties of these orders as columns
    """

    def __init__(self):
        # Assign an attribute ".data" to all new instances of Order
        self.data = Olist().get_data()

    def get_wait_time(self):
        """
        Returns a DataFrame with:
        order_id, wait_time, expected_wait_time, delay_vs_expected, order_status
        """
        orders = self.data['orders'].copy()
        orders = orders[orders['order_status'] == 'delivered'].copy()

        date_columns = [
            'order_purchase_timestamp',
            'order_delivered_customer_date',
            'order_estimated_delivery_date',
        ]
        for c in date_columns:
            orders[c] = pd.to_datetime(orders[c])

        orders['wait_time'] = (
            orders['order_delivered_customer_date'] - orders['order_purchase_timestamp']
        ) / np.timedelta64(1, 'D')

        orders['expected_wait_time'] = (
            orders['order_estimated_delivery_date'] - orders['order_purchase_timestamp']
        ) / np.timedelta64(1, 'D')

        orders['delay_vs_expected'] = (
            orders['order_delivered_customer_date'] - orders['order_estimated_delivery_date']
        ) / np.timedelta64(1, 'D')
        orders['delay_vs_expected'] = orders['delay_vs_expected'].apply(lambda x: max(x, 0))

        return orders[['order_id', 'wait_time', 'expected_wait_time', 'delay_vs_expected', 'order_status']]

    def get_review_score(self):
        """
        Returns a DataFrame with:
        order_id, dim_is_five_star, dim_is_one_star, review_score
        """
        reviews = self.data['order_reviews'].copy()

        reviews['dim_is_five_star'] = reviews['review_score'].apply(lambda x: 1 if x == 5 else 0)
        reviews['dim_is_one_star'] = reviews['review_score'].apply(lambda x: 1 if x == 1 else 0)

        return reviews[['order_id', 'dim_is_five_star', 'dim_is_one_star', 'review_score']]

    def get_number_items(self):
        """
        Returns a DataFrame with:
        order_id, number_of_items
        """
        order_items = self.data['order_items'].copy()

        return order_items.groupby('order_id', as_index=False).agg(number_of_items=('order_item_id', 'count'))

    def get_number_sellers(self):
        """
        Returns a DataFrame with:
        order_id, number_of_sellers
        """
        order_items = self.data['order_items'].copy()

        return order_items.groupby('order_id', as_index=False).agg(number_of_sellers=('seller_id', 'nunique'))
