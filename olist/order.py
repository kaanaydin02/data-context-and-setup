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

    def get_price_and_freight(self):
        """
        Returns a DataFrame with:
        order_id, price, freight_value
        """
        order_items = self.data['order_items'].copy()

        return order_items.groupby('order_id', as_index=False).agg({'price': 'sum', 'freight_value': 'sum'})

    def get_training_data(self, with_distance_seller_customer=False):
        """
        Returns a clean DataFrame (no NaN), with the columns:
        order_id, wait_time, expected_wait_time, delay_vs_expected, order_status,
        dim_is_five_star, dim_is_one_star, review_score, number_of_items,
        number_of_sellers, price, freight_value
        and optionally distance_seller_customer
        """
        training_set = (
            self.get_wait_time()
            .merge(self.get_review_score(), on='order_id')
            .merge(self.get_number_items(), on='order_id')
            .merge(self.get_number_sellers(), on='order_id')
            .merge(self.get_price_and_freight(), on='order_id')
        )

        if with_distance_seller_customer:
            training_set = training_set.merge(self.get_distance_seller_customer(), on='order_id')

        return training_set.dropna()

    def get_distance_seller_customer(self):
        """
        Returns a DataFrame with:
        order_id, distance_seller_customer
        """
        from olist.utils import haversine_distance

        orders = self.data['orders']
        order_items = self.data['order_items']
        sellers = self.data['sellers']
        customers = self.data['customers']
        geo = self.data['geolocation']

        geo = geo.groupby('geolocation_zip_code_prefix', as_index=False).first()

        sellers_geo = sellers.merge(
            geo, left_on='seller_zip_code_prefix', right_on='geolocation_zip_code_prefix'
        )
        customers_geo = customers.merge(
            geo, left_on='customer_zip_code_prefix', right_on='geolocation_zip_code_prefix'
        )

        matching = order_items[['order_id', 'seller_id']].merge(
            orders[['order_id', 'customer_id']], on='order_id'
        )
        matching = matching.merge(
            sellers_geo[['seller_id', 'geolocation_lat', 'geolocation_lng']], on='seller_id'
        )
        matching = matching.merge(
            customers_geo[['customer_id', 'geolocation_lat', 'geolocation_lng']],
            on='customer_id',
            suffixes=('_seller', '_customer'),
        )

        matching['distance_seller_customer'] = haversine_distance(
            matching['geolocation_lng_seller'],
            matching['geolocation_lat_seller'],
            matching['geolocation_lng_customer'],
            matching['geolocation_lat_customer'],
        )

        return matching.groupby('order_id', as_index=False)['distance_seller_customer'].mean()
