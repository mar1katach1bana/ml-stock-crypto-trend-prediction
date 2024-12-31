from binance.client import Client
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Union
import os
from dotenv import load_dotenv

from .base_collector import BaseDataCollector

class BinanceDataCollector(BaseDataCollector):
    """Data collector for Binance cryptocurrency exchange."""
    
    def __init__(self):
        super().__init__()
        load_dotenv()
        
        # Initialize Binance client with API keys from environment variables
        api_key = os.getenv('BINANCE_API_KEY')
        api_secret = os.getenv('BINANCE_API_SECRET')
        self.client = Client(api_key, api_secret) if api_key and api_secret else Client()
        
        # Mapping for interval strings
        self.interval_map = {
            '1m': Client.KLINE_INTERVAL_1MINUTE,
            '5m': Client.KLINE_INTERVAL_5MINUTE,
            '15m': Client.KLINE_INTERVAL_15MINUTE,
            '1h': Client.KLINE_INTERVAL_1HOUR,
            '4h': Client.KLINE_INTERVAL_4HOUR,
            '1d': Client.KLINE_INTERVAL_1DAY,
            '1w': Client.KLINE_INTERVAL_1WEEK,
        }
    
    def fetch_data(self,
                  symbol: str,
                  start_date: Optional[Union[str, datetime]] = None,
                  end_date: Optional[Union[str, datetime]] = None,
                  interval: str = "1d",
                  **kwargs) -> pd.DataFrame:
        """
        Fetch cryptocurrency data from Binance.
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT', 'ETHUSDT')
            start_date: Start date for historical data
            end_date: End date for historical data
            interval: Data interval ('1m', '5m', '15m', '1h', '4h', '1d', '1w')
            **kwargs: Additional arguments for Binance API
            
        Returns:
            DataFrame containing historical data
        """
        # Set default dates if not provided
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=365)
            
        # Convert dates to datetime if they're strings
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d")
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, "%Y-%m-%d")
            
        # Convert interval to Binance format
        binance_interval = self.interval_map.get(interval)
        if not binance_interval:
            raise ValueError(f"Invalid interval: {interval}")
            
        # Fetch klines (candlestick) data
        klines = self.client.get_historical_klines(
            symbol,
            binance_interval,
            start_date.strftime("%d %b %Y %H:%M:%S"),
            end_date.strftime("%d %b %Y %H:%M:%S")
        )
        
        # Convert to DataFrame
        data = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        # Convert timestamp to datetime
        data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
        data.set_index('timestamp', inplace=True)
        
        # Convert string values to float
        for col in ['open', 'high', 'low', 'close', 'volume']:
            data[col] = data[col].astype(float)
            
        # Validate and return data
        if self.validate_data(data):
            self.data = data
            return data
        else:
            raise ValueError(f"Invalid or empty data received for symbol {symbol}")
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate the fetched data.
        
        Args:
            data: DataFrame to validate
            
        Returns:
            True if data is valid, False otherwise
        """
        # Check if data is empty
        if data.empty:
            return False
            
        # Check for required columns
        required_columns = {'open', 'high', 'low', 'close', 'volume'}
        if not required_columns.issubset(set(data.columns)):
            return False
            
        # Check for missing values in required columns
        if data[list(required_columns)].isnull().values.any():
            return False
            
        return True
