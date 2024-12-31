import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Union

from .base_collector import BaseDataCollector

class YahooDataCollector(BaseDataCollector):
    """Data collector for Yahoo Finance."""
    
    def __init__(self):
        super().__init__()
        
    def fetch_data(self,
                  symbol: str,
                  start_date: Optional[Union[str, datetime]] = None,
                  end_date: Optional[Union[str, datetime]] = None,
                  interval: str = "1d",
                  **kwargs) -> pd.DataFrame:
        """
        Fetch data from Yahoo Finance.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL', 'MSFT')
            start_date: Start date for historical data
            end_date: End date for historical data
            interval: Data interval ('1d', '1wk', '1mo')
            **kwargs: Additional arguments for yfinance
            
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
            
        # Fetch data using yfinance
        ticker = yf.Ticker(symbol)
        data = ticker.history(
            start=start_date,
            end=end_date,
            interval=interval
        )
        
        # Validate and standardize the data
        if self.validate_data(data):
            self.data = self._standardize_columns(data)
            return self.data
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
        required_columns = {'Open', 'High', 'Low', 'Close', 'Volume'}
        if not required_columns.issubset(set(data.columns)):
            return False
            
        # Check for missing values
        if data[list(required_columns)].isnull().values.any():
            return False
            
        return True
