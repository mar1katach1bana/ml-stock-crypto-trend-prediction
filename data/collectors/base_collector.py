from abc import ABC, abstractmethod
import pandas as pd
from typing import Optional, Union, Dict, Any
from datetime import datetime

class BaseDataCollector(ABC):
    """Base class for all data collectors."""
    
    def __init__(self):
        self.data: Optional[pd.DataFrame] = None
        
    @abstractmethod
    def fetch_data(self, 
                  symbol: str,
                  start_date: Optional[Union[str, datetime]] = None,
                  end_date: Optional[Union[str, datetime]] = None,
                  interval: str = "1d",
                  **kwargs) -> pd.DataFrame:
        """
        Fetch data for the given symbol and time range.
        
        Args:
            symbol: The trading symbol to fetch data for
            start_date: Start date for historical data
            end_date: End date for historical data
            interval: Time interval between data points
            **kwargs: Additional arguments specific to the data source
            
        Returns:
            DataFrame containing the fetched data
        """
        pass
    
    @abstractmethod
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate the fetched data.
        
        Args:
            data: DataFrame to validate
            
        Returns:
            True if data is valid, False otherwise
        """
        pass
    
    def _standardize_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize column names across different data sources.
        
        Args:
            data: DataFrame with source-specific column names
            
        Returns:
            DataFrame with standardized column names
        """
        # Standard column mapping
        standard_columns = {
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
            'Adj Close': 'adj_close'
        }
        
        # Rename columns if they exist
        for old_name, new_name in standard_columns.items():
            if old_name in data.columns:
                data = data.rename(columns={old_name: new_name})
                
        return data
