import os
import yaml
import pandas as pd
import yfinance as yf
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from alpha_vantage.timeseries import TimeSeries
from binance.client import Client
from fredapi import Fred
import tweepy
import praw
import redis
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class DataLoader:
    """Data loader class for fetching market data from various sources."""
    
    def __init__(self, config_path: str):
        """Initialize DataLoader with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.cache = self._initialize_cache()
        
    def _initialize_cache(self) -> Optional[redis.Redis]:
        """Initialize Redis cache if enabled in config."""
        if self.config['cache']['enabled']:
            try:
                return redis.Redis(
                    host=self.config['cache']['host'],
                    port=self.config['cache']['port'],
                    decode_responses=True
                )
            except Exception as e:
                logger.warning(f"Failed to initialize Redis cache: {e}")
                return None
        return None

    def fetch_stock_data(self, symbols: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """Fetch stock data from configured sources."""
        if symbols is None:
            symbols = self.config['stock_data']['yahoo_finance']['symbols']
        
        data = {}
        for symbol in symbols:
            cache_key = f"stock_{symbol}"
            
            # Try to get from cache first
            if self.cache:
                cached_data = self.cache.get(cache_key)
                if cached_data:
                    data[symbol] = pd.read_json(cached_data)
                    continue
            
            # Fetch from Yahoo Finance
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(
                    period="max",
                    interval=self.config['stock_data']['yahoo_finance']['interval']
                )
                
                # Cache the data
                if self.cache:
                    self.cache.setex(
                        cache_key,
                        self.config['cache']['ttl'],
                        df.to_json()
                    )
                
                data[symbol] = df
                
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {e}")
                continue
                
        return data

    def fetch_crypto_data(self, symbols: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """Fetch cryptocurrency data from configured sources."""
        if symbols is None:
            symbols = self.config['crypto_data']['binance']['symbols']
        
        data = {}
        try:
            client = Client(
                self.config['crypto_data']['binance']['api_key'],
                self.config['crypto_data']['binance']['api_secret']
            )
            
            for symbol in symbols:
                cache_key = f"crypto_{symbol}"
                
                # Try to get from cache first
                if self.cache:
                    cached_data = self.cache.get(cache_key)
                    if cached_data:
                        data[symbol] = pd.read_json(cached_data)
                        continue
                
                # Fetch from Binance
                klines = client.get_historical_klines(
                    symbol,
                    self.config['crypto_data']['binance']['interval'],
                    self.config['crypto_data']['binance']['start_date']
                )
                
                df = pd.DataFrame(klines, columns=[
                    'timestamp', 'open', 'high', 'low', 'close',
                    'volume', 'close_time', 'quote_asset_volume',
                    'number_of_trades', 'taker_buy_base_asset_volume',
                    'taker_buy_quote_asset_volume', 'ignore'
                ])
                
                # Convert timestamp to datetime
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                
                # Cache the data
                if self.cache:
                    self.cache.setex(
                        cache_key,
                        self.config['cache']['ttl'],
                        df.to_json()
                    )
                
                data[symbol] = df
                
        except Exception as e:
            logger.error(f"Error fetching crypto data: {e}")
            
        return data

    def fetch_economic_data(self) -> Dict[str, pd.DataFrame]:
        """Fetch economic indicators from FRED."""
        data = {}
        try:
            fred = Fred(api_key=self.config['economic_data']['fred']['api_key'])
            
            for series in self.config['economic_data']['fred']['series']:
                cache_key = f"fred_{series}"
                
                # Try to get from cache first
                if self.cache:
                    cached_data = self.cache.get(cache_key)
                    if cached_data:
                        data[series] = pd.read_json(cached_data)
                        continue
                
                # Fetch from FRED
                df = pd.DataFrame(
                    fred.get_series(series),
                    columns=[series]
                )
                
                # Cache the data
                if self.cache:
                    self.cache.setex(
                        cache_key,
                        self.config['cache']['ttl'],
                        df.to_json()
                    )
                
                data[series] = df
                
        except Exception as e:
            logger.error(f"Error fetching economic data: {e}")
            
        return data

    def fetch_social_sentiment(self) -> Dict[str, pd.DataFrame]:
        """Fetch social media sentiment data."""
        data = {}
        
        # Twitter sentiment
        try:
            auth = tweepy.OAuthHandler(
                self.config['alternative_data']['twitter']['api_key'],
                self.config['alternative_data']['twitter']['api_secret']
            )
            api = tweepy.API(auth)
            
            for term in self.config['alternative_data']['twitter']['search_terms']:
                cache_key = f"twitter_{term}"
                
                # Try to get from cache first
                if self.cache:
                    cached_data = self.cache.get(cache_key)
                    if cached_data:
                        data[f"twitter_{term}"] = pd.read_json(cached_data)
                        continue
                
                # Fetch from Twitter
                tweets = api.search_tweets(
                    q=term,
                    lang=self.config['alternative_data']['twitter']['lang'],
                    count=self.config['alternative_data']['twitter']['max_results']
                )
                
                df = pd.DataFrame([
                    {
                        'created_at': tweet.created_at,
                        'text': tweet.text,
                        'user': tweet.user.screen_name,
                        'retweets': tweet.retweet_count,
                        'likes': tweet.favorite_count
                    }
                    for tweet in tweets
                ])
                
                # Cache the data
                if self.cache:
                    self.cache.setex(
                        cache_key,
                        self.config['cache']['ttl'],
                        df.to_json()
                    )
                
                data[f"twitter_{term}"] = df
                
        except Exception as e:
            logger.error(f"Error fetching Twitter data: {e}")
        
        # Reddit sentiment
        try:
            reddit = praw.Reddit(
                client_id=self.config['alternative_data']['reddit']['client_id'],
                client_secret=self.config['alternative_data']['reddit']['client_secret'],
                user_agent='ml_stock_trend_prediction'
            )
            
            for subreddit in self.config['alternative_data']['reddit']['subreddits']:
                cache_key = f"reddit_{subreddit}"
                
                # Try to get from cache first
                if self.cache:
                    cached_data = self.cache.get(cache_key)
                    if cached_data:
                        data[f"reddit_{subreddit}"] = pd.read_json(cached_data)
                        continue
                
                # Fetch from Reddit
                posts = reddit.subreddit(subreddit).top(
                    time_filter=self.config['alternative_data']['reddit']['timeframe'],
                    limit=self.config['alternative_data']['reddit']['limit']
                )
                
                df = pd.DataFrame([
                    {
                        'created_utc': datetime.fromtimestamp(post.created_utc),
                        'title': post.title,
                        'score': post.score,
                        'num_comments': post.num_comments,
                        'upvote_ratio': post.upvote_ratio
                    }
                    for post in posts
                ])
                
                # Cache the data
                if self.cache:
                    self.cache.setex(
                        cache_key,
                        self.config['cache']['ttl'],
                        df.to_json()
                    )
                
                data[f"reddit_{subreddit}"] = df
                
        except Exception as e:
            logger.error(f"Error fetching Reddit data: {e}")
            
        return data

    def save_to_local(self, data: Dict[str, pd.DataFrame], data_type: str):
        """Save data to local storage in parquet format."""
        base_path = Path(self.config['storage']['local'][f'{data_type}_data'])
        base_path.mkdir(parents=True, exist_ok=True)
        
        for key, df in data.items():
            file_path = base_path / f"{key}.parquet"
            df.to_parquet(
                file_path,
                compression=self.config['storage']['local']['compression']
            )
            logger.info(f"Saved {key} data to {file_path}")

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize DataLoader
    loader = DataLoader('config/data_sources.yaml')
    
    # Fetch and save data
    stock_data = loader.fetch_stock_data()
    loader.save_to_local(stock_data, 'raw')
    
    crypto_data = loader.fetch_crypto_data()
    loader.save_to_local(crypto_data, 'raw')
    
    economic_data = loader.fetch_economic_data()
    loader.save_to_local(economic_data, 'external')
    
    sentiment_data = loader.fetch_social_sentiment()
    loader.save_to_local(sentiment_data, 'external')
