"""Caching strategies and cache management for weather data."""

from __future__ import annotations

import json
import pickle
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
import hashlib

import pandas as pd
import polars as pl
import logging

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    import diskcache as dc
    DISKCACHE_AVAILABLE = True
except ImportError:
    DISKCACHE_AVAILABLE = False

logger = logging.getLogger(__name__)


class CacheBackend(ABC):
    """Abstract base class for cache backends."""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        pass
        
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache with optional TTL."""
        pass
        
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        pass
        
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        pass
        
    @abstractmethod
    async def clear(self) -> bool:
        """Clear all cache entries."""
        pass


class MemoryCache(CacheBackend):
    """Simple in-memory cache backend."""
    
    def __init__(self, max_size: int = 1000) -> None:
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.max_size = max_size
        
    async def get(self, key: str) -> Optional[Any]:
        """Get value from memory cache."""
        if key not in self.cache:
            return None
            
        entry = self.cache[key]
        
        # Check if expired
        if entry.get("expires_at") and datetime.utcnow() > entry["expires_at"]:
            del self.cache[key]
            return None
            
        return entry["value"]
        
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in memory cache."""
        # Evict oldest entries if cache is full
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k]["created_at"])
            del self.cache[oldest_key]
            
        expires_at = None
        if ttl:
            expires_at = datetime.utcnow() + timedelta(seconds=ttl)
            
        self.cache[key] = {
            "value": value,
            "created_at": datetime.utcnow(),
            "expires_at": expires_at
        }
        
        return True
        
    async def delete(self, key: str) -> bool:
        """Delete key from memory cache."""
        if key in self.cache:
            del self.cache[key]
            return True
        return False
        
    async def exists(self, key: str) -> bool:
        """Check if key exists in memory cache."""
        if key not in self.cache:
            return False
            
        # Check expiration
        entry = self.cache[key]
        if entry.get("expires_at") and datetime.utcnow() > entry["expires_at"]:
            del self.cache[key]
            return False
            
        return True
        
    async def clear(self) -> bool:
        """Clear all entries from memory cache."""
        self.cache.clear()
        return True
        
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        now = datetime.utcnow()
        expired_count = sum(
            1 for entry in self.cache.values()
            if entry.get("expires_at") and now > entry["expires_at"]
        )
        
        return {
            "total_entries": len(self.cache),
            "max_size": self.max_size,
            "expired_entries": expired_count,
            "memory_usage_estimate": sum(
                len(str(entry)) for entry in self.cache.values()
            )
        }


class RedisCache(CacheBackend):
    """Redis cache backend."""
    
    def __init__(self, 
                 host: str = "localhost",
                 port: int = 6379,
                 db: int = 0,
                 password: Optional[str] = None,
                 prefix: str = "weather_cache:") -> None:
        if not REDIS_AVAILABLE:
            raise ImportError("Redis not available. Install with: pip install redis")
            
        self.redis_client = redis.Redis(
            host=host, 
            port=port, 
            db=db, 
            password=password,
            decode_responses=True
        )
        self.prefix = prefix
        
    def _get_key(self, key: str) -> str:
        """Get prefixed cache key."""
        return f"{self.prefix}{key}"
        
    async def get(self, key: str) -> Optional[Any]:
        """Get value from Redis cache."""
        try:
            value = self.redis_client.get(self._get_key(key))
            if value is not None:
                return json.loads(value)
        except redis.RedisError as e:
            logger.error(f"Redis get error: {e}")
        return None
        
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in Redis cache."""
        try:
            serialized_value = json.dumps(value, default=str)
            return self.redis_client.set(
                self._get_key(key), 
                serialized_value, 
                ex=ttl
            )
        except (redis.RedisError, TypeError, ValueError) as e:
            logger.error(f"Redis set error: {e}")
            return False
            
    async def delete(self, key: str) -> bool:
        """Delete key from Redis cache."""
        try:
            return bool(self.redis_client.delete(self._get_key(key)))
        except redis.RedisError as e:
            logger.error(f"Redis delete error: {e}")
            return False
            
    async def exists(self, key: str) -> bool:
        """Check if key exists in Redis cache."""
        try:
            return bool(self.redis_client.exists(self._get_key(key)))
        except redis.RedisError as e:
            logger.error(f"Redis exists error: {e}")
            return False
            
    async def clear(self) -> bool:
        """Clear all prefixed keys from Redis cache."""
        try:
            keys = self.redis_client.keys(f"{self.prefix}*")
            if keys:
                return bool(self.redis_client.delete(*keys))
            return True
        except redis.RedisError as e:
            logger.error(f"Redis clear error: {e}")
            return False


class DiskCache(CacheBackend):
    """Disk-based cache backend using diskcache."""
    
    def __init__(self, cache_dir: str = "./cache", size_limit: int = 1024**3) -> None:  # 1GB default
        if not DISKCACHE_AVAILABLE:
            raise ImportError("DiskCache not available. Install with: pip install diskcache")
            
        self.cache = dc.Cache(cache_dir, size_limit=size_limit)
        
    async def get(self, key: str) -> Optional[Any]:
        """Get value from disk cache."""
        try:
            return self.cache.get(key)
        except dc.CacheError as e:
            logger.error(f"DiskCache get error: {e}")
            return None
            
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in disk cache."""
        try:
            expire = None
            if ttl:
                expire = datetime.utcnow() + timedelta(seconds=ttl)
            return self.cache.set(key, value, expire=expire)
        except dc.CacheError as e:
            logger.error(f"DiskCache set error: {e}")
            return False
            
    async def delete(self, key: str) -> bool:
        """Delete key from disk cache."""
        try:
            return self.cache.delete(key)
        except dc.CacheError as e:
            logger.error(f"DiskCache delete error: {e}")
            return False
            
    async def exists(self, key: str) -> bool:
        """Check if key exists in disk cache."""
        try:
            return key in self.cache
        except dc.CacheError as e:
            logger.error(f"DiskCache exists error: {e}")
            return False
            
    async def clear(self) -> bool:
        """Clear all entries from disk cache."""
        try:
            self.cache.clear()
            return True
        except dc.CacheError as e:
            logger.error(f"DiskCache clear error: {e}")
            return False


class CacheManager:
    """High-level cache manager with multiple backends and strategies."""
    
    def __init__(self, 
                 backend: Optional[CacheBackend] = None,
                 default_ttl: int = 3600) -> None:  # 1 hour default
        self.backend = backend or MemoryCache()
        self.default_ttl = default_ttl
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0
        }
        
    async def get_weather_data(self, 
                               location: str, 
                               provider: str,
                               timestamp: Optional[datetime] = None) -> Optional[Dict[str, Any]]:
        """Get cached weather data."""
        cache_key = self._generate_weather_key(location, provider, timestamp)
        
        result = await self.backend.get(cache_key)
        if result is not None:
            self.cache_stats["hits"] += 1
            return result
        else:
            self.cache_stats["misses"] += 1
            return None
            
    async def cache_weather_data(self,
                                location: str,
                                provider: str,
                                data: Dict[str, Any],
                                timestamp: Optional[datetime] = None,
                                ttl: Optional[int] = None) -> bool:
        """Cache weather data."""
        cache_key = self._generate_weather_key(location, provider, timestamp)
        
        success = await self.backend.set(
            cache_key, 
            data, 
            ttl or self.default_ttl
        )
        
        if success:
            self.cache_stats["sets"] += 1
            
        return success
        
    async def get_processed_data(self, 
                                processing_key: str,
                                parameters: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        """Get cached processed data results."""
        cache_key = self._generate_processing_key(processing_key, parameters)
        
        result = await self.backend.get(cache_key)
        if result is not None:
            self.cache_stats["hits"] += 1
            
            # Handle DataFrame serialization
            if isinstance(result, dict) and "dataframe_type" in result:
                return self._deserialize_dataframe(result)
            return result
        else:
            self.cache_stats["misses"] += 1
            return None
            
    async def cache_processed_data(self,
                                  processing_key: str,
                                  result: Any,
                                  parameters: Optional[Dict[str, Any]] = None,
                                  ttl: Optional[int] = None) -> bool:
        """Cache processed data results."""
        cache_key = self._generate_processing_key(processing_key, parameters)
        
        # Handle DataFrame serialization
        if isinstance(result, (pd.DataFrame, pl.DataFrame)):
            serialized_result = self._serialize_dataframe(result)
        else:
            serialized_result = result
            
        success = await self.backend.set(
            cache_key,
            serialized_result,
            ttl or self.default_ttl
        )
        
        if success:
            self.cache_stats["sets"] += 1
            
        return success
        
    async def cache_analysis_result(self,
                                   analysis_type: str,
                                   data_hash: str,
                                   result: Dict[str, Any],
                                   ttl: Optional[int] = None) -> bool:
        """Cache analysis results."""
        cache_key = f"analysis:{analysis_type}:{data_hash}"
        
        success = await self.backend.set(
            cache_key,
            result,
            ttl or (self.default_ttl * 2)  # Analysis results cached longer
        )
        
        if success:
            self.cache_stats["sets"] += 1
            
        return success
        
    async def get_analysis_result(self,
                                 analysis_type: str,
                                 data_hash: str) -> Optional[Dict[str, Any]]:
        """Get cached analysis results."""
        cache_key = f"analysis:{analysis_type}:{data_hash}"
        
        result = await self.backend.get(cache_key)
        if result is not None:
            self.cache_stats["hits"] += 1
        else:
            self.cache_stats["misses"] += 1
            
        return result
        
    async def invalidate_location(self, location: str) -> int:
        """Invalidate all cache entries for a location."""
        # This is a simplified implementation
        # In practice, you'd need to track keys by location
        count = 0
        if hasattr(self.backend, 'cache') and hasattr(self.backend.cache, 'keys'):
            # For memory cache
            keys_to_delete = [
                key for key in self.backend.cache.keys() 
                if location in key
            ]
            for key in keys_to_delete:
                if await self.backend.delete(key):
                    count += 1
                    self.cache_stats["deletes"] += 1
                    
        return count
        
    async def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate cache entries matching a pattern."""
        count = 0
        
        # For Redis backend
        if isinstance(self.backend, RedisCache):
            try:
                keys = self.backend.redis_client.keys(f"{self.backend.prefix}*{pattern}*")
                if keys:
                    deleted = self.backend.redis_client.delete(*keys)
                    count = deleted
                    self.cache_stats["deletes"] += deleted
            except redis.RedisError as e:
                logger.error(f"Pattern invalidation error: {e}")
                
        # For memory cache
        elif hasattr(self.backend, 'cache'):
            keys_to_delete = [
                key for key in self.backend.cache.keys()
                if pattern in key
            ]
            for key in keys_to_delete:
                if await self.backend.delete(key):
                    count += 1
                    self.cache_stats["deletes"] += 1
                    
        return count
        
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.cache_stats["hits"] + self.cache_stats["misses"]
        hit_rate = self.cache_stats["hits"] / total_requests if total_requests > 0 else 0
        
        stats = {
            "hit_rate": hit_rate,
            "total_requests": total_requests,
            **self.cache_stats
        }
        
        # Add backend-specific stats if available
        if hasattr(self.backend, 'get_stats'):
            stats["backend_stats"] = self.backend.get_stats()
            
        return stats
        
    async def clear_all(self) -> bool:
        """Clear all cache entries."""
        success = await self.backend.clear()
        if success:
            # Reset stats
            for key in self.cache_stats:
                self.cache_stats[key] = 0
                
        return success
        
    def _generate_weather_key(self, 
                             location: str, 
                             provider: str,
                             timestamp: Optional[datetime] = None) -> str:
        """Generate cache key for weather data."""
        if timestamp:
            # Round timestamp to nearest hour for better cache hits
            rounded_time = timestamp.replace(minute=0, second=0, microsecond=0)
            time_str = rounded_time.isoformat()
        else:
            time_str = "current"
            
        return f"weather:{provider}:{location}:{time_str}"
        
    def _generate_processing_key(self, 
                                processing_key: str,
                                parameters: Optional[Dict[str, Any]] = None) -> str:
        """Generate cache key for processed data."""
        if parameters:
            # Create hash of parameters for consistent key
            param_str = json.dumps(parameters, sort_keys=True, default=str)
            param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]
            return f"processing:{processing_key}:{param_hash}"
        else:
            return f"processing:{processing_key}"
            
    def _serialize_dataframe(self, df: Union[pd.DataFrame, pl.DataFrame]) -> Dict[str, Any]:
        """Serialize DataFrame for caching."""
        if isinstance(df, pd.DataFrame):
            return {
                "dataframe_type": "pandas",
                "data": df.to_json(orient='records', date_format='iso'),
                "index": df.index.tolist(),
                "columns": df.columns.tolist()
            }
        elif isinstance(df, pl.DataFrame):
            return {
                "dataframe_type": "polars",
                "data": df.to_pandas().to_json(orient='records', date_format='iso'),
                "columns": df.columns
            }
        else:
            raise ValueError(f"Unsupported DataFrame type: {type(df)}")
            
    def _deserialize_dataframe(self, data: Dict[str, Any]) -> Union[pd.DataFrame, pl.DataFrame]:
        """Deserialize DataFrame from cache."""
        df_type = data["dataframe_type"]
        
        if df_type == "pandas":
            df = pd.read_json(data["data"], orient='records')
            if "index" in data:
                df.index = data["index"]
            return df
        elif df_type == "polars":
            pandas_df = pd.read_json(data["data"], orient='records')
            return pl.from_pandas(pandas_df)
        else:
            raise ValueError(f"Unknown DataFrame type: {df_type}")
            
    def calculate_data_hash(self, data: Union[pd.DataFrame, pl.DataFrame, Dict, List]) -> str:
        """Calculate hash of data for cache keys."""
        if isinstance(data, pd.DataFrame):
            # Use DataFrame content hash
            return hashlib.md5(
                pd.util.hash_pandas_object(data, index=True).values.tobytes()
            ).hexdigest()[:16]
        elif isinstance(data, pl.DataFrame):
            # Convert to pandas for hashing
            return self.calculate_data_hash(data.to_pandas())
        else:
            # Use JSON representation hash
            json_str = json.dumps(data, sort_keys=True, default=str)
            return hashlib.md5(json_str.encode()).hexdigest()[:16]
