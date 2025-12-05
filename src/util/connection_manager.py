"""
Centralized connection manager for MongoDB, MySQL, and file loading operations.
This module provides a single interface for all database connections and file I/O.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Any, List
import pandas as pd
import numpy as np
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database

LOGGER = logging.getLogger(__name__)


class ConnectionManager:
    """
    Centralized manager for all database connections and file operations.
    Provides singleton-like behavior with connection pooling and caching.
    """
    
    _instance = None
    _mongodb_clients: Dict[str, MongoClient] = {}
    _mysql_connections: Dict[str, Any] = {}
    _config: Optional[Dict[str, Any]] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConnectionManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        self._load_config()
    
    def _load_config(self):
        """Load configuration from config/config.json."""
        config_path = Path(__file__).parent.parent.parent / "config" / "config.json"
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self._config = json.load(f)
            LOGGER.debug(f"Loaded configuration from {config_path}")
        except Exception as e:
            LOGGER.error(f"Failed to load config from {config_path}: {e}")
            self._config = {}
    
    def get_config(self, config_type: str = "") -> Dict[str, Any]:
        """
        Get the loaded configuration.
        
        Args:
            config_type: Optional filter - "mongodb", "local_mongodb", "weather_record", etc.
                        If empty, returns full config
        
        Returns:
            Configuration dict or empty dict if not found
        """
        if not config_type:
            return self._config or {}
        return self._config.get(config_type, {})
    
    def read_config(self, config_type: str = "config", file_name: str = "") -> Dict[str, Any]:
        """
        Read configuration (compatibility method for config_parser.ConfigManger).
        
        Args:
            config_type: Type of config (default: "config")
            file_name: Optional file name (not used, always uses config/config.json)
        
        Returns:
            Configuration dict
        """
        if config_type == "config":
            return self.get_config()
        return {}
    
    def update_config(self, config_type: str = "", data: Dict[str, Any] = {}):
        """
        Update configuration file (compatibility method for config_parser.ConfigManger).
        
        Args:
            config_type: Type of config ("global_variables", "alarms_state", etc.)
            data: Data to write (must be dict or JSON string)
        """
        config_dir = Path(__file__).parent.parent.parent / "config"
        
        if config_type == "global_variables":
            file_path = config_dir / "app" / "global_variables.json"
        elif config_type == "alarms_state":
            file_path = config_dir / "app" / "alarms_state.json"
        else:
            LOGGER.warning(f"Unknown config type for update: {config_type}")
            return
        
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Handle data as dict or JSON string
            if isinstance(data, str):
                json_data = data
            else:
                json_data = json.dumps(data, indent=2, ensure_ascii=False)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(json_data)
            LOGGER.info(f"Updated config file: {file_path}")
        except Exception as e:
            LOGGER.error(f"Failed to update config file {file_path}: {e}")
    
    # ============================================================================
    # MongoDB Connection Methods
    # ============================================================================
    
    def get_mongodb_client(self, connection_type: str = "cloud") -> Optional[MongoClient]:
        """
        Get or create a MongoDB client.
        
        Args:
            connection_type: "cloud" for cloud MongoDB, "local" for local MongoDB
        
        Returns:
            MongoClient instance or None if connection fails
        """
        if connection_type in self._mongodb_clients:
            return self._mongodb_clients[connection_type]
        
        try:
            if connection_type == "cloud":
                mongo_cfg = self._config.get("mongodb", {})
                connection_string = mongo_cfg.get("connection_string")
                if not connection_string:
                    LOGGER.error("MongoDB cloud connection string not found in config")
                    return None
                
                client = MongoClient(connection_string, serverSelectionTimeoutMS=10000)
                # Test connection
                client.server_info()
                self._mongodb_clients[connection_type] = client
                LOGGER.info(f"✅ Connected to MongoDB cloud: {mongo_cfg.get('db_name', 'unknown')}")
                return client
            
            elif connection_type == "local":
                local_cfg = self._config.get("local_mongodb", {})
                local_uri = local_cfg.get("LOCAL_MONGO_URI", "mongodb://localhost:27017")
                
                client = MongoClient(local_uri, serverSelectionTimeoutMS=10000)
                # Test connection
                client.server_info()
                self._mongodb_clients[connection_type] = client
                LOGGER.info(f"✅ Connected to MongoDB local: {local_uri}")
                return client
            
            else:
                LOGGER.error(f"Unknown MongoDB connection type: {connection_type}")
                return None
                
        except Exception as e:
            LOGGER.error(f"Failed to connect to MongoDB ({connection_type}): {e}")
            return None
    
    def get_mongodb_collection(self, connection_type: str = "cloud", 
                              db_name: Optional[str] = None,
                              collection_name: Optional[str] = None) -> Optional[Collection]:
        """
        Get a MongoDB collection.
        
        Args:
            connection_type: "cloud" or "local"
            db_name: Database name (if None, uses default from config)
            collection_name: Collection name (if None, uses default from config)
        
        Returns:
            Collection instance or None if connection fails
        """
        client = self.get_mongodb_client(connection_type)
        if not client:
            return None
        
        try:
            if connection_type == "cloud":
                mongo_cfg = self._config.get("mongodb", {})
                db_name = db_name or mongo_cfg.get("db_name", "ampcome")
                collection_name = collection_name or mongo_cfg.get("collection_name", "foodorder")
            else:
                local_cfg = self._config.get("local_mongodb", {})
                db_name = db_name or local_cfg.get("LOCAL_MONGO_DB", "localGokhana")
                collection_name = collection_name or collection_name  # Keep as-is if provided
            
            db: Database = client[db_name]
            collection: Collection = db[collection_name]
            LOGGER.debug(f"Accessing MongoDB collection: {db_name}.{collection_name}")
            return collection
            
        except Exception as e:
            LOGGER.error(f"Failed to get MongoDB collection: {e}")
            return None
    
    def get_mongo_names(self, foodcourt_id: str, restaurant_id: str) -> tuple[str, str]:
        """
        Get foodcourt name and restaurant name from MongoDB.
        Uses caching to avoid repeated queries.
        
        Args:
            foodcourt_id: Foodcourt ID
            restaurant_id: Restaurant ID
        
        Returns:
            Tuple of (foodcourt_name, restaurant_name)
        """
        from bson import ObjectId
        
        # Try local MongoDB first
        try:
            local_cfg = self._config.get("local_mongodb", {})
            client = self.get_mongodb_client("local")
            
            if client:
                local_db = client[local_cfg.get("LOCAL_MONGO_DB", "localGokhana")]
                
                # Get restaurant name
                restaurant_name = restaurant_id
                try:
                    restaurant_objid = ObjectId(restaurant_id) if len(restaurant_id) == 24 else restaurant_id
                    restaurant_coll = local_db.get_collection("restaurant_data")
                    restaurant_doc = restaurant_coll.find_one({"_id": restaurant_objid})
                    if restaurant_doc and "data" in restaurant_doc and "name" in restaurant_doc["data"]:
                        restaurant_name = restaurant_doc["data"]["name"]
                except Exception as e:
                    LOGGER.debug(f"Could not fetch restaurant name for {restaurant_id}: {e}")
                
                # Get foodcourt name
                foodcourt_name = foodcourt_id
                try:
                    foodcourt_objid = ObjectId(foodcourt_id) if len(foodcourt_id) == 24 else foodcourt_id
                    fc_coll_name = local_cfg.get("FOOD_COURT_COLL", "food_court_record").strip()
                    foodcourt_coll = local_db.get_collection(fc_coll_name)
                    foodcourt_doc = foodcourt_coll.find_one({"_id": foodcourt_objid})
                    if foodcourt_doc and "data" in foodcourt_doc and "name" in foodcourt_doc["data"]:
                        foodcourt_name = foodcourt_doc["data"]["name"]
                except Exception as e:
                    LOGGER.debug(f"Could not fetch foodcourt name for {foodcourt_id}: {e}")
                
                return foodcourt_name, restaurant_name
        except Exception as e:
            LOGGER.debug(f"Failed to get names from local MongoDB: {e}")
        
        # Fallback to IDs if MongoDB lookup fails
        return foodcourt_id, restaurant_id
    
    def close_mongodb_connection(self, connection_type: str = "cloud"):
        """Close a MongoDB connection."""
        if connection_type in self._mongodb_clients:
            try:
                self._mongodb_clients[connection_type].close()
                del self._mongodb_clients[connection_type]
                LOGGER.debug(f"Closed MongoDB connection: {connection_type}")
            except (SystemExit, KeyboardInterrupt):
                raise  # Re-raise these
            except Exception as e:
                # Suppress warnings during Python shutdown (sys.meta_path is None)
                error_msg = str(e)
                if "sys.meta_path" in error_msg or "Python is likely shutting down" in error_msg:
                    # This is harmless - connections will be closed automatically during shutdown
                    LOGGER.debug(f"MongoDB connection cleanup skipped (Python shutting down): {connection_type}")
                else:
                    LOGGER.warning(f"Error closing MongoDB connection: {e}")
    
    def close_all_mongodb_connections(self):
        """Close all MongoDB connections."""
        for connection_type in list(self._mongodb_clients.keys()):
            self.close_mongodb_connection(connection_type)
    
    # ============================================================================
    # MySQL Connection Methods
    # ============================================================================
    
    def get_mysql_connection(self, connection_key: str = "weather_record") -> Optional[Any]:
        """
        Get or create a MySQL connection.
        
        Args:
            connection_key: Key in config for MySQL connection (default: "weather_record")
        
        Returns:
            MySQL connection object or None if connection fails
        """
        if connection_key in self._mysql_connections:
            return self._mysql_connections[connection_key]
        
        try:
            import importlib
            mysql_connector = importlib.import_module("mysql.connector")
        except ImportError:
            LOGGER.warning("mysql-connector-python not installed; MySQL connections disabled.")
            return None
        
        try:
            mysql_cfg = self._config.get(connection_key, {})
            if not mysql_cfg:
                LOGGER.error(f"MySQL config '{connection_key}' not found in config")
                return None
            
            conn = mysql_connector.connect(
                host=mysql_cfg.get("host", "localhost"),
                user=mysql_cfg.get("user", "root"),
                password=mysql_cfg.get("password", ""),
                database=mysql_cfg.get("db_name", ""),
                autocommit=True
            )
            
            self._mysql_connections[connection_key] = conn
            LOGGER.info(f"✅ Connected to MySQL: {mysql_cfg.get('db_name', 'unknown')}")
            return conn
            
        except Exception as e:
            LOGGER.error(f"Failed to connect to MySQL ({connection_key}): {e}")
            return None
    
    def fetch_weather_data(self, city_id: str, start_date: str, end_date: str,
                          connection_key: str = "weather_record") -> pd.DataFrame:
        """
        Fetch weather data from MySQL.
        
        Args:
            city_id: City ID
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            connection_key: MySQL connection key in config
        
        Returns:
            DataFrame with weather data
        """
        mysql_conn = self.get_mysql_connection(connection_key)
        if not mysql_conn:
            LOGGER.warning("MySQL connection unavailable; returning empty DataFrame.")
            return pd.DataFrame()
        
        try:
            table_name = f"city_{city_id}"
            query = f"""
                SELECT date, temperature, humidity, precipitation
                FROM {table_name}
                WHERE date BETWEEN %s AND %s
                ORDER BY date
            """
            
            cursor = mysql_conn.cursor(dictionary=True)
            cursor.execute(query, (start_date, end_date))
            rows = cursor.fetchall()
            cursor.close()
            
            if rows:
                df = pd.DataFrame(rows)
                df["date"] = pd.to_datetime(df["date"])
                return df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            LOGGER.warning(f"Failed to fetch weather data for city {city_id}: {e}")
            return pd.DataFrame()
    
    def close_mysql_connection(self, connection_key: str = "weather_record"):
        """Close a MySQL connection."""
        if connection_key in self._mysql_connections:
            try:
                self._mysql_connections[connection_key].close()
                del self._mysql_connections[connection_key]
                LOGGER.debug(f"Closed MySQL connection: {connection_key}")
            except Exception as e:
                LOGGER.warning(f"Error closing MySQL connection: {e}")
    
    def close_all_mysql_connections(self):
        """Close all MySQL connections."""
        for connection_key in list(self._mysql_connections.keys()):
            self.close_mysql_connection(connection_key)
    
    def initialize_all_connections(self):
        """
        Initialize all required database connections upfront.
        This ensures connections are established before processing starts.
        
        Returns:
            bool: True if all critical connections succeeded, False otherwise
        """
        LOGGER.info("Initializing database connections...")
        
        success = True
        
        # Initialize MongoDB local connection (most critical for enrichment)
        try:
            local_client = self.get_mongodb_client("local")
            if not local_client:
                LOGGER.error("Failed to initialize MongoDB local connection")
                success = False
            else:
                LOGGER.info("✅ MongoDB local connection initialized")
        except Exception as e:
            LOGGER.error(f"Error initializing MongoDB local connection: {e}")
            success = False
        
        # Initialize MySQL connection (for weather data)
        try:
            mysql_conn = self.get_mysql_connection("weather_record")
            if not mysql_conn:
                LOGGER.warning("MySQL connection not available (weather data will be skipped)")
                # Don't fail the pipeline if MySQL is unavailable
            else:
                LOGGER.info("✅ MySQL connection initialized")
        except Exception as e:
            LOGGER.warning(f"Error initializing MySQL connection: {e} (weather data will be skipped)")
            # Don't fail the pipeline if MySQL is unavailable
        
        # Initialize MongoDB cloud connection (optional, for entityrecord fallback)
        try:
            cloud_client = self.get_mongodb_client("cloud")
            if cloud_client:
                LOGGER.info("✅ MongoDB cloud connection initialized")
            # Don't fail if cloud connection is unavailable (local is primary)
        except Exception as e:
            LOGGER.debug(f"Cloud MongoDB connection not available (will use local): {e}")
        
        if success:
            LOGGER.info("✅ All critical database connections initialized successfully")
        else:
            LOGGER.warning("⚠️  Some database connections failed to initialize")
        
        return success
    
    # ============================================================================
    # File Loading Methods
    # ============================================================================
    
    def load_csv(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """
        Load a CSV file.
        
        Args:
            file_path: Path to CSV file
            **kwargs: Additional arguments to pass to pd.read_csv
        
        Returns:
            DataFrame or empty DataFrame if file doesn't exist
        """
        try:
            if not file_path.exists():
                LOGGER.warning(f"CSV file not found: {file_path}")
                return pd.DataFrame()
            
            df = pd.read_csv(file_path, **kwargs)
            LOGGER.debug(f"Loaded CSV: {file_path} ({len(df)} rows)")
            return df
            
        except Exception as e:
            LOGGER.error(f"Failed to load CSV {file_path}: {e}")
            return pd.DataFrame()
    
    def load_excel(self, file_path: Path, sheet_name: Optional[str] = None, **kwargs) -> pd.DataFrame:
        """
        Load an Excel file.
        
        Args:
            file_path: Path to Excel file
            sheet_name: Sheet name (if None, loads first sheet)
            **kwargs: Additional arguments to pass to pd.read_excel
        
        Returns:
            DataFrame or empty DataFrame if file doesn't exist
        """
        try:
            if not file_path.exists():
                LOGGER.warning(f"Excel file not found: {file_path}")
                return pd.DataFrame()
            
            df = pd.read_excel(file_path, sheet_name=sheet_name, **kwargs)
            LOGGER.debug(f"Loaded Excel: {file_path} ({len(df)} rows)")
            return df
            
        except Exception as e:
            LOGGER.error(f"Failed to load Excel {file_path}: {e}")
            return pd.DataFrame()
    
    def load_parquet(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """
        Load a Parquet file.
        
        Args:
            file_path: Path to Parquet file
            **kwargs: Additional arguments to pass to pd.read_parquet
        
        Returns:
            DataFrame or empty DataFrame if file doesn't exist
        """
        try:
            if not file_path.exists():
                LOGGER.warning(f"Parquet file not found: {file_path}")
                return pd.DataFrame()
            
            df = pd.read_parquet(file_path, **kwargs)
            LOGGER.debug(f"Loaded Parquet: {file_path} ({len(df)} rows)")
            return df
            
        except Exception as e:
            LOGGER.error(f"Failed to load Parquet {file_path}: {e}")
            return pd.DataFrame()
    
    def load_json(self, file_path: Path, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Load a JSON file.
        
        Args:
            file_path: Path to JSON file
            **kwargs: Additional arguments to pass to json.load
        
        Returns:
            Dict or None if file doesn't exist
        """
        try:
            if not file_path.exists():
                LOGGER.warning(f"JSON file not found: {file_path}")
                return None
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f, **kwargs)
            LOGGER.debug(f"Loaded JSON: {file_path}")
            return data
            
        except Exception as e:
            LOGGER.error(f"Failed to load JSON {file_path}: {e}")
            return None
    
    def load_pickle(self, file_path: Path) -> Optional[Any]:
        """
        Load a Pickle file.
        
        Args:
            file_path: Path to Pickle file
        
        Returns:
            Unpickled object or None if file doesn't exist
        """
        try:
            import pickle
            
            if not file_path.exists():
                LOGGER.warning(f"Pickle file not found: {file_path}")
                return None
            
            with open(file_path, 'rb') as f:
                obj = pickle.load(f)
            LOGGER.debug(f"Loaded Pickle: {file_path}")
            return obj
            
        except Exception as e:
            LOGGER.error(f"Failed to load Pickle {file_path}: {e}")
            return None
    
    # ============================================================================
    # Cleanup Methods
    # ============================================================================
    
    def close_all_connections(self):
        """Close all database connections."""
        try:
            self.close_all_mongodb_connections()
            self.close_all_mysql_connections()
            LOGGER.info("All database connections closed")
        except (SystemExit, KeyboardInterrupt):
            raise
        except Exception as e:
            # Suppress errors during Python shutdown - connections will be closed automatically
            error_msg = str(e)
            if "sys.meta_path" not in error_msg and "Python is likely shutting down" not in error_msg:
                LOGGER.debug(f"Error during connection cleanup (Python shutting down): {e}")
    
    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.close_all_connections()
        except (SystemExit, KeyboardInterrupt):
            raise
        except Exception:
            # Suppress errors during Python shutdown - connections will be closed automatically
            # sys.meta_path being None is normal during shutdown
            pass


# Global instance
_connection_manager: Optional[ConnectionManager] = None


def get_connection_manager() -> ConnectionManager:
    """Get the global ConnectionManager instance."""
    global _connection_manager
    if _connection_manager is None:
        _connection_manager = ConnectionManager()
    return _connection_manager

