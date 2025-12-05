"""
DEPRECATED: This module is deprecated. Use connection_manager.ConnectionManager instead.

For backward compatibility, ConfigManger now delegates to ConnectionManager.
New code should use: connection_manager.get_connection_manager().get_config()

This file helps in parsing config.json file in config folder.
Set the config.json path using "_file_name" variable.
Set the MongoDB uri and api keys in config file.
"""

import json
import logging
import warnings

warnings.warn(
    "config_parser.ConfigManger is deprecated. "
    "Use connection_manager.get_connection_manager().get_config() instead.",
    DeprecationWarning,
    stacklevel=2
)

class ConfigManger:
    """
    DEPRECATED: Use connection_manager.ConnectionManager instead.
    This class is kept for backward compatibility only.
    """
    def read_config(self, type='', file_name=''):    
        # Delegate to ConnectionManager for backward compatibility
        try:
            from src.util.connection_manager import get_connection_manager
            conn_mgr = get_connection_manager()
            return conn_mgr.read_config(type, file_name)
        except Exception as error:
            logging.warning(f"Failed to use ConnectionManager, falling back to old method: {error}")
            # Fallback to old method
            _app_var={}
            _file_name=''
            if  type=='config':
                _file_name= 'config/config.json'
            try:
                with open(_file_name) as config_file:
                    _app_var= json.load(config_file)
            except Exception as err:
                logging.info(err)
            return _app_var
        
    def update_config(self,type='',data={}): 
        # Delegate to ConnectionManager for backward compatibility
        try:
            from src.util.connection_manager import get_connection_manager
            conn_mgr = get_connection_manager()
            conn_mgr.update_config(type, data)
        except Exception as error:
            logging.warning(f"Failed to use ConnectionManager, falling back to old method: {error}")
            # Fallback to old method
            _file_name=''
            if  type=='global_variables':
                _file_name='config/app/global_variables.json'
            if  type=='alarms_state':
                _file_name='config/app/alarms_state.json'
            try:
                with open(_file_name,'w') as config_file:
                    config_file.write(data)
            except Exception as err:
                logging.info("----------"+str(err))