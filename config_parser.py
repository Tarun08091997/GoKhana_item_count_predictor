"""
This file helps in parsing config.json file in config folder.
Set the config.json path using "_file_name" variable.
Set the MongoDB uri and api keys in config file.
"""

import json
import logging

class ConfigManger:
    def read_config(self, type='', file_name=''):    
        _app_var={}
        _file_name=''
        if  type=='config':
            _file_name= 'config/config.json'
        try:
            with open(_file_name) as config_file:
                _app_var= json.load(config_file)
        except Exception as error:
            logging.info(error)
        return _app_var
        
    def update_config(self,type='',data={}): 
        _file_name=''
        if  type=='global_variables':
            _file_name='config/app/global_variables.json'
        if  type=='alarms_state':
            _file_name='config/app/alarms_state.json'
        try:
            with open(_file_name,'w') as config_file:
                config_file.write(data)
        except Exception as error:
            logging.info("----------"+str(error))