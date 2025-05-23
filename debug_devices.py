#!/usr/bin/env python3
"""
Debug script to test the devices page specifically
"""

import sys
import traceback

try:
    print("Testing devices page...")
    from openwearables.ui.app import app
    
    with app.test_client() as client:
        response = client.get('/devices')
        print(f"Response status: {response.status_code}")
        if response.status_code != 200:
            print(f"Response data: {response.data.decode()}")
    
except Exception as e:
    print(f"Error: {e}")
    print("Full traceback:")
    traceback.print_exc() 