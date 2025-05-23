#!/usr/bin/env python3
"""
Debug script to test app loading and identify the exact error
"""

import sys
import traceback

try:
    print("Testing imports...")
    from openwearables.ui.app import app, load_config
    print("✓ App imported successfully")
    
    print("Testing config loading...")
    config = load_config()
    print(f"✓ Config loaded: {len(config)} keys")
    
    print("Testing template rendering...")
    with app.test_client() as client:
        response = client.get('/')
        print(f"Response status: {response.status_code}")
        if response.status_code != 200:
            print(f"Response data: {response.data.decode()}")
    
except Exception as e:
    print(f"Error: {e}")
    print("Full traceback:")
    traceback.print_exc() 