#!/usr/bin/env python3
"""
OpenWearables Startup Script

A clean entry point that avoids module import conflicts.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

def main(mock=None, port=None, debug=None, host=None):
    """Main entry point for OpenWearables."""
    
    # Set up basic logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger("OpenWearables.Startup")
    
    # If called directly, parse arguments
    if mock is None or port is None:
        parser = argparse.ArgumentParser(description='OpenWearables Platform')
        parser.add_argument('--port', type=int, default=5000, help='Port to run on')
        parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
        parser.add_argument('--mock', action='store_true', help='Run in mock mode')
        parser.add_argument('--debug', action='store_true', help='Enable debug mode')
        
        args = parser.parse_args()
        
        mock = args.mock
        port = args.port
        debug = args.debug
        host = args.host
    
    # Set defaults if not provided
    port = port or 5000
    host = host or '0.0.0.0'
    debug = debug or False
    mock = mock or False
    
    # Create necessary directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs('config', exist_ok=True)
    
    # Set environment variables
    os.environ['OPENWEARABLES_MOCK'] = 'true' if mock else 'false'
    
    # Import and run the app
    try:
        from openwearables.ui.app import app, socketio
        
        logger.info(f"Starting OpenWearables on {host}:{port} (mock: {mock})")
        
        if debug:
            socketio.run(app, host=host, port=port, debug=True)
        else:
            socketio.run(app, host=host, port=port, debug=False)
            
    except OSError as e:
        if "Address already in use" in str(e):
            logger.error(f"Port {port} is already in use.")
            logger.info("Try a different port with --port <number>")
            logger.info("On macOS, disable AirPlay Receiver in System Preferences > Sharing")
        else:
            logger.error(f"Server error: {e}")
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error(f"Error: {e}")

if __name__ == '__main__':
    main() 