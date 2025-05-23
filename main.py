#!/usr/bin/env python3
"""
OpenWearables Platform - Main Entry Point

A comprehensive AI-powered wearable health monitoring platform with real-time
sensor data processing, advanced health analytics, and intelligent insights.

Usage:
    python main.py [--mock] [--port PORT] [--debug]
    
Examples:
    python main.py --mock --port 5000      # Run with mock data on port 5000
    python main.py --debug                 # Run in debug mode
    python main.py                         # Run in production mode
"""

import sys
import argparse
import logging
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from start import main as start_main

def main():
    """Main entry point for OpenWearables platform."""
    parser = argparse.ArgumentParser(
        description="OpenWearables - AI-Powered Wearable Health Monitoring Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --mock --port 5000     Start with mock data on port 5000
  %(prog)s --debug               Start in debug mode with real sensors
  %(prog)s                       Start in production mode

For more information, visit: https://github.com/llamasearchai/OpenWearables
        """
    )
    
    parser.add_argument(
        '--mock', 
        action='store_true', 
        help='Run with mock sensor data (recommended for development and demos)'
    )
    
    parser.add_argument(
        '--port', 
        type=int, 
        default=5000, 
        help='Port to run the web interface on (default: 5000)'
    )
    
    parser.add_argument(
        '--debug', 
        action='store_true', 
        help='Run in debug mode with verbose logging'
    )
    
    parser.add_argument(
        '--host', 
        default='0.0.0.0', 
        help='Host to bind to (default: 0.0.0.0)'
    )
    
    parser.add_argument(
        '--version', 
        action='version', 
        version='OpenWearables v1.0.0'
    )
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger('OpenWearables.Main')
    
    # Display startup banner
    print("=" * 60)
    print("OpenWearables Platform v1.0.0")
    print("AI-Powered Wearable Health Monitoring")
    print("=" * 60)
    print(f"Mode: {'Mock Data' if args.mock else 'Real Sensors'}")
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Debug: {'Enabled' if args.debug else 'Disabled'}")
    print("=" * 60)
    
    # Start the application
    try:
        start_main(
            mock=args.mock,
            port=args.port,
            debug=args.debug,
            host=args.host
        )
    except KeyboardInterrupt:
        logger.info("Shutdown requested by user")
        print("\nGracefully shutting down OpenWearables...")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 