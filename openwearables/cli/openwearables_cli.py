"""
OpenWearables Command Line Interface

A comprehensive CLI tool for managing the OpenWearables health monitoring platform.
"""

import os
import sys
import json
import time
import click
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timedelta

import openwearables
from openwearables.core.architecture import OpenWearablesCore

# Configure logging for CLI
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Global core instance
core_instance: Optional[OpenWearablesCore] = None

@click.group()
@click.version_option(version=openwearables.__version__)
@click.option('--config', '-c', default=None, help='Path to configuration file')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.pass_context
def cli(ctx, config: Optional[str], verbose: bool):
    """OpenWearables CLI - AI-Powered Wearable Health Monitoring Platform"""
    global core_instance
    
    # Set up context
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose
    ctx.obj['config'] = config or openwearables.get_config_path()
    
    # Configure logging based on verbosity
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger("OpenWearables").setLevel(logging.DEBUG)

@cli.command()
@click.option('--force', is_flag=True, help='Overwrite existing configuration')
@click.pass_context
def init(ctx, force: bool):
    """Initialize OpenWearables configuration and directories."""
    config_path = ctx.obj['config']
    
    click.echo(f"Initializing OpenWearables at {config_path}")
    
    # Check if config already exists
    if os.path.exists(config_path) and not force:
        click.echo(f"Configuration already exists at {config_path}")
        click.echo("Use --force to overwrite existing configuration")
        return
    
    try:
        # Create directories
        data_dir = openwearables.get_data_path()
        log_dir = os.path.join(os.path.dirname(data_dir), "logs")
        config_dir = os.path.dirname(config_path)
        
        for directory in [data_dir, log_dir, config_dir]:
            os.makedirs(directory, exist_ok=True)
            click.echo(f"Created directory: {directory}")
        
        # Create default configuration
        default_config = {
            "database": {"path": os.path.join(data_dir, "wearables.db")},
            "sensors": ["ecg", "ppg", "accelerometer", "gyroscope", "temperature"],
            "sampling_rates": {
                "ecg": 250,
                "ppg": 100,
                "accelerometer": 50,
                "gyroscope": 50,
                "temperature": 1
            },
            "models": {
                "arrhythmia_detection": "openwearables/arrhythmia-detection",
                "stress_analysis": "openwearables/stress-analysis",
                "activity_recognition": "openwearables/activity-recognition",
                "health_assessment": "microsoft/DialoGPT-medium"
            },
            "processing": {
                "window_size": 10,
                "overlap": 0.5,
                "features": ["time_domain", "frequency_domain", "wavelet"]
            },
            "privacy": {
                "encryption": True,
                "anonymization": True,
                "data_retention": 90
            },
            "logging": {
                "level": "INFO",
                "file": os.path.join(log_dir, "openwearables.log")
            },
            "user_profile": {
                "name": "",
                "age": None,
                "gender": "",
                "height": None,
                "weight": None,
                "medical_conditions": "",
                "medications": ""
            }
        }
        
        # Save configuration
        with open(config_path, 'w') as f:
            json.dump(default_config, f, indent=2)
        
        click.echo(f"Created configuration file: {config_path}")
        click.echo("OpenWearables initialized successfully!")
        click.echo("Run 'openwearables start' to begin monitoring")
        
    except Exception as e:
        click.echo(f"Error initializing OpenWearables: {e}", err=True)
        sys.exit(1)

@cli.command()
@click.option('--daemon', '-d', is_flag=True, help='Run in daemon mode')
@click.option('--port', '-p', default=5000, help='Web interface port')
@click.pass_context
def start(ctx, daemon: bool, port: int):
    """Start the OpenWearables monitoring system."""
    global core_instance
    
    config_path = ctx.obj['config']
    
    if not os.path.exists(config_path):
        click.echo("Configuration not found. Run 'openwearables init' first.")
        sys.exit(1)
    
    try:
        click.echo("Starting OpenWearables monitoring system...")
        
        # Initialize core
        core_instance = OpenWearablesCore(config_path)
        
        # Start the system
        core_instance.start()
        
        click.echo("OpenWearables started successfully!")
        click.echo(f"Web interface available at: http://localhost:{port}")
        
        if not daemon:
            click.echo("Press Ctrl+C to stop")
            try:
                # Start web interface in separate thread
                from openwearables.ui.app import app
                app.run(host='0.0.0.0', port=port, debug=False)
            except KeyboardInterrupt:
                click.echo("\nShutting down...")
                if core_instance:
                    core_instance.stop()
        else:
            click.echo("Running in daemon mode. Use 'openwearables stop' to stop.")
            
    except Exception as e:
        click.echo(f"Error starting OpenWearables: {e}", err=True)
        sys.exit(1)

@cli.command()
@click.pass_context
def stop(ctx):
    """Stop the OpenWearables monitoring system."""
    global core_instance
    
    try:
        if core_instance:
            core_instance.stop()
            click.echo("OpenWearables stopped successfully")
        else:
            click.echo("OpenWearables is not running")
            
    except Exception as e:
        click.echo(f"Error stopping OpenWearables: {e}", err=True)
        sys.exit(1)

@cli.command()
@click.pass_context
def status(ctx):
    """Show system status and health metrics."""
    global core_instance
    
    config_path = ctx.obj['config']
    
    try:
        if not core_instance:
            if os.path.exists(config_path):
                core_instance = OpenWearablesCore(config_path)
            else:
                click.echo("Configuration not found. Run 'openwearables init' first.")
                return
        
        status_info = core_instance.get_system_status()
        
        click.echo("OpenWearables System Status")
        click.echo("=" * 40)
        click.echo(f"Version: {status_info['version']}")
        click.echo(f"Status: {'Running' if status_info['running'] else 'Stopped'}")
        click.echo(f"Device: {status_info['device']}")
        click.echo(f"Sensors: {', '.join(status_info['sensors'])}")
        click.echo(f"Models: {', '.join(status_info['models'])}")
        click.echo(f"Database: {status_info['database_path']}")
        click.echo(f"Config: {status_info['config_path']}")
        
        if status_info['running']:
            # Get latest readings
            latest_readings = core_instance.get_latest_readings()
            if latest_readings:
                click.echo("\nLatest Readings:")
                click.echo("-" * 20)
                for sensor, reading in latest_readings.items():
                    click.echo(f"{sensor}: {reading}")
        
    except Exception as e:
        click.echo(f"Error getting status: {e}", err=True)
        sys.exit(1)

@cli.command()
@click.option('--sensors', '-s', help='Comma-separated list of sensors to monitor')
@click.option('--duration', '-d', default=60, help='Monitoring duration in seconds')
@click.option('--output', '-o', help='Output file for data export')
@click.pass_context
def monitor(ctx, sensors: Optional[str], duration: int, output: Optional[str]):
    """Start monitoring with specific sensors for a duration."""
    global core_instance
    
    config_path = ctx.obj['config']
    
    try:
        if not core_instance:
            core_instance = OpenWearablesCore(config_path)
        
        # Parse sensors if provided
        if sensors:
            sensor_list = [s.strip() for s in sensors.split(',')]
            click.echo(f"Monitoring sensors: {', '.join(sensor_list)}")
        else:
            sensor_list = core_instance.config.get('sensors', [])
            click.echo(f"Monitoring all configured sensors: {', '.join(sensor_list)}")
        
        # Start monitoring
        if not core_instance.is_running():
            core_instance.start()
        
        click.echo(f"Monitoring for {duration} seconds...")
        
        start_time = time.time()
        end_time = start_time + duration
        
        with click.progressbar(length=duration, label='Monitoring') as bar:
            while time.time() < end_time:
                time.sleep(1)
                bar.update(1)
        
        click.echo("Monitoring completed!")
        
        # Export data if output specified
        if output:
            exported_data = core_instance.export_data(start_time, end_time, "json")
            if exported_data:
                with open(output, 'w') as f:
                    f.write(exported_data)
                click.echo(f"Data exported to: {output}")
        
    except Exception as e:
        click.echo(f"Error during monitoring: {e}", err=True)
        sys.exit(1)

@cli.command()
@click.option('--days', '-d', default=7, help='Number of days to include in report')
@click.option('--format', '-f', type=click.Choice(['json', 'csv']), default='json', help='Output format')
@click.option('--output', '-o', help='Output file path')
@click.pass_context
def export(ctx, days: int, format: str, output: Optional[str]):
    """Export health data and analysis results."""
    global core_instance
    
    config_path = ctx.obj['config']
    
    try:
        if not core_instance:
            core_instance = OpenWearablesCore(config_path)
        
        # Calculate time range
        end_time = time.time()
        start_time = end_time - (days * 86400)
        
        click.echo(f"Exporting {days} days of data...")
        
        # Export data
        exported_data = core_instance.export_data(start_time, end_time, format)
        
        if exported_data:
            if output:
                with open(output, 'w') as f:
                    f.write(exported_data)
                click.echo(f"Data exported to: {output}")
            else:
                click.echo(exported_data)
        else:
            click.echo("No data to export")
            
    except Exception as e:
        click.echo(f"Error exporting data: {e}", err=True)
        sys.exit(1)

@cli.command()
@click.option('--days', '-d', default=7, help='Number of days for health summary')
@click.pass_context
def health(ctx, days: int):
    """Generate health summary and insights."""
    global core_instance
    
    config_path = ctx.obj['config']
    
    try:
        if not core_instance:
            core_instance = OpenWearablesCore(config_path)
        
        click.echo(f"Generating health summary for the last {days} days...")
        
        health_summary = core_instance.get_health_summary(days)
        
        click.echo("Health Summary")
        click.echo("=" * 40)
        click.echo(f"Period: {days} days")
        click.echo(f"Generated: {datetime.fromtimestamp(health_summary['generated_at']).strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Display metrics
        metrics = health_summary.get('metrics', {})
        if metrics:
            click.echo("\nHealth Metrics:")
            click.echo("-" * 20)
            for metric_type, data in metrics.items():
                click.echo(f"{metric_type}:")
                click.echo(f"  Average: {data['average']:.2f}")
                click.echo(f"  Range: {data['minimum']:.2f} - {data['maximum']:.2f}")
                click.echo(f"  Count: {data['count']}")
        
        # Display recent analyses
        analyses = health_summary.get('analyses', {})
        if analyses:
            click.echo("\nRecent Analyses:")
            click.echo("-" * 20)
            for analysis_type, data in analyses.items():
                timestamp = datetime.fromtimestamp(data['timestamp']).strftime('%Y-%m-%d %H:%M')
                click.echo(f"{analysis_type} ({timestamp}): Confidence {data['confidence']:.2f}")
        
        # Display alerts
        alerts = health_summary.get('alerts', {})
        if alerts:
            click.echo("\nAlerts:")
            click.echo("-" * 20)
            for alert_type, alert_list in alerts.items():
                for alert in alert_list:
                    latest = datetime.fromtimestamp(alert['latest']).strftime('%Y-%m-%d %H:%M')
                    click.echo(f"{alert_type}: {alert['count']} ({alert['severity']}) - Latest: {latest}")
        
        if not metrics and not analyses and not alerts:
            click.echo("No health data available for the specified period.")
            
    except Exception as e:
        click.echo(f"Error generating health summary: {e}", err=True)
        sys.exit(1)

@cli.command()
@click.option('--key', help='Configuration key to set')
@click.option('--value', help='Configuration value to set')
@click.option('--show', is_flag=True, help='Show current configuration')
@click.pass_context
def config(ctx, key: Optional[str], value: Optional[str], show: bool):
    """View or modify configuration settings."""
    config_path = ctx.obj['config']
    
    try:
        if not os.path.exists(config_path):
            click.echo("Configuration not found. Run 'openwearables init' first.")
            return
        
        # Load current config
        with open(config_path, 'r') as f:
            current_config = json.load(f)
        
        if show:
            click.echo("Current Configuration:")
            click.echo("=" * 40)
            click.echo(json.dumps(current_config, indent=2))
            return
        
        if key and value:
            # Set configuration value
            keys = key.split('.')
            config_section = current_config
            
            # Navigate to the correct section
            for k in keys[:-1]:
                if k not in config_section:
                    config_section[k] = {}
                config_section = config_section[k]
            
            # Set the value
            try:
                # Try to parse as JSON first
                config_section[keys[-1]] = json.loads(value)
            except json.JSONDecodeError:
                # If not JSON, treat as string
                config_section[keys[-1]] = value
            
            # Save updated config
            with open(config_path, 'w') as f:
                json.dump(current_config, f, indent=2)
            
            click.echo(f"Configuration updated: {key} = {value}")
            
        elif key:
            # Get configuration value
            keys = key.split('.')
            config_section = current_config
            
            try:
                for k in keys:
                    config_section = config_section[k]
                click.echo(f"{key}: {json.dumps(config_section, indent=2)}")
            except KeyError:
                click.echo(f"Configuration key not found: {key}")
        else:
            click.echo("Use --show to view configuration or --key/--value to modify")
            
    except Exception as e:
        click.echo(f"Error managing configuration: {e}", err=True)
        sys.exit(1)

@cli.command()
@click.pass_context
def version(ctx):
    """Show version information."""
    click.echo(f"OpenWearables version {openwearables.__version__}")
    click.echo(f"Author: {openwearables.__author__}")
    click.echo(f"License: {openwearables.__license__}")

def main():
    """Main CLI entry point."""
    try:
        cli()
    except KeyboardInterrupt:
        click.echo("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        click.echo(f"Unexpected error: {e}", err=True)
        sys.exit(1)

if __name__ == '__main__':
    main() 