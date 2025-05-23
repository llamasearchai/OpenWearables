# OpenWearables UI

This directory contains the web interface for the OpenWearables platform, a comprehensive wearable AI system for health monitoring and analysis.

## Features

- Real-time vital signs monitoring (ECG, PPG, temperature)
- Heart rate variability (HRV) analysis
- Activity recognition and tracking
- Health analytics and insights
- Stress level monitoring
- Interactive data visualization
- System configuration and settings
- Privacy and security controls

## Dependencies

- Flask (web framework)
- Plotly.js (data visualization)
- jQuery (DOM manipulation)
- OpenWearables core package (or run in mock mode for UI development)

## Running the UI

### With Core Integration

To run with full OpenWearables core integration:

```bash
python app.py
```

The UI will automatically detect and initialize the core components.

### Mock Mode (Development)

For UI development without the full system:

```bash
OPENWEARABLES_MOCK=true python app.py
```

This enables mock data mode for frontend development.

### Environment Variables

- `OPENWEARABLES_CONFIG`: Path to configuration file (default: `config/default.json`)
- `OPENWEARABLES_MOCK`: Set to `true` to run in mock mode without core integration
- `FLASK_DEBUG`: Set to `true` for development mode with auto-reload
- `PORT`: Port to run the server on (default: 5000)

## File Structure

```
ui/
├── app.py                 # Main Flask application
├── templates/
│   ├── base.html         # Base template with navigation
│   ├── index.html        # Main dashboard
│   ├── analysis.html     # Detailed analysis page
│   ├── settings.html     # Configuration page
│   └── error.html        # Error page template
├── static/
│   ├── css/
│   │   ├── reset.css     # CSS reset
│   │   ├── variables.css # CSS variables
│   │   ├── layout.css    # Layout styles
│   │   └── components.css # Component styles
│   ├── js/
│   │   ├── dashboard.js  # Dashboard functionality
│   │   ├── analysis.js   # Analysis page scripts
│   │   └── settings.js   # Settings page scripts
│   └── img/              # Images and icons
└── README.md            # This file
```

## Development Mode

For UI development without the core system, run with `OPENWEARABLES_MOCK=true`. This will simulate data and system behavior.

The mock mode provides:
- Simulated real-time vital signs data
- Mock analysis results
- Fake system status
- Sample health insights

## API Endpoints

The UI communicates with the following API endpoints:

- `GET /api/system/status` - System status
- `POST /start` - Start monitoring
- `POST /stop` - Stop monitoring
- `GET /api/data/latest` - Latest sensor data
- `GET /api/analysis/latest` - Latest analysis results
- `GET /api/health/summary` - Health summary data
- `POST /api/settings` - Update settings

## Dependencies

```text:openwearables/ui/requirements.txt
Flask>=2.3.0
Jinja2>=3.1.0
Werkzeug>=2.3.0
```

## Browser Support

- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

Modern browsers with ES6+ support required for full functionality.