/**
 * OpenWearables Dashboard Manager
 * Handles real-time data visualization, WebSocket connectivity, and interactive features
 */

class DashboardManager {
    constructor() {
        this.socket = null;
        this.charts = new Map();
        this.dataBuffers = new Map();
        this.isMonitoring = false;
        this.refreshInterval = null;
        this.animationFrame = null;
        
        // Configuration
        this.config = {
            websocketUrl: this.getWebSocketUrl(),
            refreshRate: 1000, // 1 second
            maxDataPoints: 300, // 5 minutes at 1 second intervals
            chartUpdateInterval: 100, // 100ms for smooth animation
            reconnectInterval: 5000,
            maxReconnectAttempts: 10
        };
        
        this.reconnectAttempts = 0;
        this.lastHeartbeat = Date.now();
        
        // Initialize data buffers
        this.initializeDataBuffers();
        
        // Bind methods
        this.handleVisibilityChange = this.handleVisibilityChange.bind(this);
        this.handleResize = this.handleResize.bind(this);
        this.updateCharts = this.updateCharts.bind(this);
    }
    
    /**
     * Initialize the dashboard
     */
    async init() {
        try {
            console.log('Initializing OpenWearables Dashboard...');
            
            // Set up event listeners
            this.setupEventListeners();
            
            // Initialize WebSocket connection
            await this.initializeWebSocket();
            
            // Initialize charts
            this.initializeCharts();
            
            // Start monitoring
            this.startMonitoring();
            
            // Start heartbeat
            this.startHeartbeat();
            
            console.log('Dashboard initialized successfully');
        } catch (error) {
            console.error('Failed to initialize dashboard:', error);
            this.showError('Failed to initialize dashboard. Please refresh the page.');
        }
    }
    
    /**
     * Initialize data buffers for different metrics
     */
    initializeDataBuffers() {
        const metrics = ['heart_rate', 'spo2', 'temperature', 'hrv', 'activity'];
        
        metrics.forEach(metric => {
            this.dataBuffers.set(metric, {
                timestamps: [],
                values: [],
                maxSize: this.config.maxDataPoints
            });
        });
    }
    
    /**
     * Set up event listeners
     */
    setupEventListeners() {
        // Start/Stop monitoring buttons
        const startBtn = document.getElementById('start-monitoring');
        const stopBtn = document.getElementById('stop-monitoring');
        
        if (startBtn) {
            startBtn.addEventListener('click', () => this.startMonitoring());
        }
        
        if (stopBtn) {
            stopBtn.addEventListener('click', () => this.stopMonitoring());
        }
        
        // Export data button
        const exportBtn = document.getElementById('export-data');
        if (exportBtn) {
            exportBtn.addEventListener('click', () => this.exportData());
        }
        
        // Chart timeframe selector
        const timeframeSelect = document.getElementById('chart-timeframe');
        if (timeframeSelect) {
            timeframeSelect.addEventListener('change', (e) => {
                this.updateChartTimeframe(e.target.value);
            });
        }
        
        // Window events
        window.addEventListener('resize', this.handleResize);
        document.addEventListener('visibilitychange', this.handleVisibilityChange);
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.ctrlKey || e.metaKey) {
                switch (e.key) {
                    case 'e':
                        e.preventDefault();
                        this.exportData();
                        break;
                    case 'm':
                        e.preventDefault();
                        this.isMonitoring ? this.stopMonitoring() : this.startMonitoring();
                        break;
                }
            }
        });
    }
    
    /**
     * Initialize WebSocket connection
     */
    async initializeWebSocket() {
        return new Promise((resolve, reject) => {
            try {
                this.socket = io(this.config.websocketUrl, {
                    transports: ['websocket', 'polling'],
                    timeout: 10000,
                    forceNew: true
                });
                
                this.socket.on('connect', () => {
                    console.log('WebSocket connected successfully');
                    this.reconnectAttempts = 0;
                    this.updateConnectionStatus('connected');
                    resolve();
                });
                
                this.socket.on('disconnect', (reason) => {
                    console.warn('WebSocket disconnected:', reason);
                    this.updateConnectionStatus('disconnected');
                    this.attemptReconnect();
                });
                
                this.socket.on('connect_error', (error) => {
                    console.error('WebSocket connection error:', error);
                    this.updateConnectionStatus('error');
                    reject(error);
                });
                
                // Data event handlers
                this.socket.on('vital_signs', (data) => this.handleVitalSigns(data));
                this.socket.on('health_score', (data) => this.handleHealthScore(data));
                this.socket.on('activity_data', (data) => this.handleActivityData(data));
                this.socket.on('alert', (data) => this.handleAlert(data));
                this.socket.on('system_status', (data) => this.handleSystemStatus(data));
                
                // Heartbeat
                this.socket.on('heartbeat', () => {
                    this.lastHeartbeat = Date.now();
                });
                
            } catch (error) {
                reject(error);
            }
        });
    }
    
    /**
     * Initialize charts
     */
    initializeCharts() {
        this.initializeRealtimeChart();
        this.initializeHealthScoreChart();
        this.initializeActivityCharts();
    }
    
    /**
     * Initialize real-time monitoring chart
     */
    initializeRealtimeChart() {
        const container = document.getElementById('realtime-chart');
        if (!container) return;
        
        const layout = {
            title: {
                text: 'Real-time Vital Signs',
                font: { size: 16, color: getComputedStyle(document.body).getPropertyValue('--text-primary') }
            },
            xaxis: {
                title: 'Time',
                type: 'date',
                range: [new Date(Date.now() - 30000), new Date()],
                color: getComputedStyle(document.body).getPropertyValue('--text-secondary')
            },
            yaxis: {
                title: 'Heart Rate (BPM)',
                color: getComputedStyle(document.body).getPropertyValue('--text-secondary')
            },
            yaxis2: {
                title: 'SpO₂ (%)',
                overlaying: 'y',
                side: 'right',
                color: getComputedStyle(document.body).getPropertyValue('--text-secondary')
            },
            plot_bgcolor: 'transparent',
            paper_bgcolor: 'transparent',
            font: { color: getComputedStyle(document.body).getPropertyValue('--text-primary') },
            margin: { l: 50, r: 50, t: 50, b: 50 },
            showlegend: true,
            legend: {
                x: 0,
                y: 1,
                bgcolor: 'rgba(0,0,0,0)',
                bordercolor: 'rgba(0,0,0,0)'
            }
        };
        
        const config = {
            responsive: true,
            displayModeBar: false,
            scrollZoom: false
        };
        
        // Initialize with empty data
        const data = [
            {
                x: [],
                y: [],
                type: 'scatter',
                mode: 'lines',
                name: 'Heart Rate',
                line: { color: '#ef4444', width: 2 },
                yaxis: 'y'
            },
            {
                x: [],
                y: [],
                type: 'scatter',
                mode: 'lines',
                name: 'SpO₂',
                line: { color: '#0ea5e9', width: 2 },
                yaxis: 'y2'
            }
        ];
        
        Plotly.newPlot(container, data, layout, config);
        this.charts.set('realtime', { container, data, layout });
    }
    
    /**
     * Initialize health score chart
     */
    initializeHealthScoreChart() {
        // Health score is updated via DOM manipulation
        // Could add a radial chart here if needed
    }
    
    /**
     * Initialize activity charts
     */
    initializeActivityCharts() {
        // Activity progress bars are updated via DOM manipulation
        // Could add detailed activity charts here if needed
    }
    
    /**
     * Start monitoring
     */
    startMonitoring() {
        if (this.isMonitoring) return;
        
        this.isMonitoring = true;
        this.updateMonitoringButtons();
        
        // Start chart updates
        this.startChartUpdates();
        
        // Emit start monitoring event
        if (this.socket && this.socket.connected) {
            this.socket.emit('start_monitoring');
        }
        
        this.showToast('Monitoring started', 'success');
        console.log('Monitoring started');
    }
    
    /**
     * Stop monitoring
     */
    stopMonitoring() {
        if (!this.isMonitoring) return;
        
        this.isMonitoring = false;
        this.updateMonitoringButtons();
        
        // Stop chart updates
        this.stopChartUpdates();
        
        // Emit stop monitoring event
        if (this.socket && this.socket.connected) {
            this.socket.emit('stop_monitoring');
        }
        
        this.showToast('Monitoring stopped', 'info');
        console.log('Monitoring stopped');
    }
    
    /**
     * Start chart update loop
     */
    startChartUpdates() {
        const updateLoop = () => {
            if (this.isMonitoring) {
                this.updateCharts();
                this.animationFrame = requestAnimationFrame(updateLoop);
            }
        };
        updateLoop();
    }
    
    /**
     * Stop chart updates
     */
    stopChartUpdates() {
        if (this.animationFrame) {
            cancelAnimationFrame(this.animationFrame);
            this.animationFrame = null;
        }
    }
    
    /**
     * Update charts with latest data
     */
    updateCharts() {
        if (!this.isMonitoring) return;
        
        const realtimeChart = this.charts.get('realtime');
        if (realtimeChart) {
            this.updateRealtimeChart(realtimeChart);
        }
    }
    
    /**
     * Update real-time chart
     */
    updateRealtimeChart(chart) {
        const now = new Date();
        const timeRange = this.getTimeRange();
        
        const hrBuffer = this.dataBuffers.get('heart_rate');
        const spo2Buffer = this.dataBuffers.get('spo2');
        
        if (hrBuffer && spo2Buffer) {
            // Filter data within time range
            const hrData = this.filterDataByTimeRange(hrBuffer, timeRange);
            const spo2Data = this.filterDataByTimeRange(spo2Buffer, timeRange);
            
            // Update chart data
            const update = {
                x: [hrData.timestamps, spo2Data.timestamps],
                y: [hrData.values, spo2Data.values]
            };
            
            // Update layout for time range
            const layoutUpdate = {
                'xaxis.range': [new Date(now - timeRange), now]
            };
            
            Plotly.restyle(chart.container, update, [0, 1]);
            Plotly.relayout(chart.container, layoutUpdate);
        }
    }
    
    /**
     * Handle vital signs data
     */
    handleVitalSigns(data) {
        console.log('Received vital signs:', data);
        
        // Add data to buffers
        const timestamp = new Date(data.timestamp || Date.now());
        
        Object.entries(data).forEach(([key, value]) => {
            if (key !== 'timestamp' && this.dataBuffers.has(key)) {
                this.addDataPoint(key, timestamp, value);
            }
        });
        
        // Update UI elements
        this.updateVitalSignsDisplay(data);
    }
    
    /**
     * Handle health score data
     */
    handleHealthScore(data) {
        console.log('Received health score:', data);
        this.updateHealthScoreDisplay(data);
    }
    
    /**
     * Handle activity data
     */
    handleActivityData(data) {
        console.log('Received activity data:', data);
        this.updateActivityDisplay(data);
    }
    
    /**
     * Handle alerts
     */
    handleAlert(data) {
        console.log('Received alert:', data);
        this.addAlert(data);
        this.updateAlertsCount();
    }
    
    /**
     * Handle system status updates
     */
    handleSystemStatus(data) {
        console.log('Received system status:', data);
        this.updateSystemStatus(data);
    }
    
    /**
     * Add data point to buffer
     */
    addDataPoint(metric, timestamp, value) {
        const buffer = this.dataBuffers.get(metric);
        if (!buffer) return;
        
        buffer.timestamps.push(timestamp);
        buffer.values.push(value);
        
        // Maintain buffer size
        if (buffer.timestamps.length > buffer.maxSize) {
            buffer.timestamps.shift();
            buffer.values.shift();
        }
    }
    
    /**
     * Update vital signs display
     */
    updateVitalSignsDisplay(data) {
        // Update heart rate
        if (data.heart_rate !== undefined) {
            this.updateElement('heart-rate-value', `${Math.round(data.heart_rate)}`);
            this.updateTrend('heart-rate-trend', data.heart_rate_trend);
        }
        
        // Update SpO2
        if (data.spo2 !== undefined) {
            this.updateElement('spo2-value', `${Math.round(data.spo2)}%`);
            this.updateTrend('spo2-trend', data.spo2_trend);
        }
        
        // Update temperature
        if (data.temperature !== undefined) {
            this.updateElement('temperature-value', `${data.temperature.toFixed(1)}°C`);
            this.updateTrend('temperature-trend', data.temperature_trend);
        }
        
        // Update HRV
        if (data.hrv !== undefined) {
            this.updateElement('hrv-value', `${Math.round(data.hrv)} ms`);
            this.updateTrend('hrv-trend', data.hrv_trend);
        }
    }
    
    /**
     * Update health score display
     */
    updateHealthScoreDisplay(data) {
        if (data.overall_score !== undefined) {
            this.updateElement('health-score-value', Math.round(data.overall_score));
            this.updateProgressBar('health-score-progress', data.overall_score);
            
            // Update score label
            const label = this.getHealthScoreLabel(data.overall_score);
            this.updateElement('health-score-label', label);
        }
        
        // Update breakdown scores
        if (data.cardiovascular !== undefined) {
            this.updateElement('cardiovascular-score', Math.round(data.cardiovascular));
        }
        
        if (data.respiratory !== undefined) {
            this.updateElement('respiratory-score', Math.round(data.respiratory));
        }
        
        if (data.activity !== undefined) {
            this.updateElement('activity-score', Math.round(data.activity));
        }
        
        if (data.stress !== undefined) {
            this.updateElement('stress-score', Math.round(data.stress));
        }
    }
    
    /**
     * Update activity display
     */
    updateActivityDisplay(data) {
        if (data.steps !== undefined) {
            this.updateElement('steps-value', this.formatNumber(data.steps));
            this.updateProgressBar('steps-progress', (data.steps / 10000) * 100);
        }
        
        if (data.calories !== undefined) {
            this.updateElement('calories-value', Math.round(data.calories));
            this.updateProgressBar('calories-progress', (data.calories / 2000) * 100);
        }
        
        if (data.distance !== undefined) {
            this.updateElement('distance-value', `${data.distance.toFixed(1)} km`);
            this.updateProgressBar('distance-progress', (data.distance / 10) * 100);
        }
        
        if (data.active_minutes !== undefined) {
            this.updateElement('active-minutes-value', Math.round(data.active_minutes));
            this.updateProgressBar('active-minutes-progress', (data.active_minutes / 150) * 100);
        }
    }
    
    /**
     * Update system status
     */
    updateSystemStatus(data) {
        const indicator = document.getElementById('system-status-indicator');
        const text = document.getElementById('system-status-text');
        
        if (indicator && text) {
            indicator.className = `status-indicator ${data.status}`;
            text.textContent = data.message || this.getStatusMessage(data.status);
        }
    }
    
    /**
     * Add alert to alerts list
     */
    addAlert(data) {
        const alertsList = document.getElementById('alerts-list');
        if (!alertsList) return;
        
        // Remove empty state if it exists
        const emptyState = alertsList.querySelector('.empty-state');
        if (emptyState) {
            emptyState.remove();
        }
        
        // Create alert element
        const alertElement = document.createElement('div');
        alertElement.className = `alert-item ${data.severity}`;
        alertElement.innerHTML = `
            <div class="alert-item-icon">
                ${this.getAlertIcon(data.severity)}
            </div>
            <div class="alert-item-content">
                <div class="alert-item-title">${data.title}</div>
                <div class="alert-item-message">${data.message}</div>
                <div class="alert-item-time">${this.formatTime(new Date())}</div>
            </div>
        `;
        
        // Add to top of list
        alertsList.insertBefore(alertElement, alertsList.firstChild);
        
        // Limit number of alerts shown
        const alerts = alertsList.querySelectorAll('.alert-item');
        if (alerts.length > 10) {
            alerts[alerts.length - 1].remove();
        }
    }
    
    /**
     * Export data
     */
    async exportData() {
        try {
            const data = this.collectExportData();
            const blob = new Blob([JSON.stringify(data, null, 2)], { 
                type: 'application/json' 
            });
            
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `openwearables-data-${this.formatDateForFilename(new Date())}.json`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
            
            this.showToast('Data exported successfully', 'success');
        } catch (error) {
            console.error('Export failed:', error);
            this.showToast('Export failed', 'error');
        }
    }
    
    /**
     * Utility methods
     */
    
    getWebSocketUrl() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        return `${protocol}//${window.location.host}`;
    }
    
    getTimeRange() {
        const timeframe = document.getElementById('chart-timeframe')?.value || '30s';
        const ranges = {
            '30s': 30 * 1000,
            '1m': 60 * 1000,
            '5m': 5 * 60 * 1000,
            '15m': 15 * 60 * 1000
        };
        return ranges[timeframe] || ranges['30s'];
    }
    
    filterDataByTimeRange(buffer, timeRange) {
        const now = Date.now();
        const cutoff = now - timeRange;
        
        const filtered = { timestamps: [], values: [] };
        
        for (let i = 0; i < buffer.timestamps.length; i++) {
            if (buffer.timestamps[i].getTime() >= cutoff) {
                filtered.timestamps.push(buffer.timestamps[i]);
                filtered.values.push(buffer.values[i]);
            }
        }
        
        return filtered;
    }
    
    updateElement(id, value) {
        const element = document.getElementById(id);
        if (element) {
            element.textContent = value;
        }
    }
    
    updateProgressBar(id, percentage) {
        const progressBar = document.getElementById(id);
        if (progressBar) {
            progressBar.style.width = `${Math.min(100, Math.max(0, percentage))}%`;
        }
    }
    
    updateTrend(id, trend) {
        const element = document.getElementById(id);
        if (!element) return;
        
        element.className = `vital-trend trend-${trend || 'stable'}`;
        
        const icons = {
            up: '<svg viewBox="0 0 24 24" width="12" height="12"><path d="M7 14l5-5 5 5z"/></svg>',
            down: '<svg viewBox="0 0 24 24" width="12" height="12"><path d="M7 10l5 5 5-5z"/></svg>',
            stable: '<svg viewBox="0 0 24 24" width="12" height="12"><path d="M7 14l5-5 5 5z"/></svg>'
        };
        
        const labels = {
            up: 'Rising',
            down: 'Falling',
            stable: 'Stable'
        };
        
        element.innerHTML = `${icons[trend] || icons.stable} ${labels[trend] || labels.stable}`;
    }
    
    updateMonitoringButtons() {
        const startBtn = document.getElementById('start-monitoring');
        const stopBtn = document.getElementById('stop-monitoring');
        
        if (startBtn && stopBtn) {
            if (this.isMonitoring) {
                startBtn.style.display = 'none';
                stopBtn.style.display = 'inline-flex';
            } else {
                startBtn.style.display = 'inline-flex';
                stopBtn.style.display = 'none';
            }
        }
    }
    
    updateConnectionStatus(status) {
        const statusText = document.getElementById('system-status-text');
        const statusIndicator = document.getElementById('system-status-indicator');
        
        if (statusText && statusIndicator) {
            const messages = {
                connected: 'Connected',
                disconnected: 'Disconnected',
                error: 'Connection Error'
            };
            
            statusText.textContent = messages[status] || 'Unknown';
            statusIndicator.className = `status-indicator ${status === 'connected' ? 'active' : 'warning'}`;
        }
    }
    
    formatNumber(num) {
        return new Intl.NumberFormat().format(num);
    }
    
    formatTime(date) {
        return new Intl.DateTimeFormat('en-US', {
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit'
        }).format(date);
    }
    
    formatDateForFilename(date) {
        return date.toISOString().slice(0, 19).replace(/:/g, '-');
    }
    
    getHealthScoreLabel(score) {
        if (score >= 90) return 'Excellent';
        if (score >= 80) return 'Very Good';
        if (score >= 70) return 'Good';
        if (score >= 60) return 'Fair';
        return 'Needs Attention';
    }
    
    getStatusMessage(status) {
        const messages = {
            active: 'System Operating Normally',
            warning: 'System Performance Warning',
            error: 'System Error Detected'
        };
        return messages[status] || 'Unknown Status';
    }
    
    getAlertIcon(severity) {
        const icons = {
            critical: '<svg viewBox="0 0 24 24" width="20" height="20"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/></svg>',
            warning: '<svg viewBox="0 0 24 24" width="20" height="20"><path d="M1 21h22L12 2 1 21zm12-3h-2v-2h2v2zm0-4h-2v-4h2v4z"/></svg>',
            info: '<svg viewBox="0 0 24 24" width="20" height="20"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/></svg>'
        };
        return icons[severity] || icons.info;
    }
    
    collectExportData() {
        const data = {
            timestamp: new Date().toISOString(),
            version: '1.0.0',
            data_buffers: {}
        };
        
        this.dataBuffers.forEach((buffer, metric) => {
            data.data_buffers[metric] = {
                timestamps: buffer.timestamps.map(t => t.toISOString()),
                values: buffer.values
            };
        });
        
        return data;
    }
    
    showToast(message, type = 'info') {
        // Use the existing toast notification system
        if (window.notifications) {
            window.notifications.show(message, type);
        } else {
            console.log(`Toast: ${message} (${type})`);
        }
    }
    
    showError(message) {
        this.showToast(message, 'error');
    }
    
    handleVisibilityChange() {
        if (document.hidden) {
            // Page is hidden, reduce update frequency
            this.config.chartUpdateInterval = 1000;
        } else {
            // Page is visible, restore normal frequency
            this.config.chartUpdateInterval = 100;
        }
    }
    
    handleResize() {
        // Debounce resize events
        clearTimeout(this.resizeTimeout);
        this.resizeTimeout = setTimeout(() => {
            this.charts.forEach((chart) => {
                if (chart.container) {
                    Plotly.Plots.resize(chart.container);
                }
            });
        }, 250);
    }
    
    startHeartbeat() {
        setInterval(() => {
            if (this.socket && this.socket.connected) {
                this.socket.emit('heartbeat');
            }
            
            // Check for stale connection
            if (Date.now() - this.lastHeartbeat > 30000) {
                console.warn('Heartbeat timeout, attempting reconnect');
                this.attemptReconnect();
            }
        }, 10000);
    }
    
    attemptReconnect() {
        if (this.reconnectAttempts >= this.config.maxReconnectAttempts) {
            console.error('Max reconnection attempts reached');
            this.showError('Connection lost. Please refresh the page.');
            return;
        }
        
        this.reconnectAttempts++;
        console.log(`Attempting reconnection ${this.reconnectAttempts}/${this.config.maxReconnectAttempts}`);
        
        setTimeout(() => {
            if (this.socket) {
                this.socket.connect();
            }
        }, this.config.reconnectInterval);
    }
    
    updateChartTimeframe(timeframe) {
        console.log(`Updating chart timeframe to: ${timeframe}`);
        // Chart will be updated on next render cycle
    }
    
    updateAlertsCount() {
        const alertsCount = document.getElementById('alerts-count');
        const notificationCount = document.getElementById('notification-count');
        const alertsList = document.getElementById('alerts-list');
        
        if (alertsList) {
            const alertItems = alertsList.querySelectorAll('.alert-item');
            const count = alertItems.length;
            
            if (alertsCount) {
                alertsCount.textContent = count;
                alertsCount.style.display = count > 0 ? 'flex' : 'none';
            }
            
            if (notificationCount) {
                notificationCount.textContent = count;
                notificationCount.classList.toggle('show', count > 0);
            }
        }
    }
    
    /**
     * Clean up resources
     */
    destroy() {
        this.stopMonitoring();
        
        if (this.socket) {
            this.socket.disconnect();
        }
        
        if (this.refreshInterval) {
            clearInterval(this.refreshInterval);
        }
        
        if (this.animationFrame) {
            cancelAnimationFrame(this.animationFrame);
        }
        
        // Remove event listeners
        window.removeEventListener('resize', this.handleResize);
        document.removeEventListener('visibilitychange', this.handleVisibilityChange);
        
        console.log('Dashboard destroyed');
    }
}

// Export for global use
window.DashboardManager = DashboardManager;