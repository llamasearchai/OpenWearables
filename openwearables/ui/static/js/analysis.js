// Analysis Page JavaScript
document.addEventListener('DOMContentLoaded', function() {
    // Initialize analysis page
    initializeAnalysisPage();

    // Set up event listeners
    setupEventListeners();

    // Check system status and start data loop if running
    checkSystemStatus();
});

// Global variables
let isSystemRunning = false;
let updateInterval = null;
let selectedTimeRange = '24h';
let anomaliesData = [];

function initializeAnalysisPage() {
    console.log('Initializing analysis page...');
    
    // Create empty charts
    createEmptyCharts();
}

function setupEventListeners() {
    // Time range selector
    const timeRange = document.getElementById('timeRange');
    if (timeRange) {
        timeRange.addEventListener('change', function() {
            selectedTimeRange = this.value;
            loadAnalysisData();
        });
    }
    
    // Anomaly filter
    const anomalyFilter = document.getElementById('anomalyFilter');
    if (anomalyFilter) {
        anomalyFilter.addEventListener('change', function() {
            filterAnomalies(this.value);
        });
    }
}

function checkSystemStatus() {
    // Get system status from API
    fetch('/api/system/status')
        .then(response => response.json())
        .then(data => {
            updateSystemStatus(data);
            
            // Load initial data
            loadAnalysisData();
            
            // If system is running, start update loop
            if (data.running) {
                startDataUpdateLoop();
            }
        })
        .catch(error => {
            console.error('Error checking system status:', error);
        });
}

function updateSystemStatus(status) {
    isSystemRunning = status.running;
    
    // Update status indicator in sidebar
    const statusIndicator = document.querySelector('#systemStatusIndicator .status-indicator');
    const statusValue = document.querySelector('#systemStatusIndicator .status-value');
    
    if (statusIndicator && statusValue) {
        if (isSystemRunning) {
            statusIndicator.classList.add('active');
            statusValue.textContent = 'Running';
        } else {
            statusIndicator.classList.remove('active');
            statusValue.textContent = 'Stopped';
        }
    }
}

function startDataUpdateLoop() {
    // Clear existing interval if any
    if (updateInterval) {
        clearInterval(updateInterval);
    }
    
    // Set interval for regular updates (less frequent for analysis page)
    updateInterval = setInterval(loadAnalysisData, 30000); // 30 seconds
}

function stopDataUpdateLoop() {
    if (updateInterval) {
        clearInterval(updateInterval);
        updateInterval = null;
    }
}

function loadAnalysisData() {
    // Convert time range to days parameter
    let days = 1;
    switch (selectedTimeRange) {
        case '7d': days = 7; break;
        case '30d': days = 30; break;
        case 'custom': days = 30; break; // Default for custom range
        default: days = 1; // 24h
    }
    
    // Fetch health summary data
    fetch(`/api/health/summary?days=${days}`)
        .then(response => response.json())
        .then(data => {
            updateAnalysisCharts(data);
            updateAnalysisSummaries(data);
            updateAnomalies(data.anomalies || []);
            updateHealthAssessment(data.assessment || {});
        })
        .catch(error => {
            console.error('Error loading analysis data:', error);
        });
}

function updateAnalysisCharts(data) {
    // Update heart rate chart
    updateHeartRateChart(data.heart_rate || []);
    
    // Update HRV chart
    updateHrvChart(data.hrv || []);
    
    // Update SpO2 chart
    updateSpo2Chart(data.spo2 || []);
    
    // Update activity charts
    updateActivityCharts(data.activity || {});
}

function updateAnalysisSummaries(data) {
    // Update cardiac summary
    const cardiacSummary = document.getElementById('cardiacSummary');
    if (cardiacSummary && data.summaries && data.summaries.cardiac) {
        cardiacSummary.innerHTML = data.summaries.cardiac;
    }
    
    // Update respiratory summary
    const respiratorySummary = document.getElementById('respiratorySummary');
    if (respiratorySummary && data.summaries && data.summaries.respiratory) {
        respiratorySummary.innerHTML = data.summaries.respiratory;
    }
    
    // Update activity summary
    const activitySummary = document.getElementById('activitySummary');
    if (activitySummary && data.summaries && data.summaries.activity) {
        activitySummary.innerHTML = data.summaries.activity;
    }
}

function updateAnomalies(anomalies) {
    anomaliesData = anomalies;
    filterAnomalies(document.getElementById('anomalyFilter')?.value || 'all');
}

function filterAnomalies(filterType) {
    const anomaliesContainer = document.getElementById('anomaliesContainer');
    if (!anomaliesContainer) return;
    
    if (!anomaliesData || anomaliesData.length === 0) {
        anomaliesContainer.innerHTML = `
            <div class="no-anomalies">
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="48" height="48">
                    <path fill="none" d="M0 0h24v24H0z"/>
                    <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z" fill="currentColor"/>
                </svg>
                <p>No anomalies detected in the selected time range.</p>
            </div>
        `;
        return;
    }
    
    // Filter anomalies based on type
    let filteredAnomalies = anomaliesData;
    if (filterType !== 'all') {
        filteredAnomalies = anomaliesData.filter(a => a.type === filterType);
    }
    
    if (filteredAnomalies.length === 0) {
        anomaliesContainer.innerHTML = `
            <div class="no-anomalies">
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="48" height="48">
                    <path fill="none" d="M0 0h24v24H0z"/>
                    <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z" fill="currentColor"/>
                </svg>
                <p>No ${filterType} anomalies detected in the selected time range.</p>
            </div>
        `;
        return;
    }
    
    // Create anomalies HTML
    let anomaliesHTML = '<div class="anomalies-list">';
    
    filteredAnomalies.forEach(anomaly => {
        const severityClass = anomaly.severity === 'high' ? 'high-severity' : 
                             anomaly.severity === 'medium' ? 'medium-severity' : 'low-severity';
        
        const date = new Date(anomaly.timestamp * 1000);
        const formattedDate = date.toLocaleString();
        
        anomaliesHTML += `
            <div class="anomaly-item ${severityClass}">
                <div class="anomaly-header">
                    <span class="anomaly-type">${capitalizeFirstLetter(anomaly.type)}</span>
                    <span class="anomaly-time">${formattedDate}</span>
                </div>
                <div class="anomaly-content">
                    <h3>${anomaly.title}</h3>
                    <p>${anomaly.description}</p>
                    ${anomaly.recommendation ? `<p class="anomaly-recommendation"><strong>Recommendation:</strong> ${anomaly.recommendation}</p>` : ''}
                </div>
                <div class="anomaly-footer">
                    <span class="severity-badge ${severityClass}">${capitalizeFirstLetter(anomaly.severity)} Severity</span>
                </div>
            </div>
        `;
    });
    
    anomaliesHTML += '</div>';
    anomaliesContainer.innerHTML = anomaliesHTML;
}

function updateHealthAssessment(assessment) {
    const healthAssessmentContainer = document.getElementById('healthAssessmentContainer');
    if (!healthAssessmentContainer) return;
    
    if (!assessment || Object.keys(assessment).length === 0) {
        healthAssessmentContainer.innerHTML = `
            <div class="loading-assessment">
                <div class="spinner"></div>
                <p>Generating comprehensive health assessment...</p>
            </div>
        `;
        return;
    }
    
    // Create assessment HTML
    let assessmentHTML = `
        <div class="health-assessment">
            <div class="assessment-header">
                <div class="overall-status ${assessment.overall_status?.toLowerCase() || 'unknown'}-status">
                    <h3>Overall Health Status: ${assessment.overall_status || 'Unknown'}</h3>
                    <p>${assessment.summary || ''}</p>
                </div>
                <div class="assessment-score">
                    <div class="score-circle">
                        <span class="score-value">${assessment.score || 0}</span>
                    </div>
                    <span class="score-label">Health Score</span>
                </div>
            </div>
            
            <div class="assessment-sections">
    `;
    
    // Add cardiac section
    if (assessment.cardiac) {
        assessmentHTML += `
            <div class="assessment-section">
                <h3>Cardiac Health</h3>
                <div class="section-status ${assessment.cardiac.status?.toLowerCase() || 'unknown'}-status">
                    <span>${assessment.cardiac.status || 'Unknown'}</span>
                </div>
                <p>${assessment.cardiac.summary || ''}</p>
                <ul>
                    ${assessment.cardiac.details?.map(detail => `<li>${detail}</li>`).join('') || ''}
                </ul>
            </div>
        `;
    }
    
    // Add respiratory section
    if (assessment.respiratory) {
        assessmentHTML += `
            <div class="assessment-section">
                <h3>Respiratory Health</h3>
                <div class="section-status ${assessment.respiratory.status?.toLowerCase() || 'unknown'}-status">
                    <span>${assessment.respiratory.status || 'Unknown'}</span>
                </div>
                <p>${assessment.respiratory.summary || ''}</p>
                <ul>
                    ${assessment.respiratory.details?.map(detail => `<li>${detail}</li>`).join('') || ''}
                </ul>
            </div>
        `;
    }
    
    // Add activity section
    if (assessment.activity) {
        assessmentHTML += `
            <div class="assessment-section">
                <h3>Physical Activity</h3>
                <div class="section-status ${assessment.activity.status?.toLowerCase() || 'unknown'}-status">
                    <span>${assessment.activity.status || 'Unknown'}</span>
                </div>
                <p>${assessment.activity.summary || ''}</p>
                <ul>
                    ${assessment.activity.details?.map(detail => `<li>${detail}</li>`).join('') || ''}
                </ul>
            </div>
        `;
    }
    
    // Add recommendations
    if (assessment.recommendations && assessment.recommendations.length > 0) {
        assessmentHTML += `
            <div class="assessment-section recommendations">
                <h3>Recommendations</h3>
                <ul class="recommendation-list">
                    ${assessment.recommendations.map(rec => `
                        <li class="recommendation-item">
                            <div class="recommendation-header">
                                <span class="recommendation-title">${rec.title}</span>
                                <span class="priority-badge ${rec.priority}-priority">${capitalizeFirstLetter(rec.priority || 'normal')}</span>
                            </div>
                            <p>${rec.description}</p>
                        </li>
                    `).join('')}
                </ul>
            </div>
        `;
    }
    
    assessmentHTML += `
            </div>
        </div>
    `;
    
    healthAssessmentContainer.innerHTML = assessmentHTML;
}

function createEmptyCharts() {
    // Create heart rate chart
    const heartRateChart = document.getElementById('heartRateChart');
    if (heartRateChart) {
        Plotly.newPlot(heartRateChart, [
            {
                x: [],
                y: [],
                type: 'scatter',
                mode: 'lines',
                name: 'Heart Rate',
                line: {
                    color: 'rgba(255, 69, 58, 1)',
                    width: 2
                }
            }
        ], {
            title: 'Heart Rate Trend',
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: { color: 'white' },
            margin: { t: 40, r: 10, b: 40, l: 50 },
            xaxis: {
                showgrid: true,
                gridcolor: 'rgba(255, 255, 255, 0.1)',
                gridwidth: 1,
                title: 'Time'
            },
            yaxis: {
                showgrid: true,
                gridcolor: 'rgba(255, 255, 255, 0.1)',
                gridwidth: 1,
                title: 'Heart Rate (bpm)',
                range: [40, 160]
            }
        }, {
            responsive: true
        });
    }
    
    // Create HRV chart
    const hrvChart = document.getElementById('hrvChart');
    if (hrvChart) {
        Plotly.newPlot(hrvChart, [
            {
                x: [],
                y: [],
                type: 'scatter',
                mode: 'lines',
                name: 'SDNN',
                line: {
                    color: 'rgba(94, 92, 230, 1)',
                    width: 2
                }
            },
            {
                x: [],
                y: [],
                type: 'scatter',
                mode: 'lines',
                name: 'RMSSD',
                line: {
                    color: 'rgba(255, 159, 10, 1)',
                    width: 2
                }
            }
        ], {
            title: 'Heart Rate Variability',
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: { color: 'white' },
            margin: { t: 40, r: 10, b: 40, l: 50 },
            xaxis: {
                showgrid: true,
                gridcolor: 'rgba(255, 255, 255, 0.1)',
                gridwidth: 1,
                title: 'Time'
            },
            yaxis: {
                showgrid: true,
                gridcolor: 'rgba(255, 255, 255, 0.1)',
                gridwidth: 1,
                title: 'HRV (ms)'
            },
            legend: {
                orientation: 'h',
                y: 1.1
            }
        }, {
            responsive: true
        });
    }
    
    // Create SpO2 chart
    const spo2TrendChart = document.getElementById('spo2TrendChart');
    if (spo2TrendChart) {
        Plotly.newPlot(spo2TrendChart, [
            {
                x: [],
                y: [],
                type: 'scatter',
                mode: 'lines',
                name: 'SpO2',
                line: {
                    color: 'rgba(52, 199, 89, 1)',
                    width: 2
                }
            }
        ], {
            title: 'Blood Oxygen Saturation (SpO2)',
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: { color: 'white' },
            margin: { t: 40, r: 10, b: 40, l: 50 },
            xaxis: {
                showgrid: true,
                gridcolor: 'rgba(255, 255, 255, 0.1)',
                gridwidth: 1,
                title: 'Time'
            },
            yaxis: {
                showgrid: true,
                gridcolor: 'rgba(255, 255, 255, 0.1)',
                gridwidth: 1,
                title: 'SpO2 (%)',
                range: [85, 100]
            },
            shapes: [
                {
                    type: 'rect',
                    xref: 'paper',
                    yref: 'y',
                    x0: 0,
                    y0: 95,
                    x1: 1,
                    y1: 100,
                    fillcolor: 'rgba(52, 199, 89, 0.1)',
                    opacity: 0.5,
                    line: { width: 0 }
                },
                {
                    type: 'rect',
                    xref: 'paper',
                    yref: 'y',
                    x0: 0,
                    y0: 90,
                    x1: 1,
                    y1: 95,
                    fillcolor: 'rgba(255, 204, 0, 0.1)',
                    opacity: 0.5,
                    line: { width: 0 }
                },
                {
                    type: 'rect',
                    xref: 'paper',
                    yref: 'y',
                    x0: 0,
                    y0: 0,
                    x1: 1,
                    y1: 90,
                    fillcolor: 'rgba(255, 69, 58, 0.1)',
                    opacity: 0.5,
                    line: { width: 0 }
                }
            ]
        }, {
            responsive: true
        });
    }
    
    // Create activity distribution chart
    const activityDistributionChart = document.getElementById('activityDistributionChart');
    if (activityDistributionChart) {
        Plotly.newPlot(activityDistributionChart, [
            {
                values: [25, 25, 25, 25],
                labels: ['Resting', 'Walking', 'Running', 'Other'],
                type: 'pie',
                textinfo: 'percent',
                textposition: 'inside',
                marker: {
                    colors: [
                        'rgba(175, 82, 222, 0.8)',
                        'rgba(90, 200, 250, 0.8)',
                        'rgba(255, 149, 0, 0.8)',
                        'rgba(142, 142, 147, 0.8)'
                    ]
                }
            }
        ], {
            title: 'Activity Distribution',
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: { color: 'white' },
            margin: { t: 40, r: 10, b: 10, l: 10 }
        }, {
            responsive: true
        });
    }
    
    // Create activity trend chart
    const activityTrendChart = document.getElementById('activityTrendChart');
    if (activityTrendChart) {
        Plotly.newPlot(activityTrendChart, [
            {
                x: [],
                y: [],
                type: 'bar',
                name: 'Activity Minutes',
                marker: {
                    color: 'rgba(90, 200, 250, 0.8)'
                }
            }
        ], {
            title: 'Daily Activity Duration',
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: { color: 'white' },
            margin: { t: 40, r: 10, b: 40, l: 50 },
            xaxis: {
                showgrid: false,
                title: 'Date'
            },
            yaxis: {
                showgrid: true,
                gridcolor: 'rgba(255, 255, 255, 0.1)',
                gridwidth: 1,
                title: 'Minutes'
            }
        }, {
            responsive: true
        });
    }
}

function updateHeartRateChart(heartRateData) {
    const heartRateChart = document.getElementById('heartRateChart');
    if (!heartRateChart || !heartRateData || !heartRateData.length) return;
    
    const timestamps = heartRateData.map(d => new Date(d.timestamp * 1000));
    const values = heartRateData.map(d => d.value);
    
    Plotly.update(heartRateChart, {
        x: [timestamps],
        y: [values]
    });
}

function updateHrvChart(hrvData) {
    const hrvChart = document.getElementById('hrvChart');
    if (!hrvChart || !hrvData || !hrvData.length) return;
    
    const timestamps = hrvData.map(d => new Date(d.timestamp * 1000));
    const sdnnValues = hrvData.map(d => d.SDNN);
    const rmssdValues = hrvData.map(d => d.RMSSD);
    
    Plotly.update(hrvChart, {
        x: [timestamps, timestamps],
        y: [sdnnValues, rmssdValues]
    });
}

function updateSpo2Chart(spo2Data) {
    const spo2TrendChart = document.getElementById('spo2TrendChart');
    if (!spo2TrendChart || !spo2Data || !spo2Data.length) return;
    
    const timestamps = spo2Data.map(d => new Date(d.timestamp * 1000));
    const values = spo2Data.map(d => d.value);
    
    Plotly.update(spo2TrendChart, {
        x: [timestamps],
        y: [values]
    });
}

function updateActivityCharts(activityData) {
    // Update activity distribution chart
    const activityDistributionChart = document.getElementById('activityDistributionChart');
    if (activityDistributionChart && activityData.distribution) {
        const labels = Object.keys(activityData.distribution);
        const values = Object.values(activityData.distribution);
        
        Plotly.update(activityDistributionChart, {
            labels: [labels],
            values: [values]
        });
    }
    
    // Update activity trend chart
    const activityTrendChart = document.getElementById('activityTrendChart');
    if (activityTrendChart && activityData.daily) {
        const dates = activityData.daily.map(d => new Date(d.date));
        const minutes = activityData.daily.map(d => d.minutes);
        
        Plotly.update(activityTrendChart, {
            x: [dates],
            y: [minutes]
        });
    }
}

// Helper functions
function capitalizeFirstLetter(string) {
    if (!string) return '';
    return string.charAt(0).toUpperCase() + string.slice(1);
}