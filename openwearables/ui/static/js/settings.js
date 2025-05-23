// Settings Page JavaScript
document.addEventListener('DOMContentLoaded', function() {
    // Initialize settings page
    initializeSettingsPage();

    // Set up event listeners
    setupEventListeners();

    // Check system status
    checkSystemStatus();
});

function initializeSettingsPage() {
    console.log('Initializing settings page...');
}

function setupEventListeners() {
    // Save settings button
    const saveSettingsBtn = document.getElementById('saveSettingsBtn');
    if (saveSettingsBtn) {
        saveSettingsBtn.addEventListener('click', saveSettings);
    }
    
    // Multiple select enhancement
    const multiSelects = document.querySelectorAll('select[multiple]');
    multiSelects.forEach(select => {
        // Add click event to make multiple selects more user-friendly
        select.addEventListener('mousedown', function(e) {
            e.preventDefault();
            
            const select = this;
            const option = e.target.closest('option');
            
            if (option) {
                const optionIndex = Array.from(select.options).indexOf(option);
                
                if (select.options[optionIndex].selected) {
                    select.options[optionIndex].selected = false;
                } else {
                    select.options[optionIndex].selected = true;
                }
                
                const event = new Event('change');
                select.dispatchEvent(event);
            }
            
            return false;
        });
    });
}

function checkSystemStatus() {
    // Get system status from API
    fetch('/api/system/status')
        .then(response => response.json())
        .then(data => {
            updateSystemStatus(data);
        })
        .catch(error => {
            console.error('Error checking system status:', error);
        });
}

function updateSystemStatus(status) {
    // Update status indicator in sidebar
    const statusIndicator = document.querySelector('#systemStatusIndicator .status-indicator');
    const statusValue = document.querySelector('#systemStatusIndicator .status-value');
    
    if (statusIndicator && statusValue) {
        if (status.running) {
            statusIndicator.classList.add('active');
            statusValue.textContent = 'Running';
        } else {
            statusIndicator.classList.remove('active');
            statusValue.textContent = 'Stopped';
        }
    }
}

function saveSettings() {
    // Collect settings from forms
    const systemSettings = collectSystemSettings();
    const userProfile = collectUserProfile();
    
    // Combine settings
    const settings = {
        system: systemSettings,
        user_profile: userProfile
    };
    
    // Send settings to API
    fetch('/api/settings', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(settings)
    })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                showNotification('Settings saved successfully', 'success');
                
                if (data.message) {
                    showNotification(data.message, 'info');
                }
            } else {
                showNotification('Failed to save settings: ' + (data.error || 'Unknown error'), 'error');
            }
        })
        .catch(error => {
            console.error('Error saving settings:', error);
            showNotification('Error saving settings. See console for details.', 'error');
        });
}

function collectSystemSettings() {
    return {
        device_name: document.getElementById('deviceName')?.value,
        use_mlx: document.getElementById('useMlx')?.checked,
        logging_level: document.getElementById('loggingLevel')?.value,
        sensors: collectEnabledSensors(),
        sampling_rate: parseInt(document.getElementById('samplingRate')?.value || '250'),
        processing: {
            window_size: parseInt(document.getElementById('windowSize')?.value || '10'),
            overlap: parseInt(document.getElementById('windowOverlap')?.value || '50') / 100,
            features: collectSelectedFeatures()
        },
        privacy: {
            encryption: document.getElementById('enableEncryption')?.checked,
            anonymization: document.getElementById('enableAnonymization')?.checked,
            data_retention: parseInt(document.getElementById('dataRetention')?.value || '90')
        }
    };
}

function collectUserProfile() {
    return {
        name: document.getElementById('userName')?.value,
        age: document.getElementById('userAge')?.value ? parseInt(document.getElementById('userAge').value) : null,
        gender: document.getElementById('userGender')?.value,
        height: document.getElementById('userHeight')?.value ? parseFloat(document.getElementById('userHeight').value) : null,
        weight: document.getElementById('userWeight')?.value ? parseFloat(document.getElementById('userWeight').value) : null,
        medical_conditions: document.getElementById('medicalConditions')?.value,
        medications: document.getElementById('medications')?.value
    };
}

function collectEnabledSensors() {
    const sensors = [];
    
    if (document.getElementById('sensorEcg')?.checked) sensors.push('ecg');
    if (document.getElementById('sensorPpg')?.checked) sensors.push('ppg');
    if (document.getElementById('sensorAccel')?.checked) sensors.push('accelerometer');
    if (document.getElementById('sensorGyro')?.checked) sensors.push('gyroscope');
    if (document.getElementById('sensorTemp')?.checked) sensors.push('temperature');
    
    return sensors;
}

function collectSelectedFeatures() {
    const featureTypes = document.getElementById('featureTypes');
    if (!featureTypes) return ['time_domain'];
    
    const selectedFeatures = [];
    for (let i = 0; i < featureTypes.options.length; i++) {
        if (featureTypes.options[i].selected) {
            selectedFeatures.push(featureTypes.options[i].value);
        }
    }
    
    return selectedFeatures.length > 0 ? selectedFeatures : ['time_domain'];
}

function showNotification(message, type = 'info') {
    // Create notification element if it doesn't exist
    let notification = document.getElementById('notification');
    if (!notification) {
        notification = document.createElement('div');
        notification.id = 'notification';
        notification.className = 'notification';
        document.body.appendChild(notification);
        
        // Add styles inline if not already in CSS
        notification.style.position = 'fixed';
        notification.style.top = '20px';
        notification.style.right = '20px';
        notification.style.padding = '12px 20px';
        notification.style.borderRadius = '8px';
        notification.style.zIndex = '1000';
        notification.style.boxShadow = '0 4px 12px rgba(0, 0, 0, 0.15)';
        notification.style.transition = 'opacity 0.3s ease';
    }
    
    // Set type-specific styles
    switch (type) {
        case 'success':
            notification.style.backgroundColor = 'rgba(52, 199, 89, 0.9)';
            break;
        case 'error':
            notification.style.backgroundColor = 'rgba(255, 69, 58, 0.9)';
            break;
        case 'warning':
            notification.style.backgroundColor = 'rgba(255, 204, 0, 0.9)';
            notification.style.color = '#000';
            break;
        default: // info
            notification.style.backgroundColor = 'rgba(0, 122, 255, 0.9)';
    }
    
    // Set message
    notification.textContent = message;
    
    // Show notification
    notification.style.opacity = '1';
    
    // Hide after 3 seconds
    setTimeout(() => {
        notification.style.opacity = '0';
        setTimeout(() => {
            notification.remove();
        }, 300);
    }, 3000);
}