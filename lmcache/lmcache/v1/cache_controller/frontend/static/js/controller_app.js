// Controller Dashboard JavaScript
// Global variables
let controllerBaseUrl = "";
let isConnected = false;
let currentInstances = [];
let currentWorkers = [];
let currentKeyPool = [];
let envVariablesData = null;

// Initialize after DOM is loaded
window.addEventListener('DOMContentLoaded', () => {
    // Initialize current time display
    updateCurrentTime();
    setInterval(updateCurrentTime, 1000);

    // Connect button event
    document.getElementById('connectControllerBtn').addEventListener('click', connectToController);

    // Refresh all button
    document.getElementById('refreshAllBtn').addEventListener('click', refreshAllData);

    // Tab switching event
    document.querySelectorAll('.nav-link').forEach(tab => {
        tab.addEventListener('shown.bs.tab', (event) => {
            if (isConnected) {
                const tabId = event.target.getAttribute('data-bs-target').replace('#', '');
                loadTabData(tabId);
            }
        });
    });

    // Instance management
    document.getElementById('refreshInstancesBtn').addEventListener('click', loadInstances);
    document.getElementById('saveInstanceBtn').addEventListener('click', addInstance);

    // Worker management
    document.getElementById('refreshWorkersBtn').addEventListener('click', loadWorkers);
    document.getElementById('instanceFilter').addEventListener('change', loadWorkers);

    // Metrics
    document.getElementById('refreshMetricsBtn').addEventListener('click', loadMetrics);

    // Threads
    document.getElementById('refreshThreadsBtn').addEventListener('click', loadThreads);

    // Environment
    document.getElementById('envSearchInput').addEventListener('input', filterEnvVariables);

    // Auto-connect on startup
    const urlParams = new URLSearchParams(window.location.search);
    const urlHostParam = urlParams.get('host');
    const urlPortParam = urlParams.get('port');
    
    // Get host and port from URL parameters, or use current page's hostname and port
    const autoHost = urlHostParam || window.location.hostname;
    const autoPort = urlPortParam || window.location.port || (window.location.protocol === 'https:' ? '443' : '80');
    
    // Always set input values and attempt to connect
    document.getElementById('controllerHostInput').value = autoHost;
    document.getElementById('controllerPortInput').value = autoPort;
    setTimeout(() => connectToController(), 1000);
});

// Update current time display
function updateCurrentTime() {
    const now = new Date();
    const timeString = now.toLocaleTimeString('en-US', {
        hour12: false,
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit'
    });
    document.getElementById('currentTime').textContent = timeString;
}

// Connect to Controller
async function connectToController() {
    const host = document.getElementById('controllerHostInput').value.trim();
    const port = document.getElementById('controllerPortInput').value.trim();
    
    if (!host || !port) {
        alert('Please enter controller host and port');
        return;
    }

    // Construct the base URL
    const protocol = window.location.protocol;
    controllerBaseUrl = `${protocol}//${host}:${port}`;
    const statusElement = document.getElementById('connectionStatus');
    
    try {
        statusElement.textContent = 'Connecting...';
        statusElement.className = 'badge bg-warning';
        
        // Test connection with a simple health check
        const response = await fetch(`${controllerBaseUrl}/health`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ instance_id: 'test' })
        });
        
        if (response.ok) {
            isConnected = true;
            statusElement.textContent = 'Connected';
            statusElement.className = 'badge bg-success';
            
            // Load initial data
            loadOverview();
            loadInstances();
            loadWorkers();
            
            // Update URL with connection parameters
            const newUrl = new URL(window.location);
            newUrl.searchParams.set('host', host);
            newUrl.searchParams.set('port', port);
            window.history.replaceState({}, '', newUrl);
            
        } else {
            throw new Error('Connection failed');
        }
    } catch (error) {
        console.error('Connection error:', error);
        statusElement.textContent = 'Connection Failed';
        statusElement.className = 'badge bg-danger';
        isConnected = false;
        alert('Failed to connect to controller: ' + error.message);
    }
}

// Refresh all data
async function refreshAllData() {
    if (!isConnected) {
        alert('Please connect to controller first');
        return;
    }

    const activeTab = document.querySelector('.tab-pane.active').id;
    loadTabData(activeTab);
}

// Load data for active tab
function loadTabData(tabId) {
    if (!isConnected) return;

    switch (tabId) {
        case 'overview':
            loadOverview();
            break;
        case 'instances':
            loadInstances();
            break;
        case 'workers':
            loadWorkers();
            break;
        case 'metrics':
            loadMetrics();
            break;
        case 'threads':
            loadThreads();
            break;
        case 'env':
            loadEnvironment();
            break;
    }
}

// Load overview data
async function loadOverview() {
    if (!isConnected) return;

    const systemStatusElement = document.getElementById('systemStatus');
    const quickStatsElement = document.getElementById('quickStats');
    const recentActivitiesElement = document.getElementById('recentActivities');

    try {
        // Load system status
        const healthResponse = await fetch(`${controllerBaseUrl}/health`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ instance_id: 'system' })
        });

        if (healthResponse.ok) {
            const healthData = await healthResponse.json();
            systemStatusElement.innerHTML = `
                <div class="text-success">
                    <i class="bi bi-check-circle-fill fs-1"></i>
                    <p class="mt-2">Controller is running</p>
                    <small class="text-muted">Event ID: ${healthData.event_id}</small>
                </div>
            `;
        }

        // Load quick stats (instance count, worker count, key count)
        const response = await fetch(`${controllerBaseUrl}/query_worker_info`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ instance_id: 'all', worker_ids: [] })
        });

        // Load key stats
        const keyStatsResponse = await fetch(`${controllerBaseUrl}/controller/key-stats`, {
            method: 'GET',
            headers: {'Content-Type': 'application/json'}
        });

        if (response.ok && keyStatsResponse.ok) {
            const data = await response.json();
            const keyStatsData = await keyStatsResponse.json();
            const instanceCount = new Set(data.worker_infos.map(w => w.instance_id)).size;
            const workerCount = data.worker_infos.length;
            const keyCount = keyStatsData.total_key_count;
            
            quickStatsElement.innerHTML = `
                <div class="row">
                    <div class="col-4">
                        <div class="card bg-light mb-2">
                            <div class="card-body p-2">
                                <h6 class="card-title mb-0">Instances</h6>
                                <h3 class="mb-0">${instanceCount}</h3>
                            </div>
                        </div>
                    </div>
                    <div class="col-4">
                        <div class="card bg-light mb-2">
                            <div class="card-body p-2">
                                <h6 class="card-title mb-0">Workers</h6>
                                <h3 class="mb-0">${workerCount}</h3>
                            </div>
                        </div>
                    </div>
                    <div class="col-4">
                        <div class="card bg-light mb-2">
                            <div class="card-body p-2">
                                <h6 class="card-title mb-0">Total Keys</h6>
                                <h3 class="mb-0">${keyCount}</h3>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        }

        // Recent activities (placeholder)
        recentActivitiesElement.innerHTML = `
            <div class="list-group">
                <div class="list-group-item">
                    <small class="text-muted">Just now</small>
                    <p class="mb-1">Controller dashboard loaded</p>
                </div>
                <div class="list-group-item">
                    <small class="text-muted">2 minutes ago</small>
                    <p class="mb-1">Health check performed</p>
                </div>
            </div>
        `;

    } catch (error) {
        console.error('Error loading overview:', error);
        systemStatusElement.innerHTML = `<div class="alert alert-danger">Error: ${error.message}</div>`;
    }
}

// Load instances
async function loadInstances() {
    if (!isConnected) return;

    const tableBody = document.getElementById('instancesTableBody');
    tableBody.innerHTML = '<tr><td colspan="7" class="text-center"><div class="spinner-border" role="status"></div></td></tr>';

    try {
        const response = await fetch(`${controllerBaseUrl}/query_worker_info`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ instance_id: 'all' })
        });

        // Load key stats for instances
        const keyStatsResponse = await fetch(`${controllerBaseUrl}/controller/key-stats`, {
            method: 'GET',
            headers: {'Content-Type': 'application/json'}
        });

        if (!response.ok || !keyStatsResponse.ok) {
            throw new Error('Failed to fetch instances or key stats');
        }

        const data = await response.json();
        const keyStatsData = await keyStatsResponse.json();
        currentInstances = data.worker_infos;
        
        // Create a map of instance key counts from key stats
        const instanceKeyCounts = new Map();
        keyStatsData.instances.forEach(instance => {
            instanceKeyCounts.set(instance.instance_id, instance.key_count);
        });
        
        // Group workers by instance
        const instancesMap = new Map();
        currentInstances.forEach(worker => {
            if (!instancesMap.has(worker.instance_id)) {
                instancesMap.set(worker.instance_id, {
                    instance_id: worker.instance_id,
                    ip: worker.ip,
                    workers: [],
                    last_heartbeat: worker.last_heartbeat_time,
                    key_count: instanceKeyCounts.get(worker.instance_id) || 0
                });
            }
            instancesMap.get(worker.instance_id).workers.push(worker);
            // Update latest heartbeat
            if (worker.last_heartbeat_time > instancesMap.get(worker.instance_id).last_heartbeat) {
                instancesMap.get(worker.instance_id).last_heartbeat = worker.last_heartbeat_time;
            }
        });

        // Update instance filter dropdown
        const instanceFilter = document.getElementById('instanceFilter');
        instanceFilter.innerHTML = '<option value="">All Instances</option>';
        instancesMap.forEach((instance, instanceId) => {
            const option = document.createElement('option');
            option.value = instanceId;
            option.textContent = instanceId;
            instanceFilter.appendChild(option);
        });

        // Update HTML table header to include Key Count column
        const tableHeader = document.querySelector('#instances thead tr');
        if (tableHeader && !tableHeader.innerHTML.includes('Key Count')) {
            tableHeader.innerHTML = `
                <th>Instance ID</th>
                <th>IP Address</th>
                <th>Status</th>
                <th>Worker Count</th>
                <th>Key Count</th>
                <th>Last Heartbeat</th>
                <th>Actions</th>
            `;
        }

        // Populate table
        tableBody.innerHTML = '';
        instancesMap.forEach((instance, instanceId) => {
            const row = document.createElement('tr');
            const now = Math.floor(Date.now() / 1000);
            const timeDiff = now - instance.last_heartbeat;
            const status = timeDiff < 60 ? 'Active' : timeDiff < 300 ? 'Warning' : 'Inactive';
            const statusClass = timeDiff < 60 ? 'status-active' : timeDiff < 300 ? 'status-warning' : 'status-inactive';
            
            const lastHeartbeat = new Date(instance.last_heartbeat * 1000).toLocaleTimeString();
            
            row.innerHTML = `
                <td><strong>${instanceId}</strong></td>
                <td>${instance.ip}</td>
                <td><span class="${statusClass}">${status}</span></td>
                <td>${instance.workers.length}</td>
                <td>${instance.key_count}</td>
                <td>${lastHeartbeat}</td>
                <td>
                    <button class="btn btn-sm btn-info view-instance" data-instance="${instanceId}">View</button>
                    <button class="btn btn-sm btn-danger remove-instance" data-instance="${instanceId}">Remove</button>
                </td>
            `;
            tableBody.appendChild(row);
        });

        // Add event listeners to buttons
        document.querySelectorAll('.view-instance').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const instanceId = e.target.dataset.instance;
                alert(`Viewing instance: ${instanceId}`);
                // In a real implementation, this would navigate to instance details
            });
        });

        document.querySelectorAll('.remove-instance').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const instanceId = e.target.dataset.instance;
                if (confirm(`Are you sure you want to remove instance ${instanceId}?`)) {
                    removeInstance(instanceId);
                }
            });
        });

    } catch (error) {
        console.error('Error loading instances:', error);
        tableBody.innerHTML = `<tr><td colspan="7" class="text-center text-danger">Error: ${error.message}</td></tr>`;
    }
}

// Add instance (placeholder - would need backend implementation)
async function addInstance() {
    const instanceId = document.getElementById('newInstanceId').value.trim();
    const instanceIp = document.getElementById('newInstanceIp').value.trim();

    if (!instanceId || !instanceIp) {
        alert('Please fill all fields');
        return;
    }

    // This is a placeholder - in reality, you would need a backend endpoint to add instances
    alert(`Would add instance: ${instanceId} with IP: ${instanceIp}`);
    
    // Close modal
    const modal = bootstrap.Modal.getInstance(document.getElementById('addInstanceModal'));
    modal.hide();
    
    // Clear form
    document.getElementById('newInstanceId').value = '';
    document.getElementById('newInstanceIp').value = '';
    
    // Refresh instances list
    loadInstances();
}

// Remove instance (placeholder)
async function removeInstance(instanceId) {
    // This is a placeholder - in reality, you would need a backend endpoint to remove instances
    alert(`Would remove instance: ${instanceId}`);
    loadInstances();
}

// Load workers
async function loadWorkers() {
    if (!isConnected) return;

    const tableBody = document.getElementById('workersTableBody');
    const instanceFilter = document.getElementById('instanceFilter').value;
    
    tableBody.innerHTML = '<tr><td colspan="8" class="text-center"><div class="spinner-border" role="status"></div></td></tr>';

    try {
        const requestBody = instanceFilter ? 
            { instance_id: instanceFilter, worker_ids: [] } : 
            { instance_id: 'all', worker_ids: [] };
        
        const response = await fetch(`${controllerBaseUrl}/query_worker_info`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(requestBody)
        });

        if (!response.ok) {
            throw new Error('Failed to fetch workers');
        }

        const data = await response.json();
        currentWorkers = data.worker_infos;
        
        // Get detailed worker info with key counts
        const workersWithKeyCounts = await Promise.all(
            currentWorkers.map(async (worker) => {
                try {
                    // Get detailed worker info including key count
                    const workerDetailResponse = await fetch(`${controllerBaseUrl}/controller/workers?instance_id=${worker.instance_id}&worker_id=${worker.worker_id}`, {
                        method: 'GET',
                        headers: {'Content-Type': 'application/json'}
                    });
                    
                    if (workerDetailResponse.ok) {
                        const workerDetail = await workerDetailResponse.json();
                        return {
                            ...worker,
                            key_count: workerDetail.key_count || 0
                        };
                    }
                } catch (error) {
                    console.warn(`Failed to get key count for worker ${worker.instance_id}/${worker.worker_id}:`, error);
                }
                
                // Fallback to 0 if key count is not available
                return {
                    ...worker,
                    key_count: 0
                };
            })
        );
        
        // Update HTML table header to include Key Count column
        const tableHeader = document.querySelector('#workers thead tr');
        if (tableHeader && !tableHeader.innerHTML.includes('Key Count')) {
            tableHeader.innerHTML = `
                <th>Instance ID</th>
                <th>Worker ID</th>
                <th>IP</th>
                <th>Port</th>
                <th>Status</th>
                <th>Key Count</th>
                <th>Last Heartbeat</th>
                <th>Actions</th>
            `;
        }
        
        // Populate table
        tableBody.innerHTML = '';
        workersWithKeyCounts.forEach(worker => {
            const row = document.createElement('tr');
            const now = Math.floor(Date.now() / 1000);
            const timeDiff = now - worker.last_heartbeat_time;
            const status = timeDiff < 60 ? 'Active' : timeDiff < 300 ? 'Warning' : 'Inactive';
            const statusClass = timeDiff < 60 ? 'status-active' : timeDiff < 300 ? 'status-warning' : 'status-inactive';
            
            const lastHeartbeat = new Date(worker.last_heartbeat_time * 1000).toLocaleTimeString();
            
            row.innerHTML = `
                <td>${worker.instance_id}</td>
                <td>${worker.worker_id}</td>
                <td>${worker.ip}</td>
                <td>${worker.port}</td>
                <td><span class="${statusClass}">${status}</span></td>
                <td>${worker.key_count}</td>
                <td>${lastHeartbeat}</td>
                <td>
                    <button class="btn btn-sm btn-info view-worker" data-instance="${worker.instance_id}" data-worker="${worker.worker_id}">View</button>
                </td>
            `;
            tableBody.appendChild(row);
        });

    } catch (error) {
        console.error('Error loading workers:', error);
        tableBody.innerHTML = `<tr><td colspan="8" class="text-center text-danger">Error: ${error.message}</td></tr>`;
    }
}

// Load log level
async function removeKey(key) {
    // This is a placeholder - in reality, you would need a backend endpoint to remove keys
    alert(`Would remove key: ${key}`);
    loadKeyPool();
}

// Load metrics
async function loadMetrics() {
    if (!isConnected) return;

    const contentDiv = document.getElementById('metricsContent');
    contentDiv.textContent = 'Loading...';

    try {
        // Note: This endpoint might not exist in the current controller
        // You would need to implement a metrics endpoint
        const response = await fetch(`${controllerBaseUrl}/metrics`);
        
        if (response.ok) {
            const metrics = await response.text();
            contentDiv.textContent = metrics;
        } else {
            contentDiv.textContent = 'Metrics endpoint not available. Would need to implement /metrics endpoint.';
        }
    } catch (error) {
        contentDiv.textContent = `Failed to load metrics: ${error.message}`;
    }
}

// Load environment variables
async function loadEnvironment() {
    if (!isConnected) return;

    const contentDiv = document.getElementById('envContent');
    const searchInput = document.getElementById('envSearchInput');
    contentDiv.textContent = 'Loading...';
    searchInput.value = '';

    try {
        // Call /env API to get environment variables
        const response = await fetch(`${controllerBaseUrl}/env`);
        
        if (!response.ok) {
            throw new Error('Failed to fetch environment variables');
        }

        envVariablesData = await response.json();
        
        // Format for display
        if (typeof envVariablesData === 'object' && envVariablesData !== null) {
            const formattedText = Object.entries(envVariablesData)
                .map(([key, value]) => `${key}=${value}`)
                .join('\n');
            contentDiv.textContent = formattedText;
        } else {
            contentDiv.textContent = 'No environment variables found or invalid data format';
            envVariablesData = null;
        }
    } catch (error) {
        console.error('Error loading environment variables:', error);
        contentDiv.textContent = `Failed to load environment variables: ${error.message}`;
        envVariablesData = null;
    }
}

// Filter environment variables
function filterEnvVariables() {
    const searchInput = document.getElementById('envSearchInput');
    const contentDiv = document.getElementById('envContent');
    const searchTerm = searchInput.value.toLowerCase();

    if (!envVariablesData) return;

    if (typeof envVariablesData === 'object') {
        const filteredEntries = Object.entries(envVariablesData).filter(([key, value]) => {
            const line = `${key}=${value}`;
            return line.toLowerCase().includes(searchTerm);
        });
        const formattedText = filteredEntries
            .map(([key, value]) => `${key}=${value}`)
            .join('\n');
        contentDiv.textContent = formattedText;
    }
}

// Load threads
async function loadThreads() {
    if (!isConnected) return;

    const contentDiv = document.getElementById('threadsContent');
    contentDiv.textContent = 'Loading...';

    try {
        const response = await fetch(`${controllerBaseUrl}/threads`);
        
        if (!response.ok) {
            throw new Error('Failed to fetch threads');
        }

        // Try to parse as JSON first
        const responseText = await response.text();
        
        let formattedText;
        try {
            // Try to parse as JSON
            const threadsData = JSON.parse(responseText);
            
            // Format threads data as text
            formattedText = '';
            threadsData.forEach((thread, index) => {
                formattedText += `Thread: ${thread.function_name}\n`;
                // Add thread details in a format similar to stack trace
                formattedText += `  thread_id: ${thread.thread_id}\n`;
                formattedText += `  name: ${thread.name}\n`;
                formattedText += `  state: ${thread.state}\n`;
                formattedText += `  cpu_time: ${thread.cpu_time}\n`;
                formattedText += `  memory_usage: ${thread.memory_usage}\n`;
                
                // Add separator between threads
                if (index < threadsData.length - 1) {
                    formattedText += '\n\n';
                }
            });
        } catch (jsonError) {
            // If not JSON, use the text as-is
            formattedText = responseText;
        }
        
        contentDiv.textContent = formattedText;
    } catch (error) {
        console.error('Error loading threads:', error);
        contentDiv.textContent = `Failed to load threads: ${error.message}`;
    }
}