// Global variables
let currentNode = null;
let currentProxy = null;
let proxyNodes = {};
let allProxies = []; // Store all proxies for filtering

// Initialize after DOM is loaded
window.addEventListener('DOMContentLoaded', () => {
    // Initialize proxy selector
    loadProxies();

    // Proxy search input event
    const proxySearchInput = document.getElementById('proxySearchInput');
    const proxyDropdown = document.getElementById('proxyDropdown');
    
    proxySearchInput.addEventListener('focus', () => {
        filterProxies();
        proxyDropdown.classList.add('show');
    });
    
    proxySearchInput.addEventListener('input', () => {
        filterProxies();
    });
    
    // Close dropdown when clicking outside
    document.addEventListener('click', (e) => {
        if (!proxySearchInput.contains(e.target) && !proxyDropdown.contains(e.target)) {
            proxyDropdown.classList.remove('show');
        }
    });

    // Proxy selection event (kept for compatibility)
    document.getElementById('proxySelector').addEventListener('change', (e) => {
        const proxyName = e.target.value;
        if (proxyName) {
            currentProxy = proxyNodes[proxyName];
            loadTargetNodes(proxyName);
        } else {
            currentProxy = null;
            document.getElementById('targetSelector').disabled = true;
            document.getElementById('targetSelector').innerHTML = '<option value="">-- Select Target --</option>';
        }
    });

    // Target selection event
    document.getElementById('targetSelector').addEventListener('change', (e) => {
        const nodeId = e.target.value;
        if (nodeId) {
            currentNode = JSON.parse(nodeId);
            document.getElementById('currentNode').textContent =
                `${currentNode.name} (${currentNode.host}:${currentNode.port})`;

            // Refresh active tab
            refreshActiveTab();
        } else {
            currentNode = null;
            document.getElementById('currentNode').textContent = 'No Node Selected';
            clearAllTabs();
        }
    });

    // Tab switching event
    document.querySelectorAll('.nav-link').forEach(tab => {
        tab.addEventListener('shown.bs.tab', () => {
            if (currentNode) {
                refreshActiveTab();
            }
        });
    });

    // Set log level button
    document.getElementById('setLogLevelBtn').addEventListener('click', setLogLevel);

    // Config management buttons
    document.getElementById('getConfigBtn').addEventListener('click', getConfig);
    document.getElementById('setConfigBtn').addEventListener('click', setConfig);

    // Refresh page button
    document.getElementById('refreshPageBtn').addEventListener('click', refreshCurrentPage);

    // Node management buttons
    document.getElementById('addNodeBtn').addEventListener('click', addNode);
    document.getElementById('updateNodeBtn').addEventListener('click', updateNode);

    // Refresh nodes button
    document.getElementById('refreshNodesBtn').addEventListener('click', refreshNodes);

    // Environment search input
    document.getElementById('envSearchInput').addEventListener('input', filterEnvVariables);

    // Environment filter buttons
    document.getElementById('filterNodesBtn').addEventListener('click', filterNodesByEnv);
    document.getElementById('clearFilterBtn').addEventListener('click', clearEnvFilter);

    // Refresh current page function
    function refreshCurrentPage() {
        if (currentNode) {
            refreshActiveTab();
        } else {
            alert('Please select a target node first');
        }
    }

    // Load node management list
    document.getElementById('node-management-tab').addEventListener('shown.bs.tab', () => {
        loadNodeListForManagement();
    });
});

// Load proxy list
async function loadProxies() {
    try {
        const response = await fetch('/api/proxies');
        const data = await response.json();

        const selector = document.getElementById('proxySelector');
        selector.innerHTML = '<option value="">-- Select Proxy --</option>';

        proxyNodes = {};
        allProxies = [];

        data.proxies.forEach(proxy => {
            const option = document.createElement('option');
            option.value = proxy.name;
            option.textContent = `${proxy.name} (${proxy.host}:${proxy.port})`;
            selector.appendChild(option);

            proxyNodes[proxy.name] = proxy;
            allProxies.push(proxy);
        });
        
        // Initialize dropdown with all proxies
        filterProxies();
    } catch (error) {
        console.error('Failed to load proxies:', error);
    }
}

// Filter proxies based on search input
function filterProxies() {
    const searchInput = document.getElementById('proxySearchInput');
    const dropdown = document.getElementById('proxyDropdown');
    const searchTerm = searchInput.value.toLowerCase();
    
    dropdown.innerHTML = '';
    
    const filteredProxies = allProxies.filter(proxy => {
        const proxyText = `${proxy.name} (${proxy.host}:${proxy.port})`.toLowerCase();
        return proxyText.includes(searchTerm);
    });
    
    if (filteredProxies.length === 0) {
        const noResultItem = document.createElement('div');
        noResultItem.className = 'dropdown-item disabled';
        noResultItem.textContent = 'No matching proxies found';
        dropdown.appendChild(noResultItem);
    } else {
        filteredProxies.forEach(proxy => {
            const item = document.createElement('a');
            item.className = 'dropdown-item';
            item.href = '#';
            item.textContent = `${proxy.name} (${proxy.host}:${proxy.port})`;
            item.dataset.proxyName = proxy.name;
            
            item.addEventListener('click', (e) => {
                e.preventDefault();
                selectProxy(proxy);
            });
            
            dropdown.appendChild(item);
        });
    }
}

// Select a proxy
function selectProxy(proxy) {
    const searchInput = document.getElementById('proxySearchInput');
    const dropdown = document.getElementById('proxyDropdown');
    const selector = document.getElementById('proxySelector');
    
    // Update search input display
    searchInput.value = `${proxy.name} (${proxy.host}:${proxy.port})`;
    
    // Update hidden selector
    selector.value = proxy.name;
    
    // Close dropdown
    dropdown.classList.remove('show');
    
    // Trigger proxy selection
    currentProxy = proxyNodes[proxy.name];
    loadTargetNodes(proxy.name);
}

// Load target nodes for selected proxy
async function loadTargetNodes(proxyName, filteredNodes = null) {
    try {
        let nodes;
        if (filteredNodes) {
            nodes = filteredNodes;
        } else {
            const response = await fetch(`/api/proxies/${proxyName}/nodes`);
            const data = await response.json();
            nodes = data.nodes;
        }

        const selector = document.getElementById('targetSelector');
        selector.innerHTML = '<option value="">-- Select Target --</option>';
        selector.disabled = false;

        nodes.forEach(node => {
            const option = document.createElement('option');
            option.value = JSON.stringify(node);
            option.textContent = `${node.name} (${node.host}:${node.port})`;
            selector.appendChild(option);
        });
    } catch (error) {
        console.error('Failed to load target nodes:', error);
    }
}

// Refresh child nodes of proxy
async function refreshProxyNodes(proxyName) {
    try {
        const response = await fetch(`/api/proxies/${proxyName}/refresh`);
        const data = await response.json();
        if (data.status === "success") {
            return data.nodes;
        }
        return [];
    } catch (error) {
        console.error('Failed to refresh proxy nodes:', error);
        return [];
    }
}

// Load node list
async function loadNodes() {}

// Refresh nodes for current proxy
async function refreshNodes() {
    if (!currentProxy) return;
    await loadTargetNodes(currentProxy.name);
}

// ==== Node Management Functions ====
async function loadNodeListForManagement() {
    try {
        const response = await fetch('/api/nodes');
        const data = await response.json();
        
        const tableBody = document.getElementById('nodeListBody');
        tableBody.innerHTML = '';
        
        data.nodes.forEach(node => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${node.name}</td>
                <td>${node.host}</td>
                <td>${node.port}</td>
                <td>${node.is_proxy ? 'Yes' : 'No'}</td>
                <td>${node.proxy_id || '-'}</td>
                <td>
                    <button class="btn btn-sm btn-warning edit-node me-1" data-name="${node.name}">Edit</button>
                    <button class="btn btn-sm btn-danger delete-node" data-name="${node.name}">Delete</button>
                </td>
            `;
            tableBody.appendChild(row);
        });
        
        // Add event listeners to edit/delete buttons
        document.querySelectorAll('.edit-node').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const nodeName = e.target.dataset.name;
                const node = data.nodes.find(n => n.name === nodeName);
                if (node) {
                    document.getElementById('nodeName').value = node.name;
                    document.getElementById('nodeHost').value = node.host;
                    document.getElementById('nodePort').value = node.port;
                    
                    // Auto-set proxy fields
                    document.getElementById('isProxyCheck').checked = node.is_proxy || false;
                    document.getElementById('proxyIdInput').value = node.proxy_id || '';
                }
            });
        });
        
        document.querySelectorAll('.delete-node').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const nodeName = e.target.dataset.name;
                if (confirm(`Are you sure you want to delete node ${nodeName}?`)) {
                    deleteNode(nodeName);
                }
            });
        });
        
    } catch (error) {
        console.error('Failed to load nodes for management:', error);
        alert('Failed to load nodes: ' + error.message);
    }
}


async function addNode() {
    const name = document.getElementById('nodeName').value.trim();
    const host = document.getElementById('nodeHost').value.trim();
    const port = document.getElementById('nodePort').value.trim();
    const isProxy = document.getElementById('isProxyCheck').checked;
    const proxyId = document.getElementById('proxyIdInput').value.trim();
    
    if (!name || !host || !port) {
        alert('Please fill all fields');
        return;
    }
    
    try {
        const response = await fetch('/api/nodes', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                name, 
                host, 
                port,
                is_proxy: isProxy,
                proxy_id: proxyId || null
            })
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Failed to add node');
        }
        
        // Refresh UI
        document.getElementById('nodeName').value = '';
        document.getElementById('nodeHost').value = '';
        document.getElementById('nodePort').value = '';
        
        loadNodeListForManagement();
        loadNodes(); // Refresh node selector
        
        alert('Node added successfully');
    } catch (error) {
        console.error('Add node error:', error);
        alert('Failed to add node: ' + error.message);
    }
}


async function updateNode() {
    const name = document.getElementById('nodeName').value.trim();
    const host = document.getElementById('nodeHost').value.trim();
    const port = document.getElementById('nodePort').value.trim();
    const isProxy = document.getElementById('isProxyCheck').checked;
    const proxyId = document.getElementById('proxyIdInput').value.trim();
    
    if (!name || !host || !port) {
        alert('Please fill all fields');
        return;
    }
    
    try {
        const response = await fetch(`/api/nodes/${encodeURIComponent(name)}`, {
            method: 'PUT',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                name, 
                host, 
                port,
                is_proxy: isProxy,
                proxy_id: proxyId || null
            })
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Failed to update node');
        }
        
        // Refresh UI
        loadNodeListForManagement();
        loadNodes(); // Refresh node selector
        
        alert('Node updated successfully');
    } catch (error) {
        console.error('Update node error:', error);
        alert('Failed to update node: ' + error.message);
    }
}


async function deleteNode(nodeName) {
    try {
        const response = await fetch(`/api/nodes/${encodeURIComponent(nodeName)}`, {
            method: 'DELETE'
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Failed to delete node');
        }
        
        // Refresh UI
        loadNodeListForManagement();
        loadNodes(); // Refresh node selector
        
        // Clear form if deleting the currently edited node
        if (document.getElementById('nodeName').value === nodeName) {
            document.getElementById('nodeName').value = '';
            document.getElementById('nodeHost').value = '';
            document.getElementById('nodePort').value = '';
        }
        
        alert('Node deleted successfully');
    } catch (error) {
        console.error('Delete node error:', error);
        alert('Failed to delete node: ' + error.message);
    }
}

// Refresh active tab
function refreshActiveTab() {
    const activeTab = document.querySelector('.tab-pane.active');
    if (!activeTab) return;

    switch (activeTab.id) {
        case 'overview':
            loadOverview();
            break;
        case 'metrics':
            loadMetrics();
            break;
        case 'threads':
            loadThreads();
            break;
        case 'loglevel':
            loadLogLevel();
            break;
        case 'config':
            loadConfig();
            break;
        case 'meta':
            loadMeta();
            break;
        case 'inference':
            loadInference();
            break;
        case 'env':
            loadEnvironment();
            break;
        case 'node-management':
            loadNodeListForManagement();
            break;
    }
}

// Load overview information
async function loadOverview() {
    if (!currentNode) return;

    const contentDiv = document.getElementById('overviewContent');
    contentDiv.innerHTML = '<div class="spinner-border" role="status">'
        + '<span class="visually-hidden">Loading...</span></div>';

    // Try multiProcess /api/status first; fall back to basic info.
    try {
        const statusResp = await fetch(transformPath('api/status'));
        if (statusResp.ok) {
            const statusData = await statusResp.json();
            renderMpOverview(contentDiv, statusData);
            return;
        }
    } catch (_) {
        // not a multiProcess node – fall through
    }

    // Basic overview for inProcess / legacy nodes
    try {
        const response = await fetch(transformPath('version'));
        const versionInfo = await response.text();

        contentDiv.innerHTML = `
            <div class="card">
                <div class="card-body">
                    <h5 class="card-title">Node Information</h5>
                    <p><strong>Name:</strong> ${escapeHtmlStr(currentNode.name)}</p>
                    <p><strong>Host:</strong> ${escapeHtmlStr(currentNode.host)}</p>
                    <p><strong>Port:</strong> ${escapeHtmlStr(String(currentNode.port))}</p>
                </div>
            </div>
            <div class="card mt-3">
                <div class="card-body">
                    <h5 class="card-title">Version Information</h5>
                    <pre>${escapeHtmlStr(versionInfo)}</pre>
                </div>
            </div>
        `;
    } catch (error) {
        contentDiv.innerHTML = '<div class="alert alert-danger">'
            + 'Failed to load overview: '
            + escapeHtmlStr(error.message) + '</div>';
    }
}

// ---------------------------------------------------------------
// MultiProcess Overview renderer (ported from mp_app.js)
// ---------------------------------------------------------------
function renderMpOverview(container, data) {
    var isHealthy = data.is_healthy;
    var healthClass = isHealthy ? "healthy" : "unhealthy";
    var healthText  = isHealthy ? "Healthy" : "Unhealthy";

    var sm = data.storage_manager || {};
    var l1 = sm.l1_manager || {};
    var l1TotalBytes = l1.memory_total_bytes || 0;
    var l1UsedBytes  = l1.memory_used_bytes  || 0;
    var l1Pct = l1TotalBytes > 0
        ? Math.round((l1UsedBytes / l1TotalBytes) * 100) : 0;
    var l1Objects = l1.total_object_count || 0;
    var barColor = l1Pct > 90 ? "#dc3545"
        : l1Pct > 70 ? "#ffc107" : "#198754";

    var gpuIds      = data.registered_gpu_ids || [];
    var sessions    = data.active_sessions    || 0;
    var engineType  = data.engine_type        || "Unknown";
    var chunkSize   = data.chunk_size         || "N/A";
    var hashAlgo    = data.hash_algorithm     || "N/A";
    var numAdapters = sm.num_l2_adapters      || 0;

    var html = '<div class="row">';

    // Row 1: Health / Engine / Sessions
    html += '<div class="col-md-4 mb-3"><div class="card stat-card">'
        + '<div class="card-body"><div class="stat-label">Health</div>'
        + '<div class="mt-2"><span class="health-dot ' + healthClass + '"></span>'
        + '<span class="fs-4 fw-bold">' + healthText + '</span>'
        + '</div></div></div></div>';

    html += '<div class="col-md-4 mb-3"><div class="card stat-card">'
        + '<div class="card-body"><div class="stat-label">Engine Type</div>'
        + '<div class="stat-value fs-4">' + escapeHtmlStr(engineType) + '</div>'
        + '<small class="text-muted">Chunk: ' + escapeHtmlStr(String(chunkSize))
        + ' | Hash: ' + escapeHtmlStr(hashAlgo) + '</small>'
        + '</div></div></div>';

    html += '<div class="col-md-4 mb-3"><div class="card stat-card">'
        + '<div class="card-body"><div class="stat-label">Active Sessions</div>'
        + '<div class="stat-value">' + sessions + '</div>'
        + '</div></div></div>';

    // Row 2: GPU / L1 / L2
    html += '<div class="col-md-4 mb-3"><div class="card stat-card">'
        + '<div class="card-body"><div class="stat-label">GPU Workers</div>'
        + '<div class="stat-value">' + gpuIds.length + '</div>'
        + '<small class="text-muted">IDs: '
        + escapeHtmlStr(gpuIds.length > 0 ? gpuIds.join(", ") : "none")
        + '</small></div></div></div>';

    html += '<div class="col-md-4 mb-3"><div class="card stat-card">'
        + '<div class="card-body"><div class="stat-label">L1 Cache Usage</div>'
        + '<div class="memory-bar mt-2"><div class="bar-fill" style="width:'
        + l1Pct + '%;background-color:' + barColor + '">' + l1Pct + '%</div></div>'
        + '<small class="text-muted mt-1 d-block">'
        + formatBytesStr(l1UsedBytes) + ' / ' + formatBytesStr(l1TotalBytes)
        + ' (' + l1Objects + ' objects)</small>'
        + '</div></div></div>';

    html += '<div class="col-md-4 mb-3"><div class="card stat-card">'
        + '<div class="card-body"><div class="stat-label">L2 Adapters</div>'
        + '<div class="stat-value">' + numAdapters + '</div>'
        + '</div></div></div>';

    // Row 3: Pending & Prefetch
    var pendingLookups  = data.pending_lookup_count    || 0;
    var nextJobId       = data.next_prefetch_job_id    || 0;
    var prefetchJobIds  = data.prefetch_job_ids        || [];
    var pendingReqIds   = data.pending_request_ids     || [];

    html += '<div class="col-12 mt-2 mb-2"><h5 class="text-muted">'
        + '<i class="bi bi-hourglass-split"></i> Pending &amp; Prefetch</h5></div>';

    html += '<div class="col-md-4 mb-3"><div class="card stat-card">'
        + '<div class="card-body"><div class="stat-label">Active Prefetch Jobs</div>'
        + '<div class="stat-value">' + prefetchJobIds.length + '</div>'
        + '<small class="text-muted">next ID: ' + nextJobId;
    if (prefetchJobIds.length > 0) {
        html += ' &middot; IDs: '
            + escapeHtmlStr(prefetchJobIds.slice(0, 5).join(", "));
        if (prefetchJobIds.length > 5) {
            html += ' +' + (prefetchJobIds.length - 5) + ' more';
        }
    }
    html += '</small></div></div></div>';

    html += '<div class="col-md-4 mb-3"><div class="card stat-card">'
        + '<div class="card-body"><div class="stat-label">Pending Lookups</div>'
        + '<div class="stat-value">' + pendingLookups + '</div>'
        + '</div></div></div>';

    html += '<div class="col-md-4 mb-3"><div class="card stat-card">'
        + '<div class="card-body"><div class="stat-label">Pending Requests</div>'
        + '<div class="stat-value">' + pendingReqIds.length + '</div>';
    if (pendingReqIds.length > 0) {
        html += '<small class="text-muted">'
            + escapeHtmlStr(pendingReqIds.slice(0, 3).join(", "));
        if (pendingReqIds.length > 3) {
            html += ' +' + (pendingReqIds.length - 3) + ' more';
        }
        html += '</small>';
    }
    html += '</div></div></div>';

    // Row 4: Periodic Threads summary
    var pt       = data.periodic_threads || {};
    var ptTotal   = pt.total_count   || 0;
    var ptRunning = pt.running_count || 0;
    var ptActive  = pt.active_count  || 0;

    if (ptTotal > 0) {
        html += '<div class="col-12 mt-2 mb-2"><h5 class="text-muted">'
            + '<i class="bi bi-arrow-repeat"></i> Periodic Threads</h5></div>';

        html += '<div class="col-md-4 mb-3"><div class="card stat-card">'
            + '<div class="card-body"><div class="stat-label">Registered</div>'
            + '<div class="stat-value">' + ptTotal + '</div>'
            + '</div></div></div>';

        var runColor = ptRunning === ptTotal ? "#198754" : "#ffc107";
        html += '<div class="col-md-4 mb-3"><div class="card stat-card">'
            + '<div class="card-body"><div class="stat-label">Running</div>'
            + '<div class="stat-value" style="color:' + runColor + '">'
            + ptRunning + ' / ' + ptTotal + '</div>'
            + '</div></div></div>';

        var actColor = ptActive === ptRunning ? "#198754" : "#dc3545";
        html += '<div class="col-md-4 mb-3"><div class="card stat-card">'
            + '<div class="card-body"><div class="stat-label">Active</div>'
            + '<div class="stat-value" style="color:' + actColor + '">'
            + ptActive + ' / ' + ptRunning + '</div>'
            + '</div></div></div>';
    }

    // Row 5: Hit Statistics
    html += renderMpHitStats(data.hit_stats);

    html += '</div>'; // close .row
    container.innerHTML = html;
}

function renderMpHitStats(stats) {
    if (!stats) return "";

    var hitRate  = stats.hit_rate || 0;
    var hitPct   = Math.round(hitRate * 100);
    var hitColor = hitPct >= 80 ? "#198754"
        : hitPct >= 50 ? "#ffc107" : "#dc3545";

    var totalReqs       = stats.total_requests        || 0;
    var totalTokens     = stats.total_tokens          || 0;
    var retrievedTokens = stats.total_retrieved_tokens || 0;

    var html = '<div class="col-12 mt-2 mb-2"><h5 class="text-muted">'
        + '<i class="bi bi-bullseye"></i> Hit Statistics</h5></div>';

    html += '<div class="col-md-4 mb-3"><div class="card stat-card">'
        + '<div class="card-body"><div class="stat-label">GPU Hit Rate</div>'
        + '<div class="stat-value" style="color:' + hitColor + '">'
        + hitPct + '%</div>'
        + '<div class="memory-bar mt-2"><div class="bar-fill" style="width:'
        + hitPct + '%;background-color:' + hitColor + '">' + hitPct + '%</div></div>'
        + '<small class="text-muted mt-1 d-block">'
        + formatTokenCountStr(retrievedTokens) + ' / '
        + formatTokenCountStr(totalTokens) + ' tokens</small>'
        + '</div></div></div>';

    html += '<div class="col-md-4 mb-3"><div class="card stat-card">'
        + '<div class="card-body"><div class="stat-label">Total Requests</div>'
        + '<div class="stat-value">' + totalReqs + '</div>'
        + '<small class="text-muted">'
        + formatTokenCountStr(totalTokens) + ' tokens total</small>'
        + '</div></div></div>';

    html += '<div class="col-md-4 mb-3"><div class="card stat-card">'
        + '<div class="card-body"><div class="stat-label">GPU Retrieved</div>'
        + '<div class="stat-value">'
        + formatTokenCountStr(retrievedTokens) + '</div>'
        + '<small class="text-muted">tokens written to GPU</small>'
        + '</div></div></div>';

    return html;
}

function formatBytesStr(bytes) {
    if (bytes >= 1073741824) return (bytes / 1073741824).toFixed(2) + " GB";
    if (bytes >= 1048576)    return (bytes / 1048576).toFixed(1)    + " MB";
    if (bytes >= 1024)       return (bytes / 1024).toFixed(1)       + " KB";
    return bytes + " B";
}

function formatTokenCountStr(count) {
    if (count >= 1000000) return (count / 1000000).toFixed(1) + "M";
    if (count >= 1000)    return (count / 1000).toFixed(1)    + "K";
    return String(count);
}

function escapeHtmlStr(str) {
    var div = document.createElement("div");
    div.appendChild(document.createTextNode(str));
    return div.innerHTML;
}

// Load metrics information
async function loadMetrics() {
    if (!currentNode) return;

    const contentDiv = document.getElementById('metricsContent');
    contentDiv.textContent = 'Loading...';

    try {
        const response = await fetch(transformPath('metrics'));
        const metrics = await response.text();
        contentDiv.textContent = metrics;
    } catch (error) {
        contentDiv.textContent = `Failed to load metrics: ${error.message}`;
    }
}

// Load threads information
async function loadThreads() {
    if (!currentNode) return;
    const contentDiv = document.getElementById('threadsContent');
    contentDiv.textContent = 'Loading...';

    try {
        const response = await fetch(transformPath('threads'));
        const threads = await response.text();
        contentDiv.textContent = threads;
    } catch (error) {
        contentDiv.textContent = `Failed to load threads: ${error.message}`;
    }
}

// Load log level
async function loadLogLevel() {
    if (!currentNode) return;

    const contentDiv = document.getElementById('logLevelContent');
    const loggerInput = document.getElementById('loggerInput');

    contentDiv.textContent = 'Loading...';
    loggerInput.value = '';

    try {
        const response = await fetch(transformPath('loglevel'));

        const text = await response.text();

        contentDiv.textContent = text;
    } catch (error) {
        contentDiv.textContent = `Failed to load log levels: ${error.message}`;
    }
}

// Set log level
async function setLogLevel() {
    if (!currentNode) return;

    const loggerInput = document.getElementById('loggerInput');
    const levelSelector = document.getElementById('logLevelSelector');

    const loggerName = loggerInput.value.trim();
    const level = levelSelector.value;

    try {
        let url;
        // Encode socket path if needed
        const portOrSocket = encodeURIComponent(encodeURIComponent(currentNode.port));

        if (!level) {
            // Read log level if no level is selected
            url = transformPath('loglevel');
            if (loggerName) {
                url += `?logger_name=${encodeURIComponent(loggerName)}`;
            }
            const response = await fetch(url);
            const text = await response.text();
            alert(text);
        } else {
            // Set log level if level is selected
            if (!loggerName) {
                alert('Please enter a Logger name');
                return;
            }
            url = transformPath('loglevel');
            url += `?logger_name=${encodeURIComponent(loggerName)}&level=${level}`;
            const response = await fetch(url, { method: 'GET' });

            const text = await response.text();
            alert(text);

            if (response.ok) {
                loadLogLevel();
            }
        }
    } catch (error) {
        alert(`Failed to manage log level: ${error.message}`);
    }
}

// Load configuration
async function loadConfig() {
    if (!currentNode) return;

    const contentDiv = document.getElementById('configContent');
    const configKeyInput = document.getElementById('configKeyInput');
    const configValueInput = document.getElementById('configValueInput');

    contentDiv.textContent = 'Loading...';
    configKeyInput.value = '';
    configValueInput.value = '';

    try {
        const response = await fetch(transformPath('conf'));
        const text = await response.text();
        contentDiv.textContent = text;
    } catch (error) {
        contentDiv.textContent = `Failed to load configuration: ${error.message}`;
    }
}

// Get configuration
async function getConfig() {
    if (!currentNode) return;

    const configKeyInput = document.getElementById('configKeyInput');
    const configKey = configKeyInput.value.trim();

    try {
        let url = transformPath('conf');
        if (configKey) {
            url += `?key=${encodeURIComponent(configKey)}`;
        }
        const response = await fetch(url);
        const text = await response.text();
        alert(text);
    } catch (error) {
        alert(`Failed to get configuration: ${error.message}`);
    }
}

// Set configuration
async function setConfig() {
    if (!currentNode) return;

    const configKeyInput = document.getElementById('configKeyInput');
    const configValueInput = document.getElementById('configValueInput');

    const configKey = configKeyInput.value.trim();
    const configValue = configValueInput.value.trim();

    if (!configKey) {
        alert('Please enter a configuration key');
        return;
    }

    if (!configValue) {
        alert('Please enter a configuration value');
        return;
    }

    try {
        const url = transformPath('conf') + `?key=${encodeURIComponent(configKey)}&value=${encodeURIComponent(configValue)}`;
        const response = await fetch(url, { method: 'GET' });
        const text = await response.text();
        alert(text);

        if (response.ok) {
            loadConfig();
        }
    } catch (error) {
        alert(`Failed to set configuration: ${error.message}`);
    }
}

// Load meta information
async function loadMeta() {
    if (!currentNode) return;

    const contentDiv = document.getElementById('metaContent');
    contentDiv.textContent = 'Loading...';

    try {
        const response = await fetch(transformPath('meta'));
        const text = await response.text();
        contentDiv.textContent = text;
    } catch (error) {
        contentDiv.textContent = `Failed to load meta information: ${error.message}`;
    }
}

// Load inference information
async function loadInference() {
    if (!currentNode) return;

    const contentDiv = document.getElementById('inferenceContent');
    contentDiv.textContent = 'Loading...';

    try {
        const response = await fetch(transformPath('inference_info'));
        const text = await response.text();
        contentDiv.textContent = text;
    } catch (error) {
        contentDiv.textContent = `Failed to load inference information: ${error.message}`;
    }
}

// Load environment variables
let envVariablesData = null; // Store as object instead of string
async function loadEnvironment() {
    if (!currentNode) return;

    const contentDiv = document.getElementById('envContent');
    const searchInput = document.getElementById('envSearchInput');
    contentDiv.textContent = 'Loading...';
    searchInput.value = '';

    try {
        const response = await fetch(transformPath('env'));
        const text = await response.text();
        
        // Parse JSON and format for display
        try {
            envVariablesData = JSON.parse(text);
            // Format as KEY=VALUE lines for display
            const formattedText = Object.entries(envVariablesData)
                .map(([key, value]) => `${key}=${value}`)
                .join('\n');
            contentDiv.textContent = formattedText;
        } catch (e) {
            // Fallback to plain text if not JSON
            envVariablesData = text;
            contentDiv.textContent = text;
        }
    } catch (error) {
        contentDiv.textContent = `Failed to load environment variables: ${error.message}`;
        envVariablesData = null;
    }
}

// Filter environment variables based on search input
function filterEnvVariables() {
    const searchInput = document.getElementById('envSearchInput');
    const contentDiv = document.getElementById('envContent');
    const searchTerm = searchInput.value.toLowerCase();

    if (!envVariablesData) {
        return;
    }

    if (!searchTerm) {
        // Show all variables
        if (typeof envVariablesData === 'object') {
            const formattedText = Object.entries(envVariablesData)
                .map(([key, value]) => `${key}=${value}`)
                .join('\n');
            contentDiv.textContent = formattedText;
        } else {
            contentDiv.textContent = envVariablesData;
        }
        return;
    }

    // Filter based on search term
    if (typeof envVariablesData === 'object') {
        const filteredEntries = Object.entries(envVariablesData).filter(([key, value]) => {
            const line = `${key}=${value}`;
            return line.toLowerCase().includes(searchTerm);
        });
        const formattedText = filteredEntries
            .map(([key, value]) => `${key}=${value}`)
            .join('\n');
        contentDiv.textContent = formattedText;
    } else {
        // Fallback for plain text
        const lines = envVariablesData.split('\n');
        const filteredLines = lines.filter(line => 
            line.toLowerCase().includes(searchTerm)
        );
        contentDiv.textContent = filteredLines.join('\n');
    }
}

// Clear all tab contents
function clearAllTabs() {
    document.getElementById('overviewContent').innerHTML = 'Please select a target node first';
    document.getElementById('metricsContent').textContent = 'Please select a target node first';
    document.getElementById('threadsContent').textContent = 'Please select a target node first';
    document.getElementById('logLevelContent').textContent = 'Please select a target node first';
    document.getElementById('configContent').textContent = 'Please select a target node first';
    document.getElementById('metaContent').textContent = 'Please select a target node first';
    document.getElementById('inferenceContent').textContent = 'Please select a target node first';
    document.getElementById('envContent').textContent = 'Please select a target node first';
    document.getElementById('loggerInput').value = '';
    document.getElementById('configKeyInput').value = '';
    document.getElementById('configValueInput').value = '';
    document.getElementById('envSearchInput').value = '';
    envVariablesData = null;
}

function transformPath(path) {
    if (!currentNode) return path;

    // When proxy_id equals the node's own name the node IS the proxy
    // (multiProcess child node) – use a single proxy2 hop.
    if (
        currentNode.proxy_id
        && proxyNodes[currentNode.proxy_id]
        && currentNode.proxy_id !== currentNode.name
    ) {
        const proxyNode = proxyNodes[currentNode.proxy_id];
        return `/proxy2/${proxyNode.name}/proxy2/${currentNode.name}/${path}`;
    }
    return `/proxy2/${currentNode.name}/${path}`;
}

// Filter nodes by environment variable and show matching proxies
async function filterNodesByEnv() {
    const envFilter = document.getElementById('envFilterInput').value.trim();
    if (!envFilter) {
        alert('Please enter an environment variable filter (e.g., TAG or TAG=TaijiDS)');
        return;
    }

    // Parse filter condition
    let filterKey, filterValue;
    const filterParts = envFilter.split('=');
    
    if (filterParts.length === 1) {
        // Only key provided, filter by key existence
        filterKey = filterParts[0].trim();
        filterValue = null;
    } else if (filterParts.length === 2) {
        // Key=Value provided, filter by exact match
        filterKey = filterParts[0].trim();
        filterValue = filterParts[1].trim();
    } else {
        alert('Invalid filter format. Please use KEY or KEY=VALUE format (e.g., TAG or TAG=TaijiDS)');
        return;
    }

    try {
        // Show loading indicator
        const proxySearchInput = document.getElementById('proxySearchInput');
        const proxySelector = document.getElementById('proxySelector');
        const targetSelector = document.getElementById('targetSelector');
        const proxyDropdown = document.getElementById('proxyDropdown');
        
        proxySearchInput.value = 'Filtering proxies...';
        proxySearchInput.disabled = true;
        proxyDropdown.classList.remove('show');
        targetSelector.innerHTML = '<option value="">-- Select Target --</option>';
        targetSelector.disabled = true;

        // Get all proxies
        const response = await fetch('/api/proxies');
        const data = await response.json();
        const fetchedProxies = data.proxies;

        // Check each proxy's nodes for matching environment variables
        const matchingProxies = new Map(); // Map<proxyName, matchingNodes[]>
        
        for (const proxy of fetchedProxies) {
            try {
                // Get nodes for this proxy
                const nodesResponse = await fetch(`/api/proxies/${proxy.name}/nodes`);
                const nodesData = await nodesResponse.json();
                const nodes = nodesData.nodes;

                // Check each node's environment variables
                const matchingNodes = [];
                const checkPromises = nodes.map(async (node) => {
                    try {
                        // Build the path to check env
                        let envPath;
                        if (node.proxy_id && proxyNodes[node.proxy_id]) {
                            const proxyNode = proxyNodes[node.proxy_id];
                            envPath = `/proxy2/${proxyNode.name}/proxy2/${node.name}/env`;
                        } else {
                            envPath = `/proxy2/${node.name}/env`;
                        }

                        const envResponse = await fetch(envPath);
                        if (!envResponse.ok) {
                            console.warn(`Failed to fetch env for node ${node.name}`);
                            return null;
                        }

                        const envText = await envResponse.text();
                        
                        // Parse JSON response
                        let envData;
                        try {
                            envData = JSON.parse(envText);
                        } catch (e) {
                            console.warn(`Failed to parse env JSON for node ${node.name}:`, e);
                            return null;
                        }
                        
                        // Check if the environment variable matches
                        if (filterValue === null) {
                            // Only key provided, check if key exists
                            if (filterKey in envData) {
                                return node;
                            }
                        } else {
                            // Key=Value provided, match exact value
                            if (filterKey in envData && envData[filterKey] === filterValue) {
                                return node;
                            }
                        }
                        
                        return null;
                    } catch (error) {
                        console.error(`Error checking node ${node.name}:`, error);
                        return null;
                    }
                });

                const results = await Promise.all(checkPromises);
                const validNodes = results.filter(node => node !== null);
                
                if (validNodes.length > 0) {
                    matchingProxies.set(proxy.name, validNodes);
                }
            } catch (error) {
                console.error(`Error checking proxy ${proxy.name}:`, error);
            }
        }

        // Update proxy selector with filtered proxies
        proxySearchInput.value = '';
        proxySearchInput.disabled = false;

        if (matchingProxies.size === 0) {
            const filterDesc = filterValue === null ? filterKey : `${filterKey}=${filterValue}`;
            alert(`No nodes found with ${filterDesc}`);
            // Restore original proxy list
            loadProxies();
        } else {
            // Update global allProxies with filtered results
            allProxies = [];
            let totalNodes = 0;
            const proxyNodeMap = new Map(); // Store matching nodes for each proxy
            
            matchingProxies.forEach((nodes, proxyName) => {
                const proxy = fetchedProxies.find(p => p.name === proxyName);
                if (proxy) {
                    allProxies.push(proxy);
                    proxyNodeMap.set(proxyName, nodes);
                    totalNodes += nodes.length;
                }
            });
            
            // Rebuild filtered proxy list
            filterProxies();
            
            const filterDesc = filterValue === null ? filterKey : `${filterKey}=${filterValue}`;
            alert(`Found ${matchingProxies.size} proxy(ies) with ${totalNodes} matching node(s) for ${filterDesc}`);
        }
    } catch (error) {
        console.error('Failed to filter nodes:', error);
        alert('Failed to filter nodes: ' + error.message);
        // Restore original proxy list
        loadProxies();
    }
}

// Clear environment filter
function clearEnvFilter() {
    document.getElementById('envFilterInput').value = '';
    
    // Restore original proxy list
    loadProxies();
    
    // Clear target selector
    const targetSelector = document.getElementById('targetSelector');
    targetSelector.innerHTML = '<option value="">-- Select Target --</option>';
    targetSelector.disabled = true;
}
