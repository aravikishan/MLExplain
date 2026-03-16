/**
 * MLExplain -- Frontend JavaScript
 *
 * Handles form submissions, Chart.js rendering for feature importance
 * bar charts, confusion matrix heatmaps, and model comparison charts.
 */

/* ========================================================================
   Colour Palette
   ======================================================================== */
var COLORS = {
    primary: '#1a56db',
    primaryLight: '#93b4f5',
    accent: '#059669',
    danger: '#dc2626',
    gray: '#94a3b8',
    chartBg: [
        'rgba(26, 86, 219, 0.7)',
        'rgba(5, 150, 105, 0.7)',
        'rgba(220, 38, 38, 0.7)',
        'rgba(217, 119, 6, 0.7)',
        'rgba(124, 58, 237, 0.7)',
    ],
    chartBorder: [
        '#1a56db', '#059669', '#dc2626', '#d97706', '#7c3aed',
    ],
};


/* ========================================================================
   Feature Importance Bar Chart
   ======================================================================== */
function renderFeatureImportanceChart(canvasId, importanceData) {
    var canvas = document.getElementById(canvasId);
    if (!canvas || !importanceData || importanceData.length === 0) return null;

    // Take top 20
    var data = importanceData.slice(0, 20);
    var labels = data.map(function(d) { return d.feature; });
    var values = data.map(function(d) { return d.importance; });

    return new Chart(canvas.getContext('2d'), {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Importance',
                data: values,
                backgroundColor: COLORS.chartBg[0],
                borderColor: COLORS.chartBorder[0],
                borderWidth: 1,
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            plugins: {
                legend: { display: false },
                title: { display: false },
            },
            scales: {
                x: {
                    beginAtZero: true,
                    title: { display: true, text: 'Importance Score' },
                    grid: { color: '#e2e8f0' },
                },
                y: {
                    grid: { display: false },
                },
            },
        },
    });
}


/* ========================================================================
   Confusion Matrix Heatmap
   ======================================================================== */
function renderConfusionMatrixChart(canvasId, cmData, labels) {
    var canvas = document.getElementById(canvasId);
    if (!canvas || !cmData || cmData.length === 0) return null;

    var ctx = canvas.getContext('2d');
    var n = cmData.length;
    var cellSize = Math.min(60, Math.floor(400 / n));
    var padding = 80;

    canvas.width = n * cellSize + padding + 40;
    canvas.height = n * cellSize + padding + 40;

    // Find max value for colour scaling
    var maxVal = 0;
    for (var i = 0; i < n; i++) {
        for (var j = 0; j < n; j++) {
            if (cmData[i][j] > maxVal) maxVal = cmData[i][j];
        }
    }
    if (maxVal === 0) maxVal = 1;

    ctx.font = '12px sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';

    // Draw cells
    for (var i = 0; i < n; i++) {
        for (var j = 0; j < n; j++) {
            var val = cmData[i][j];
            var intensity = val / maxVal;
            var r = Math.round(26 + (248 - 26) * (1 - intensity));
            var g = Math.round(86 + (250 - 86) * (1 - intensity));
            var b = Math.round(219 + (252 - 219) * (1 - intensity));

            var x = padding + j * cellSize;
            var y = padding + i * cellSize;

            ctx.fillStyle = 'rgb(' + r + ',' + g + ',' + b + ')';
            ctx.fillRect(x, y, cellSize - 2, cellSize - 2);

            // Text
            ctx.fillStyle = intensity > 0.5 ? '#fff' : '#1e293b';
            ctx.fillText(val, x + cellSize / 2 - 1, y + cellSize / 2 - 1);
        }
    }

    // Axis labels
    ctx.fillStyle = '#1e293b';
    ctx.font = '11px sans-serif';
    for (var i = 0; i < n; i++) {
        var label = labels && labels[i] ? labels[i] : String(i);
        // Short label
        if (label.length > 10) label = label.substring(0, 10) + '..';
        // X axis (predicted)
        ctx.save();
        ctx.translate(padding + i * cellSize + cellSize / 2, padding - 10);
        ctx.rotate(-0.5);
        ctx.fillText(label, 0, 0);
        ctx.restore();
        // Y axis (actual)
        ctx.fillText(label, padding - 10, padding + i * cellSize + cellSize / 2);
    }

    // Axis titles
    ctx.font = 'bold 12px sans-serif';
    ctx.fillStyle = '#64748b';
    ctx.fillText('Predicted', padding + (n * cellSize) / 2, padding - 35);
    ctx.save();
    ctx.translate(15, padding + (n * cellSize) / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Actual', 0, 0);
    ctx.restore();

    return null;  // canvas-based, no Chart instance
}


/* ========================================================================
   Comparison Chart (grouped bar)
   ======================================================================== */
function renderComparisonChart(canvasId, comparisons) {
    var canvas = document.getElementById(canvasId);
    if (!canvas || !comparisons || comparisons.length === 0) return null;

    var labels = comparisons.map(function(c) { return c.algorithm; });
    var metrics = ['accuracy', 'precision', 'recall', 'f1_score'];
    var metricLabels = ['Accuracy', 'Precision', 'Recall', 'F1 Score'];

    var datasets = metrics.map(function(metric, idx) {
        return {
            label: metricLabels[idx],
            data: comparisons.map(function(c) { return c[metric] || 0; }),
            backgroundColor: COLORS.chartBg[idx],
            borderColor: COLORS.chartBorder[idx],
            borderWidth: 1,
        };
    });

    return new Chart(canvas.getContext('2d'), {
        type: 'bar',
        data: { labels: labels, datasets: datasets },
        options: {
            responsive: true,
            plugins: {
                legend: { position: 'top' },
                title: { display: false },
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1.0,
                    title: { display: true, text: 'Score' },
                    grid: { color: '#e2e8f0' },
                },
                x: {
                    grid: { display: false },
                },
            },
        },
    });
}


/* ========================================================================
   Hyperparameter Fields
   ======================================================================== */
var HYPERPARAM_DEFAULTS = {
    max_depth: { type: 'number', default: 5, min: 1, max: 50 },
    min_samples_split: { type: 'number', default: 2, min: 2, max: 20 },
    n_estimators: { type: 'number', default: 100, min: 10, max: 500 },
    C: { type: 'number', default: 1.0, min: 0.01, max: 100, step: 0.1 },
    kernel: { type: 'select', default: 'rbf', options: ['rbf', 'linear', 'poly'] },
    n_neighbors: { type: 'number', default: 5, min: 1, max: 50 },
    metric: { type: 'select', default: 'euclidean', options: ['euclidean', 'manhattan', 'minkowski'] },
    solver: { type: 'select', default: 'lbfgs', options: ['lbfgs', 'liblinear', 'saga'] },
    max_iter: { type: 'number', default: 1000, min: 100, max: 10000 },
};

function updateHyperparamFields() {
    var select = document.getElementById('algorithmSelect');
    if (!select) return;
    var option = select.options[select.selectedIndex];
    var params = JSON.parse(option.getAttribute('data-params') || '[]');
    var container = document.getElementById('hyperparamFields');
    if (!container) return;
    container.innerHTML = '';

    params.forEach(function(param) {
        var config = HYPERPARAM_DEFAULTS[param] || { type: 'number', default: 0 };
        var div = document.createElement('div');
        div.style.marginBottom = '0.5rem';

        var label = document.createElement('label');
        label.textContent = param;
        label.style.fontSize = '0.85rem';
        label.style.display = 'block';
        label.style.marginBottom = '0.2rem';
        div.appendChild(label);

        var input;
        if (config.type === 'select') {
            input = document.createElement('select');
            input.className = 'form-control';
            input.setAttribute('data-param', param);
            config.options.forEach(function(opt) {
                var o = document.createElement('option');
                o.value = opt;
                o.textContent = opt;
                if (opt === config.default) o.selected = true;
                input.appendChild(o);
            });
        } else {
            input = document.createElement('input');
            input.type = 'number';
            input.className = 'form-control';
            input.setAttribute('data-param', param);
            input.value = config.default;
            if (config.min !== undefined) input.min = config.min;
            if (config.max !== undefined) input.max = config.max;
            if (config.step !== undefined) input.step = config.step;
        }
        div.appendChild(input);
        container.appendChild(div);
    });
}


/* ========================================================================
   Train Form Handler
   ======================================================================== */
function handleTrain(e) {
    e.preventDefault();

    var dataset = document.getElementById('datasetSelect').value;
    var algorithm = document.getElementById('algorithmSelect').value;
    var name = document.getElementById('experimentName').value || (algorithm + ' on ' + dataset);
    var testRatio = parseFloat(document.getElementById('testRatio').value) || 0.2;
    var randomState = parseInt(document.getElementById('randomState').value) || 42;

    // Collect hyperparameters
    var hyperparameters = {};
    var paramInputs = document.querySelectorAll('#hyperparamFields [data-param]');
    paramInputs.forEach(function(input) {
        var key = input.getAttribute('data-param');
        var val = input.value;
        // Try to parse as number
        var num = parseFloat(val);
        hyperparameters[key] = isNaN(num) ? val : num;
    });

    var payload = {
        dataset: dataset,
        algorithm: algorithm,
        name: name,
        test_ratio: testRatio,
        random_state: randomState,
        hyperparameters: hyperparameters,
    };

    document.getElementById('loadingOverlay').style.display = 'flex';
    document.getElementById('trainBtn').disabled = true;

    fetch('/api/train', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
    })
    .then(function(response) { return response.json(); })
    .then(function(data) {
        document.getElementById('loadingOverlay').style.display = 'none';
        document.getElementById('trainBtn').disabled = false;

        if (data.error) {
            alert('Training failed: ' + data.error);
            return;
        }

        // Show results
        var panel = document.getElementById('resultsPanel');
        panel.style.display = 'block';

        var result = data.result || {};
        document.getElementById('metricAccuracy').textContent =
            (result.accuracy * 100).toFixed(2) + '%';
        document.getElementById('metricPrecision').textContent =
            result.precision.toFixed(4);
        document.getElementById('metricRecall').textContent =
            result.recall.toFixed(4);
        document.getElementById('metricF1').textContent =
            result.f1_score.toFixed(4);

        // Meta info
        var meta = document.getElementById('resultMeta');
        meta.innerHTML = '<p>Train: ' + result.train_samples + ' samples | ' +
            'Test: ' + result.test_samples + ' samples | ' +
            'Time: ' + result.training_time_ms.toFixed(1) + ' ms</p>';

        // Update explain link
        document.getElementById('explainLink').href = '/explain/' + data.id;

        // Render charts
        if (result.feature_importance) {
            renderFeatureImportanceChart('importanceChart', result.feature_importance);
        }
        if (result.confusion_matrix) {
            // Fetch labels
            fetch('/api/datasets/' + dataset)
            .then(function(r) { return r.json(); })
            .then(function(dsInfo) {
                renderConfusionMatrixChart('confusionChart', result.confusion_matrix, dsInfo.target_names || []);
            });
        }

        // Scroll to results
        panel.scrollIntoView({ behavior: 'smooth' });
    })
    .catch(function(err) {
        document.getElementById('loadingOverlay').style.display = 'none';
        document.getElementById('trainBtn').disabled = false;
        alert('Request failed: ' + err);
    });
}


/* ========================================================================
   Compare Form Handler
   ======================================================================== */
function handleCompare(e) {
    e.preventDefault();

    var dataset = document.getElementById('compareDataset').value;
    var checkboxes = document.querySelectorAll('input[name="algorithms"]:checked');
    var algorithms = [];
    checkboxes.forEach(function(cb) { algorithms.push(cb.value); });

    if (algorithms.length === 0) {
        alert('Select at least one algorithm.');
        return;
    }

    var testRatio = parseFloat(document.getElementById('compareTestRatio').value) || 0.2;
    var randomState = parseInt(document.getElementById('compareRandomState').value) || 42;

    var payload = {
        dataset: dataset,
        algorithms: algorithms,
        test_ratio: testRatio,
        random_state: randomState,
    };

    document.getElementById('loadingOverlay').style.display = 'flex';
    document.getElementById('compareBtn').disabled = true;

    fetch('/api/compare', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
    })
    .then(function(response) { return response.json(); })
    .then(function(data) {
        document.getElementById('loadingOverlay').style.display = 'none';
        document.getElementById('compareBtn').disabled = false;

        if (data.error) {
            alert('Comparison failed: ' + data.error);
            return;
        }

        var panel = document.getElementById('compareResults');
        panel.style.display = 'block';

        // Build comparison table
        var comparisons = data.comparisons || [];
        var html = '<div class="table-container"><table class="data-table">';
        html += '<thead><tr><th>Algorithm</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th><th>Time (ms)</th></tr></thead>';
        html += '<tbody>';
        comparisons.forEach(function(c) {
            if (c.error) {
                html += '<tr><td>' + c.algorithm + '</td><td colspan="5" class="error">' + c.error + '</td></tr>';
            } else {
                html += '<tr>';
                html += '<td><strong>' + c.algorithm + '</strong></td>';
                html += '<td>' + (c.accuracy * 100).toFixed(2) + '%</td>';
                html += '<td>' + c.precision.toFixed(4) + '</td>';
                html += '<td>' + c.recall.toFixed(4) + '</td>';
                html += '<td>' + c.f1_score.toFixed(4) + '</td>';
                html += '<td>' + c.training_time_ms.toFixed(1) + '</td>';
                html += '</tr>';
            }
        });
        html += '</tbody></table></div>';
        document.getElementById('compareTable').innerHTML = html;

        // Render comparison chart
        var validComparisons = comparisons.filter(function(c) { return !c.error; });
        renderComparisonChart('compareChart', validComparisons);

        panel.scrollIntoView({ behavior: 'smooth' });
    })
    .catch(function(err) {
        document.getElementById('loadingOverlay').style.display = 'none';
        document.getElementById('compareBtn').disabled = false;
        alert('Request failed: ' + err);
    });
}


/* ========================================================================
   Delete Experiment
   ======================================================================== */
function deleteExperiment(id) {
    if (!confirm('Delete experiment #' + id + '?')) return;

    fetch('/api/experiments/' + id, { method: 'DELETE' })
    .then(function(response) { return response.json(); })
    .then(function(data) {
        if (data.error) {
            alert(data.error);
        } else {
            window.location.reload();
        }
    })
    .catch(function(err) {
        alert('Delete failed: ' + err);
    });
}
