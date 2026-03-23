document.addEventListener('DOMContentLoaded', async () => {
    try {
        const response = await fetch('/results/benchmark.json');
        if (!response.ok) throw new Error('Data not found');
        const data = await response.json();
        
        const fp32 = data.find(d => d.Model === 'FP32');
        const qat = data.find(d => d.Model === 'QAT-INT8');
        const ptq4 = data.find(d => d.Model === 'PTQ-INT4');
        
        // Populate Top Cards
        document.getElementById('fp32-val').innerText = fp32.Perplexity.toFixed(2);
        
        if(qat) {
            document.getElementById('qat-val').innerText = qat.Perplexity.toFixed(2);
            const delta = (qat.Perplexity - fp32.Perplexity).toFixed(2);
            const el = document.getElementById('qat-delta');
            el.innerText = `+${delta}`;
            el.parentElement.className = 'card-delta';
            if (delta > 0) el.parentElement.style.color = '#ef4444'; // Red if ppl increased
        }

        const memSaved = fp32['Size (MB)'] - qat['Size (MB)'];
        const memSavedPct = (memSaved / fp32['Size (MB)'] * 100).toFixed(0);
        document.getElementById('mem-val').innerText = `${memSavedPct}%`;
        document.getElementById('mem-delta').innerText = `${memSaved.toFixed(2)} MB`;

        // Calculate retention
        const tpRetention = ((qat['Tokens/sec'] / fp32['Tokens/sec']) * 100).toFixed(1);
        document.getElementById('tp-retention').innerText = `${tpRetention}%`;
        
        const ptq4Ret = ((ptq4['Tokens/sec'] / fp32['Tokens/sec']) * 100).toFixed(1);
        document.getElementById('ptq4-ret').innerText = `${ptq4Ret}%`;

        // Populate Table
        const tbody = document.getElementById('table-body');
        data.forEach(row => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td>${row.Model}</td>
                <td>${row['Latency (ms)']}</td>
                <td>${row['RAM (MB)']}</td>
                <td>${row.Perplexity}</td>
            `;
            tbody.appendChild(tr);
        });

        // Hydrate Benchmarks View
        try {
            const reportRes = await fetch('/api/report');
            const reportData = await reportRes.json();
            document.getElementById('benchmarks-pre').innerText = JSON.stringify(reportData, null, 4);
        } catch (e) {
            document.getElementById('benchmarks-pre').innerText = "Failed to load report.json";
        }

        // Hydrate Settings View
        try {
            const configRes = await fetch('/api/config');
            const configData = await configRes.json();
            const listEl = document.getElementById('settings-list');
            listEl.innerHTML = '';
            
            // Render primitive config values
            for (const [key, val] of Object.entries(configData)) {
                if (typeof val !== 'object') {
                    listEl.innerHTML += `<li><strong>${key}:</strong> ${val}</li>`;
                } else {
                    for (const [subk, subv] of Object.entries(val)) {
                        listEl.innerHTML += `<li><strong>${key}.${subk}:</strong> ${subv}</li>`;
                    }
                }
            }
        } catch (e) {
            document.getElementById('settings-list').innerHTML = "<li>Failed to load config.yaml</li>";
        }

        // Initialize Area Chart (Chart.js)
        const ctx = document.getElementById('mainChart').getContext('2d');
        const gradient = ctx.createLinearGradient(0, 0, 0, 200);
        gradient.addColorStop(0, 'rgba(74, 122, 255, 0.4)');
        gradient.addColorStop(1, 'rgba(74, 122, 255, 0.0)');

        const mainChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: Array(12).fill(''),
                datasets: [{
                    label: 'Tokens/Sec History',
                    data: Array(12).fill(0),
                    borderColor: '#4a7aff',
                    backgroundColor: gradient,
                    borderWidth: 3,
                    fill: true,
                    tension: 0.4,
                    pointRadius: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: { duration: 500 },
                plugins: { legend: { display: false } },
                scales: {
                    x: { grid: { display: false, color: '#27272a' } },
                    y: { 
                        grid: { color: '#27272a', borderDash: [5, 5] },
                        ticks: { callback: value => value / 1000 + 'k' }
                    }
                }
            }
        });

        const analyticsCtx = document.getElementById('analyticsChart').getContext('2d');
        const secondaryGradient = analyticsCtx.createLinearGradient(0, 0, 0, 200);
        secondaryGradient.addColorStop(0, 'rgba(34, 197, 94, 0.4)');
        secondaryGradient.addColorStop(1, 'rgba(34, 197, 94, 0.0)');

        const rxChart = new Chart(analyticsCtx, {
            type: 'bar',
            data: {
                labels: Array(12).fill(''),
                datasets: [{
                    label: 'SQLite Database Ingest (ms)',
                    data: Array(12).fill(0),
                    backgroundColor: '#22c55e',
                    borderRadius: 4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: { duration: 500 },
                plugins: { legend: { display: false } },
                scales: {
                    x: { grid: { display: false, color: '#27272a' } },
                    y: { grid: { color: '#27272a', borderDash: [5, 5] }}
                }
            }
        });

        const heatmap = document.getElementById('heatmap');
        function drawHeatmap() {
            heatmap.innerHTML = '';
            for (let i = 0; i < 64; i++) {
                const cell = document.createElement('div');
                cell.className = 'heat-cell';
                if (Math.random() > 0.4) {
                    const heatLvl = Math.floor(Math.random() * 5) + 1;
                    cell.classList.add(`heat-${heatLvl}`);
                }
                heatmap.appendChild(cell);
            }
        }
        drawHeatmap();

        // Live metrics polling every 5s
        setInterval(async () => {
            try {
                const res = await fetch('/api/live');
                const liveData = await res.json();
                const hist = liveData.history;
                if (hist && hist.length > 0) {
                    // Update chart
                    const chartData = mainChart.data.datasets[0].data;
                    const rxData = rxChart.data.datasets[0].data;
                    for (let i = 0; i < hist.length; i++) {
                        chartData[chartData.length - hist.length + i] = hist[i];
                        rxData[rxData.length - hist.length + i] = 64 / hist[i] * 1000; // Recalculate latency ms
                    }
                    mainChart.update();
                    rxChart.update();
                    
                    // Update main throughput display to current value
                    const latestTp = hist[hist.length - 1];
                    document.querySelector('.card-analytics .card-value.sm').innerHTML = 
                        `${latestTp.toFixed(0).toLocaleString()}<span class="unit">tok/s</span>`;
                    
                    // Animate heatmap
                    drawHeatmap();
                }
            } catch(e) {}
        }, 5000);

        // Toast System
        window.showToast = function(msg) {
            const container = document.getElementById('toast-container');
            const toast = document.createElement('div');
            toast.className = 'toast';
            toast.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#22c55e" stroke-width="2"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg> ${msg}`;
            container.appendChild(toast);
            setTimeout(() => {
                toast.style.animation = 'slideOut 0.3s ease forwards';
                setTimeout(() => toast.remove(), 300);
            }, 3000);
        };

        // UI Interactions
        document.querySelector('.icon-btn').addEventListener('click', () => {
            showToast("No new notifications from the ML pipeline.");
        });

        const viewsMap = {
            "Dashboard": "view-dashboard",
            "Models": "view-models",
            "Precisions": "view-precisions",
            "Benchmarks": "view-benchmarks",
            "Analytics": "view-analytics",
            "Settings": "view-settings"
        };

        document.querySelectorAll('.nav-item').forEach(item => {
            item.addEventListener('click', (e) => {
                e.preventDefault();
                const viewNameText = item.innerText.trim();
                
                // Flexible matching
                let targetId = null;
                for (const [key, val] of Object.entries(viewsMap)) {
                    if (viewNameText.includes(key)) {
                        targetId = val;
                        break;
                    }
                }
                
                if(!targetId) return;
                
                // Toggle active nav class
                document.querySelectorAll('.nav-item').forEach(el => el.classList.remove('active'));
                item.classList.add('active');
                
                // Toggle active view content
                document.querySelectorAll('.view-content').forEach(view => {
                    view.classList.remove('active');
                    view.style.display = 'none'; // Force style update
                });
                
                const targetView = document.getElementById(targetId);
                if (targetView) {
                    targetView.classList.add('active');
                    targetView.style.display = 'block'; // Force style update
                }
            });
        });

        document.querySelector('.share-btn').addEventListener('click', () => {
            navigator.clipboard.writeText(window.location.href);
            showToast("Dashboard URL copied to clipboard!");
        });

        document.querySelectorAll('.btn-outline').forEach(btn => {
            btn.addEventListener('click', () => showToast("Configuration modal opened."));
        });

        document.querySelector('.settings-dots').addEventListener('click', () => showToast("Matrix settings opened."));
        document.querySelectorAll('.see-more').forEach(el => {
            el.addEventListener('click', (e) => { e.preventDefault(); showToast("Detailed report generating..."); });
        });

        // Model Hot-Swapping
        const modelBtns = document.querySelectorAll('.toggle-switch button');
        modelBtns[0].addEventListener('click', () => {
            modelBtns[1].classList.remove('active');
            modelBtns[0].classList.add('active');
            fetch('/api/set_model?m=FP32');
            showToast("Switched live inference back to FP32 Baseline");
            if(fp32) document.getElementById('fp32-val').innerText = fp32.Perplexity.toFixed(2);
        });
        modelBtns[1].addEventListener('click', () => {
            modelBtns[0].classList.remove('active');
            modelBtns[1].classList.add('active');
            fetch('/api/set_model?m=QAT-INT8');
            showToast("Switched live inference to QAT-INT8");
            if(qat) document.getElementById('fp32-val').innerText = qat.Perplexity.toFixed(2);
        });

    } catch (err) {
        console.error("Error loading dashboard data:", err);
    }
});
