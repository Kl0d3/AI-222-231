document.getElementById('testBtn').addEventListener('click', () => {
    const x = parseFloat(document.getElementById('featureX').value) || 0;
    const y = parseFloat(document.getElementById('featureY').value) || 0;
    const output = document.getElementById('resultOutput');

    // Simple synthetic isolation forest threshold logic for demo
    // Normal data centered around (0,0) with radius ~ 3.0
    const distance = Math.sqrt(x * x + y * y);
    const anomalyScore = Math.min(1, Math.max(0, (distance - 1.5) / 2.5));

    if (anomalyScore > 0.6) {
        output.style.borderColor = '#ef4444';
        output.style.backgroundColor = '#fef2f2';
        output.innerHTML = `🚨 <strong>Anomaly Detected!</strong><br>Score: ${anomalyScore.toFixed(3)} (High outlier probability)`;
    } else {
        output.style.borderColor = '#22c55e';
        output.style.backgroundColor = '#f0fdf4';
        output.innerHTML = `✅ <strong>Normal Data Point</strong><br>Score: ${anomalyScore.toFixed(3)} (Low anomaly risk)`;
    }
});
