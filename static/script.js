const video = document.getElementById('video');
const predictionText = document.getElementById('prediction');
const confidenceText = document.getElementById('confidence');
const predictionList = document.getElementById('prediction-list');

// Initialize webcam
navigator.mediaDevices.getUserMedia({ video: true })
.then(stream => { video.srcObject = stream; })
.catch(err => { 
    console.error("Error accessing webcam:", err);
    alert("Cannot access webcam. Please allow camera access.");
});

// History and trend
let history = [];
let trendData = [];
const maxHistory = 20;  // rolling window for trend

// Map expression to engagement score
const scoreMap = {
    "happiness": 1,
    "neutral": 1,
    "surprise": 0,
    "anger": -1,
    "disgust": -1,
    "fear": -1,
    "sadness": -1
};

// Initialize Chart.js
const ctx = document.getElementById('trendChart').getContext('2d');
const trendChart = new Chart(ctx, {
    type: 'line',
    data: {
        labels: [],
        datasets: [{
            label: 'Engagement Score',
            data: [],
            borderColor: '#3f51b5',
            backgroundColor: 'rgba(63,81,181,0.2)',
            tension: 0.4,
            fill: true
        }]
    },
    options: {
        responsive: true,
        scales: {
            y: {
                min: -1,
                max: 1,
                title: { display: true, text: 'Score (-1 Burnout → +1 Engaged)' }
            },
            x: {
                title: { display: true, text: 'Recent Frames' }
            }
        }
    }
});

// Capture frame and predict
async function predictFrame() {
    if (video.readyState !== 4) return;

    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    canvas.toBlob(async function(blob) {
        const formData = new FormData();
        formData.append('file', blob, 'frame.jpg');

        try {
            const response = await fetch('/predict/', { method: 'POST', body: formData });
            if (!response.ok) throw new Error(`Server error: ${response.statusText}`);
            const data = await response.json();

            // Update current prediction
            predictionText.innerText = `Expression: ${data.prediction}`;
            confidenceText.innerText = `Confidence: ${data.confidence}%`;

            // Update history
            history.unshift(`${data.prediction} (${data.confidence}%)`);
            if (history.length > 5) history.pop();

            predictionList.innerHTML = "";
            history.forEach(pred => {
                const li = document.createElement('li');
                const expr = pred.split(" ")[0].toLowerCase();
                if(expr === "happiness" || expr === "neutral") li.className = "positive";
                else if(expr === "anger" || expr === "disgust" || expr === "fear" || expr === "sadness") li.className = "negative";
                else li.className = "neutral";
                li.innerText = pred;
                predictionList.appendChild(li);
            });

            // Update trend chart
            const score = scoreMap[data.prediction.toLowerCase()] ?? 0;
            trendData.unshift(score);
            if (trendData.length > maxHistory) trendData.pop();

            trendChart.data.labels = trendData.map((_, idx) => idx + 1).reverse();
            trendChart.data.datasets[0].data = [...trendData].reverse();
            trendChart.update();

        } catch (err) {
            console.error("Prediction error:", err);
        }
    }, 'image/jpeg');
}

// Run prediction every second
setInterval(predictFrame, 1000);
