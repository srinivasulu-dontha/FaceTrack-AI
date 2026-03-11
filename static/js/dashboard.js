// dashboard.js
document.addEventListener("DOMContentLoaded", () => {
  const trainBtn = document.getElementById("trainBtn");
  const trainProgress = document.getElementById("trainProgress");
  const trainMsg = document.getElementById("trainMsg");

  async function pollStatus() {
    try {
      const res = await fetch("/train_status");
      const data = await res.json();
      if (trainProgress) {
        trainProgress.style.width = data.progress + "%";
        trainProgress.textContent = data.progress + "%";
      }
      if (trainMsg) trainMsg.textContent = data.message || "";
      return data;
    } catch (e) {
      console.error(e);
      return null;
    }
  }

  if (trainBtn) {
    trainBtn.addEventListener("click", async () => {
      trainBtn.disabled = true;
      const start = await fetch("/train_model");
      if (!start.ok && start.status !== 202) {
        alert("Failed to start training");
        trainBtn.disabled = false;
        return;
      }
      if (trainMsg) trainMsg.textContent = "Training started…";
      const t = setInterval(async () => {
        const s = await pollStatus();
        if (s && s.progress >= 100) {
          clearInterval(t);
          trainBtn.disabled = false;
          alert("Training completed successfully!");
        }
      }, 1500);
    });
  }

  // Attendance chart
  let chart = null;
  async function updateChart() {
    try {
      const res = await fetch("/attendance_stats");
      const data = await res.json();
      const ctx = document.getElementById("attendanceChart") && document.getElementById("attendanceChart").getContext("2d");
      if (!ctx) return;
      if (!chart) {
        chart = new Chart(ctx, {
          type: "bar",
          data: {
            labels: data.dates,
            datasets: [{
              label: "Attendance Count",
              data: data.counts,
              backgroundColor: "rgba(96,165,250,0.7)",
              borderColor: "rgba(96,165,250,1)",
              borderWidth: 1,
              borderRadius: 6
            }]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
              legend: { labels: { color: "#94a3b8", font: { size: 11 } } }
            },
            scales: {
              x: { ticks: { color: "#64748b", font: { size: 10 } }, grid: { color: "rgba(51,65,85,.5)" } },
              y: { ticks: { color: "#64748b" }, grid: { color: "rgba(51,65,85,.5)" }, beginAtZero: true }
            }
          }
        });
      } else {
        chart.data.labels = data.dates;
        chart.data.datasets[0].data = data.counts;
        chart.update();
      }
    } catch (e) { console.error(e); }
  }
  updateChart();
  setInterval(updateChart, 15000);

  // Initial status poll
  pollStatus();
});
