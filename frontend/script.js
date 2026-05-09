const runBtn = document.querySelector("#runBtn");
const statusEl = document.querySelector("#status");
let chart;

const format = (value) => Number(value).toFixed(4);

function setStatus(message, type = "") {
  statusEl.textContent = message;
  statusEl.className = `status ${type}`;
}

function renderChart(result) {
  const labels = result.recent_predictions.map((row) => row.Date);
  const actual = result.recent_predictions.map((row) => row.Actual);
  const predicted = result.recent_predictions.map((row) => row.Predicted);

  if (chart) chart.destroy();
  chart = new Chart(document.querySelector("#chart"), {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "Actual",
          data: actual,
          borderColor: "#0f766e",
          backgroundColor: "rgba(15, 118, 110, 0.08)",
          borderWidth: 3,
          pointRadius: 3,
          tension: 0.25,
        },
        {
          label: "Predicted",
          data: predicted,
          borderColor: "#2563eb",
          backgroundColor: "rgba(37, 99, 235, 0.08)",
          borderWidth: 3,
          pointRadius: 3,
          tension: 0.25,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: {
        mode: "index",
        intersect: false,
      },
      plugins: {
        legend: {
          position: "bottom",
          labels: {
            boxWidth: 12,
            boxHeight: 12,
            usePointStyle: true,
          },
        },
        tooltip: {
          padding: 12,
          displayColors: true,
        },
      },
      scales: {
        x: {
          grid: { display: false },
          ticks: { color: "#64748b" },
        },
        y: {
          beginAtZero: false,
          grid: { color: "#e8eef5" },
          ticks: { color: "#64748b" },
        },
      },
    },
  });
}

function renderTables(result) {
  document.querySelector("#summary").innerHTML = `
    <tr><td>Best model</td><td>${result.model_name}</td></tr>
    <tr><td>Rows used</td><td>${result.df_rows}</td></tr>
    <tr><td>Training examples</td><td>${result.training_examples}</td></tr>
    <tr><td>Lookback used</td><td>${result.lookback_used}</td></tr>
    <tr><td>Date range</td><td>${result.date_start} to ${result.date_end}</td></tr>
  `;

  document.querySelector("#models").innerHTML = result.model_results
    .sort((a, b) => a["Max Error"] - b["Max Error"])
    .map(
      (row) => `
        <tr>
          <td>${row.Model}</td>
          <td>${format(row.MAE)}</td>
          <td>${format(row.RMSE)}</td>
          <td>${format(row["Max Error"])}</td>
        </tr>
      `
    )
    .join("");
}

runBtn.addEventListener("click", async () => {
  const apiUrl = document.querySelector("#apiUrl").value.replace(/\/$/, "");
  const file = document.querySelector("#csvFile").files[0];
  if (!file) {
    setStatus("Please choose a CSV file first.", "error");
    return;
  }

  const formData = new FormData();
  formData.append("file", file);
  formData.append("lookback", document.querySelector("#lookback").value);
  formData.append("test_ratio", document.querySelector("#testRatio").value);

  setStatus("Training models and preparing dashboard...");
  runBtn.disabled = true;
  runBtn.textContent = "Training...";

  try {
    const response = await fetch(`${apiUrl}/predict`, {
      method: "POST",
      body: formData,
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || "Prediction failed");

    document.querySelector("#futurePrice").textContent = format(result.future_price);
    document.querySelector("#mae").textContent = format(result.mae);
    document.querySelector("#rmse").textContent = format(result.rmse);
    document.querySelector("#maxError").textContent = format(result.max_error);
    renderChart(result);
    renderTables(result);
    setStatus("Prediction complete. The dashboard is updated with the latest validation results.", "ok");
  } catch (error) {
    setStatus(error.message, "error");
  } finally {
    runBtn.disabled = false;
    runBtn.textContent = "Run Prediction";
  }
});
