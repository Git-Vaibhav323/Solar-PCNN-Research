# JMeter Load Testing Guide — Solar AR-PCNN

This guide walks through installing dependencies, starting the inference server,
running the JMeter test plan, and exporting the HTML report for submission.

---

## 1. Install Python dependencies

Run once from the project root:

```powershell
.\.venv\Scripts\Activate.ps1
pip install fastapi "uvicorn[standard]" pydantic
```

Or install the full requirements file:

```powershell
pip install -r requirements.txt
```

---

## 2. Verify the server starts

```powershell
.\.venv\Scripts\Activate.ps1
python api/run_server.py
```

Expected output:

```
============================================================
 Solar AR-PCNN Inference Server
============================================================
  URL  : http://127.0.0.1:8000
  Docs : http://127.0.0.1:8000/docs
  Health: http://127.0.0.1:8000/health

  JMeter target endpoint:
    POST http://127.0.0.1:8000/predict

  Press Ctrl+C to stop.
============================================================
[API] Model loaded from artifacts\checkpoints\ar_pcnn_cd_multitask_best.pt on cpu
```

Quick smoke-test in a second terminal:

```powershell
Invoke-RestMethod -Method POST `
  -Uri http://localhost:8000/predict `
  -ContentType "application/json" `
  -Body (Get-Content jmeter\sample_payload.json -Raw)
```

Expected response shape:

```json
{
  "predicted_power_w": 148.3,
  "predicted_power_kw": 0.1483,
  "alpha": 0.82,
  "residual_w": -12.4,
  "physics_w": 162.5,
  "external_kw": 5.94,
  "latency_ms": 3.2
}
```

Also browse the interactive API docs at: http://localhost:8000/docs

---

## 3. Install JMeter

1. Download the latest binary from https://jmeter.apache.org/download_jmeter.cgi  
   (e.g. `apache-jmeter-5.6.3.zip`)
2. Extract to a folder, e.g. `C:\tools\jmeter`
3. Add `C:\tools\jmeter\bin` to your `PATH`, or use the full path in commands below.
4. Java 8+ is required — verify with `java -version`.

---

## 4. Create the results output folder

```powershell
New-Item -ItemType Directory -Path jmeter\results -Force
```

---

## 5. Run the JMeter test (non-GUI / command-line mode)

**Keep the server running in one terminal.** Open a second terminal:

```powershell
cd e:\Solar-PCNN-Research

jmeter -n `
  -t jmeter\solar_pcnn_load_test.jmx `
  -l jmeter\results\results.jtl `
  -e -o jmeter\results\html_report
```

Flag breakdown:

| Flag | Meaning |
|------|---------|
| `-n` | Non-GUI (headless) mode |
| `-t` | Path to test plan `.jmx` |
| `-l` | Path to write raw results `.jtl` |
| `-e` | Generate HTML report after test |
| `-o` | Output folder for HTML report (must not exist yet) |

The test runs three thread groups sequentially:

| Thread Group | Users | Duration |
|---|---|---|
| TG1 – Light  | 10  | 60 s |
| TG2 – Medium | 50  | 60 s |
| TG3 – Stress | 100 | 60 s |

Total wall-clock time: ~3–4 minutes.

---

## 6. Customise load parameters (optional)

Override variables on the command line without editing the `.jmx`:

```powershell
jmeter -n `
  -t jmeter\solar_pcnn_load_test.jmx `
  -l jmeter\results\results.jtl `
  -e -o jmeter\results\html_report `
  -JSERVER_HOST=localhost `
  -JSERVER_PORT=8000 `
  -JRAMP_UP_SEC=15 `
  -JDURATION_SEC=120
```

---

## 7. Open the HTML report

After the test finishes:

```powershell
Start-Process jmeter\results\html_report\index.html
```

The dashboard contains:

| Section | What it shows |
|---------|---------------|
| Statistics | Requests/sec, avg/min/max/90th/95th/99th percentile latency, error rate |
| Response Time Over Time | Latency trend during the test |
| Throughput Over Time | Predictions per second |
| Active Threads Over Time | Concurrency ramp-up |
| Response Time Percentiles | Distribution of latency across all requests |
| Errors | Any HTTP non-200 responses with details |

---

## 8. Key metrics to report

| Metric | Where in report |
|--------|----------------|
| Average response time (ms) | Statistics table — "Average" column |
| Throughput (req/s) | Statistics table — "Throughput" column |
| 90th percentile latency | Statistics table — "90th pct" column |
| 95th / 99th percentile latency | Statistics table — "95th pct" / "99th pct" |
| Error rate (%) | Statistics table — "Error %" column |
| Peak concurrent users | Active Threads Over Time graph |

---

## 9. Re-running the test

If `html_report` folder already exists, delete it first:

```powershell
Remove-Item -Recurse -Force jmeter\results\html_report
jmeter -n -t jmeter\solar_pcnn_load_test.jmx -l jmeter\results\results.jtl -e -o jmeter\results\html_report
```

---

## 10. API endpoints reference

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/health` | Liveness check — returns `{"status": "ok"}` |
| `GET`  | `/model-info` | Checkpoint path, feature list, device |
| `POST` | `/predict` | Single-sample inference (JMeter target) |
| `POST` | `/predict/batch` | Batch inference (up to 256 samples) |
| `GET`  | `/docs` | Interactive Swagger UI |

### POST /predict — request body

```json
{
  "GHI": 650.0,
  "DNI": 520.0,
  "DHI": 130.0,
  "Temperature": 28.5,
  "Relative Humidity": 45.0,
  "Pressure": 1013.25,
  "Surface Albedo": 0.2,
  "Wind Speed": 3.5,
  "Wind Direction": 180.0,
  "Dew Point": 15.2,
  "Precipitable Water": 1.8,
  "Solar Zenith Angle": 35.0,
  "hour_sin": 0.9659258262890682,
  "hour_cos": 0.25881904510252074,
  "P_theoretical": 162.5
}
```

`P_theoretical` is optional — if omitted the server estimates it from GHI.  
`hour` (integer 0–23) can be used instead of `hour_sin`/`hour_cos`.

### POST /predict — response body

```json
{
  "predicted_power_w": 148.3,
  "predicted_power_kw": 0.1483,
  "alpha": 0.82,
  "residual_w": -12.4,
  "physics_w": 162.5,
  "external_kw": 5.94,
  "latency_ms": 3.2
}
```

| Field | Meaning |
|-------|---------|
| `predicted_power_w` | AR-PCNN physics-path output in Watts |
| `predicted_power_kw` | Same value in kilowatts |
| `alpha` | Physics gate (0 = ignore physics, 1 = trust physics fully) |
| `residual_w` | Learned correction term in Watts |
| `physics_w` | Raw theoretical physics estimate in Watts |
| `external_kw` | Cross-domain head output for 10 kW system in kW |
| `latency_ms` | Server-side inference time in milliseconds |
