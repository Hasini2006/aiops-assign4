
# AI Image Captioning: Observability with Prometheus & Grafana
### DASHBOARD VIDEOS - https://drive.google.com/file/d/1sIUYszO2mnckKE7-BSixhNV0TilqGw0k/view?usp=share_link, https://drive.google.com/file/d/1QbgoA4wSqK1sE_5ofRfmUqew74THNIAV/view?usp=share_link
### FULL VIDEO - https://drive.google.com/file/d/1ooDYqTrnT0AHrmgaD9IuSV3NwWHSAWQ1/view?usp=share_link
A CPU-only image-captioning app built with Streamlit and **`Salesforce/blip-image-captioning-base`**. It is
instrumented with `prometheus_client` and exposes its metrics on a separate `/metrics` port. Prometheus scrapes it
together with `node_exporter`, so application spikes can be matched with CPU, RAM, disk and network usage.
Alert rules fire through Alertmanager, and a Grafana dashboard that follows the 7 Commandments of
Plotting shows everything in real time.

```
Browser ──► Streamlit app :8501 ──► /metrics :18000 ─┐
                                                      ├──► Prometheus :9090 ──► Grafana :3000
             node_exporter :9100 (CPU/RAM/disk/net) ──┘          │
                                                                 └──► Alertmanager :9093
```

---

## Repository structure

```
aiops-assign4/
├── app.py                  # Streamlit app: Single + Bulk (ZIP) captioning, including all Prometheus instrumentation
├── requirements.txt        # Python libraries (installed in aiops-venv)
├── monitoring/
│   ├── prometheus.yml      # scrape jobs: prometheus (9090), node (9100), caption_app (18000); rules; Alertmanager target
│   ├── rules.yml           # 4 recording rules + 11 alerting rules (loaded by prometheus.yml)
│   └── alertmanager.yml    # routing, inhibit rule
├── grafana/
│   └── dashboard.json      # Grafana dashboard export (5 s refresh, 5 sections)
├── test-images/            # sample images used for the demo, few taken from online and few were captured by myself, includes few non-images aswell. 
├── test-images.zip         # the same images zipped
├── dashboard-video         # ~1 min recording of the live dashboard at 5 s refresh
├── AI_DISCLOSURE.md        # AI-use disclosure 
├── .gitignore
└── README.md
```

I have only kept the Prometheus, Alertmanager, node_exporter and Grafana configuration files but not the installations.
---

## Setup
```bash
source ~/aiops-venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu  
# pip install torch                                                  
pip install -r requirements.txt
```
The BLIP model (~1 GB) downloads to `~/.cache/huggingface` the first time the app page is opened.
---

## To Run 
Use one terminal per component. 

**Terminal 1: node_exporter**

```bash
cd monitoring/node_exporter-1.12.1.linux-arm64
./node_exporter
```

**Terminal 2: Alertmanager**

```bash
cd monitoring/alertmanager-0.34.1.linux-arm64
./alertmanager --config.file=$HOME/Desktop/aiops-assign4/monitoring/alertmanager.yml
```

**Terminal 3: Prometheus**

`rules.yml` is found automatically because it sits next to `prometheus.yml`.

```bash
cd monitoring/prometheus-3.15.0.linux-arm64
./prometheus --config.file=$HOME/Desktop/aiops-assign4/monitoring/prometheus.yml
```

**Terminal 4: The captioning app**

```bash
source ~/aiops-venv/bin/activate
cd ~/Desktop/aiops-assign4
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

### Open in browser

| What | URL | Expected |
|---|---|---|
| Captioning app | http://localhost:8501 | Single image and Bulk (ZIP) tabs |
| App metrics | http://localhost:18000/metrics | application metrics + `process_*` |
| node_exporter metrics | http://localhost:9100/metrics | `node_*` metrics |
| Prometheus targets | http://localhost:9090/targets | `prometheus`, `node`, `caption_app` all **UP** |
| Prometheus rules | http://localhost:9090/rules | `caption-recording` and `caption-alerts` groups, all OK |
| Prometheus alerts | http://localhost:9090/alerts | 11 alert rules |
| Alertmanager | http://localhost:9093 | firing alerts, silences, status |
| Grafana dashboard | http://localhost:3000/d/caption-obs | live dashboard, 5 s refresh |

