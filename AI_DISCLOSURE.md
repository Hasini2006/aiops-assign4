# AI Disclosure

## ChatGPT

- I used chatgpt to understand how to structure a Streamlit app, how script reruns work, and how to manage state and cached resources.
- I used chatgpt to understand how to load and use the pretrained BLIP image-captioning model.
- I used chatgpt to learn the commands to start Prometheus, Alertmanager, node_exporter, Grafana, and the Streamlit app.
- I used chatgpt to understand the configuration and syntax of `prometheus.yml`, `alertmanager.yml`, and node_exporter setup.
- **PromQL queries:** I used  chatgpt to help write queries for system resource usage and client-level metrics in `grafana/dashboard.json` and `monitoring/rules.yml`.

## Claude

- I used Claude to generate the README, and format the AI_DISCLOSURE file, including the repository structure, setup instructions.
- I used Claude to review my metrics and help me ideate new ones.

### System Resource Usage

I used AI to help write queries for:
- Overall CPU utilisation and CPU usage by mode.
- Memory utilisation.
- Disk read and write throughput.
- Network receive and transmit rates.
- Root filesystem usage.

### Per-client Metrics

I used AI to help write queries for:
- Grouping request counters by client IP, session ID, browser, and mode.
- Images processed by each client IP.
- Request rates by client IP.
- Sessions started by browser.