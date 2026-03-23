import streamlit as st
import json
import os
import plotly.graph_objects as go
import yaml

st.set_page_config(page_title="QAT vs PTQ Dashboard", layout="wide")
st.title("QAT vs PTQ: Memory-Accuracy Tradeoff Analysis")

config_path = os.path.join(os.path.dirname(__file__), 'configs', 'config.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
res_dir = os.path.join(os.path.dirname(__file__), config['output']['results_dir'])
bench_file = os.path.join(res_dir, 'benchmark.json')
report_file = os.path.join(res_dir, 'report.json')

if not os.path.exists(bench_file):
    st.warning("`benchmark.json` not found. Please execute the pipeline with `python cli.py --mode all` first.")
    st.stop()
    
with open(bench_file, 'r') as f:
    data = json.load(f)

# Sidebar mapping
st.sidebar.header("Filter Models")
color_map = {"FP32": "blue", "QAT-INT8": "green", "PTQ-INT8": "orange", "PTQ-INT4": "red"}

selected_models = []
for row in data:
    if st.sidebar.checkbox(row['Model'], value=True):
        selected_models.append(row['Model'])
        
filtered_data = [x for x in data if x['Model'] in selected_models]

if not filtered_data:
    st.warning("Please select at least one model variant from the sidebar.")
    st.stop()

models = [x['Model'] for x in filtered_data]
ppl = [x['Perplexity'] for x in filtered_data]
sizes = [x['Size (MB)'] for x in filtered_data]
latency = [x['Latency (ms)'] for x in filtered_data]
colors = [color_map.get(m, "gray") for m in models]

col1, col2, col3 = st.columns(3)

with col1:
    fig_ppl = go.Figure(data=[go.Bar(x=models, y=ppl, marker_color=colors)])
    fig_ppl.update_layout(title="Perplexity (Lower is Better)", yaxis_title="Perplexity")
    st.plotly_chart(fig_ppl, use_container_width=True)
    
with col2:
    fig_size = go.Figure(data=[go.Bar(x=models, y=sizes, marker_color=colors)])
    fig_size.update_layout(title="Model Size (MB) (Lower is Better)", yaxis_title="MB")
    st.plotly_chart(fig_size, use_container_width=True)

with col3:
    fig_lat = go.Figure(data=[go.Bar(x=models, y=latency, marker_color=colors)])
    fig_lat.update_layout(title="Latency (ms) (Lower is Better)", yaxis_title="ms")
    st.plotly_chart(fig_lat, use_container_width=True)

if os.path.exists(report_file):
    with open(report_file, 'r') as f:
        rep = json.load(f)
    diff = rep['key_findings']['ptq_int8_degradation_pct'] - rep['key_findings']['qat_int8_degradation_pct']
    
    st.markdown("---")
    st.subheader("Results Summary")
    st.info(f"💡 Emphasizing simulated quantization during training (**QAT-INT8**) achieved **{diff:.2f}% less** perplexity degradation overall compared to simple conversion (**PTQ-INT8**).")
