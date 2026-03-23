import os
import json
import plotly.graph_objects as go
import yaml

def main():
    config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    res_dir = os.path.join(os.path.dirname(__file__), '..', config['output']['results_dir'])
    bench_file = os.path.join(res_dir, 'benchmark.json')
    
    if not os.path.exists(bench_file):
        print("Waiting setup: benchmark.json not found for visualizer.")
        return
        
    with open(bench_file, 'r') as f:
        data = json.load(f)
        
    models = [x['Model'] for x in data]
    ppl = [x['Perplexity'] for x in data]
    size = [x['Size (MB)'] for x in data]
    
    # Perplexity vs Precision
    fig1 = go.Figure(data=[go.Bar(x=models, y=ppl, marker_color=['blue', 'green', 'orange', 'red'])])
    fig1.update_layout(title="Perplexity by Model Variant (Lower is Better)")
    fig1.write_html(os.path.join(res_dir, "perplexity_chart.html"))
    
    # Model Size vs Precision
    fig2 = go.Figure(data=[go.Bar(x=models, y=size, marker_color=['blue', 'green', 'orange', 'red'])])
    fig2.update_layout(title="Model Size by Variant (MB) (Lower is Better)")
    fig2.write_html(os.path.join(res_dir, "size_chart.html"))
    
    print("Static HTML visualizations generated via Plotly.")

if __name__ == "__main__":
    main()
