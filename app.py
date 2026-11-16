try:
    import spaces
except ImportError:
    pass

"""
ECG 12-Lead Cardiac Arrhythmia Pre-Screener - Hugging Face Space
"""
import os, json, time, numpy as np, torch, torch.nn as nn, wfdb, pandas as pd, requests
from scipy import signal
from PIL import Image
import gradio as gr
import plotly.graph_objects as go
from plotly.subplots import make_subplots

os.chdir(os.path.dirname(os.path.abspath(__file__)))

LEAD_NAMES = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]
CONDITIONS = ["Normal SR","Atrial Fibrillation","STEMI / Acute MI","Bradycardia","Tachycardia"]
SAMPLES = {"Normal Sinus Rhythm (PTB-XL 00001)": "sample_ecgs/records100/00000/00001_lr",
           "Acute MI Alert (PTB-XL 00008)": "sample_ecgs/records100/00000/00008_lr"}
PTBXL_DB_URL = "https://physionet.org/files/ptb-xl/1.0.3/ptbxl_database.csv"
WEIGHTS_PATH = "ecg_resnet_ptbxl.pt"
NORM_TARGET = 0.85
MI_TARGET = 0.80

class ECG1DResNet(nn.Module):
    def __init__(self, in_channels=12, num_classes=5):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        self.block1 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(inplace=True),
        )
        self.shortcut = nn.Conv1d(64, 64, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(64, num_classes), nn.Sigmoid(),
        )
    def forward(self, x):
        x = self.stem(x)
        out = self.block1(x) + self.shortcut(x)
        out = torch.relu(out)
        return self.classifier(out)

model = ECG1DResNet()
model.eval()

def _has_code(codes_str, code):
    if pd.isna(codes_str): return False
    return code in str(codes_str)

def download_ptbxl_database():
    r = requests.get(PTBXL_DB_URL, timeout=30)
    r.raise_for_status()
    return pd.read_csv(__import__("io").StringIO(r.text), index_col="ecg_id")

def bandpass_filter(ecg, lowcut=0.5, highcut=40.0, fs=500, order=4):
    nyq = 0.5 * fs
    b, a = signal.butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return signal.filtfilt(b, a, ecg, axis=0)

def extract_biomarkers(ecg, fs=500):
    ecg_f = bandpass_filter(ecg, fs=fs)
    lead_ii = ecg_f[:, 1]
    peaks, _ = signal.find_peaks(lead_ii, distance=int(0.4 * fs), prominence=0.1)
    if len(peaks) > 1:
        rr_intervals = np.diff(peaks) / fs
        bpm = 60.0 / np.mean(rr_intervals)
        rr_irregularity = np.std(rr_intervals) / np.mean(rr_intervals) if np.mean(rr_intervals) > 0 else 0.0
    else:
        bpm = 75.0
        rr_irregularity = 0.0
    n_samples = ecg_f.shape[0]
    baseline_end = int(0.1 * fs)
    st_start = int(0.3 * fs)
    st_end = int(0.5 * fs)
    st_leads = [1, 5, 9, 10, 11]
    st_z_scores, st_deviations_mv = [], []
    for li in st_leads:
        lead_signal = ecg_f[:, li]
        baseline = np.mean(lead_signal[:baseline_end]) if baseline_end > 0 else 0.0
        st_segment = np.mean(lead_signal[st_start:st_end]) if st_end > st_start else 0.0
        raw_dev = st_segment - baseline
        st_deviations_mv.append(raw_dev)
        noise_floor = np.std(lead_signal[:baseline_end]) if baseline_end > 1 else 1.0
        noise_floor = max(noise_floor, 1e-6)
        st_z_scores.append(raw_dev / noise_floor)
    st_max_z = float(max(abs(z) for z in st_z_scores)) if st_z_scores else 0.0
    st_deviation = float(max(abs(d) for d in st_deviations_mv)) if st_deviations_mv else 0.0
    return {"bpm": float(bpm), "rr_irregularity": float(rr_irregularity), "st_deviation": st_deviation, "st_max_z": st_max_z}

def run_inference(sig, fs=500):
    t0 = time.perf_counter()
    biomarkers = extract_biomarkers(sig, fs=fs)
    x = torch.from_numpy(sig.T).float().unsqueeze(0)
    with torch.no_grad():
        nn_probs = model(x).numpy()[0]
    latency_ms = (time.perf_counter() - t0) * 1000.0
    triage = {"bpm": biomarkers["bpm"], "rr_irregularity": biomarkers["rr_irregularity"],
              "st_deviation": biomarkers["st_deviation"], "st_max_z": biomarkers["st_max_z"],
              "nn_probs": [float(nn_probs[0]), float(nn_probs[1]), float(nn_probs[2]), float(nn_probs[3]), float(nn_probs[4])],
              "nn_conditions": CONDITIONS, "latency_ms": round(latency_ms, 2),
              "recommendation": "Routine follow-up"}
    if triage["nn_probs"][0] > NORM_TARGET:
        triage["recommendation"] = "LOW RISK - Normal Sinus Rhythm"
    elif triage["nn_probs"][2] > MI_TARGET or biomarkers["st_max_z"] > 2.5:
        triage["recommendation"] = "HIGH PRIORITY - Urgent Cardiologist Review"
    elif biomarkers["rr_irregularity"] > 0.12 or triage["nn_probs"][1] > 0.7:
        triage["recommendation"] = "Urgent: Possible atrial fibrillation - ECG confirmation"
    elif biomarkers["bpm"] < 50:
        triage["recommendation"] = "Urgent: Bradycardia - clinical review"
    elif biomarkers["bpm"] > 120:
        triage["recommendation"] = "Urgent: Tachycardia - clinical review"
    else:
        triage["recommendation"] = "Routine follow-up"
    return triage

def generate_interactive_plot(signal_data, title_text):
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    # Ensure the data is oriented as (12 leads, N samples)
    if signal_data.shape[0] != 12 and signal_data.shape[1] == 12:
        signal_data = signal_data.T
        
    n_samples = signal_data.shape[1]
    time_axis = np.linspace(0, 10, n_samples)
    
    fig = make_subplots(rows=4, cols=3, subplot_titles=LEAD_NAMES)
    
    for i in range(12):
        row, col = (i // 3) + 1, (i % 3) + 1
        fig.add_trace(go.Scatter(x=time_axis, y=signal_data[i], mode="lines", name=LEAD_NAMES[i], line=dict(width=1.5)), row=row, col=col)
        
    fig.update_layout(height=850, title_text=title_text, showlegend=False, margin=dict(l=40, r=40, t=60, b=40), dragmode="zoom", autosize=True)
    fig.update_xaxes(matches=None, showticklabels=True, zeroline=False)
    fig.update_yaxes(matches=None, showticklabels=True, zeroline=False)
    return fig


def process_ecg(sample_choice, uploaded_image):
    if uploaded_image is not None:
        img = Image.open(uploaded_image).convert("RGB")
        arr = np.array(img)
        if arr.ndim == 3 and arr.shape[2] >= 3:
            sig = arr[:, :, :3].mean(axis=2).reshape(-1, 1)
            sig = np.tile(sig, (1, 12))
            fs = 500
        else:
            sig = np.zeros((1000, 12)); fs = 500
        label = "Uploaded ECG Strip"
    else:
        sig, fields = wfdb.rdsamp(SAMPLES[sample_choice])
        fs = int(fields["fs"])
        label = sample_choice
        print(f"DEBUG: Loaded signal shape {sig.shape}, Max value: {np.max(sig)}", flush=True)

    triage = run_inference(sig, fs=fs)
    fig = generate_interactive_plot(sig, "12-Lead ECG: " + label)

    prob_text = "\n".join(f"- **{c}**: {p:.1%}" for c, p in zip(triage["nn_conditions"], triage["nn_probs"]))
    rec = triage["recommendation"]
    
    summary = f"""### Triage Report

**Sample:** {label}
**Inference Latency:** {triage['latency_ms']:.1f} ms

---
#### Vital Signs & Biomarkers
* **Heart Rate:** {triage['bpm']:.1f} bpm
* **RR Irregularity:** {triage['rr_irregularity']:.4f}
* **ST Deviation:** {triage['st_deviation']:.4f} mV
* **ST Max Z-Score:** {triage['st_max_z']:.2f}

---
#### Neural Network Probabilities
{prob_text}

---
#### Clinical Recommendation
**{rec}**
"""
    return fig, summary

with gr.Blocks(title="ECG 12-Lead Pre-Screener") as demo:
    gr.Markdown("# ECG 12-Lead Cardiac Arrhythmia Pre-Screener")
    gr.Markdown("Upload a 12-lead ECG strip or select a PTB-XL sample for automated triage screening.")
    with gr.Row():
        with gr.Column(scale=1):
            sample_dropdown = gr.Dropdown(choices=list(SAMPLES.keys()), value=list(SAMPLES.keys())[0], label="PTB-XL Sample")
            upload_img = gr.Image(type="filepath", label="Upload ECG Strip Image (optional)")
            btn = gr.Button("Run Triage", variant="primary")
        with gr.Column(scale=2):
            plot_out = gr.Plot(label="12-Lead Waveform")
            text_out = gr.Markdown(label="Triage Report")
    btn.click(fn=process_ecg, inputs=[sample_dropdown, upload_img], outputs=[plot_out, text_out])
    gr.Markdown("---\nClinical Decision Support Disclaimer: This tool is for educational and research purposes only. It is NOT a substitute for professional medical advice.")

def _noop():
    return None
if "spaces" in globals():
    _noop = spaces.GPU(_noop)

if __name__ == "__main__":
    # Load weights after UI is built
    if os.path.exists(WEIGHTS_PATH):
        print(f"Loading trained weights from {WEIGHTS_PATH}...")
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=torch.device("cpu")))
    model.eval()
    demo.launch(server_name="0.0.0.0", ssr_mode=False)
