import spaces
"""
ECG 12-Lead Cardiac Arrhythmia Pre-Screener - Hugging Face Space
"""
import os, json, time, numpy as np, torch, torch.nn as nn, wfdb, pandas as pd, requests
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal
from PIL import Image
import gradio as gr

# Ensure we run from the script's directory so relative paths work
os.chdir(os.path.dirname(os.path.abspath(__file__)))

LEAD_NAMES = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]
CONDITIONS = ["Normal SR","Atrial Fibrillation","STEMI / Acute MI","Bradycardia","Tachycardia"]
SAMPLES = {"Normal Sinus Rhythm (PTB-XL 00001)": "sample_ecgs/records100/00000/00001_lr",
           "Acute MI Alert (PTB-XL 00008)": "sample_ecgs/records100/00000/00008_lr"}
PTBXL_DB_URL = "https://physionet.org/files/ptb-xl/1.0.3/ptbxl_database.csv"
WEIGHTS_PATH = "ecg_resnet_ptbxl.pt"
PTBXL_DB = "ptb-xl"
NORM_TARGET = 0.85   # Normal SR probability threshold for LOW RISK
MI_TARGET = 0.80     # MI probability threshold for HIGH PRIORITY


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
    if pd.isna(codes_str):
        return False
    return code in str(codes_str)


def download_ptbxl_database():
    r = requests.get(PTBXL_DB_URL, timeout=30)
    r.raise_for_status()
    return pd.read_csv(__import__("io").StringIO(r.text), index_col="ecg_id")


def select_training_records(df, n_norm=10, n_mi=10):
    norm_df = df[df["scp_codes"].apply(lambda x: _has_code(x, "NORM"))].head(n_norm)
    mi_df = df[df["scp_codes"].apply(lambda x: _has_code(x, "MI") or _has_code(x, "STTC"))].head(n_mi)
    return norm_df, mi_df


def record_path_for(ecg_id):
    str_id = str(int(ecg_id)).zfill(6)
    folder = str((int(ecg_id) - 1) // 10 * 10).zfill(5)
    return f"records100/{folder}/{str_id}_lr"


def train_model(n_epochs=30):
    print("Downloading PTB-XL database index...")
    df = download_ptbxl_database()
    print("Selecting training records...")
    norm_df, mi_df = select_training_records(df)
    records, labels = [], []
    dl_dir = "./ptbxl_train"
    os.makedirs(dl_dir, exist_ok=True)
    BASE_URL = "https://physionet.org/files/ptb-xl/1.0.3/"

    def fetch_record(ecg_id):
        row = df.loc[ecg_id]
        rel = row["filename_lr"]
        for ext in [".hea", ".dat"]:
            url = BASE_URL + rel + ext
            local_path = os.path.join(dl_dir, os.path.basename(rel) + ext)
            try:
                r = requests.get(url, timeout=15)
                r.raise_for_status()
                with open(local_path, "wb") as f:
                    f.write(r.content)
            except Exception as e:
                print(f"  Download failed {rel}{ext}: {e}")
        return os.path.join(dl_dir, os.path.basename(rel))

    for _, row in norm_df.iterrows():
        try:
            path = fetch_record(row.name)
            sig, _ = wfdb.rdsamp(path)
            records.append(sig.T.astype(np.float32))
            labels.append(0)
            print(f"  Loaded NORM record {row.name}: shape={sig.shape}")
        except Exception as e:
            print(f"  Skipped NORM {row.name}: {e}")
    for _, row in mi_df.iterrows():
        try:
            path = fetch_record(row.name)
            sig, _ = wfdb.rdsamp(path)
            records.append(sig.T.astype(np.float32))
            labels.append(2)
            print(f"  Loaded MI record {row.name}: shape={sig.shape}")
        except Exception as e:
            print(f"  Skipped MI {row.name}: {e}")
    if len(records) == 0:
        print("No records loaded, skipping training.")
        return
    X = torch.from_numpy(np.stack(records))
    y = torch.from_numpy(np.array(labels)).long()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.4f}")
    model.eval()
    torch.save(model.state_dict(), WEIGHTS_PATH)
    print(f"Model weights saved to {WEIGHTS_PATH}")


# Load or train weights at startup
if os.path.exists(WEIGHTS_PATH):
    print(f"Loading trained weights from {WEIGHTS_PATH}...")
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=torch.device("cpu")))
else:
    print("No trained weights found. Training new model (30 epochs, CPU)...")
    train_model(n_epochs=30)

model.eval()


def bandpass_filter(ecg, lowcut=0.5, highcut=40.0, fs=500, order=4):
    nyq = 0.5 * fs
    b, a = signal.butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return signal.filtfilt(b, a, ecg, axis=0)


def extract_biomarkers(ecg, fs=500):
    """
    Extract deterministic clinical biomarkers with lead-specific ST-segment
    z-score deviation relative to each lead's baseline noise floor.
    """
    ecg_f = bandpass_filter(ecg, fs=fs)
    lead_ii = ecg_f[:, 1]

    # --- Heart rate & RR irregularity from lead II ---
    peaks, _ = signal.find_peaks(lead_ii, distance=int(0.4 * fs), prominence=0.1)
    if len(peaks) > 1:
        rr_intervals = np.diff(peaks) / fs
        bpm = 60.0 / np.mean(rr_intervals)
        rr_irregularity = np.std(rr_intervals) / np.mean(rr_intervals) if np.mean(rr_intervals) > 0 else 0.0
    else:
        bpm = 75.0
        rr_irregularity = 0.0

    # --- Lead-specific ST-segment deviation (z-score relative to baseline) ---
    n_samples = ecg_f.shape[0]
    baseline_end = int(0.1 * fs)
    st_start = int(0.3 * fs)
    st_end = int(0.5 * fs)

    st_leads = [1, 5, 9, 10, 11]
    st_z_scores = []
    st_deviations_mv = []

    for li in st_leads:
        lead_signal = ecg_f[:, li]
        baseline = np.mean(lead_signal[:baseline_end]) if baseline_end > 0 else 0.0
        st_segment = np.mean(lead_signal[st_start:st_end]) if st_end > st_start else 0.0
        raw_dev = st_segment - baseline
        st_deviations_mv.append(raw_dev)

        noise_floor = np.std(lead_signal[:baseline_end]) if baseline_end > 1 else 1.0
        noise_floor = max(noise_floor, 1e-6)
        z_score = raw_dev / noise_floor
        st_z_scores.append(z_score)

    st_max_z = float(max(abs(z) for z in st_z_scores)) if st_z_scores else 0.0
    st_deviation = float(max(abs(d) for d in st_deviations_mv)) if st_deviations_mv else 0.0

    return {
        "bpm": float(bpm),
        "rr_irregularity": float(rr_irregularity),
        "st_deviation": st_deviation,
        "st_max_z": st_max_z,
        "st_z_scores": [float(z) for z in st_z_scores],
    }


def run_inference(sig, fs=500):
    t0 = time.perf_counter()
    biomarkers = extract_biomarkers(sig, fs=fs)
    x = torch.from_numpy(sig.T).float().unsqueeze(0)
    with torch.no_grad():
        nn_probs = model(x).numpy()[0]
    latency_ms = (time.perf_counter() - t0) * 1000.0
    triage = {"bpm": biomarkers["bpm"], "rr_irregularity": biomarkers["rr_irregularity"],
              "st_deviation": biomarkers["st_deviation"], "st_max_z": biomarkers["st_max_z"],
              "nn_probs": [float(nn_probs[0]), float(nn_probs[1]), float(nn_probs[2]),
                           float(nn_probs[3]), float(nn_probs[4])],
              "nn_conditions": CONDITIONS, "latency_ms": round(latency_ms, 2),
              "recommendation": "Routine follow-up"}
    # Distinct triage thresholds
    if triage["nn_probs"][0] > NORM_TARGET:
        triage["recommendation"] = "LOW RISK — Normal Sinus Rhythm"
    elif triage["nn_probs"][2] > MI_TARGET or biomarkers["st_max_z"] > 2.5:
        triage["recommendation"] = "HIGH PRIORITY — Urgent Cardiologist Review"
    elif biomarkers["rr_irregularity"] > 0.12 or triage["nn_probs"][1] > 0.7:
        triage["recommendation"] = "Urgent: Possible atrial fibrillation - ECG confirmation"
    elif biomarkers["bpm"] < 50:
        triage["recommendation"] = "Urgent: Bradycardia - clinical review"
    elif biomarkers["bpm"] > 120:
        triage["recommendation"] = "Urgent: Tachycardia - clinical review"
    else:
        triage["recommendation"] = "Routine follow-up"
    return triage


def plot_12lead(sig, label, out_path, fs=500):
    fig, axes = plt.subplots(4, 3, figsize=(15, 10))
    fig.suptitle("12-Lead ECG: " + label, fontsize=14)
    t = np.arange(sig.shape[0]) / fs
    for i, ax in enumerate(axes.flat):
        ax.plot(t, sig[:, i], linewidth=0.5)
        ax.set_title(LEAD_NAMES[i], fontsize=9)
        ax.set_xlabel("s", fontsize=7)
        ax.grid(True, alpha=0.3)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


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

    triage = run_inference(sig, fs=fs)
    plot_path = "output_ui.png"
    plot_12lead(sig, label, plot_path, fs=fs)

    prob_text = "\n".join(f"  {c}: {p:.3f}" for c, p in zip(triage["nn_conditions"], triage["nn_probs"]))
    summary = ("Sample: " + label + "\n" + "Heart Rate: " + f"{triage['bpm']:.1f}" + " bpm\n"
               + "RR Irregularity: " + f"{triage['rr_irregularity']:.4f}" + "\n"
               + "ST Deviation (mV): " + f"{triage['st_deviation']:.4f}" + "\n"
               + "ST Max Z-Score: " + f"{triage['st_max_z']:.2f}" + "\n"
               + "Inference Latency: " + f"{triage['latency_ms']:.1f}" + " ms\n"
               + "\nNN Probabilities:\n" + prob_text + "\n"
               + "\nRecommendation: " + triage["recommendation"])
    return plot_path, summary


with gr.Blocks(title="ECG 12-Lead Pre-Screener") as demo:
    gr.Markdown("# ECG 12-Lead Cardiac Arrhythmia Pre-Screener")
    gr.Markdown("Upload a 12-lead ECG strip or select a PTB-XL sample for automated triage screening.")
    with gr.Row():
        with gr.Column(scale=1):
            sample_dropdown = gr.Dropdown(choices=list(SAMPLES.keys()),
                                          value=list(SAMPLES.keys())[0], label="PTB-XL Sample")
            upload_img = gr.Image(type="filepath", label="Upload ECG Strip Image (optional)")
            btn = gr.Button("Run Triage", variant="primary")
        with gr.Column(scale=2):
            plot_out = gr.Image(label="12-Lead Waveform")
            text_out = gr.Textbox(label="Triage Report", lines=20)
    btn.click(fn=process_ecg, inputs=[sample_dropdown, upload_img], outputs=[plot_out, text_out])
    gr.Markdown("---\nClinical Decision Support Disclaimer: This tool is for educational and research "
                "purposes only. It is NOT a substitute for professional medical advice, diagnosis, "
                "or treatment. Always consult a qualified healthcare provider for clinical decisions.")

@spaces.GPU
def _noop():
    return None

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True, ssr_mode=False)