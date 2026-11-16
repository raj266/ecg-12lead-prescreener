---
title: ECG 12-Lead Prescreener
emoji: 🫀
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 5.0.0
app_file: app.py
pinned: false
thumbnail: https://huggingface.co/spaces/hraj266/ecg-12lead-prescreener/resolve/main/arch.jpg
tags:
  - medical-ai
  - ecg
  - cardiology
  - pytorch
  - ptb-xl
---

# 🫀 ECG 12-Lead Cardiac Arrhythmia Pre-Screener

Upload a 12-lead ECG strip or select a PTB-XL sample for automated triage screening.

### Features
* **PyTorch 1D-ResNet Classifier**: Trained on the PhysioNet PTB-XL dataset for <10ms CPU inference.
* **Deterministic Clinical Biomarker Engine**: R-R interval AFib irregularity, BPM, Lead V1-V6/II/aVF ST-Segment Z-Score Analyzer.
* **MCP Protocol Ready**: Built with a Gradio 5 Dual-Mode Server for both Web UI and external AI Agent connectivity.

*Clinical Decision Support Disclaimer: This tool is for educational and research purposes only. It is NOT a substitute for professional medical advice, diagnosis, or treatment.*
