# ACCU-RATES

**ACCU-RATES** is a Python-based Streamlit tool for enzyme kinetics analysis, calculating initial reaction rates (v₀), Michaelis constant (Kₘ), and limiting rate (V). ACCU-RATES is free for non-commercial use at https://accu-rates.i3s.up.pt.

-----

## Overview

ACCU-RATES analyzes product accumulation or substrate depletion curves (>2 time points) using the Michaelis-Menten equation to fit progress curves and estimate v₀, Kₘ, and V. It’s robust with noisy data, ideal for enzyme inhibitor discovery, synthetic biology, and biomarker assays.

-----

## Features

  - Calculates v₀, Kₘ, and V from progress curves.
  - Handles noisy or sparse data.
  - Supports Excel-like ODS file uploads.
  - Applications in inhibitor discovery and synthetic biology.

-----

## Getting Started

### Running ACCU-RATES Locally (Linux)

Set up and run ACCU-RATES on Linux using **Python 3.11**.

#### 1\. Check Prerequisites

Ensure tools are installed:

  - **Git**:
    Verify:
    ```bash
    git --version
    ```
    Install: `sudo apt-get install git`
  - **Python 3.11**:
    Verify:
    ```bash
    python3.11 --version
    ```
    Install:
    ```bash
    sudo apt-get update
    sudo apt-get install -y python3.11 python3.11-dev python3.11-venv
    ```
  - **pip**:
    Verify:
    ```bash
    python3.11 -m pip --version
    ```
    Install: `sudo apt-get install python3-pip`
 
#### 2\. Clone the Repository

Clone:

```bash
git clone https://github.com/pmartins2106/ACCU-RATES.git
cd ACCU-RATES
```

#### 3\. Set Up Virtual Environment

Create with Python 3.11:

```bash
python3.11 -m venv venv
source venv/bin/activate
```

#### 4\. Install Dependencies

Install:

```bash
pip3 install beautifulsoup4==4.13.3 matplotlib==3.10.1 numpy==1.24.2 pandas==1.5.3 scipy==1.15.2 streamlit==1.41.1 odfpy
```

#### 5\. Run the App

Run with Python 3.11:

```bash
python3.11 -m streamlit run ACCU-RATES.py
```