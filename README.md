# **SemViQA: A Semantic Question Answering System for Vietnamese Information Fact-Checking**  

### **Authors**:  
*Nam V. Nguyen, Dien X. Tran, Thanh T. Tran, Anh T. Hoang, Tai V. Duong, Di T. Le, Phuc-Lu Le*  

<p align="center">
    <a href="#-quick-start">🚀 Quick Start</a> •
    <a href="#-about">📌 About</a> •
    <a href="#-training">🏋️‍♂️ Training</a> •
    <a href="#-pipeline">🧪 Pipeline</a> •
    <a href="#-citation">📖 Citation</a>
</p>  

---

## 📌 **About**  

Misinformation is a growing problem, exacerbated by the increasing use of **Large Language Models (LLMs)** like GPT and Gemini. This issue is even more critical for **low-resource languages like Vietnamese**, where existing fact-checking methods struggle with **semantic ambiguity, homonyms, and complex linguistic structures**.  

To address these challenges, we introduce **SemViQA**, a novel **Vietnamese fact-checking framework** integrating:  

- **Semantic-based Evidence Retrieval (SER)**: Combines **TF-IDF** with a **Question Answering Token Classifier (QATC)** to enhance retrieval precision while reducing inference time.  
- **Two-step Verdict Classification (TVC)**: Uses hierarchical classification optimized with **Cross-Entropy and Focal Loss**, improving claim verification across three categories:  
  - **Supported** ✅  
  - **Refuted** ❌  
  - **Not Enough Information (NEI)** 🤷‍♂️  

### **🏆 Achievements**
- **1st place** in the **UIT Data Science Challenge** 🏅  
- **State-of-the-art** performance on:  
  - **ISE-DSC01** → **78.97% strict accuracy**  
  - **ViWikiFC** → **80.82% strict accuracy**  
- **SemViQA Faster**: **7x speed improvement** over the standard model 🚀  

These results establish **SemViQA** as a **benchmark for Vietnamese fact verification**, advancing efforts to combat misinformation and ensure **information integrity**.  

---

## 🚀 **Quick Start**  

### 📥 **Installation**  

#### **1️⃣ Clone this repository**  
```bash
git clone https://github.com/DAVID-NGUYEN-S16/SemViQA.git
cd SemViQA
```

#### **2️⃣ Set up Python environment**  
We recommend using **Python 3.11** in a virtual environment (`venv`) or **Anaconda**.  

**Using `venv`:**  
```bash
python -m venv semviqa_env
source semviqa_env/bin/activate  # On MacOS/Linux
semviqa_env\Scripts\activate      # On Windows
```

**Using `Anaconda`:**  
```bash
conda create -n semviqa_env python=3.11 -y
conda activate semviqa_env
```

#### **3️⃣ Install dependencies**  
```bash
pip install --upgrade pip
pip install transformers==4.42.3
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

#### **4️⃣ Install FlashAttention (Optional, for performance boost)**  
Choose the version based on your Torch version from the [FlashAttention Releases](https://github.com/Dao-AILab/flash-attention/releases).  
Example for **Torch 2.1 + CUDA 11.8**:  
```bash
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu118torch2.1cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
```

---

## 🏋️‍♂️ **Training**  

Train different components of **SemViQA** using the provided scripts:  

### **1️⃣ Three-Class Classification Training**  
```bash
bash scripts/tc.sh
```

### **2️⃣ Binary Classification Training**  
```bash
bash scripts/bc.sh
```

### **3️⃣ QATC Model Training**  
```bash
bash scripts/qatc.sh
```

---

## 🧪 **Pipeline**  

Use the trained models to **predict test data**:  
```bash
bash scripts/pipeline.sh
```

---

## 📖 **Citation**  

If you use **SemViQA** in your research, please cite our work:  

```bibtex
@article{semviqa2024,
  title={SemViQA: A Semantic Question Answering System for Vietnamese Information Fact-Checking},
  author={Nguyen, Nam V. and Tran, Dien X. and Tran, Thanh T. and Hoang, Anh T. and Duong, Tai V. and Le, Di T. and Le, Phuc-Lu},
  year={2024},
  journal={arXiv preprint arXiv:2402.xxxx}
}
``` 