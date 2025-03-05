# **SemViQA: A Semantic Question Answering System for Vietnamese Information Fact-Checking**  

### **Authors**:  
[**Nam V. Nguyen**](https://github.com/DAVID-NGUYEN-S16), [**Dien X. Tran**](https://github.com/xndien2004), Thanh T. Tran, Anh T. Hoang, Tai V. Duong, Di T. Le, Phuc-Lu Le 

<p align="center">
    <a href="#-about">📌 About</a> •
    <a href="#-checkpoints">🔍 Checkpoints</a> •
    <a href="#-quick-start">🚀 Quick Start</a> •
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
## 🔍 Checkpoints
We are making our **SemViQA** experiment checkpoints publicly available to support the **Vietnamese fact-checking research community**. By sharing these models, we aim to:  

- **Facilitate reproducibility**: Allow researchers and developers to validate and build upon our results.  
- **Save computational resources**: Enable fine-tuning or transfer learning on top of **pre-trained and fine-tuned models** instead of training from scratch.  
- **Encourage further improvements**: Provide a strong baseline for future advancements in **Vietnamese misinformation detection**.  
 

<table>
  <tr>
    <th>Method</th>
    <th>Model</th>
    <th>ViWikiFC</th>
    <th>ISE-DSC01</th>
  </tr>
  <tr>
    <td rowspan="3"><strong>TC</strong></td>
    <td>InfoXLM<sub>large</sub></td>
    <td><a href="https://huggingface.co/xuandin/infoxlm-large_viwiki_3class_cross">Link</a></td>
    <td><a href="https://huggingface.co/xuandin/infoxlm-large_isedsc_3class_cross">Link</a></td>
  </tr>
  <tr>
    <td>XLM-R<sub>large</sub></td>
    <td><a href="https://huggingface.co/xuandin/xlm-roberta-large_viwiki_3class_cross">Link</a></td>
    <td><a href="https://huggingface.co/xuandin/xlm-roberta-large_isedsc_3class_cross">Link</a></td>
  </tr>
  <tr>
    <td>Ernie-M<sub>large</sub></td>
    <td><a href="https://huggingface.co/xuandin/ernie-m-large-mnli-xnli_viwiki_3class_cross">Link</a></td>
    <td><a href="https://huggingface.co/xuandin/ernie-m-large-mnli-xnli_isedsc_3class_cross">Link</a></td> 
  </tr>
  <tr>
    <td rowspan="3"><strong>BC</strong></td>
    <td>InfoXLM<sub>large</sub></td>
    <td><a href="https://huggingface.co/xuandin/infoxlm-large_viwiki_2class_focal">Link</a></td>
    <td><a href="https://huggingface.co/xuandin/infoxlm-large_isedsc_2class_focal">Link</a></td>
  </tr>
  <tr>
    <td>XLM-R<sub>large</sub></td>
    <td><a href="https://huggingface.co/xuandin/xlm-roberta-large_viwiki_2class_focal">Link</a></td>
    <td><a href="https://huggingface.co/xuandin/xlm-roberta-large_isedsc_2class_focal">Link</a></td>
  </tr>
  <tr>
    <td>Ernie-M<sub>large</sub></td>
    <td><a href="https://huggingface.co/xuandin/ernie-m-large-mnli-xnli_viwiki_2class_focal">Link</a></td>
    <td><a href="https://huggingface.co/xuandin/ernie-m-large-mnli-xnli_isedsc_2class_focal">Link</a></td>
  </tr>
  <tr>
    <td rowspan="2"><strong>QATC</strong></td>
    <td>InfoXLM<sub>large</sub></td>
    <td><a href="https://huggingface.co/xuandin/infoxlm-large_viwiki_qatc">Link</a></td>
    <td><a href="https://huggingface.co/xuandin/infoxlm-large_isedsc_qatc">Link</a></td>
  </tr>
  <tr>
    <td>ViMRC<sub>large</sub></td>
    <td><a href="https://huggingface.co/xuandin/vi-mrc-large_viwiki_qatc">Link</a></td>
    <td><a href="https://huggingface.co/xuandin/vi-mrc-large_isedsc_qatc">Link</a></td>
  </tr>
  <tr>
    <td rowspan="2"><strong>QA origin</strong></td>
    <td>InfoXLM<sub>large</sub></td>
    <td><a href="https://huggingface.co/xuandin/infoxlm-large_viwiki_evidence_origin">Link</a></td>
    <td><a href="https://huggingface.co/xuandin/infoxlm-large_isedsc_evidence_origin">Link</a></td>
  </tr>
  <tr>
    <td>ViMRC<sub>large</sub></td>
    <td><a href="https://huggingface.co/xuandin/vi-mrc-large_viwiki_evidence_origin">Link</a></td>
    <td><a href="https://huggingface.co/xuandin/vi-mrc-large_isedsc_evidence_origin">Link</a></td>
  </tr>
</table>

 

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

## Acknowledgment  
Our development is based on our previous works:  
- [Check-Fact-Question-Answering-System](https://github.com/DAVID-NGUYEN-S16/Check-Fact-Question-Answering-System)  
- [Extract-Evidence-Question-Answering](https://github.com/DAVID-NGUYEN-S16/Extract-evidence-question-answering)  

**SemViQA** is the final version we have developed for verifying fact-checking in Vietnamese, achieving state-of-the-art (SOTA) performance compared to any other system for Vietnamese.

## 📖 **Citation**  

If you use **SemViQA** in your research, please cite our work:  

```bibtex
@article{semviqa2024,
  title={SemViQA: A Semantic Question Answering System for Vietnamese Information Fact-Checking},
  author={Nam V. Nguyen, Dien X. Tran, Thanh T. Tran, Anh T. Hoang, Tai V. Duong, Di T. Le, Phuc-Lu Le},
  year={2025},
  journal={https://arxiv.org/abs/2503.00955}
}
```  
