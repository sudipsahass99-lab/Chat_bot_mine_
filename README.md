
# 🧭 IIT KGP Senior Chatbot  
### *Domain-Specialized AI using Novel Multi-Source Knowledge Curation + DPO-Enhanced RAG Pipeline*

<img width="1143" alt="demo" src="https://github.com/user-attachments/assets/155f709c-2046-4fd1-b2a2-5750f20fd01d">

🎥 **Demo Video:**  
https://drive.google.com/file/d/1eLQiKs_EtX5sU7vwGZCw6j9dVoledzdw/view?usp=sharing

---

## 📌 Overview

This project builds a **domain-specific assistant for IIT Kharagpur**, designed to answer campus-specific queries far better than any general LLM.  
We demonstrate how **targeted datasets + lightweight alignment (DPO) + RAG** can enable a **1.5B model** to outperform much larger foundational models on a specialized domain.

This repo includes:

✔️ A **novel multi-source IIT KGP knowledge dataset**  
✔️ A **Direct Preference Optimization (DPO)** pipeline for alignment  
✔️ A **full retrieval system (RAG)** using FAISS + Sentence Transformers  
✔️ A **Flask-based chat interface**  
✔️ Utilities for scraping, cleaning, chunking, and embedding documents  
✔️ End-to-end training and inference code

---

# 📚 1. Dataset: Curating IIT KGP’s Online Footprint

Our dataset consolidates *hard-to-find* KGP-specific information scattered across:

### **1. IITKGP Subreddit (`r/iitkgp`)**
- Raw JSON:  
  `iitkgp_subreddit_final_data.json`
- Includes posts + full threaded comments
- Cleaned + normalized + enriched metadata

### **2. Preference Dataset for DPO (Ranking Data)**
- Built from subreddit JSON  
- Uses scoring based on:
  - TF-IDF keyword relevance  
  - SBERT semantic similarity  
  - Informational novelty  
- Contains:
  - `prompt` (original post)  
  - `highest_score_comment` (best chosen answer)  
  - `lowest_score_comment` (least relevant & rejected answer)  

📄 File: `preference_data_reddit.csv`

---

### **3. IIT KGP Official Website Scrape**
Text files extracted from:
- Academic sections  
- Hostels  
- Admin rules  
- Calendar  
- Infrastructure info  
- About pages  

📁 Folder: `iit-kgp/`

---

### **4. MetaKGP Wiki**
- Academic articles  
- Hostel pages  
- KGP culture  
- Places, policies, procedures  

📁 Folder: `meta-kgp/`

---

### **5. Apna KGP Resources**
- Commonly asked questions  
- Student club info  
- Hostel lists  
- Internal FAQs

📁 Folder: `iit-kgp/` (subfolder)

---

## 📊 Dataset Summary

| Source           | # Documents | Format         |
|------------------|-------------|----------------|
| IIT KGP Website  | 1777        | `.txt`         |
| MetaKGP Wiki     | 1410        | `.txt`         |
| r/iitkgp Reddit  | 10,453      | `.json` threads |
| **Total**        | **13,640**  | Mixed          |

Total chunks after preprocessing: **126,713**  
Stored using FAISS for efficient semantic search.

---

# ⚙️ System Architecture

            ┌────────────────────────┐
            │    Datasets (13k+)     │
            └───────────┬────────────┘
                        │
                        ▼
        ┌──────────────────────────────────┐
        │ Preprocessing + Cleaning          │
        │ (Text extraction, JSON parsing)   │
        └───────────────────┬──────────────┘
                            │
                            ▼
      ┌──────────────────────────────────────┐
      │ Chunking + Embedding (SBERT)         │
      │ Stored in FAISS Vector DB            │
      └──────────────┬───────────────────────┘
                     │
                     ▼
 ┌──────────────────────────────────────────────┐
 │ Alignment: Direct Preference Optimization     │
 │ (Qwen2.5-1.5B + Reddit preference data)       │
 └─────────────┬────────────────────────────────┘
               │
               ▼
   ┌──────────────────────────────────────┐
   │ RAG Inference Pipeline                │
   │ (Context retrieval → LLM generation)  │
   └──────────────────┬────────────────────┘
                      │
                      ▼
             Flask Web Chatbot UI


---

# 🏋️ 3.How to Run

```bash
python app.py
  ```

## Contribution: Demonstrates that carefully curated domain-specific datasets combined with DPO alignment can make smaller models (1.5B parameters) outperform larger general-purpose models in specialized tasks.
