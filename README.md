# Internal Consistency Detection for Misinformation Analysis

**Author:** Atefeh Heydari, PhD  
**Project Type:** NLP / Misinformation Detection / Research Prototype

---

## Overview

This repository contains a research prototype for detecting **internal consistency** in news articles and analysing its relationship with **veracity assessment**. The project investigates whether inconsistencies within the content of a news article can serve as useful signals for misinformation detection.

The system combines **text-based features**, **word embeddings**, and **CNN-based modelling** to classify news articles according to their internal consistency while supporting veracity-related analysis.

---

## Research Context

This work was developed as part of my doctoral research in Artificial Intelligence at the University of Manchester, focusing on computational approaches for analysing misinformation in online news content.

---

## Main Contributions

- Developed a research pipeline for **internal consistency detection** in news articles  
- Built a **CNN-based model** using text representations and linguistic features  
- Conducted experiments analysing the relationship between **internal consistency signals and news veracity**  
- Prepared a labelled dataset including:
  - internal consistency labels  
  - veracity labels  
  - cleaned article text for experimentation

---

## Repository Structure

IC-Misinformation/
├── data/               # Dataset and external embedding files
├── src/                # Source code for preprocessing, modelling, and experiments
├── requirements.txt    # Python dependencies
└── README.md

---

## Dataset

The repository includes a labelled dataset in the `data` folder:

- `label` → internal consistency label  
  - `0` = IC  
  - `1` = NIC  

- `label_fk` → veracity label  
  - `0` = real  
  - `1` = fake  

- `Cleaned` → cleaned article text used for modelling

---

## External Embeddings Required

Due to file size limitations, the following pretrained embeddings are not included in the repository and should be downloaded separately and placed inside the `data/` folder:

- `GoogleNews-vectors-negative300.bin`
- `glove.6B.100d.txt`

---

## Method Overview

The project uses:

- Python
- CNN-based text classification
- pretrained word embeddings
- feature-based news analysis

The typical workflow is:

1. Prepare and clean article text  
2. Load pretrained embeddings  
3. Train internal consistency detection model  
4. Evaluate consistency predictions  
5. Analyse the relationship between consistency and veracity

---

## Notes

This repository is shared as a **research prototype and portfolio project** derived from doctoral research experiments.
