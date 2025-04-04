# 🧠 InterGPS: Geometry Theorem Prediction System

This project is part of a symbolic solver pipeline that automates geometric problem solving. Specifically, this component focuses on predicting candidate sequences of geometric theorems that can be used to solve a given problem based on its logic forms (from diagrams and text).

## 📌 What This Project Does

Given a geometry word problem and its accompanying diagram, this module:
- Parses logic forms from both text and diagram representations.
- Uses a language model (Gemini or BART) to predict sequences of relevant theorems that could help solve the problem.
- Outputs multiple candidate theorem sequences per problem, in a standardized format.
- Supports rate-limit-aware querying to Gemini API with retry logic.

## 🏗️ Project Structure

```
theorem_predict/
│
├── eval.py                          # Main script to generate predicted theorem sequences using Gemini API
├── results/
│   └── test/
│       └── pred_seqs_test_gemini.json   # Output file containing predictions
│
├── ../diagram_parser/
│   └── diagram_logic_forms.json     # Logic extracted from diagrams
│
├── ../text_parser/
│   └── text_logic_forms.json        # Logic extracted from word problem text
│
└── models/
    └── tp_model_best.pt             # (Optional) Pretrained BART model checkpoint
```

## 📥 Input

Each problem consists of:
- **Diagram Logic Forms**: parsed geometric relationships and elements from the figure.
- **Text Logic Forms**: formalized version of the word problem content.
- The goal: identify relevant theorems to apply.

These logic forms are stored as:
- `diagram_logic_forms.json`
- `text_logic_forms.json`

## 📤 Output Format

The model outputs a dictionary where each entry contains:
- `id`: Problem ID
- `num_seqs`: Number of predicted sequences
- `seq`: A list of candidate theorem sequences

Example:
```json
{
  "2401": {
    "id": "2401",
    "num_seqs": 5,
    "seq": [[4, 16], [3, 15], [4, 17], [3, 4, 16], [3, 4, 17]]
  }
}
```

## 📚 Theorem Definitions

Each theorem is represented by an ID (1–17) and corresponds to a standard geometric principle (e.g., triangle sum, congruent triangles, angle bisector).

## 🤖 Model Used

- **Gemini 1.5 Pro** (via Google Generative AI API): Generates theorem sequences using prompt engineering.
- *(Optional)* **facebook/bart-base**: Pretrained transformer-based fallback model.

## 🚀 How to Run

1. Make sure you have your Gemini API key.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the script:
   ```bash
   python eval.py
   ```

## 🔐 Rate Limit Handling

The script automatically retries Gemini API calls if it hits quota limits, with a delay of 45 seconds per retry (up to 5 attempts).
