# Building-LLM-From-Scratch

A hands-on, notebook-driven walkthrough of building a GPT-style LLM from scratch — starting with text preprocessing, tokenization, and data loading.

## Prerequisites

- Python 3.10
- macOS: [Homebrew](https://brew.sh/) installed
- Windows: none extra — the official installer below is enough

## 1. Install Python 3.10

Check if it's already installed:

```bash
python3.10 --version
```

### macOS

Install via Homebrew:

```bash
brew install python@3.10
```

### Windows

1. Download the Python 3.10 installer from the [official Python downloads page](https://www.python.org/downloads/release/python-31011/) (choose "Windows installer (64-bit)").
2. Run the installer. **Check "Add python.exe to PATH"** before clicking Install.
3. Verify the install by opening Command Prompt or PowerShell:

```powershell
python --version   # should print Python 3.10.x
```

If you have multiple Python versions installed on Windows, use the [py launcher](https://docs.python.org/3/using/windows.html#launcher) instead to target 3.10 specifically:

```powershell
py -3.10 --version
```

## 2. Create and activate the virtual environment

From the project root:

### macOS / Linux

```bash
python3.10 -m venv llm
source llm/bin/activate
```

### Windows (Command Prompt)

```powershell
py -3.10 -m venv llm
llm\Scripts\activate.bat
```

### Windows (PowerShell)

```powershell
py -3.10 -m venv llm
llm\Scripts\Activate.ps1
```

> If PowerShell blocks the activation script with an execution-policy error, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned` first, then retry.

Your shell prompt should now show `(llm)`. Verify the interpreter version:

```bash
python --version   # should print Python 3.10.x
```

To leave the environment later:

```bash
deactivate
```

## 3. Install dependencies

With the `llm` environment activated:

```bash
pip install --upgrade pip
pip install jupyter torch numpy tiktoken
```

## 4. How to run the notebook

1. Activate the `llm` virtual environment (step 2 above).
2. Open the project folder in VS Code (or run `jupyter notebook` from the terminal).
3. Open `Stage-1/step-1:working-with-textdata.ipynb`.
4. Select the `llm` virtual environment as the notebook kernel (VS Code: top-right kernel picker → Python Environments → `llm`).
5. Run cells **top to bottom** in order — later cells depend on variables defined earlier (e.g. `vocab`, `tokenizer`, `raw_text`).

## Notebook overview — `Stage-1/step-1:working-with-textdata.ipynb`

| Block | What it does |
|---|---|
| **Tokenizing text** | Reads `the-verdict.txt` and splits the raw text into word/punctuation tokens using regular expressions. |
| **Converting tokens into token IDs** | Builds a `vocab` dictionary mapping each unique token to a unique integer ID. |
| **SimpleTokenizerV1** | A basic tokenizer class that encodes text → token IDs and decodes IDs → text, using the fixed `vocab`. Fails on unknown words. |
| **Adding special context tokens** | Extends the vocabulary with `<|unk|>` (unknown word placeholder) and `<|endoftext|>` (marks boundaries between unrelated documents). |
| **SimpleTokenizerV2** | An upgraded tokenizer that falls back to `<|unk|>` for out-of-vocabulary words instead of crashing. |
| **BytePair encoding (BPE)** | Uses the `tiktoken` GPT-2 tokenizer, which breaks unfamiliar words into subword units instead of needing an unknown-token placeholder — the same tokenizer real GPT models use. |
| **Data sampling with a sliding window** | Demonstrates how input–target token pairs are built by sliding a fixed-size context window across the encoded text. |
| **Dataset loading using PyTorch** | `GPTDatasetV1` (a PyTorch `Dataset`) and `create_dataloader_v1` (builds a `DataLoader`) batch the tokenized text into `(input, target)` tensor pairs for training, using `max_length` (context size) and `stride` (window shift) parameters. |
| **Creating token embeddings** | Converts token IDs into learnable dense vectors via `torch.nn.Embedding` — the starting representation the LLM will optimize during training. |
| **Encoding word positions** | Adds a separate positional embedding (based on token position in the sequence) to the token embedding, since embeddings alone don't encode word order. |

## Project structure

```
Building-LLM/
├── README.md
├── .gitignore
├── llm/                                        # Python 3.10 virtual environment (not committed)
└── Stage-1/
    ├── step-1:working-with-textdata.ipynb       # Main notebook
    └── the-verdict.txt                          # Sample training text
```

## Notes

- `the-verdict.txt` is the short story used as sample training data throughout Stage 1.
- The `llm/` virtual environment folder is excluded from version control via `.gitignore` — each contributor should create their own using the steps above.
