# Nano-LLM: Training a Language Model from Scratch on Consumer Hardware

This repository documents the process and provides the code for training a small-scale (\~24M parameter) GPT-2 style Language Model from scratch on a personal laptop (an Acer Predator Helios Neo 16 with an NVIDIA RTX 40-series GPU). The model is trained on the `TinyStories` dataset and is capable of generating simple, coherent, story-like text.

The primary goal of this project was to explore the end-to-end LLM training pipeline on consumer-grade hardware, from data preparation and tokenization to training and interactive inference.

## 🤖 Model Details

| Attribute             | Value                                 |
| --------------------- | ------------------------------------- |
| **Architecture** | GPT-2 Style (Decoder-only)            |
| **Parameters** | \~24.17 Million                        |
| **Training Dataset** | Ronen Eldan's `TinyStories`             |
| **Vocabulary Size** | 10,000 (Custom BPE)                   |
| **Context Length** | 256 tokens                            |
| **Layers** | 6                                     |
| **Attention Heads** | 8                                     |
| **Embedding Size** | 512                                   |

## ✨ Features

  - **Custom Tokenizer:** Script to train a `ByteLevelBPETokenizer` from scratch.
  - **From-Scratch Training:** A complete training script using PyTorch and the Hugging Face `Trainer` API.
  - **Interactive Chat:** A command-line interface (`chat_cli.py`) to interact with the trained model.
  - **Pre-trained Model:** The final trained model (`nanollm-final-model`) is included in this repository via Git LFS.

## 🚀 Getting Started

Follow these steps to set up the project and run the model on your own machine.

### 1\. Prerequisites

  - An NVIDIA GPU with CUDA support is highly recommended.
  - Python 3.10+
  - Git and [Git LFS](https://git-lfs.github.com/) installed.

### 2\. Installation

First, clone the repository and install Git LFS hooks.

```bash
# Clone the repository
git clone https://github.com/rajchauhan28/nano-llm.git
cd nano-llm

# Pull the large model files tracked by LFS
git lfs pull
```

Next, create a Python virtual environment and install the required dependencies.

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets accelerate sentencepiece protobuf
```

## 💻 Usage

The repository includes the final trained model, so you can start chatting with it immediately. Retraining is optional.

### Chat with the Model

To interact with the pre-trained nano LLM, run the command-line chat script:

```bash
python chat_cli.py
```

Type your sentences and press Enter. To exit, type `exit` or `quit`.

#### Example Interaction

```
Loading model... This might take a moment.
Model loaded on cuda. Type 'exit' or 'quit' to end the chat.
------------------------------
You: A little fox was lost in the woods
Assistant: . The fox was sad. He missed his mommy and daddy. The fox walked and walked. He saw a big tree. He said, "I will climb this tree to find my mommy and daddy." The fox climbed the tree. He saw his mommy and daddy. He was happy.

You: what is the capital of france
Assistant: ard. He likes to be the best captured in the world. He wears it all the time.
```

### (Optional) Retraining the Model

If you wish to train the model from scratch yourself, follow these steps:

1.  **Train the Tokenizer:**
    This script will create the `nanollm-tokenizer` directory with `vocab.json` and `merges.txt`.

    ```bash
    # This step is not necessary as the tokenizer is already provided
    # python train_tokenizer.py
    ```

2.  **Train the LLM:**
    This will start the full training process and save checkpoints in the `nanollm-results` directory.
    **Warning:** This requires a capable NVIDIA GPU and will take many hours.

    ```bash
    # python train_llm.py
    ```

## ⚠️ Limitations

This is a **base model**, not an instruction-tuned or chat-fine-tuned model.

  - **No Factual Knowledge:** The model was trained *only* on the `TinyStories` dataset. It has **no knowledge of real-world facts** and will not be able to answer questions like "What is the capital of France?".
  - **Story-Teller:** It is designed to generate simple, creative stories in the style of its training data. It excels at continuing story-like prompts.

## 📂 Project Structure

```
.
├── README.md
├── chat_cli.py             # Main script to chat with the model
├── train_llm.py            # Script to train the LLM from scratch
├── train_tokenizer.py      # Script to create the tokenizer
├── .gitignore
├── .gitattributes          # Git LFS configuration
└── nanollm-final-model/      # Final, trained model and tokenizer files
    ├── model.safetensors
    ├── config.json
    ├── tokenizer.json
    └── ...
```

## License

This project is licensed under the MIT License. See the LICENSE file for details.

-----
