# BeanGuard

BeanGuard detects disease in bean leaves. Upload a photo of a leaf and a small web app predicts its condition, shows a confidence score for each class, and suggests what to do next.

It recognises three classes:

| Class | Meaning |
| --- | --- |
| `angular_leaf_spot` | Angular leaf spot (fungal) |
| `bean_rust` | Bean rust (fungal) |
| `healthy` | No disease detected |

If the top prediction is below 60% confidence, the app asks for a clearer, closer photo of a single leaf.

> **Note:** BeanGuard is an educational demo, not expert advice. Check with an agricultural extension officer before treating crops.

## Getting started

Tested with Python 3.12.

```bash
git clone https://github.com/WazDevZm/bean_guard.git
cd bean_guard
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Then open the local URL printed in the terminal. Sample leaf photos to try are in `test_images/`.

## How it works

The model is a ResNet18 pretrained on ImageNet and fine-tuned on the [Makerere iBean dataset](https://huggingface.co/datasets/AI-Lab-Makerere/beans), which has 1,034 training, 133 validation and 128 test images.

Training ran in three stages:

1. Train only the final layer for 3 epochs (learning rate 1e-3).
2. Unfreeze the whole network and train for 5 epochs (learning rate 1e-4).
3. Continue with a smaller learning rate (1e-5).

The highest validation accuracy in the notebook logs is 98.5%, against a 33.8% baseline from always guessing the most common class.

On the 128-image test set, `best_model.pt` scores 96.1% (123 of 128 correct):

| Class | Correct |
| --- | --- |
| `angular_leaf_spot` | 40 of 43 |
| `bean_rust` | 42 of 43 |
| `healthy` | 41 of 42 |

## Project files

| File | Purpose |
| --- | --- |
| `app.py` | Gradio web app that runs predictions |
| `bean_guard.ipynb` | Data exploration and model training |
| `best_model.pt` | Trained model weights |
| `labels.json` | Class names, in model output order |
| `test_images/` | 15 sample photos from the test set, 5 per class |
| `requirements.txt` | Python dependencies |

## Author

Built by Wazingwa Mugala.
