import json
from pathlib import Path
import torch
import gradio as gr
from torchvision import models, transforms

BASE = Path(__file__).parent

labels = json.load(open(BASE / "labels.json"))

model = models.resnet18(weights = None)
model.fc = torch.nn.Linear(model.fc.in_features, len(labels))
model.load_state_dict(torch.load(BASE / "best_model.pt", map_location="cpu"))
model.eval()


tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

ADVICE = {
    "angular_leaf_spot" : "Angular leaf spot (fungal). Remove infected leaves, avoid overhead watering, rotate crops, and ask an extension officer for advice. ",
    "bean_rust" : "Bean rust (fungal). Remove infected leaves, improve air flow between plants, and ask an extension officer about suitable fungicides. ",
    "healthy": "The leaf looks healthy. Keep monitoring regularly. ",
}

def predict (img):
    if img is None:
        return {}, "Please upload a photo of a bean leaf. "
    x = tf (img.convert("RGB")). unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1) [0]
    scores = {labels[i]: float(probs[i]) for i in range(len(labels))}
    top = max(scores, key=scores.get)
    note = ADVICE.get(top, " ")
    if scores[top]< 0.6:
        note = "Low Confidence - try a clearer, closer photo of a single leaf. " + note
    return scores, note

demo = gr. Interface(
    fn=predict,
    inputs = gr.Image(type="pil", label = "Bean leaf photo"),
    outputs = [gr.Label(num_top_classes= 3, label = "Prediction"),
               gr.Textbox(label= "What to do")],
    title = "BeanGuard: Bean Leaf Disease Detector",
    description = ("Upload a photo of a bean leaf. A ResNet18 model fine-tuned on the Makerere iBean dataset predicts angular leaf spot, bean rust or healthy, this is an Educational Demo and not expert advice-- by Wazingwa Mugala"),

) 

if __name__ == "__main__" :
    demo.launch()