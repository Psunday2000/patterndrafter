from flask import Flask, render_template, send_from_directory, redirect, url_for, request
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import time

# Generator Model Function
class Generator (nn.Module):
    def __init__(self):
        super().__init__()
        self.label_emb = nn.Embedding(10, 10)
        self.gen_model = nn.Sequential(
            nn.Linear(100 + 10, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 784),
            nn.Tanh()
        )

    def forward(self, x, labels):
        label_input = self.label_emb(labels)
        x = torch.cat((x, label_input), dim=1)
        out = self.gen_model(x)
        out = out.view(x.size(0), 1, 28, 28)
        return out

# Pattern Drafter Flask Application
app = Flask(__name__)

# Load the generator model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
generator = Generator().to(device)
generator.load_state_dict(torch.load('generator.pth', map_location=device))
generator.eval()

# Generate a new pattern
@app.route('/draft', methods=['GET'])
def draft_pattern():
    # Generate noise and labels
    noise = torch.randn(1, 100).to(device)
    labels = torch.tensor([3], dtype=torch.long).to(device)

    # Draft the Pattern
    with torch.no_grad():
        generated_image = generator(noise, labels).cpu().detach().numpy()

    # Reshape and scale the image
    generated_image = (generated_image.squeeze() * 0.5 + 0.5) * 255
    generated_image = generated_image.astype(np.uint8)

    # Save the image 
    image = Image.fromarray(generated_image, mode='L')
    image.save('static/draft_pattern.png')

    return redirect(url_for('index', generated=time.time()))

@app.route('/')
def index():
    generated = request.args.get('generated', False)
    return render_template('index.html', generated=generated)

# Serve the Image
@app.route('/images/<filename>')
def images(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    app.run(debug=True)
