from flask import Flask, render_template, request, send_file, url_for
import os
from models.tacotron2 import Tacotron2Wave
from models.fastpitch import FastPitch2Wave
import torch
from scipy.io.wavfile import write

app = Flask(__name__)

# Load models
models = {
    "Pretrained Tacotron2 mse Model": Tacotron2Wave('pretrained/tacotron2_ar_mse.pth'),
    "Pretrained Tacotron2 adv Model": Tacotron2Wave('pretrained/tacotron2_ar_adv.pth'),
    "Pretrained FastPitch mse Model": FastPitch2Wave('pretrained/fastpitch_ar_mse.pth'),
    "Pretrained FastPitch adv Model": FastPitch2Wave('pretrained/fastpitch_ar_adv.pth'),

    "Our Tacotron2 12 Epoch": Tacotron2Wave('checkpoints/exp_tc2/states_450.pth'),
    "Our FastPitch 12 Epoch": FastPitch2Wave('checkpoints/exp_fp_adv/states.pth'),
    "Our FastPitch 40 Epoch": FastPitch2Wave('checkpoints/exp_fp_adv_40_epoch/states.pth')
 
}

@app.route('/')
def index():
    return render_template('index.html', models=models.keys(), audio_url=None)


@app.route('/execute', methods=['POST'])
def execute():
    model_name = request.form['model']
    text = request.form['text']

    if not text:
        return "Error: Text field is empty.", 400

    # Select the model based on the user's choice
    model = models.get(model_name)
    if not model:
        return "Error: Invalid model selected.", 400
    model.cuda()
    # Generate the wave
    wave = model.tts(text, vowelizer='shakkelha')
    
    # Assuming wave is a PyTorch tensor, convert it to a NumPy array
    wave_np = wave.cpu().numpy() if wave.is_cuda else wave.numpy()

    # Define the sample rate (e.g., 22050 Hz in your case)
    sample_rate = 22050

    # Normalize the audio data to ensure it's within the range [-1, 1]
    wave_np = wave_np / max(abs(wave_np))

    # Convert to 16-bit PCM format for saving as WAV
    wave_int16 = (wave_np * 32767).astype('int16')
    
    # Save the wave to a file
    output_path = "static/output.wav"
    # Save the waveform to a .wav file
    write(output_path, sample_rate, wave_int16)
    
    # Return the index page with the audio URL
    audio_url = url_for('static', filename='output.wav')
    return render_template('index.html', models=models.keys(), audio_url=audio_url)

if __name__ == '__main__':
    app.run(debug=True)
