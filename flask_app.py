from flask import Flask, render_template, request, send_file, url_for
import os
from models.tacotron2 import Tacotron2Wave
from models.fastpitch import FastPitch2Wave

app = Flask(__name__)

# Load models
models = {
    "FastPitch2Wave Model 1": FastPitch2Wave('checkpoints/exp_fp_adv_40_epoch/states.pth'),
    "FastPitch2Wave Model 2": FastPitch2Wave('checkpoints/exp_fp_adv/states.pth'),
    "Tacotron2Wave Model 1": Tacotron2Wave('checkpoints/exp_tc2/states_450.pth'),
    "Tacotron2Wave Model 2": Tacotron2Wave('pretrained/tacotron2_ar_mse.pth')
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

    # Save the wave to a file
    output_path = "static/output.wav"
    wave.save(output_path)

    # Return the index page with the audio URL
    audio_url = url_for('static', filename='output.wav')
    return render_template('index.html', models=models.keys(), audio_url=audio_url)

if __name__ == '__main__':
    app.run(debug=True)
