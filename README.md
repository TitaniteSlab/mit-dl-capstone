# Chess Neural Network

A chess engine neural network.

## Train the model

`notebooks/` contains the Jupyter notebooks used for training. These are inteded to be run on Google Colab, and the output model is uploaded to Google Drive.

## Build the engine

First build the executable:

```bash
make build-pyinstaller
chmod +x dist/engine1
```

Outputs a single executable to `dist/engine1`. This still requires a model to run, which should be downloaded from Google Drive and placed in the same folder as `dist/model.pt`.

## Run the engine

In your chess program (e.g., BanksiaGUI) set the engine path to `dist/engine1` and the working directory to `dist/`
