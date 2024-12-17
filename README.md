# Smart Camera

## How to Run

We assume you have git, conda and npm installed.

Clone the repository and create a conda environment

```bash
git clone https://github.com/qwercxzsda/smart-camera.git
cd smart-camera
conda env create -f environment.yml
conda activate smart-camera
```

Naviate to [hailo.ai](https://hailo.ai/developer-zone/software-downloads/) and download the HailoRT python wheel file.
Then, install the wheel file.

```bash
pip install [path to wheel file]
```

Build the web client

```bash
cd src/client
npm install
npm run build
```

Run the server

```bash
cd ../
uvicorn main:app --host 0.0.0.0 --port 8000
```

To use the webcam, you need to access the server using https. We recommend using [ngrok](https://ngrok.com/) to create a
secure tunnel to your localhost.

```bash
ngrok http 8000
```

That's it! You can now access the server by visiting the ngrok link through any web browser.
