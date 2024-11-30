# Generalised Deepfake detection

## Directions
- Model available at the `./ml/model.py` file
- Hosting available at the `./app.py` file
  
## Getting Started
> Due to the size of the model and the nature of our competition, we did not include the following within our public github at the moment
> 1. The weights
> 2. The model's training and inference code
> 3. The model's hyperparameters
>    
> Please contact us directly if you wish for more details.

### Building the demo locally
You can run the application on docker by running 

`docker build -t howreal:latest . && docker run -p 80:8501 -d howreal:latest`

This will run the demo application and make it available on `localhost:80`

## Running Only the model
You can call the InferenceWorker at `./ml/InferenceWorker`. 

Simply call the following

```py
from ml.inference_worker import InferenceWorker
worker = InferenceWorker()
worker.load_model()
image = PIL.load(image_path)
pred, overlay = worker.predict_and_explain(image)
```
