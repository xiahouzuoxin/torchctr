import sys
import argparse
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from .model_def import BaseServingModel
# import other ServingModel classes here if needed

# Add your serving models here
serving_models = {
    # 'name': {'path': 'path/to/model', 'serving_class': BaseServingModel, 'dep_paths': 'path/to/dependencies/if/needed'}
    # 'dnn': {'path': 'examples/ckpt/checkpoint.010568.ckpt', 'serving_class': BaseServingModel},
}

class ServeRequest(BaseModel):
    '''
    features: dict of features. For example, {'user_id': [1, 2, 3], 'item_id': [1, 2, 3]}
    '''
    features: dict

app = FastAPI()

@app.on_event('startup')
async def init_models():
    for name, model in serving_models.items():
        if model.get('dep_paths', None):
            sys.path.append(model['dep_paths'])
        serving_class = model.get('serving_class', BaseServingModel)
        if isinstance(serving_class, str):
            serving_class = eval(serving_class)
        serving_models[name]['model'] = serving_class(model['path'])
        print(f'Model {name} loaded from {model["path"]} successfully')

@app.post('/{name}/predict')
async def predict(name: str, req: ServeRequest):
    '''
    name: str, model name
    req: ServeRequest, request object. For example, {'features': {'user_id': [1, 2, 3], 'item_id': [1, 2, 3]}}
    '''
    if name not in serving_models:
        raise HTTPException(status_code=404, detail='Model not found')
    
    try:
        model = serving_models[name]['model']
        df = pd.DataFrame(req.features)
        return model.predict(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get('/{name}/health')
async def health(name: str):
    '''
    Check the health of the model.
    '''
    if name not in serving_models:
        raise HTTPException(status_code=404, detail='Model not found')
    return {'status': 'ok'}

@app.get('/health')
async def health():
    '''
    Check the health of the server.
    '''
    return {'status': 'ok'}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--path', type=str, default=None)
    parser.add_argument('--serving_class', type=str, default='BaseServingModel', help='Serving model class name')
    parser.add_argument('--dep_paths', type=str, default=None, help='Comma separated list of dependencies paths, for example, model class definition')
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()

    if args.name is not None:
        assert args.path is not None, 'Model path must be provided if name is specified'

        if args.name in serving_models:
            print(f'Model {args.name} already exists, will be updated.')
        serving_models[args.name] = {'path': args.path, 'serving_class': args.serving_class, 'dep_paths': args.dep_paths}

    if len(serving_models) == 0:
        print('No serving models found. Please add serving models to `serving_models` dictionary, exiting...')
        sys.exit(1)

    uvicorn.run(app, host='0.0.0.0', port=args.port)
