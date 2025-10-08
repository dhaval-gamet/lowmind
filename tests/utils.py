import pickle

class Dropout:
    def __init__(self, p=0.5):
        self.p = p
        self.training = True
    
    def __call__(self, x):
        if not self.training or self.p == 0:
            return x
        
        mask = (np.random.random(x.shape) > self.p) / (1 - self.p)
        return x * Tensor(mask, device=x.device)
    
    def eval(self):
        self.training = False
    
    def train(self):
        self.training = True

class torch:
    class no_grad:
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass

def save_model(model, filepath):
    """Save model parameters"""
    model_params = {}
    for name, param in model.named_parameters():
        model_params[name] = param.data
    
    with open(filepath, 'wb') as f:
        pickle.dump(model_params, f)
    print(f"✅ Model saved to {filepath}")

def load_model(model, filepath):
    """Load model parameters"""
    with open(filepath, 'rb') as f:
        model_params = pickle.load(f)
    
    for name, param in model.named_parameters():
        if name in model_params:
            param.data = model_params[name]
    print(f"✅ Model loaded from {filepath}")