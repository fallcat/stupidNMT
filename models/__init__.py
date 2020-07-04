'''
Initialize the models module
'''
from models.transformer import Transformer
from models.new_transformer import NewTransformer

MODELS = {
    'transformer': Transformer,
    'new_transformer': NewTransformer,
}
