"""
Utility functions to make your models/datasets/evaluators go.
"""

def to_dict(obj):
    return {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}
