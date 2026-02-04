
try:
    import pandas as pd
    print("pandas imported")
    import sklearn
    print("sklearn imported")
    import matplotlib
    print("matplotlib imported")
    import numpy as np
    print("numpy imported")
except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"Error: {e}")
