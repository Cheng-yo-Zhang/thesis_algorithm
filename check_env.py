import sys
print(f"Python version: {sys.version}")
try:
    import pandas
    print("pandas imported")
    import matplotlib
    print("matplotlib imported")
    import numpy
    print("numpy imported")
    import sklearn
    print("sklearn imported")
    print("Environment OK")
except Exception as e:
    print(f"Import failed: {e}")
