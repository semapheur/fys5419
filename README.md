### Tips

#### Display Latex from Python code in Jupyter notebook

```python
from IPython.display import Latex

def print_latex(string: str) -> None:
  display(Latex(rf"$${string}$$"))
```