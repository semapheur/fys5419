### Tips

#### Display Latex from Python code in Jupyter notebook

```python
from IPython.display import Latex

def print_latex(string: str) -> None:
  display(Latex(rf"$${string}$$"))
```

#### Regex for converting Latex verbatim with lstlisting

```
\\begin{figure}.*?\n\\centering\n\\begin{verbatim}\n([\s\S]*?)\\end{verbatim}\n\\caption\[\]{(.*?)}\n\\label{(.*?)}\n\\end{figure}
```
