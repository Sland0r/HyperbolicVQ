with open("NLP/eval_recall.py", "r") as f:
    text = f.read()

import re
old = r"""            # codes may be \[n_q, batch\], but we want \[batch, n_q\]
            if codes\.size\(-1\) == batch_idx\.size\(0\) or \(?len\(codes\.shape\) == 2 and\s+codes\.size\(1\) == batch_idx\.size\(0\)\)?:
                codes = codes\.transpose\(0, 1\)
"""

new = """            codes = codes.squeeze(-1) if len(codes.shape) > 2 else codes
            if codes.size(1) == batch_idx.size(0) and codes.size(0) != batch_idx.size(0):
                codes = codes.transpose(0, 1)
"""

text = re.sub(old, new, text)

with open("NLP/eval_recall.py", "w") as f:
    f.write(text)
