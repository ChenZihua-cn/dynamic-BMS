# prior/io.py
"""先验参数的落盘与读回。

上游 BMS 把校准结果存在 ``Prior/*.dat`` 里，用 ``fit_prior.read_prior_par``
读取。本项目沿用同样的"算一次、存一份、到处读"的用法，但把落盘格式固定
下来，并把解析写得宽容一些——上游 .dat 的具体分隔符版本不一，这里按
``key value`` / ``key: value`` / ``key = value`` / JSON / Python 字面量
五种写法逐层尝试，任何一种都能读进来。
"""

import ast
import json
from pathlib import Path


def read_prior_par(path):
    """从文件读取先验超参数字典。

    依次尝试：JSON → Python 字面量 → 逐行解析。逐行解析会跳过 ``#`` 注释
    与空行，并兼容 ``key value``、``key: value``、``key = value``、CSV
    四种分隔写法。

    Parameters
    ----------
    path : str or Path
        先验文件路径。支持 ``.json`` / ``.dat`` / ``.txt``。

    Returns
    -------
    dict
        ``{键: float}``。

    Raises
    ------
    ValueError
        文件中没有解析出任何 ``键-数值`` 对。
    """
    text = Path(path).read_text()

    # 1) JSON
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return dict([(str(k), float(v)) for k, v in obj.items()])
    except Exception:
        pass

    # 2) Python 字面量（上游 .dat 有些是 repr 出来的 dict）
    try:
        obj = ast.literal_eval(text.strip())
        if isinstance(obj, dict):
            return dict([(str(k), float(v)) for k, v in obj.items()])
    except Exception:
        pass

    # 3) 逐行解析
    out = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        if ':' in line:
            key, val = line.split(':', 1)
        elif '=' in line:
            key, val = line.split('=', 1)
        elif ',' in line:
            key, val = line.split(',', 1)
        else:
            parts = line.split()
            if len(parts) < 2:
                continue
            key, val = parts[0], parts[1]
        key = key.strip().strip('"\'')
        val = val.strip().strip('"\'').rstrip(',')
        try:
            out[key] = float(val)
        except ValueError:
            continue
    if not out:
        raise ValueError('No key-value pairs parsed from %s' % path)
    return out


def write_prior_par(path, prior_par, header=None):
    """把先验参数字典写到文件。

    ``.json`` 后缀写 JSON（带 2 空格缩进，方便 diff）；其余后缀写上游风格
    的 ``key value`` 纯文本，可被 :func:`read_prior_par` 原样读回。

    Returns
    -------
    Path
        写入的文件路径。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == '.json':
        with open(path, 'w') as fh:
            json.dump(prior_par, fh, indent=2, sort_keys=True)
            fh.write('\n')
    else:
        lines = []
        if header:
            for h in str(header).splitlines():
                lines.append('# %s' % h)
        for k in sorted(prior_par):
            lines.append('%s %r' % (k, float(prior_par[k])))
        path.write_text('\n'.join(lines) + '\n')
    return path


def export_registry(data_dir, fmt='json'):
    """把登记表里所有先验导出到 ``data_dir``，供外部工具/其他语言读取。

    Returns
    -------
    list of Path
        写出的文件列表。
    """
    from .registry import PRIOR_REGISTRY
    data_dir = Path(data_dir)
    written = []
    for name, spec in PRIOR_REGISTRY.items():
        target = data_dir / ('%s.%s' % (name, fmt))
        written.append(write_prior_par(target, spec.build(),
                                       header='%s — %s' % (name, spec.source)))
    return written
