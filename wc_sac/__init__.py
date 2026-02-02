<<<<<<< HEAD
from tensorflow.python.util import deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

from wc_sac.sac.saclag import sac
from wc_sac.sac.wcsac import sac
=======
"""
wc_sac 包入口。

注意：
- 数据预处理/环境模块不应该强依赖 TensorFlow。
- 原始实现会在 import wc_sac 时立刻 import tensorflow 与训练代码，导致在仅做数据处理时也无法使用。

因此这里把 TensorFlow 与训练入口改为可选导入：只有在你确实运行训练脚本时才需要安装 tf1.x。
"""

try:
    from tensorflow.python.util import deprecation as deprecation  # type: ignore

    deprecation._PRINT_DEPRECATION_WARNINGS = False
except Exception:
    # 允许在未安装 TensorFlow 的环境下使用 dataset/env 等模块
    pass

# 训练入口：可选（需要 TensorFlow）
try:
    from wc_sac.sac.saclag import sac as saclag  # noqa: F401
    from wc_sac.sac.wcsac import sac as wcsac  # noqa: F401
except Exception:
    pass
>>>>>>> 90bd2a2252c4e34920844c44c02da949381d401c
