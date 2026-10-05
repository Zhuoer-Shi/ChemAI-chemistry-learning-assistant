#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ChemAI 三端原型的统一启动入口（发布环境使用）。

等价于 `python server.py`，方便托管平台自动识别启动文件。
"""

from server import main

if __name__ == "__main__":
    main()
