class ExpFilter:
    """
    一个指数滤波器类

    属性:
        _alpha (float): 滤波系数
        _filtered (float): 滤波后的输出值
        _max_val (float): 输出的最大值限制（默认为-1.0，表示无限制）

    方法:
        __init__(self, alpha: float, max_val: float = -1.0): 
            初始化滤波器，设置滤波系数和最大值限制，并初始化滤波输出值

        reset(self, alpha: float = -1.0) -> None: 
            重置滤波器，如果提供了新的 alpha 值则更新，同时重置滤波输出值

        apply(self, exp: float, sample: float) -> float: 
            应用滤波操作，根据输入的指数 exp、样本值 sample 进行滤波计算，并处理最大值限制，返回滤波后的结果

        filtered(self) -> float: 
            获取当前滤波后的输出值

        update_base(self, alpha: float) -> None: 
            更新滤波系数
    """
    def __init__(self, alpha: float, max_val: float = -1.0) -> None:
        """
        初始化滤波器

        参数:
            alpha (float): 滤波系数
            max_val (float, 可选): 输出的最大值限制，默认为-1.0（无限制）
        """
        self._alpha = alpha
        self._filtered = -1.0
        self._max_val = max_val

    def reset(self, alpha: float = -1.0) -> None:
        """
        重置滤波器

        参数:
            alpha (float, 可选): 新的滤波系数，如果为-1.0则保持不变
        """
        if alpha!= -1.0:
            self._alpha = alpha
        self._filtered = -1.0

    def apply(self, exp: float, sample: float) -> float:
        """
        对输入样本应用滤波操作

        参数:
            exp (float): 指数
            sample (float): 输入样本值

        返回:
            float: 滤波后的结果
        """
        if self._filtered == -1.0:
            self._filtered = sample
        else:
            a = self._alpha**exp
            self._filtered = a * self._filtered + (1 - a) * sample

        if self._max_val!= -1.0 and self._filtered > self._max_val:
            self._filtered = self._max_val

        return self._filtered

    def filtered(self) -> float:
        """
        获取当前滤波后的输出值

        返回:
            float: 滤波后的输出值
        """
        return self._filtered

    def update_base(self, alpha: float) -> None:
        """
        更新滤波系数

        参数:
            alpha (float): 新的滤波系数
        """
        self._alpha = alpha