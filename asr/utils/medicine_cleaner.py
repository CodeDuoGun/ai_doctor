"""
中药材剂量提取和清洗模块
从 ASR 文本中提取结构化的药材剂量单位，过滤噪音
"""

import re
from dataclasses import dataclass, field
from typing import Optional
from utils.log import logger


@dataclass
class MedicineDose:
    """单味药材剂量信息"""
    name: str           # 药材名称
    dose: str           # 剂量数值
    unit: str           # 单位（克、两、斤等）
    raw_text: str = ""  # 原始文本片段
    is_valid: bool = True


@dataclass
class CleaningResult:
    """清洗结果"""
    original_text: str                    # 原始输入文本
    cleaned_text: str = ""               # 清洗后的文本（移除噪音）
    medicines: list[MedicineDose] = field(default_factory=list)  # 提取的药材列表
    noise_removed: list[str] = field(default_factory=list)         # 被移除的噪音内容


class MedicineCleaner:
    """中药材剂量清洗器"""
    
    # 常见中药材名称（可以扩展）
    COMMON_MEDICINES = [
        # 根茎类
        "人参", "黄芪", "当归", "川芎", "白芍", "茯苓", "白术", "甘草", "党参",
        "熟地", "生地", "山药", "泽泻", "丹皮", "牡丹皮", "陈皮", "半夏", "生姜",
        "大黄", "黄连", "黄柏", "栀子", "柴胡", "葛根", "升麻", "桔梗", "杏仁",
        "桃仁", "红花", "赤芍", "地龙", "全蝎", "蜈蚣", "羌活", "独活", "威灵仙",
        "秦艽", "桑寄生", "杜仲", "续断", "牛膝", "狗脊", "骨碎补", "补骨脂",
        "肉苁蓉", "锁阳", "淫羊藿", "菟丝子", "沙苑子", "枸杞子", "菊花", "金银花",
        "连翘", "板蓝根", "大青叶", "薄荷", "荆芥", "防风", "苍耳子", "辛夷",
        "藁本", "白芷", "细辛", "川乌", "草乌", "附子", "干姜", "肉桂", "吴茱萸",
        "木香", "沉香", "檀香", "香附", "乌药", "枳壳", "枳实", "厚朴", "砂仁",
        "白豆蔻", "草豆蔻", "茴香", "丁香", "高良姜", "莱菔子", "山楂", "麦芽",
        "谷芽", "神曲", "鸡内金", "使君子", "槟榔", "苦楝皮", "南瓜子", "雷丸",
        # 果实种子类
        "酸枣仁", "柏子仁", "远志", "合欢皮", "夜交藤", "茯神", "灵芝", "天麻",
        "钩藤", "石决明", "牡蛎", "龙骨", "磁石", "朱砂", "琥珀", "珍珠母",
        # 叶类
        "桑叶", "荷叶", "竹叶", "番泻叶", "大青叶", "艾叶", "紫苏叶", "侧柏叶",
        # 花类
        "玫瑰花", "月季花", "红花", "旋覆花", "款冬花", "金银花", "野菊花", "蒲公英",
        # 皮类
        "地骨皮", "黄柏", "椿皮", "桑白皮", "五加皮", "合欢皮", "石榴皮",
        # 藤木类
        "络石藤", "海风藤", "青风藤", "雷公藤", "鸡血藤", "夜交藤", "钩藤",
        # 菌类
        "茯苓", "猪苓", "泽泻", "雷丸", "马勃", "冬虫夏草", "虫草",
        # 动物类
        "鹿茸", "鹿角", "阿胶", "龟板", "鳖甲", "蛤蚧", "乌梢蛇", "蕲蛇", "白花蛇",
        # 矿物类
        "石膏", "知母", "寒水石", "龙齿", "代赭石", "磁石", "自然铜", "雄黄",
        "朱砂", "炉甘石", "滑石", "芒硝", "玄明粉", "硼砂", "白矾", "皂矾",
        # 其他
        "乳香", "没药", "血竭", "儿茶", "冰片", "樟脑", "蟾酥", "麝香", "牛黄",
    ]
    
    # 剂量单位
    DOSE_UNITS = [
        ("克", ["g", "gram", "克", "公克"]),
        ("两", ["两", "市两"]),
        ("斤", ["斤", "市斤"]),
        ("公斤", ["kg", "公斤", "千克"]),
        ("毫克", ["mg", "毫克"]),
        ("毫升", ["ml", "毫升"]),
        ("升", ["l", "升"]),
        ("个", ["个", "枚", "颗"]),
        ("片", ["片"]),
        ("把", ["把"]),
        ("撮", ["撮"]),
        ("钱", ["钱"]),
    ]
    
    # 噪音模式
    NOISE_PATTERNS = [
        r"噪音[：:]*\s*",
        r"杂音[：:]*\s*",
        r"[嗯啊哦唉呃呀嘿哈]{2,}",  # 语气词
        r"说[完了]+",  # "说完了"等
        r"[的了吗呢啊呀哦]?\s*$",  # 句末语气词
        r"^[的了吗呢啊呀哦]+\s*",  # 句首语气词
        r"\s+",  # 多余空格
        r"[。]{2,}",  # 多个句号
        r"[？]{2,}",  # 多个问号
        r"[！]{2,}",  # 多个感叹号
        r"[\s]*[,，][\s]*",  # 逗号周围空格
        r"[\s]*[。][\s]*",  # 句号周围空格
    ]
    
    def __init__(self):
        # 构建药材正则模式
        medicine_pattern = "|".join(self.COMMON_MEDICINES)
        self.medicine_pattern = re.compile(f"({medicine_pattern})")
        
        # 构建剂量正则模式
        # 匹配格式：数字 + 单位 或 单位 + 数字
        # 如：10克、5两、3片、半斤、一两半
        self.dose_pattern = re.compile(
            r"(?:"
            r"(\d+\.?\d*)\s*(" + "|".join([u for u, _ in self.DOSE_UNITS]) + r")"  # 数字+单位
            r"|"
            r"(" + "|".join([u for u, _ in self.DOSE_UNITS]) + r")\s*(\d+\.?\d*)"  # 单位+数字
            r"|"
            r"(半|一|二|三|四|五|六|七|八|九|十)"  # 中文数字
            + r"(?:个?)?\s*(" + "|".join([u for u, _ in self.DOSE_UNITS]) + r")"  # 中文数+单位
            r"|"
            r"(\d+\.?\d*)\s*(?:克|公克|两|斤|公斤|千克|毫克|毫升|升|个|片|把|撮|钱)"  # 数字+单位简写
            r"|"
            r"(?:每|每次)\s*(\d+\.?\d*)\s*(" + "|".join([u for u, _ in self.DOSE_UNITS]) + ")"  # 每次X克
            r")",
            re.IGNORECASE
        )
        
        # 噪音正则模式
        self.noise_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.NOISE_PATTERNS
        ]
    
    def clean(self, text: str) -> CleaningResult:
        """
        清洗输入文本
        
        Args:
            text: ASR 原始文本
            
        Returns:
            CleaningResult: 包含清洗结果的对象
        """
        logger.info(f"开始清洗文本: {text}")
        
        result = CleaningResult(original_text=text)
        
        # 1. 移除噪音
        cleaned = self._remove_noise(text)
        result.cleaned_text = cleaned
        logger.info(f"移除噪音后: {cleaned}")
        
        # 2. 提取药材剂量
        medicines = self._extract_medicines(cleaned)
        result.medicines = medicines
        
        # 3. 生成最终清洗文本（保留药材信息，移除其他噪音）
        final_text = self._generate_cleaned_text(cleaned, medicines)
        result.cleaned_text = final_text
        
        logger.info(f"提取到 {len(medicines)} 味药材")
        for m in medicines:
            logger.info(f"  - {m.name}: {m.dose}{m.unit}")
        
        return result
    
    def _remove_noise(self, text: str) -> str:
        """移除文本中的噪音"""
        cleaned = text
        
        # 移除语气词和杂音
        noise_removals = []
        for pattern in self.noise_patterns:
            new_cleaned, count = pattern.subn("", cleaned)
            if count > 0:
                cleaned = new_cleaned
                noise_removals.append(f"Pattern {pattern.pattern}: {count} matches")
        
        # 清理多余空格
        cleaned = re.sub(r"\s+", " ", cleaned)
        cleaned = cleaned.strip()
        
        return cleaned
    
    def _extract_medicines(self, text: str) -> list[MedicineDose]:
        """从文本中提取药材剂量"""
        medicines = []
        
        # 1. 找到所有可能的药材+剂量组合
        # 匹配模式：药材名 + (剂量 | 空白 + 剂量)
        
        # 先找出所有药材名
        medicine_matches = list(self.medicine_pattern.finditer(text))
        
        for match in medicine_matches:
            medicine_name = match.group(1)
            
            # 查找该药材后面的剂量信息
            start_pos = match.end()
            end_pos = min(start_pos + 20, len(text))  # 在20字符范围内查找剂量
            context = text[start_pos:end_pos]
            
            # 匹配剂量
            dose_match = self.dose_pattern.search(context)
            if dose_match:
                dose_info = self._parse_dose_match(dose_match, medicine_name)
                if dose_info:
                    dose_info.raw_text = text[match.start():match.end() + dose_match.end()]
                    medicines.append(dose_info)
                    logger.info(f"提取到: {dose_info.name} {dose_info.dose}{dose_info.unit}")
            else:
                # 没有找到剂量，记录为无剂量的药材
                med = MedicineDose(
                    name=medicine_name,
                    dose="",
                    unit="",
                    raw_text=match.group(),
                    is_valid=False
                )
                medicines.append(med)
        
        # 去重（保留第一个出现的）
        seen = set()
        unique_medicines = []
        for med in medicines:
            key = med.name
            if key not in seen:
                seen.add(key)
                unique_medicines.append(med)
        
        return unique_medicines
    
    def _parse_dose_match(self, match, medicine_name: str) -> Optional[MedicineDose]:
        """解析剂量匹配结果"""
        # match groups:
        # g1: 数字, g2: 单位
        # g3: 单位, g4: 数字
        # g5: 中文数, g6: 单位
        # g7: 数字 (简写单位)
        # g8: 数字 (每次), g9: 单位 (每次)
        
        if match.group(1) and match.group(2):
            # 数字 + 单位
            return MedicineDose(
                name=medicine_name,
                dose=match.group(1),
                unit=match.group(2),
                is_valid=True
            )
        elif match.group(3) and match.group(4):
            # 单位 + 数字
            return MedicineDose(
                name=medicine_name,
                dose=match.group(4),
                unit=match.group(3),
                is_valid=True
            )
        elif match.group(5) and match.group(6):
            # 中文数 + 单位
            chinese_num = self._convert_chinese_num(match.group(5))
            return MedicineDose(
                name=medicine_name,
                dose=chinese_num,
                unit=match.group(6),
                is_valid=True
            )
        elif match.group(7):
            # 数字 + 简写单位
            unit = self._get_full_unit(match.group(7))
            if unit:
                return MedicineDose(
                    name=medicine_name,
                    dose=match.group(7),
                    unit=unit,
                    is_valid=True
                )
        elif match.group(8) and match.group(9):
            # 每次 + 数字 + 单位
            return MedicineDose(
                name=medicine_name,
                dose=match.group(8),
                unit=match.group(9),
                is_valid=True
            )
        
        return None
    
    def _convert_chinese_num(self, num: str) -> str:
        """将中文数字转换为阿拉伯数字"""
        chinese_to_arabic = {
            "半": "0.5",
            "一": "1", "二": "2", "三": "3", "四": "4",
            "五": "5", "六": "6", "七": "7", "八": "8", "九": "9", "十": "10"
        }
        return chinese_to_arabic.get(num, num)
    
    def _get_full_unit(self, unit: str) -> str:
        """将简写单位转换为全称"""
        for full, shorts in self.DOSE_UNITS:
            if unit.lower() in [s.lower() for s in shorts]:
                return full
        return unit
    
    def _generate_cleaned_text(self, text: str, medicines: list[MedicineDose]) -> str:
        """生成最终的清洗文本"""
        # 移除药材名和剂量信息，保留其他内容
        cleaned = text
        
        # 移除已识别的药材剂量文本
        for med in medicines:
            if med.raw_text:
                cleaned = cleaned.replace(med.raw_text, f"[{med.name}]")
        
        # 清理多余空格
        cleaned = re.sub(r"\s+", " ", cleaned)
        cleaned = cleaned.strip()
        
        return cleaned
    
    def extract_structure(self, text: str) -> dict:
        """
        提取结构化数据
        
        Returns:
            dict: 包含 medicines, doses, cleaned_text 的字典
        """
        result = self.clean(text)
        
        return {
            "original_text": result.original_text,
            "cleaned_text": result.cleaned_text,
            "medicines": [
                {
                    "name": m.name,
                    "dose": m.dose,
                    "unit": m.unit,
                    "is_valid": m.is_valid
                }
                for m in result.medicines
            ],
            "count": len(result.medicines)
        }


# 便捷函数
def clean_medicine_text(text: str) -> dict:
    """
    便捷清洗函数
    
    Args:
        text: ASR 原始文本
        
    Returns:
        dict: 包含清洗结果的字典
    """
    cleaner = MedicineCleaner()
    return cleaner.extract_structure(text)


if __name__ == "__main__":
    # 测试
    cleaner = MedicineCleaner()
    
    test_cases = [
        "黄芪10克，当归5克，",
        "人参15克炖汤",
        "枸杞子一把泡水喝",
        "噪音：这个那个嗯啊",
        "白术三两，半夏10克，甘草5克",
        "每次5克，每天两次",
        "熟地黄15克山药10克泽泻8克",
    ]
    
    for text in test_cases:
        print(f"\n输入: {text}")
        result = cleaner.clean(text)
        print(f"清洗后: {result.cleaned_text}")
        print(f"药材: {[m.name for m in result.medicines]}")

