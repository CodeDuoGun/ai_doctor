from pathlib import Path
from typing import List


def _build_sentences(name: str) -> List[str]:
    """
    基于单个中药“名字”构造 5 个不超过 30 字的句子，覆盖功效与用途。
    """
    name = name.strip()
    if not name:
        return []

    templates = [
        f"{name}常用于调理体质，日常保健。",
        f"{name}具有扶正固本功效，增强抵抗力。",
        f"{name}多入方剂，用于缓解相关不适。",
        f"{name}临床常配伍其他药材，协同增效。",
        f"合理使用{name}需遵医嘱，避免滥用。",
    ]

    # 确保每句长度不超过 30 字
    limited: List[str] = []
    for s in templates:
        if len(s) > 30:
            limited.append(s[:30])
        else:
            limited.append(s)
    return limited


def create_traindata(
    hotwords_path: str = "asr/hotwords.txt",
    output_path: str = "asr/trainasr_tcm.txt",
) -> None:
    """
    遍历中药药材文件 asr/hotwords.txt 中的每一行
    基于该“名字”从“功效、用途”两方面生成 5 个句子
    每个句子个数不超过 30 字，要求每个句子中必须出现完整“名字”
    所有句子最终保存为文件 asr/trainasr_tcm.txt，每个句子一行
    """
    hotwords_file = Path(hotwords_path)
    if not hotwords_file.exists():
        raise FileNotFoundError(f"未找到热词文件: {hotwords_file}")

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    lines_written = 0
    with hotwords_file.open("r", encoding="utf-8") as fin, out_file.open(
        "w", encoding="utf-8"
    ) as fout:
        for raw in fin:
            name = raw.strip()
            if not name:
                continue
            sentences = _build_sentences(name)
            for s in sentences:
                # 再次确保包含完整“名字”
                if name not in s:
                    s = f"{name}，{s}"
                fout.write(s + "\n")
                lines_written += 1

    print(f"生成训练句子完成，共写入 {lines_written} 行到 {out_file}")


if __name__ == "__main__":
    create_traindata()

