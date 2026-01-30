import pandas as pd
import json

    
def convert_mp32wav(mp3_file):
    pass


def save_hot2json():
    file = "data/药品频次.csv"
    df = pd.read_csv(file)
    res = []
    count = 0
    # 正序排序
    print(type(df["usage_count"].values[0]) )
    df = df.sort_values(by=['usage_count'], ascending=False)
    print(df["drug_name"])
    for word in df["drug_name"].values:
        if res and len(res) % 500==0:
            # save
            with open(f"asr/hotwords_{count:02d}.json", "w", encoding="utf-8") as f:
                json.dump(res, f, ensure_ascii=False, indent=2)
            res = []
            count +=1
        res.append({"text": word, "weight": 4, "lang": "zh"})


def write_hotwords():
    """"""
    file = "data/医保中草药数据_共11071条.csv"
    df = pd.read_csv(file)
    
    print(f"before unique {len(df['名称'].values)}")
    df.drop_duplicates(subset=["名称"], inplace=True)

    print(f"after unique {len(df['名称'].values)}")
    with open("asr/hotwords.txt", "w") as fp:
        for word in df["名称"].values:
            fp.write(f"{word}\n")



# if __name__ == "__main__":
#     # write_hotwords()
#     save_hot2json()