from __future__ import annotations

import csv
import subprocess
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import hf_hub_url


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
META_PATH = BASE_DIR / "meta.csv"
REPORT_DIR = BASE_DIR / "report"
TARGET_BYTES = 20 * 1024**3

FIELDS = [
    "index",
    "name",
    "category",
    "repo",
    "source_file",
    "url",
    "local_path",
    "bytes",
    "description",
]


@dataclass(frozen=True)
class Entry:
    name: str
    category: str
    repo: str
    source_file: str
    expected_bytes: int
    description: str


EXTRA_ENTRIES = [
    Entry("wdndev_webnovel_chinese_0", "小说、文学作品", "wdndev/webnovel-chinese", "data/webnovel_0.jsonl", 4183415364, "中文网文 JSONL 分片 0"),
    Entry("fjcanyue_wikipedia_zh_cn_20260501", "文献/概念说明", "fjcanyue/wikipedia-zh-cn", "wikipedia-zh-cn-20260501.json", 2393692848, "中文维基百科 JSON 语料 20260501"),
    Entry("fjcanyue_wikipedia_zh_cn_20260201", "文献/概念说明", "fjcanyue/wikipedia-zh-cn", "wikipedia-zh-cn-20260201.json", 2346115728, "中文维基百科 JSON 语料 20260201"),
    Entry("mxode_cmid_chinese_math_instruct", "数学/概念说明", "Mxode/CMID-Chinese_Math_Instruct_Dataset", "train.jsonl", 586788394, "中文数学指令 JSONL 数据集"),
    Entry("mxode_school_math_r1_distil", "数学/逻辑", "Mxode/School-Math-R1-Distil-Chinese-220K", "train.jsonl", 222148937, "中文学校数学推理 JSONL 数据集"),
    Entry("tick666_basic_math_chinese_v11", "数学", "TICK666/Basic-Math-Chinese-1M-V1.1", "Basic-Math-Chinese-1M-V1.1.json", 121445117, "中文基础数学 JSON 数据集"),
    Entry("mxode_math_chinese_deepseek_r1_10k", "数学/逻辑", "Mxode/Math-Chinese-DeepSeek-R1-10K", "train.jsonl", 39151402, "中文数学推理 JSONL 数据集"),
    Entry("ningjing_traditional_chinese_dictionary", "词典/概念说明", "NingJing0718/Traditional_Chinese_Dictionary_Preprocess", "allData2025.txt", 76328057, "繁体中文词典预处理文本"),
    Entry("ningjing_traditional_chinese_dictionary_extend", "词典/概念说明", "NingJing0718/Traditional_Chinese_Dictionary_Preprocess", "allData2025_extend.txt", 280037061, "繁体中文扩展词典文本"),
    Entry("ningjing_traditional_chinese_dictionary_synthesize", "词典/概念说明", "NingJing0718/Traditional_Chinese_Dictionary_Preprocess", "allData2025_synthesize.txt", 305534313, "繁体中文合成词典文本"),
    Entry("silk_road_50_chinese_novel_characters", "小说、文学作品/情感", "silk-road/50-Chinese-Novel-Characters", "role_in_novel_50.jsonl", 581308517, "中文小说角色对话 JSONL 数据集"),
    Entry("silk_road_alpaca_gpt4_chinese", "生活常识/概念说明", "silk-road/alpaca-data-gpt4-chinese", "Alpaca_data_gpt4_zh.jsonl", 78401244, "GPT-4 生成中文 Alpaca 指令 JSONL"),
    Entry("freedomintelligence_alpaca_gpt4_chinese", "生活常识/概念说明", "FreedomIntelligence/alpaca-gpt4-chinese", "alpaca-gpt4-chinese.json", 41144026, "中文 Alpaca GPT-4 JSON 数据集"),
    Entry("fengtc_alpaca_data_chinese_51k", "生活常识/概念说明", "fengtc/alpaca_data_chinese_51k", "alpaca_data.json", 18671162, "中文 Alpaca 51K JSON 数据集"),
    Entry("fengtc_alpaca_gpt4_data_zh", "生活常识/概念说明", "fengtc/alpaca_data_chinese_51k", "alpaca_gpt4_data_zh.json", 35240858, "中文 Alpaca GPT-4 JSON 数据集"),
    Entry("fengtc_trans_chinese_alpaca_data", "生活常识/概念说明", "fengtc/alpaca_data_chinese_51k", "trans_chinese_alpaca_data.json", 19582621, "翻译版中文 Alpaca JSON 数据集"),
    Entry("laurie_alpaca_chinese_dataset", "生活常识/概念说明", "Laurie/alpaca_chinese_dataset", "merged.json", 6474295, "合并版中文 Alpaca JSON 数据集"),
    Entry("llamafactory_dpo_zh_20k", "生活常识/情感", "llamafactory/DPO-En-Zh-20k", "dpo_zh.json", 28659101, "中文 DPO 偏好 JSON 数据集"),
    Entry("llamafactory_dpo_en_zh_20k", "生活常识/情感", "llamafactory/DPO-En-Zh-20k", "dpo_en.json", 51197315, "英中 DPO 偏好 JSON 数据集"),
    Entry("sylvan_tcm_traditional_trans", "生活常识/文献", "SylvanL/Traditional-Chinese-Medicine-Dataset-SFT", "_SFT_traditionalTrans_1959542.json", 682777905, "中医药传统转换 SFT JSON 数据集"),
    Entry("sylvan_tcm_struct_general", "生活常识/概念说明", "SylvanL/Traditional-Chinese-Medicine-Dataset-SFT", "SFT_structGeneral_310860.json", 266585970, "中医药结构化常识 SFT JSON 数据集"),
    Entry("sylvan_tcm_medical_knowledge_source1", "生活常识/文献", "SylvanL/Traditional-Chinese-Medicine-Dataset-SFT", "SFT_medicalKnowledge_source1_548404.json", 152158093, "中医药知识 SFT JSON 数据集"),
    Entry("sylvan_tcm_medical_knowledge_source2", "生活常识/文献", "SylvanL/Traditional-Chinese-Medicine-Dataset-SFT", "SFT_medicalKnowledge_source2_99334.json", 22950836, "中医药知识 SFT JSON 数据集"),
    Entry("sylvan_tcm_medical_knowledge_source3", "生活常识/文献", "SylvanL/Traditional-Chinese-Medicine-Dataset-SFT", "SFT_medicalKnowledge_source3_556540.json", 110211829, "中医药知识 SFT JSON 数据集"),
    Entry("sylvan_tcm_nlp_disease_diagnosed", "生活常识/概念说明", "SylvanL/Traditional-Chinese-Medicine-Dataset-SFT", "SFT_nlpDiseaseDiagnosed_61486.json", 108674171, "中医药疾病诊断 SFT JSON 数据集"),
    Entry("sylvan_tcm_nlp_syndrome_diagnosed", "生活常识/概念说明", "SylvanL/Traditional-Chinese-Medicine-Dataset-SFT", "SFT_nlpSyndromeDiagnosed_48665.json", 43102079, "中医药证候诊断 SFT JSON 数据集"),
    Entry("sylvan_tcm_struct_prescription", "生活常识/概念说明", "SylvanL/Traditional-Chinese-Medicine-Dataset-SFT", "SFT_structPrescription_92896.json", 42889034, "中医药处方结构化 SFT JSON 数据集"),
    Entry("pandalla_chinese_law_examples", "文献/逻辑", "pandalla/chinese_law_examples", "law_item.jsonl", 526776, "中文法律案例 JSONL 数据集"),
    Entry("tingwang_chinese_laws_civil_code", "文献/逻辑", "TingWang/Chinese_laws", "中华人民共和国民法典_20200528.txt", 330559, "中华人民共和国民法典文本"),
    Entry("tingwang_chinese_laws_criminal_law", "文献/逻辑", "TingWang/Chinese_laws", "中华人民共和国刑法_20201226.txt", 219882, "中华人民共和国刑法文本"),
    Entry("tingwang_chinese_laws_supervision_reg", "文献/逻辑", "TingWang/Chinese_laws", "中华人民共和国监察法实施条例_20250601.txt", 169424, "监察法实施条例文本"),
    Entry("tingwang_chinese_laws_criminal_procedure", "文献/逻辑", "TingWang/Chinese_laws", "中华人民共和国刑事诉讼法_20181026.txt", 117456, "刑事诉讼法文本"),
    Entry("tingwang_chinese_laws_civil_procedure", "文献/逻辑", "TingWang/Chinese_laws", "中华人民共和国民事诉讼法_20230901.txt", 104764, "民事诉讼法文本"),
    Entry("tingwang_chinese_laws_securities", "文献/逻辑", "TingWang/Chinese_laws", "中华人民共和国证券法_20191228.txt", 102242, "证券法文本"),
    Entry("tingwang_chinese_laws_company", "文献/逻辑", "TingWang/Chinese_laws", "中华人民共和国公司法_20231229.txt", 96793, "公司法文本"),
    Entry("tingwang_chinese_laws_maritime", "文献/逻辑", "TingWang/Chinese_laws", "中华人民共和国海商法_19921107.txt", 93635, "海商法文本"),
    Entry("tingwang_chinese_laws_food_safety", "生活常识/文献", "TingWang/Chinese_laws", "中华人民共和国食品安全法_20250912.txt", 91731, "食品安全法文本"),
    Entry("tingwang_chinese_laws_patent_rules", "文献/逻辑", "TingWang/Chinese_laws", "中华人民共和国专利法实施细则_20231211.txt", 78331, "专利法实施细则文本"),
    Entry("tingwang_chinese_laws_civil_aviation", "文献/逻辑", "TingWang/Chinese_laws", "中华人民共和国民用航空法_20210429.txt", 73406, "民用航空法文本"),
    Entry("tingwang_chinese_laws_hongkong_basic_law", "文献/逻辑", "TingWang/Chinese_laws", "中华人民共和国香港特别行政区基本法_19900404.txt", 69660, "香港特别行政区基本法文本"),
    Entry("zzhdbw_simplified_chinese_multi_emotion_dialogue", "情感", "zzhdbw/Simplified_Chinese_Multi-Emotion_Dialogue_Dataset", "Simplified_Chinese_Multi-Emotion_Dialogue_Dataset.csv", 289821, "简体中文多情感对话 CSV 数据集"),
    Entry("osuih_chinese_multi_emotion_dialogue", "情感", "osuih/Chinese_Multi-Emotion_Dialogue_Dataset", "data.csv", 314881, "中文多情感对话 CSV 数据集"),
    Entry("yuyijiong_multi_doc_qa_baike_001", "文献/概念说明", "yuyijiong/Multi-Doc-QA-Chinese", "old/百科-001.json", 1154713141, "中文百科多文档问答 JSON 数据集"),
    Entry("yuyijiong_multi_doc_qa_baike_002", "文献/概念说明", "yuyijiong/Multi-Doc-QA-Chinese", "old/百科-002.json", 1173148214, "中文百科多文档问答 JSON 数据集"),
    Entry("yuyijiong_multi_doc_qa_baike_003", "文献/概念说明", "yuyijiong/Multi-Doc-QA-Chinese", "old/百科-003.json", 822668212, "中文百科多文档问答 JSON 数据集"),
    Entry("yuyijiong_multi_doc_qa_baike_004", "文献/概念说明", "yuyijiong/Multi-Doc-QA-Chinese", "old/百科-004.json", 940934043, "中文百科多文档问答 JSON 数据集"),
    Entry("yuyijiong_multi_doc_qa_baike_005", "文献/概念说明", "yuyijiong/Multi-Doc-QA-Chinese", "old/百科-005.json", 933805262, "中文百科多文档问答 JSON 数据集"),
    Entry("yuyijiong_multi_doc_qa_baike_006", "文献/概念说明", "yuyijiong/Multi-Doc-QA-Chinese", "old/百科-006.json", 839800387, "中文百科多文档问答 JSON 数据集"),
    Entry("yuyijiong_multi_doc_qa_baike_007", "文献/概念说明", "yuyijiong/Multi-Doc-QA-Chinese", "old/百科-007.json", 1143874962, "中文百科多文档问答 JSON 数据集"),
    Entry("yuyijiong_multi_doc_qa_education_001", "生活常识/概念说明", "yuyijiong/Multi-Doc-QA-Chinese", "old/教育-001.json", 657973506, "中文教育多文档问答 JSON 数据集"),
    Entry("yuyijiong_multi_doc_qa_chatml", "文献/概念说明", "yuyijiong/Multi-Doc-QA-Chinese", "chatml/每条数据5个问答_max_32758_min_9275_num_4262.csv", 823103356, "中文多文档问答 ChatML CSV 数据集"),
    Entry("yuyijiong_multi_doc_qa_old_chatml", "文献/概念说明", "yuyijiong/Multi-Doc-QA-Chinese", "old_chatml/multi_doc_qa_5pos500_随机丢弃90_chatml_打乱_max_32767_min_24577_num_3903.csv", 830846374, "中文多文档问答旧版 ChatML CSV 数据集"),
]


def read_meta() -> list[dict[str, str]]:
    if not META_PATH.exists():
        return []
    with META_PATH.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def is_complete_file(path: Path, expected_bytes: int | None = None) -> bool:
    if not path.exists() or path.with_name(path.name + ".aria2").exists():
        return False
    if expected_bytes is not None and path.stat().st_size != expected_bytes:
        return False
    return True


def write_meta(rows: list[dict[str, str]]) -> None:
    with META_PATH.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def next_report_path() -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    numbers = []
    for path in REPORT_DIR.glob("*.txt"):
        try:
            numbers.append(int(path.stem))
        except ValueError:
            pass
    return REPORT_DIR / f"{max(numbers, default=0) + 1}.txt"


def slug_dir(index: int, name: str) -> Path:
    return DATA_DIR / f"{index:02d}_{name}"


def aria2_download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "aria2c",
        "--continue=true",
        "--max-connection-per-server=8",
        "--split=8",
        "--min-split-size=8M",
        "--retry-wait=10",
        "--max-tries=0",
        "--allow-overwrite=true",
        "--auto-file-renaming=false",
        "--dir",
        str(target.parent),
        "--out",
        target.name,
        url,
    ]
    subprocess.run(command, check=True)


def download(entry: Entry, target: Path) -> None:
    url = hf_hub_url(entry.repo, entry.source_file, repo_type="dataset")
    aria2_download(url, target)


def build_row(index: int, entry: Entry, target: Path) -> dict[str, str]:
    url = hf_hub_url(entry.repo, entry.source_file, repo_type="dataset")
    return {
        "index": str(index),
        "name": entry.name,
        "category": entry.category,
        "repo": entry.repo,
        "source_file": entry.source_file,
        "url": url,
        "local_path": str(target.relative_to(BASE_DIR.parent)),
        "bytes": str(target.stat().st_size),
        "description": entry.description,
    }


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for row in read_meta():
        local_path = BASE_DIR.parent / row["local_path"]
        try:
            expected_bytes = int(row["bytes"])
        except ValueError:
            expected_bytes = None
        if is_complete_file(local_path, expected_bytes):
            rows.append(row)
        else:
            print(f"dropping incomplete meta row: {row.get('name', '')} -> {local_path}")
    write_meta(rows)

    existing_names = {row["name"] for row in rows}
    next_index = max([int(row["index"]) for row in rows] or [0]) + 1
    downloaded: list[dict[str, str]] = []
    skipped: list[str] = []

    for entry in EXTRA_ENTRIES:
        if entry.name in existing_names:
            skipped.append(entry.name)
            continue

        target = slug_dir(next_index, entry.name) / Path(entry.source_file).name
        url = hf_hub_url(entry.repo, entry.source_file, repo_type="dataset")
        if not is_complete_file(target, entry.expected_bytes):
            print(f"[{next_index}] downloading {entry.name} -> {target}")
            download(entry, target)

        actual = target.stat().st_size
        if not is_complete_file(target, entry.expected_bytes):
            raise RuntimeError(f"{target} incomplete after download: {actual} != {entry.expected_bytes}")

        row = build_row(next_index, entry, target)
        rows.append(row)
        downloaded.append(row)
        existing_names.add(entry.name)
        write_meta(rows)
        next_index += 1

    total_bytes = sum(path.stat().st_size for path in DATA_DIR.rglob("*") if path.is_file())
    report = next_report_path()
    report.write_text(
        "\n".join(
            [
                "完成 dataset2/downloadTask.md 的公开中文数据集下载",
                "",
                "做了什么:",
                f"1. 新增下载 {len(downloaded)} 个公开中文数据文件到 dataset2/data。",
                f"2. 跳过已登记数据 {len(skipped)} 个，支持脚本断点续跑。",
                "3. 更新 dataset2/meta.csv，记录名称、类别、来源仓库、源文件、URL、本地路径和字节数。",
                "4. 覆盖小说/文学作品、文献、生活常识、概念说明、词典、逻辑、数学、情感等类别。",
                "",
                "结果:",
                f"meta.csv 行数: {len(rows)}",
                f"dataset2/data 总大小: {total_bytes} bytes ({total_bytes / 1024**3:.4f} GiB)",
                f"20G 目标: {'通过' if total_bytes >= TARGET_BYTES else '未通过'}",
                "",
                "新增数据清单:",
                *[
                    f"{row['index']}. {row['name']} | {row['category']} | {row['repo']} | {row['source_file']} | {row['bytes']} bytes"
                    for row in downloaded
                ],
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"meta_rows={len(rows)} total_bytes={total_bytes} report={report}")


if __name__ == "__main__":
    main()
