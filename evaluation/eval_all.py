"""
专利代理师蒸馏模型 - 综合评估脚本
"""
import json
import os
from pathlib import Path


def load_test_set(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def check_keywords(output: str, expected: list) -> dict:
    """检查输出是否包含预期关键词"""
    results = {}
    for kw in expected:
        results[kw] = kw in output
    hit_rate = sum(results.values()) / len(results) if results else 0
    return {"keywords": results, "hit_rate": hit_rate}


def evaluate_task(task_name: str, test_file: str, model_generate_fn) -> dict:
    """评估单个任务"""
    test_data = load_test_set(test_file)
    results = []

    for case in test_data:
        output = model_generate_fn(case["input"])
        keyword_check = check_keywords(output, case.get("expected_output_contains", []))
        results.append({
            "id": case["id"],
            "keyword_hit_rate": keyword_check["hit_rate"],
            "keyword_details": keyword_check["keywords"],
        })

    avg_hit_rate = sum(r["keyword_hit_rate"] for r in results) / len(results) if results else 0
    return {
        "task": task_name,
        "total_cases": len(results),
        "avg_keyword_hit_rate": avg_hit_rate,
        "details": results,
    }


def run_evaluation(model_generate_fn, output_dir: str = "./evaluation/results"):
    """运行全量评估"""
    os.makedirs(output_dir, exist_ok=True)

    test_dir = Path("./data/test")
    test_files = {
        "creativity_evaluation": test_dir / "test_creativity.json",
        "patent_drafting": test_dir / "test_drafting.json",
        "office_action_response": test_dir / "test_office_action.json",
        "client_consultation": test_dir / "test_consultation.json",
    }

    all_results = {}
    for task_name, test_file in test_files.items():
        if test_file.exists():
            print(f"Evaluating: {task_name}...")
            result = evaluate_task(task_name, str(test_file), model_generate_fn)
            all_results[task_name] = result
            print(f"  Keyword hit rate: {result['avg_keyword_hit_rate']:.2%}")

    # 保存结果
    output_path = Path(output_dir) / "eval_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to: {output_path}")
    return all_results


if __name__ == "__main__":
    # 示例：用本地模型生成
    # from transformers import AutoModelForCausalLM, AutoTokenizer
    # model = AutoModelForCausalLM.from_pretrained("./output/patent-agent-model")
    # tokenizer = AutoTokenizer.from_pretrained("./output/patent-agent-model")
    # def generate(prompt):
    #     inputs = tokenizer(prompt, return_tensors="pt")
    #     outputs = model.generate(**inputs, max_new_tokens=2048)
    #     return tokenizer.decode(outputs[0], skip_special_tokens=True)
    # run_evaluation(generate)

    print("请配置 model_generate_fn 后运行评估")
