"""
audit dpodata确定数据存在正样本长度远小于负样本后，我们需要一个过滤脚本来把短于负样本一半的脚本过滤一下

"""
import json

def filter_dpo_data(input_p, output_p):
    with open(input_p, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    filtered_data = []
    dropped_count = 0
    
    for item in data:
        c_text = item['chosen'][-1]['content'] if isinstance(item['chosen'], list) else str(item['chosen'])
        r_text = item['rejected'][-1]['content'] if isinstance(item['rejected'], list) else str(item['rejected'])
        
        # 过滤策略：
        # 1. 剔除 Chosen 长度极短（少于150字符）且 Rejected 较长的样本
        # 2. 剔除 Chosen 长度不足 Rejected 40% 的样本
        is_too_short = len(c_text) < 150 and len(r_text) > 300
        is_unbalanced = len(c_text) < len(r_text) * 0.4
        
        if is_too_short or is_unbalanced:
            dropped_count += 1
            continue
            
        filtered_data.append(item)

    with open(output_p, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 清洗完成！")
    print(f"原始样本: {len(data)} | 剔除样本: {dropped_count} | 剩余样本: {len(filtered_data)}")
    print(f"保留比例: {len(filtered_data)/len(data)*100:.1f}%")
    print(f"新数据路径: {output_p}")

if __name__ == "__main__":
    filter_dpo_data("data/align/v3_dpo_train_2600.json", "data/align/v3_dpo_filtered.json")
