import re
import hashlib
import jsonlines
from tqdm import tqdm
from datasketch import MinHash, MinHashLSH

class RigorousDedup:
    def __init__(self, threshold=0.85, num_perm=128):
        # LSH 索引，用于模糊匹配
        self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        # 存储精确哈希，已见的MD5,用于第一层过滤
        self.exact_hashes = set()
        self.num_perm = num_perm

    # 通过一些正则表达式进行代码标准化
    def _normalize_code(self, code):
        """
        严谨的标准化：移除注释、空行、统一缩进
        """
        # 1. 移除多行注释 '''...''' 或 """..."""
        code = re.sub(r'(""".*?"""|\'\'\'.*?\'\'\')', '', code, flags=re.DOTALL)  # 这里有个非贪婪匹配，加上flags这个标记点号才能匹配换行符
        # 2. 移除单行注释 #...
        code = re.sub(r'#.*', '', code)
        # 3. 移除多余空行并将多个空格合并
        lines = [line.strip() for line in code.split('\n') if line.strip()]  # 去除首尾后为空就被过滤掉了
        return "\n".join(lines)  # 列表存放每行代码，重新用换行符连接


    def get_minhash(self, text):
        """计算 MinHash 指纹"""
        # num_perm 是哈希函数的数量，越多越精确但越慢
        m = MinHash(num_perm=self.num_perm)

        # 使用 3-gram 切片提高灵敏度
        tokens = [text[i:i+3] for i in range(len(text)-2)]
        """
        滑动窗口的作用：它把代码拆解成了细小的“逻辑碎片”。
        即使你改了变量名、多了一个空格，大部分碎片依然是一样的。
        这就给了我们实现“模糊”比对的基础数据。
        """
        for t in tokens:
            m.update(t.encode('utf8'))
        return m

    def process(self, input_file, output_file):
        count_total = 0
        count_exact = 0
        count_fuzzy = 0
        
        with jsonlines.open(input_file) as reader, jsonlines.open(output_file, mode='w') as writer:
            for item in tqdm(reader, desc="Deduplicating"):
                count_total += 1
                raw_code = item['output']
                normalized_code = self._normalize_code(raw_code)
                
                # --- 第一级：精确去重 (Exact Hash) ---
                exact_hash = hashlib.md5(normalized_code.encode('utf-8')).hexdigest()
                if exact_hash in self.exact_hashes:
                    count_exact += 1
                    continue
                
                # --- 第二级：模糊去重 (MinHash + LSH) ---
                m = self.get_minhash(normalized_code)
                # 查询是否存在相似度 > threshold 的项
                # 模糊去重这里还有些问题没看明白，例如多段匹配怎么办？
                is_duplicate = self.lsh.query(m)
                
                if is_duplicate:
                    count_fuzzy += 1
                    continue
                
                # 通过所有检查，写入并记录
                self.exact_hashes.add(exact_hash)
                self.lsh.insert(f"item_{count_total}", m)
                writer.write(item)

        print(f"\n[去重报告]")
        print(f"原始总量: {count_total}")
        print(f"精确去重(逻辑完全一致)删除: {count_exact}")
        print(f"模糊去重(高度相似)删除: {count_fuzzy}")
        print(f"最终保留: {count_total - count_exact - count_fuzzy}")

if __name__ == "__main__":
    # 工业界常用阈值 0.85
    deduplicator = RigorousDedup(threshold=0.85)
    deduplicator.process("data/processed/magicoder_cleaned.jsonl", "data/processed/magicoder_final.jsonl")