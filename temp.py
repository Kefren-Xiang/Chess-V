"""
temp.py
临时文件，不用管。
"""

import csv

def print_csv_head(filename="Chess_Information_I.csv", n=5):
    try:
        with open(filename, newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            
            # 打印表头
            header = next(reader)
            print("[CSV 文件前 {} 行]".format(n))
            print(" | ".join(header))
            print("-" * 80)
            
            # 打印前n行数据
            for i, row in enumerate(reader):
                if i >= n:
                    break
                print(" | ".join(f"{col:<12}" for col in row))
                
    except FileNotFoundError:
        print(f"错误：找不到文件 {filename}")

if __name__ == "__main__":
    print_csv_head(n=5)  # 修改n的值可以查看不同行数