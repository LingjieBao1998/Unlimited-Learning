import os
import requests
from multiprocessing import Pool
from tqdm import tqdm

def download_pdb(pdb_id):
    """下载单个 PDB 文件"""
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    response = requests.get(url)

    if response.status_code == 200:
        # 创建目录以保存文件
        os.makedirs('pdb_files', exist_ok=True)
        file_path = os.path.join('pdb_files', f"{pdb_id}.pdb")

        # 保存文件
        with open(file_path, 'wb') as f:
            f.write(response.content)
        return f"{pdb_id} downloaded."
    else:
        return f"Failed to download {pdb_id}: {response.status_code}"

def main(pdb_ids):
    # 使用进程池并行下载 PDB 文件，设置 20 个进程
    with Pool(processes=30) as pool:
        results = list(tqdm(pool.imap(download_pdb, pdb_ids), total=len(pdb_ids)))

    for result in results:
        print(result)

if __name__ == "__main__":
    # 替换为你想要下载的 PDB ID 列表
    pdb_ids = []
    with open("INDEX_general_PL.2020", "r") as f:
        for _, line in enumerate(f.readlines()):
            if _ >= 6:
                pdb_ids.append(line.split()[0])
    print("len(pdb_ids)",len(pdb_ids))
    # pdb_ids = ['1A1X', '1B2A', '3C4D', '4D4D', '5E5E', '6F6F', '7G7G', '8H8H', '9I9I', '1J1K',
            #    '2L2M', '3N3O', '4P4Q', '5R5S', '6T6U', '7V7W', '8X8Y', '9Z9A', '1B2C', '1D2E']  # 示例 PDB ID 列表
    main(pdb_ids)