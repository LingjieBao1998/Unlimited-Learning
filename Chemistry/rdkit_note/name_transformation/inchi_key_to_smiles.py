import requests
from rdkit import Chem


def inchi_key_to_smiles(inchi_key_with_prefix):
    """
    输入可能带 'InChIKey=' 前缀的字符串，提取 InChIKey 并转换为 SMILES
    参数:
        inchi_key_with_prefix (str): 如 'InChIKey=FRJSECSOXKQMOD-HQRMLTQVSA-N'
    返回:
        str: 对应的 SMILES，如果查询失败则返回 None
    """
    # 1. 去掉 'InChIKey=' 前缀，提取纯净的 InChIKey
    if inchi_key_with_prefix.startswith('InChIKey='):
        inchi_key = inchi_key_with_prefix[len('InChIKey='):]  # 提取: 'FRJSECSOXKQMOD-HQRMLTQVSA-N'
    else:
        inchi_key = inchi_key_with_prefix  # 假设传入的就是纯 InChIKey

    print(f"🔍 正在查询 InChIKey: {inchi_key}")

    # 2. 通过 InChIKey 查询 PubChem CID
    url_cids = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/{inchi_key}/cids/JSON"
    try:
        response = requests.get(url_cids, timeout=10)
        if response.status_code == 200:
            data = response.json()
            cids = data.get('IdentifierList', {}).get('CID', [])
            if not cids:
                print(f"❌ 未找到 InChIKey={inchi_key} 对应的 PubChem CID")
                return None
            cid = cids[0]  # 取第一个 CID（通常是最匹配的）
            print(f"✅ 找到 CID: {cid}")

            # 3. 通过 CID 查询 SMILES
            url_smiles = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/CanonicalSMILES/JSON"
            response_smiles = requests.get(url_smiles, timeout=10)
            if response_smiles.status_code == 200:
                smiles_data = response_smiles.json()
                props = smiles_data.get('PropertyTable', {}).get('Properties', [])
                if props:
                    smiles = props[0].get('ConnectivitySMILES')
                    print(f"✅ SMILES: {smiles}")
                    return smiles
                else:
                    print(f"❌ 未找到 SMILES 数据 (CID: {cid})")
            else:
                print(f"❌ 查询 SMILES 失败，HTTP 状态码: {response_smiles.status_code}")
        else:
            print(f"❌ 查询 CID 失败，HTTP 状态码: {response.status_code}")
    except Exception as e:
        print(f"❌ 发生错误: {e}")

    return None

if __name__ == "__main__":
    inchi_key = 'InChIKey=FRJSECSOXKQMOD-HQRMLTQVSA-N'
    smiles = inchi_key_to_smiles(inchi_key)
    assert Chem.MolToSmiles(Chem.MolFromSmiles(smiles)) == Chem.MolToSmiles(Chem.MolFromSmiles('CC1=C2CCC3(CCC=C(C3CC(C2(C)C)CC1)C)C'))

    
