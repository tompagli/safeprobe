"""Shared utilities: JSON helpers, text formatting."""
import json, os
from typing import List, Dict

def ensure_json_file(path):
    if not os.path.exists(path) or os.path.getsize(path)==0:
        d=os.path.dirname(path)
        if d: os.makedirs(d, exist_ok=True)
        with open(path,"w",encoding="utf-8") as f: json.dump([],f)

def append_json_rows(path, rows):
    ensure_json_file(path)
    try:
        with open(path,"r",encoding="utf-8") as f: data=json.load(f)
        if not isinstance(data,list): data=[]
    except: data=[]
    data.extend(rows)
    with open(path,"w",encoding="utf-8") as f: json.dump(data,f,indent=2,ensure_ascii=False)

def format_asr(successful, total):
    return f"{successful/total*100:.1f}% ({successful}/{total})" if total>0 else "0.0% (0/0)"
