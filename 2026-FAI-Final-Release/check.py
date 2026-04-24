import json

file_path = "results/tournament/2026-04-24_21-19-48_player_1.json"

with open(file_path, "r") as f:
    data = json.load(f)

total_timeouts = 0
total_exceptions = 0
total_dq = 0

def scan_for_errors(node):
    global total_timeouts, total_exceptions, total_dq
    
    if isinstance(node, dict):
        # 尋找超時紀錄
        if "timeout_counts" in node:
            tc = node["timeout_counts"]
            if isinstance(tc, dict):
                total_timeouts += sum(tc.values())
            elif isinstance(tc, int):
                total_timeouts += tc
                
        # 尋找程式錯誤紀錄
        if "exception_counts" in node:
            ec = node["exception_counts"]
            if isinstance(ec, dict):
                total_exceptions += sum(ec.values())
            elif isinstance(ec, int):
                total_exceptions += ec
                
        # 尋找失格紀錄
        if "disqualified_players" in node:
            dq = node["disqualified_players"]
            if isinstance(dq, list):
                total_dq += len(dq)
            elif isinstance(dq, int):
                total_dq += dq
                
        # 繼續往下挖
        for value in node.values():
            scan_for_errors(value)
            
    elif isinstance(node, list):
        # 如果遇到陣列，把裡面的東西拿出來繼續挖
        for item in node:
            scan_for_errors(item)

# 啟動全圖掃描
scan_for_errors(data)

print("=== 穩定度檢查報告 ===")
print(f"總超時次數 (Timeouts): {total_timeouts}")
print(f"總錯誤次數 (Exceptions): {total_exceptions}")
print(f"總失格次數 (Disqualifications): {total_dq}")

if total_timeouts == 0 and total_exceptions == 0 and total_dq == 0:
    print("完美！你的 Agent 執行速度極快且沒有任何錯誤！")
else:
    print("請注意，有發生上述的異常狀況，需要再優化效能喔！")