#!/bin/bash

# 設定輸出日誌檔案
LOG_FILE="test_results.log"
echo "--- 測試開始時間: $(date) ---" > $LOG_FILE

# 輔助函式：執行並過濾結果
run_test() {
    local map_path=$1
    local method=$2
    echo "Testing: $map_path with method: $method" | tee -a $LOG_FILE
    
    # 執行指令，將 stderr 導向 stdout，並過濾出關鍵字
    python hw1.py "$map_path" --method "$method" 2>&1 | grep -E "Path Length|States Explored|Total time" | tee -a $LOG_FILE
    echo "------------------------------------------" >> $LOG_FILE
}

# 1. single 資料夾：測試 bfs 和 astar
echo ">>> Processing folder: single (bfs, astar)"
for file in maps/single/*.txt; do
    run_test "$file" "bfs"
    run_test "$file" "astar"
done

# 2. corner 資料夾：測試 astar_corner
echo ">>> Processing folder: corner (astar_corner)"
for file in maps/corner/*.txt; do
    run_test "$file" "astar_corner"
done

# 3. multi 資料夾 (1 pt 組)：測試 astar_multi
echo ">>> Processing folder: multi (astar_multi)"
MULTI_1PT=(
    "maps/multi/tinySearch.txt"
    "maps/multi/smallSearch.txt"
    "maps/multi/greedySearch.txt"
    "maps/multi/mediumDottedMaze.txt"
    "maps/multi/mediumSearch.txt"
)
for file in "${MULTI_1PT[@]}"; do
    if [ -f "$file" ]; then
        run_test "$file" "astar_multi"
    fi
done

#4. multi 資料夾 (2 pt 組)：測試 fast
echo ">>> Processing folder: multi (fast)"
MULTI_2PT=(
    "maps/multi/mediumSearch_prev.txt"
    "maps/multi/bigSearch.txt"
    "maps/multi/oddSearch.txt"
    "maps/multi/openSearch.txt"
)
for file in "${MULTI_2PT[@]}"; do
    if [ -f "$file" ]; then
        run_test "$file" "fast"
    fi
done

echo "測試完成！詳細結果請查看 $LOG_FILE"