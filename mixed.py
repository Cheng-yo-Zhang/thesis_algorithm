"""
Solomon VRPTW Instance Mixer for UAV + MCS Cooperative Charging Service
========================================================================

研究場景：
- 地面 MCS (Mobile Charging Station): 分快充/慢充
  - Urgent: 需要快充 (緊時間窗，從 R101 取)
  - Normal: 適合慢充/過夜 (寬時間窗，從 R201 取)
- UAV: 快充且機動，適合 Hard-to-Access (地面難到達的節點)

混合邏輯：
- 以 R201 為 base (整體 instance)
- Urgent 節點：替換為 R101 的時間窗
- Hard-to-Access：根據距離/位置特徵判定
- Normal：保持 R201 原始資料

Author: Operations Research Team
"""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import pandas as pd


# =============================================================================
# Enums & Configuration
# =============================================================================

class NodeType(str, Enum):
    """節點類型分類"""
    DEPOT = "depot"
    NORMAL = "normal"                   # 一般節點：慢充/過夜，使用 R201 時間窗
    URGENT = "urgent"                   # 緊急節點：快充，使用 R101 時間窗
    HARD_TO_ACCESS = "hard_to_access"   # 難達節點：UAV 服務
    
    def __str__(self) -> str:
        return self.value


@dataclass
class MixerConfig:
    """混合器配置
    
    Attributes:
        urgent_ratio: Urgent 節點比例 (從 R101 取時間窗)
        hard_to_access_ratio: Hard-to-Access 節點比例
        normal_ratio: Normal 節點比例 (自動計算為 1 - urgent - hard_to_access)
        
        hard_to_access_method: Hard-to-Access 判定方法
            - 'distance': 距離 depot 最遠的節點
            - 'peripheral': 在邊緣區域的節點
            - 'manual': 手動指定 customer IDs
        
        hard_to_access_ids: 若 method='manual'，手動指定的 customer IDs
        
        random_seed: 隨機種子 (可重現性)
    """
    urgent_ratio: float = 0.2
    hard_to_access_ratio: float = 0.1
    
    hard_to_access_method: Literal['distance', 'peripheral', 'manual'] = 'distance'
    hard_to_access_ids: list[int] | None = None
    
    random_seed: int = 42
    
    def __post_init__(self):
        self.normal_ratio = 1.0 - self.urgent_ratio - self.hard_to_access_ratio
        
        if self.normal_ratio < 0:
            raise ValueError(
                f"Ratios exceed 1.0: urgent={self.urgent_ratio}, "
                f"hard_to_access={self.hard_to_access_ratio}"
            )
        
        if self.hard_to_access_method == 'manual' and not self.hard_to_access_ids:
            raise ValueError("hard_to_access_ids required when method='manual'")


# =============================================================================
# Main Mixer Class
# =============================================================================

class SolomonInstanceMixer:
    """Solomon VRPTW Instance 混合器
    
    以 R201 (寬時間窗) 為 base，對指定節點替換 R101 (緊時間窗) 的屬性。
    
    Example:
        >>> config = MixerConfig(urgent_ratio=0.3, hard_to_access_ratio=0.2)
        >>> mixer = SolomonInstanceMixer(config)
        >>> result_df = mixer.mix("R101.csv", "R201.csv", "mixed_output.csv")
    """
    
    REQUIRED_COLUMNS = [
        "CUST NO.", "XCOORD.", "YCOORD.", "DEMAND",
        "READY TIME", "DUE DATE", "SERVICE TIME"
    ]
    
    def __init__(self, config: MixerConfig | None = None):
        self.config = config or MixerConfig()
        self.rng = np.random.default_rng(self.config.random_seed)
        
    def load_csv(self, filepath: str | Path) -> pd.DataFrame:
        """載入並驗證 Solomon CSV"""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        df = pd.read_csv(filepath)
        
        missing = set(self.REQUIRED_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        
        return df
    
    def _calculate_distances_from_depot(self, df: pd.DataFrame) -> pd.Series:
        """計算每個節點到 depot 的歐式距離"""
        depot = df.iloc[0]
        distances = np.sqrt(
            (df["XCOORD."] - depot["XCOORD."]) ** 2 +
            (df["YCOORD."] - depot["YCOORD."]) ** 2
        )
        return distances
    
    def _calculate_peripheral_score(self, df: pd.DataFrame) -> pd.Series:
        """計算邊緣程度分數 (距離整體中心 + 距離 depot)"""
        # 所有節點的重心
        center_x = df["XCOORD."].mean()
        center_y = df["YCOORD."].mean()
        
        dist_to_center = np.sqrt(
            (df["XCOORD."] - center_x) ** 2 +
            (df["YCOORD."] - center_y) ** 2
        )
        
        dist_to_depot = self._calculate_distances_from_depot(df)
        
        # 綜合分數：邊緣 + 遠離 depot
        return dist_to_center + dist_to_depot
    
    def _select_hard_to_access(
        self, 
        df: pd.DataFrame, 
        count: int,
        exclude_ids: set[int]
    ) -> list[int]:
        """選擇 Hard-to-Access 節點
        
        Args:
            df: 資料框
            count: 要選擇的數量
            exclude_ids: 已被選為其他類型的 IDs
        
        Returns:
            被選中的 customer IDs
        """
        method = self.config.hard_to_access_method
        
        if method == 'manual':
            # 手動指定
            return [
                cid for cid in self.config.hard_to_access_ids 
                if cid not in exclude_ids
            ][:count]
        
        # 排除 depot (index 0) 和已選的節點
        customers = df.iloc[1:].copy()
        customers = customers[~customers["CUST NO."].isin(exclude_ids)]
        
        if method == 'distance':
            # 距離 depot 最遠
            customers["_score"] = self._calculate_distances_from_depot(customers)
        else:  # peripheral
            # 邊緣區域
            customers["_score"] = self._calculate_peripheral_score(customers)
        
        # 選擇分數最高的
        customers = customers.sort_values("_score", ascending=False)
        selected = customers.head(count)["CUST NO."].tolist()
        
        return selected
    
    def _select_urgent(
        self,
        df_r101: pd.DataFrame,
        count: int,
        exclude_ids: set[int]
    ) -> list[int]:
        """選擇 Urgent 節點 (隨機抽取)
        
        從可用節點中隨機選擇，然後套用 R101 的時間窗
        """
        customers = df_r101.iloc[1:].copy()
        customers = customers[~customers["CUST NO."].isin(exclude_ids)]
        
        # 隨機抽取
        available_ids = customers["CUST NO."].tolist()
        if len(available_ids) <= count:
            return available_ids
        
        selected = self.rng.choice(available_ids, size=count, replace=False).tolist()
        return selected
    
    def mix(
        self,
        r101_path: str | Path,
        r201_path: str | Path,
        output_path: str | Path | None = None
    ) -> pd.DataFrame:
        """混合兩個 instance
        
        Args:
            r101_path: R101 檔案路徑 (緊時間窗來源)
            r201_path: R201 檔案路徑 (base instance)
            output_path: 輸出路徑 (可選)
        
        Returns:
            混合後的 DataFrame
        """
        # 載入資料
        df_r101 = self.load_csv(r101_path)
        df_r201 = self.load_csv(r201_path)
        
        # 驗證 customer IDs 對齊
        if not df_r101["CUST NO."].equals(df_r201["CUST NO."]):
            raise ValueError("Customer IDs do not match between R101 and R201")
        
        # 計算各類型節點數量 (排除 depot)
        n_customers = len(df_r201) - 1
        n_hard = int(round(n_customers * self.config.hard_to_access_ratio))
        n_urgent = int(round(n_customers * self.config.urgent_ratio))
        
        print(f"Total customers: {n_customers}")
        print(f"Target distribution: Hard={n_hard}, Urgent={n_urgent}, "
              f"Normal={n_customers - n_hard - n_urgent}")
        
        # 以 R201 為 base
        result = df_r201.copy()
        
        # 初始化 NODE_TYPE 欄位
        result["NODE_TYPE"] = NodeType.NORMAL.value
        result.loc[0, "NODE_TYPE"] = NodeType.DEPOT.value
        
        selected_ids: set[int] = set()
        
        # Step 1: 選擇 Hard-to-Access (基於距離/位置)
        hard_ids = self._select_hard_to_access(df_r201, n_hard, selected_ids)
        selected_ids.update(hard_ids)
        
        # Step 2: 選擇 Urgent (基於 R101 時間窗特徵)
        urgent_ids = self._select_urgent(df_r101, n_urgent, selected_ids)
        selected_ids.update(urgent_ids)
        
        # Step 3: 設定節點類型並替換時間窗
        for idx, row in result.iterrows():
            cust_id = row["CUST NO."]
            
            if cust_id == 1:  # Depot (CUST NO. = 1 in Solomon format)
                continue
            
            if cust_id in hard_ids:
                result.loc[idx, "NODE_TYPE"] = NodeType.HARD_TO_ACCESS.value
                # Hard-to-Access: 保持 R201 時間窗 (寬鬆，UAV 有彈性)
                # 但可考慮調整，例如加寬時間窗
                
            elif cust_id in urgent_ids:
                result.loc[idx, "NODE_TYPE"] = NodeType.URGENT.value
                # Urgent: 使用 R101 的緊時間窗
                r101_row = df_r101[df_r101["CUST NO."] == cust_id].iloc[0]
                result.loc[idx, "READY TIME"] = r101_row["READY TIME"]
                result.loc[idx, "DUE DATE"] = r101_row["DUE DATE"]
            
            # Normal: 保持 R201 原始資料 (已是預設)
        
        # 統計結果
        type_counts = result["NODE_TYPE"].value_counts()
        print(f"\nActual distribution:")
        for nt in NodeType:
            count = type_counts.get(nt.value, 0)
            print(f"  {nt.value}: {count}")
        
        # 儲存
        if output_path:
            output_path = Path(output_path)
            result.to_csv(output_path, index=False)
            print(f"\nSaved to: {output_path}")
            
            # 儲存 metadata
            self._save_metadata(
                output_path, 
                r101_path, r201_path,
                hard_ids, urgent_ids, result
            )
        
        return result
    
    def _save_metadata(
        self,
        output_path: Path,
        r101_path: str | Path,
        r201_path: str | Path,
        hard_ids: list[int],
        urgent_ids: list[int],
        result: pd.DataFrame
    ):
        """儲存 metadata JSON"""
        # 計算時間窗統計
        customers = result[result["NODE_TYPE"] != NodeType.DEPOT.value]
        tw_width = customers["DUE DATE"] - customers["READY TIME"]
        
        metadata = {
            "generated_at": datetime.now().isoformat(),
            "random_seed": self.config.random_seed,
            "sources": {
                "base_instance": str(r201_path),
                "tight_tw_source": str(r101_path)
            },
            "config": {
                "urgent_ratio": self.config.urgent_ratio,
                "hard_to_access_ratio": self.config.hard_to_access_ratio,
                "normal_ratio": self.config.normal_ratio,
                "hard_to_access_method": self.config.hard_to_access_method
            },
            "node_assignments": {
                "hard_to_access_ids": hard_ids,
                "urgent_ids": urgent_ids
            },
            "statistics": {
                "total_nodes": len(result),
                "by_type": result["NODE_TYPE"].value_counts().to_dict(),
                "total_demand": int(customers["DEMAND"].sum()),
                "tw_width": {
                    "mean": float(tw_width.mean()),
                    "min": int(tw_width.min()),
                    "max": int(tw_width.max())
                }
            },
            "data_hash": hashlib.md5(
                result.to_csv(index=False).encode()
            ).hexdigest()[:8]
        }
        
        meta_path = output_path.with_suffix(".meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f"Metadata saved to: {meta_path}")


# =============================================================================
# Convenience Function
# =============================================================================

def create_mixed_instance(
    r101_path: str = "R101.csv",
    r201_path: str = "R201.csv",
    output_path: str = "mixed_instance.csv",
    urgent_ratio: float = 0.2,
    hard_to_access_ratio: float = 0.1,
    hard_to_access_method: str = 'distance',
    hard_to_access_ids: list[int] | None = None,
    random_seed: int = 42
) -> pd.DataFrame:
    """便利函數：快速產生混合 instance
    
    Args:
        r101_path: R101 檔案 (緊時間窗)
        r201_path: R201 檔案 (base, 寬時間窗)
        output_path: 輸出檔案路徑
        urgent_ratio: Urgent 節點比例 (default: 0.3)
        hard_to_access_ratio: Hard-to-Access 節點比例 (default: 0.2)
        hard_to_access_method: 'distance' | 'peripheral' | 'manual'
        hard_to_access_ids: 若 method='manual'，指定的 customer IDs
        random_seed: 隨機種子
    
    Returns:
        混合後的 DataFrame
    
    Example:
        >>> df = create_mixed_instance(
        ...     urgent_ratio=0.3,
        ...     hard_to_access_ratio=0.2,
        ...     hard_to_access_method='distance'
        ... )
    """
    config = MixerConfig(
        urgent_ratio=urgent_ratio,
        hard_to_access_ratio=hard_to_access_ratio,
        hard_to_access_method=hard_to_access_method,
        hard_to_access_ids=hard_to_access_ids,
        random_seed=random_seed
    )
    
    mixer = SolomonInstanceMixer(config)
    return mixer.mix(r101_path, r201_path, output_path)


def analyze_instance(filepath: str | Path) -> dict:
    """分析 Solomon instance 統計資訊"""
    df = pd.read_csv(filepath)
    customers = df.iloc[1:]  # 排除 depot
    
    tw_width = customers["DUE DATE"] - customers["READY TIME"]
    
    depot = df.iloc[0]
    distances = np.sqrt(
        (customers["XCOORD."] - depot["XCOORD."]) ** 2 +
        (customers["YCOORD."] - depot["YCOORD."]) ** 2
    )
    
    return {
        "file": str(filepath),
        "n_customers": len(customers),
        "planning_horizon": int(df.iloc[0]["DUE DATE"]),
        "demand": {
            "total": int(customers["DEMAND"].sum()),
            "mean": float(customers["DEMAND"].mean())
        },
        "time_window": {
            "mean_width": float(tw_width.mean()),
            "min_width": int(tw_width.min()),
            "max_width": int(tw_width.max())
        },
        "distance_from_depot": {
            "mean": float(distances.mean()),
            "max": float(distances.max())
        }
    }


# =============================================================================
# CLI
# =============================================================================

def main():
    """Command-line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Mix Solomon VRPTW instances for UAV+MCS charging service"
    )
    parser.add_argument(
        "--r101", default="R101.csv",
        help="R101 file path (tight TW source)"
    )
    parser.add_argument(
        "--r201", default="R201.csv", 
        help="R201 file path (base instance)"
    )
    parser.add_argument(
        "-o", "--output", default="mixed_instance.csv",
        help="Output file path"
    )
    parser.add_argument(
        "--urgent-ratio", type=float, default=0.2,
        help="Ratio of urgent nodes (default: 0.2)"
    )
    parser.add_argument(
        "--hard-ratio", type=float, default=0.1,
        help="Ratio of hard-to-access nodes (default: 0.1)"
    )
    parser.add_argument(
        "--hard-method", choices=['distance', 'peripheral', 'manual'],
        default='distance',
        help="Method for selecting hard-to-access nodes"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--analyze", action="store_true",
        help="Analyze source instances before mixing"
    )
    
    args = parser.parse_args()
    
    if args.analyze:
        print("=" * 60)
        print("Source Instance Analysis")
        print("=" * 60)
        for path in [args.r101, args.r201]:
            stats = analyze_instance(path)
            print(f"\n{stats['file']}:")
            print(f"  Customers: {stats['n_customers']}")
            print(f"  Horizon: {stats['planning_horizon']}")
            print(f"  TW width: {stats['time_window']['mean_width']:.1f} "
                  f"[{stats['time_window']['min_width']}, {stats['time_window']['max_width']}]")
            print(f"  Avg distance from depot: {stats['distance_from_depot']['mean']:.2f}")
    
    print("\n" + "=" * 60)
    print("Creating Mixed Instance")
    print("=" * 60 + "\n")
    
    create_mixed_instance(
        r101_path=args.r101,
        r201_path=args.r201,
        output_path=args.output,
        urgent_ratio=args.urgent_ratio,
        hard_to_access_ratio=args.hard_ratio,
        hard_to_access_method=args.hard_method,
        random_seed=args.seed
    )


if __name__ == "__main__":
    main()
