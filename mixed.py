"""
Solomon VRPTW Instance Mixer for MCS Charging Service (CLI Version)
====================================================================

研究場景：
- 地面 MCS (Mobile Charging Station): 分快充/慢充
  - Urgent: 需要快充 (緊時間窗，從 R101 取)
  - Normal: 適合慢充/過夜 (寬時間窗，從 R201 取)

CLI 使用說明：
    python mixed.py --points 25 --seed 42 --urgent-ratio 0.3

Author: Operations Research Team
"""

from __future__ import annotations

import json
import hashlib
import argparse
import sys
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path

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
    
    def __str__(self) -> str:
        return self.value


@dataclass
class MixerConfig:
    """混合器配置
    
    Attributes:
        n_points: 選擇的節點數量 (必須是5的倍數，不超過100)
        urgent_ratio: Urgent 節點比例
        normal_ratio: Normal 節點比例 (自動計算為 1 - urgent)
        random_seed: 隨機種子 (可重現性)
    """
    n_points: int = 25
    urgent_ratio: float = 0.2
    random_seed: int = 42
    
    def __post_init__(self):
        # 驗證節點數量
        if self.n_points % 5 != 0:
            raise ValueError(f"節點數量必須是5的倍數，目前: {self.n_points}")
        if self.n_points < 5 or self.n_points > 100:
            raise ValueError(f"節點數量必須在 5 到 100 之間，目前: {self.n_points}")
        
        # 驗證比例
        if not 0 <= self.urgent_ratio <= 1:
            raise ValueError(f"Urgent 比例必須在 0 到 1 之間，目前: {self.urgent_ratio}")
        
        self.normal_ratio = 1.0 - self.urgent_ratio


# =============================================================================
# Main Mixer Class
# =============================================================================

class SolomonInstanceMixer:
    """Solomon VRPTW Instance 混合器
    
    以 R201 (寬時間窗) 為 base，對指定節點替換 R101 (緊時間窗) 的屬性。
    
    Example:
        >>> config = MixerConfig(n_points=25, urgent_ratio=0.3, random_seed=42)
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
    
    def _select_random_customers(
        self,
        df: pd.DataFrame,
        count: int
    ) -> list[int]:
        """隨機選擇 customer IDs"""
        customers = df.iloc[1:]  # 排除 depot
        available_ids = customers["CUST NO."].tolist()
        
        if len(available_ids) < count:
            raise ValueError(
                f"需要 {count} 個節點，但只有 {len(available_ids)} 個可用"
            )
        
        selected = self.rng.choice(available_ids, size=count, replace=False).tolist()
        return selected
    
    def _select_urgent(
        self,
        count: int,
        available_ids: list[int]
    ) -> list[int]:
        """從可用節點中隨機選擇 Urgent 節點"""
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
        
        # 隨機選擇節點
        selected_customer_ids = self._select_random_customers(
            df_r201, self.config.n_points
        )
        
        # 計算 Urgent 節點數量
        n_urgent = int(round(self.config.n_points * self.config.urgent_ratio))
        
        print(f"設定: {self.config.n_points} 點, 隨機種子: {self.config.random_seed}")
        print(f"目標分佈: Urgent={n_urgent} ({self.config.urgent_ratio:.0%}), "
              f"Normal={self.config.n_points - n_urgent} ({self.config.normal_ratio:.0%})")
        
        # 選擇 Urgent 節點
        urgent_ids = self._select_urgent(n_urgent, selected_customer_ids)
        
        # 建立結果 DataFrame (depot + 選擇的節點)
        depot_row = df_r201.iloc[[0]].copy()
        selected_rows = df_r201[df_r201["CUST NO."].isin(selected_customer_ids)].copy()
        result = pd.concat([depot_row, selected_rows], ignore_index=True)
        
        # 初始化 NODE_TYPE 欄位
        result["NODE_TYPE"] = NodeType.NORMAL.value
        result.loc[0, "NODE_TYPE"] = NodeType.DEPOT.value
        
        # 設定節點類型並替換時間窗
        for idx, row in result.iterrows():
            cust_id = row["CUST NO."]
            
            if idx == 0:  # Depot
                continue
            
            if cust_id in urgent_ids:
                result.loc[idx, "NODE_TYPE"] = NodeType.URGENT.value
                # Urgent: 使用 R101 的緊時間窗
                r101_row = df_r101[df_r101["CUST NO."] == cust_id].iloc[0]
                result.loc[idx, "READY TIME"] = r101_row["READY TIME"]
                result.loc[idx, "DUE DATE"] = r101_row["DUE DATE"]
            
            # Normal: 保持 R201 原始資料 (已是預設)
        
        # 重新編號 CUST NO. (1, 2, 3, ...)
        result["ORIGINAL_CUST_NO"] = result["CUST NO."].copy()
        result["CUST NO."] = range(1, len(result) + 1)
        
        # 統計結果
        type_counts = result["NODE_TYPE"].value_counts()
        print(f"\n實際分佈:")
        for nt in NodeType:
            count = type_counts.get(nt.value, 0)
            if count > 0:
                print(f"  {nt.value}: {count}")
        
        # 儲存
        if output_path:
            output_path = Path(output_path)
            result.to_csv(output_path, index=False)
            print(f"\n已儲存至: {output_path}")
            
            # 儲存 metadata
            self._save_metadata(
                output_path, 
                r101_path, r201_path,
                urgent_ids, selected_customer_ids, result
            )
        
        return result
    
    def _save_metadata(
        self,
        output_path: Path,
        r101_path: str | Path,
        r201_path: str | Path,
        urgent_ids: list[int],
        selected_ids: list[int],
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
                "n_points": self.config.n_points,
                "urgent_ratio": self.config.urgent_ratio,
                "normal_ratio": self.config.normal_ratio
            },
            "node_assignments": {
                "selected_customer_ids": selected_ids,
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
        print(f"Metadata 已儲存至: {meta_path}")


# =============================================================================
# Convenience Function
# =============================================================================

def create_mixed_instance(
    r101_path: str = "R101.csv",
    r201_path: str = "R201.csv",
    output_path: str = "mixed_instance.csv",
    n_points: int = 25,
    urgent_ratio: float = 0.2,
    random_seed: int = 42
) -> pd.DataFrame:
    """便利函數：快速產生混合 instance
    
    Args:
        r101_path: R101 檔案 (緊時間窗)
        r201_path: R201 檔案 (base, 寬時間窗)
        output_path: 輸出檔案路徑
        n_points: 節點數量 (必須是5的倍數，不超過100)
        urgent_ratio: Urgent 節點比例
        random_seed: 隨機種子
    
    Returns:
        混合後的 DataFrame
    
    Example:
        >>> df = create_mixed_instance(
        ...     n_points=25,
        ...     urgent_ratio=0.3,
        ...     random_seed=42
        ... )
    """
    config = MixerConfig(
        n_points=n_points,
        urgent_ratio=urgent_ratio,
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

def validate_points(value: str) -> int:
    """驗證節點數量參數"""
    try:
        ivalue = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{value}' 不是有效的整數")
    
    if ivalue % 5 != 0:
        raise argparse.ArgumentTypeError(
            f"節點數量必須是5的倍數 (例如: 5, 10, 15, ..., 100)，目前: {ivalue}"
        )
    if ivalue < 5 or ivalue > 100:
        raise argparse.ArgumentTypeError(
            f"節點數量必須在 5 到 100 之間，目前: {ivalue}"
        )
    return ivalue


def validate_ratio(value: str) -> float:
    """驗證比例參數"""
    try:
        fvalue = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{value}' 不是有效的數字")
    
    if not 0 <= fvalue <= 1:
        raise argparse.ArgumentTypeError(
            f"比例必須在 0 到 1 之間，目前: {fvalue}"
        )
    return fvalue


def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(
        description="Mixed Solomon VRPTW Instance Generator for MCS Charging Service",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  # 產生 25 個節點，urgent 佔 30%，隨機種子 42
  python mixed.py --points 25 --urgent-ratio 0.3 --seed 42

  # 產生 50 個節點，urgent 佔 20%，隨機種子 123
  python mixed.py --points 50 --urgent-ratio 0.2 --seed 123

  # 分析來源檔案後再混合
  python mixed.py --points 30 --urgent-ratio 0.4 --seed 42 --analyze
        """
    )
    
    parser.add_argument(
        "--points", "-p", "--n_points", "-n",
        type=validate_points,
        required=True,
        help="節點數量 (必須是5的倍數，範圍: 5-100)"
    )
    parser.add_argument(
        "--urgent-ratio", "-u",
        type=validate_ratio,
        required=True,
        help="Urgent 節點比例 (0-1 之間，例如: 0.3 表示 30%%)"
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        required=True,
        help="隨機種子 (用於可重現性)"
    )
    parser.add_argument(
        "--r101",
        default="R101.csv",
        help="R101 檔案路徑 (緊時間窗來源，預設: R101.csv)"
    )
    parser.add_argument(
        "--r201",
        default="R201.csv", 
        help="R201 檔案路徑 (base instance，預設: R201.csv)"
    )
    parser.add_argument(
        "-o", "--output",
        default="mixed_instance.csv",
        help="輸出檔案路徑 (預設: mixed_instance.csv)"
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="混合前分析來源 instance"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Solomon VRPTW Instance Mixer (Urgent + Normal)")
    print("=" * 60)
    
    if args.analyze:
        print("\n--- 來源 Instance 分析 ---\n")
        for path in [args.r101, args.r201]:
            try:
                stats = analyze_instance(path)
                print(f"{stats['file']}:")
                print(f"  Customers: {stats['n_customers']}")
                print(f"  Horizon: {stats['planning_horizon']}")
                print(f"  TW width: {stats['time_window']['mean_width']:.1f} "
                      f"[{stats['time_window']['min_width']}, {stats['time_window']['max_width']}]")
                print(f"  Avg distance from depot: {stats['distance_from_depot']['mean']:.2f}")
                print()
            except FileNotFoundError:
                print(f"  檔案不存在: {path}")
    
    print("\n--- 產生混合 Instance ---\n")
    
    try:
        create_mixed_instance(
            r101_path=args.r101,
            r201_path=args.r201,
            output_path=args.output,
            n_points=args.points,
            urgent_ratio=args.urgent_ratio,
            random_seed=args.seed
        )
        print("\n" + "=" * 60)
        print("完成！")
        print("=" * 60)
    except Exception as e:
        print(f"\n錯誤: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
