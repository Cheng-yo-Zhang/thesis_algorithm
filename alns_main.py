import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import List

# =====================================================================
# 1. OR 系統實體定義 (MCS 與 任務)
# =====================================================================
@dataclass
class MCS:
    """移動充電站 (Mobile Charging Station) 的物理實體"""
    mcs_id: int
    current_i: int  # 網格座標 i
    current_j: int  # 網格座標 j
    current_battery: float = 270.0  # 滿電 270 kWh
    available_time: int = 0         # 系統時間 (秒)，代表這台車何時可以接下一個任務
    
    def __repr__(self):
        return f"MCS-{self.mcs_id:02d} [Loc: ({self.current_i:02d}, {self.current_j:02d})]"

# =====================================================================
# 2. 預測大腦神經網路結構 (必須與 Colab 完全一致)
# =====================================================================
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size):
        super(ConvLSTMCell, self).__init__()
        self.hidden_dim = hidden_dim
        padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.conv = nn.Conv2d(in_channels=input_dim + hidden_dim,
                              out_channels=4 * hidden_dim,
                              kernel_size=kernel_size, padding=padding, bias=True)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)  
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i, f, o, g = torch.sigmoid(cc_i), torch.sigmoid(cc_f), torch.sigmoid(cc_o), torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

class ChargingDemandPredictor(nn.Module):
    def __init__(self):
        super(ChargingDemandPredictor, self).__init__()
        self.convlstm = ConvLSTMCell(input_dim=1, hidden_dim=16, kernel_size=(3,3))
        self.decoder = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1)
        
    def forward(self, x):
        b, seq_len, c, h, w = x.size()
        h_t = torch.zeros(b, 16, h, w).to(x.device)
        c_t = torch.zeros(b, 16, h, w).to(x.device)
        for t in range(seq_len):
            h_t, c_t = self.convlstm(x[:, t, :, :, :], (h_t, c_t))
        return torch.relu(self.decoder(h_t))

# =====================================================================
# 3. 預測與部署核心邏輯
# =====================================================================
class PredictiveDispatcher:
    def __init__(self, model_weights_path: str):
        print("🧠 1. 正在初始化本地預測大腦...")
        self.model = ChargingDemandPredictor()
        # 強制使用 CPU 載入 (因為排程演算不需要 GPU)
        self.model.load_state_dict(torch.load(model_weights_path, map_location=torch.device('cpu')))
        self.model.eval() # 切換到推論模式
        print("✅ 大腦權重載入成功！")

    def predict_and_deploy(self, recent_history_npy_path: str, num_mcs: int = 30) -> List[MCS]:
        """
        輸入過去的歷史軌跡矩陣，回傳 30 台已經部署好座標的 MCS 實體。
        """
        print(f"📊 2. 讀取測試集資料 ({recent_history_npy_path})，準備模擬 T=0 時刻...")
        # 讀取測試集 (X_test)
        X_test = np.load(recent_history_npy_path)
        
        # 擷取 T=0 當下的第一筆資料: 形狀原本是 (seq_len, 1, 100, 100)
        # 我們需要增加 Batch 維度變成 (1, seq_len, 1, 100, 100) 讓 PyTorch 讀取
        t0_history = X_test[0] 
        t0_tensor = torch.FloatTensor(np.expand_dims(t0_history, axis=0))
        
        print("⚡ 3. 執行 ConvLSTM 瞬間推論 (Inference)...")
        with torch.no_grad():
            prediction_map = self.model(t0_tensor)
            
        # 將預測結果攤平並轉為 Numpy
        pred_matrix = prediction_map.squeeze().numpy()
        
        # 找出預測需求量最大的前 num_mcs (30) 個網格索引
        # np.argsort 由小到大排，[-num_mcs:] 取最後 30 個，[::-1] 反轉由大到小
        flat_indices = np.argsort(pred_matrix.flatten())[-num_mcs:][::-1]
        
        # 將一維索引轉回 (i, j) 座標
        top_coords = [divmod(idx, 100) for idx in flat_indices]
        
        print(f"🎯 4. 預測完成！生成 {num_mcs} 個熱點座標。正在建立 MCS 車隊...")
        fleet = []
        for idx, (i, j) in enumerate(top_coords):
            mcs_unit = MCS(mcs_id=idx+1, current_i=int(i), current_j=int(j))
            fleet.append(mcs_unit)
            
        return fleet

# =====================================================================
# 4. 主程式執行區
# =====================================================================
if __name__ == "__main__":
    # 請確保 best_convlstm_nyc.pth 與 X_test.npy 都在同一個資料夾下
    dispatcher = PredictiveDispatcher(model_weights_path="best_convlstm_nyc.pth")
    
    # 執行預測部署
    mcs_fleet = dispatcher.predict_and_deploy(recent_history_npy_path="X_test.npy", num_mcs=30)
    
    # 印出部署結果
    print("\n🚀 === [Phase 3: 預先部署完成] MCS 車隊初始狀態 ===")
    for mcs in mcs_fleet[:10]: # 只印前 10 台檢查
        print(mcs)
    print("... (共 30 台)")