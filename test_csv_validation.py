"""
CSV 輸入驗證測試
測試 ChargingSchedulingProblem.load_data() 方法的驗證功能
"""

import os
import sys
import tempfile
import pytest

# 將 main.py 所在目錄加入路徑
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import ChargingSchedulingProblem


class TestCSVValidation:
    """CSV 輸入驗證測試"""
    
    def setup_method(self):
        """每個測試前重新建立 problem 實例"""
        self.problem = ChargingSchedulingProblem()
    
    def _create_temp_csv(self, content: str) -> str:
        """建立臨時 CSV 檔案並回傳路徑"""
        fd, path = tempfile.mkstemp(suffix='.csv')
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(content)
        return path
    
    def test_valid_csv_loads_successfully(self):
        """測試有效的 CSV 能正常載入"""
        csv_content = """CUST NO.,XCOORD.,YCOORD.,DEMAND,READY TIME,DUE DATE,SERVICE TIME,NODE_TYPE
1,35,35,0,0,1000,0,depot
2,20,50,5,331,410,10,normal
3,10,43,9,404,481,10,urgent"""
        path = self._create_temp_csv(csv_content)
        try:
            self.problem.load_data(path)
            assert len(self.problem.nodes) == 3
            assert self.problem.depot is not None
        finally:
            os.unlink(path)
    
    def test_missing_required_column_raises_error(self):
        """測試缺少必要欄位時拋出錯誤"""
        # 缺少 DEMAND 欄位
        csv_content = """CUST NO.,XCOORD.,YCOORD.,READY TIME,DUE DATE,SERVICE TIME,NODE_TYPE
1,35,35,0,1000,0,depot
2,20,50,331,410,10,normal"""
        path = self._create_temp_csv(csv_content)
        try:
            with pytest.raises(ValueError) as excinfo:
                self.problem.load_data(path)
            assert "DEMAND" in str(excinfo.value)
            assert "缺少必要欄位" in str(excinfo.value)
        finally:
            os.unlink(path)
    
    def test_nan_in_numeric_field_raises_error(self):
        """測試數值欄位有 NaN 時拋出錯誤"""
        csv_content = """CUST NO.,XCOORD.,YCOORD.,DEMAND,READY TIME,DUE DATE,SERVICE TIME,NODE_TYPE
1,35,35,0,0,1000,0,depot
2,20,,5,331,410,10,normal
3,10,43,9,404,481,10,urgent"""
        path = self._create_temp_csv(csv_content)
        try:
            with pytest.raises(ValueError) as excinfo:
                self.problem.load_data(path)
            assert "NaN" in str(excinfo.value)
            assert "YCOORD." in str(excinfo.value)
        finally:
            os.unlink(path)
    
    def test_invalid_node_type_raises_error(self):
        """測試無效的 NODE_TYPE 值時拋出錯誤"""
        csv_content = """CUST NO.,XCOORD.,YCOORD.,DEMAND,READY TIME,DUE DATE,SERVICE TIME,NODE_TYPE
1,35,35,0,0,1000,0,depot
2,20,50,5,331,410,10,invalid_type
3,10,43,9,404,481,10,urgent"""
        path = self._create_temp_csv(csv_content)
        try:
            with pytest.raises(ValueError) as excinfo:
                self.problem.load_data(path)
            assert "invalid_type" in str(excinfo.value)
            assert "無效的 NODE_TYPE" in str(excinfo.value)
        finally:
            os.unlink(path)
    
    def test_nan_in_node_type_raises_error(self):
        """測試 NODE_TYPE 欄位有 NaN 時拋出錯誤"""
        csv_content = """CUST NO.,XCOORD.,YCOORD.,DEMAND,READY TIME,DUE DATE,SERVICE TIME,NODE_TYPE
1,35,35,0,0,1000,0,depot
2,20,50,5,331,410,10,
3,10,43,9,404,481,10,urgent"""
        path = self._create_temp_csv(csv_content)
        try:
            with pytest.raises(ValueError) as excinfo:
                self.problem.load_data(path)
            assert "NODE_TYPE" in str(excinfo.value)
        finally:
            os.unlink(path)
    
    def test_csv_without_node_type_column_uses_default(self):
        """測試沒有 NODE_TYPE 欄位時使用預設值 'normal'"""
        csv_content = """CUST NO.,XCOORD.,YCOORD.,DEMAND,READY TIME,DUE DATE,SERVICE TIME
1,35,35,0,0,1000,0
2,20,50,5,331,410,10
3,10,43,9,404,481,10"""
        path = self._create_temp_csv(csv_content)
        try:
            self.problem.load_data(path)
            assert len(self.problem.nodes) == 3
            # 第一個節點應該是 depot
            assert self.problem.nodes[0].node_type == 'depot'
            # 其餘節點應該是 normal
            assert self.problem.nodes[1].node_type == 'normal'
            assert self.problem.nodes[2].node_type == 'normal'
        finally:
            os.unlink(path)
    
    def test_multiple_nan_values_reported(self):
        """測試多個 NaN 值都會被報告"""
        csv_content = """CUST NO.,XCOORD.,YCOORD.,DEMAND,READY TIME,DUE DATE,SERVICE TIME,NODE_TYPE
1,35,35,0,0,1000,0,depot
2,20,,5,,410,10,normal
3,10,43,,404,481,10,urgent"""
        path = self._create_temp_csv(csv_content)
        try:
            with pytest.raises(ValueError) as excinfo:
                self.problem.load_data(path)
            error_msg = str(excinfo.value)
            # 應該報告 YCOORD., READY TIME, DEMAND 的 NaN
            assert "YCOORD." in error_msg
            assert "READY TIME" in error_msg
            assert "DEMAND" in error_msg
        finally:
            os.unlink(path)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
