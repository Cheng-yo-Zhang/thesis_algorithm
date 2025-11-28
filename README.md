
## Prerequisites
* [Python 3.11+](https://www.python.org/downloads/)
* [Git](https://git-scm.com/)


## Installation & Setup
```bash=
git clone git@github.com:Cheng-yo-Zhang/Bot.git
```

## Change Directory
```bash=
cd Bot
```
## Set up a virtual environment

### Microsoft Windows

#### 在沒有anaconda環境底下
```bash=
python -version -m venv venv
```

#### 在有anaconda環境底下
```bash=
py -m venv venv
```

#### Enable virtual environment
```bash=
venv\Scripts\activate
```
#### upgrade pip.
```bash=
python -m pip install --upgrade pip
```

#### Installation kit
```bash=
pip install -r requirements.txt
```

#### Disable virtual environment
```bash=
deactivate
```

#### Remove virtual environment
```bash=
rmdir venv
```

### macOS
```bash=
python -version -m venv venv
```

#### 啟用虛擬環境
```bash=
source venv/bin/activate
```

#### 先升級pip
```bash=
python -m pip install --upgrade pip
```

#### 安裝套件
```bash=
pip install -r requirements.txt
```

#### 停用虛擬環境
```bash=
deactivate
```

#### 移除虛擬環境
```bash=
rmdir venv
python -m src.generator
```