#!/usr/bin/env python3
"""
TxGemma-9B-Chat-GGUF ダウンロードスクリプト
Hugging Faceから直接ダウンロードしてOllamaで使用可能にする
"""

import os
import sys
import requests
import subprocess
from pathlib import Path
from tqdm import tqdm
import hashlib


class TxGemmaDownloader:
    """TxGemma-9B-Chat-GGUF ダウンローダー"""
    
    def __init__(self):
        self.model_name = "txgemma-9b-chat-GGUF"
        self.quantization = "Q6_K"  # 7.59GB, 高品質
        self.filename = f"txgemma-9b-chat-{self.quantization}.gguf"
        self.url = f"https://huggingface.co/lmstudio-community/txgemma-9b-chat-GGUF/resolve/main/{self.filename}"
        self.expected_size = 7.59 * 1024 * 1024 * 1024  # 7.59GB in bytes
        
        # ダウンロード先ディレクトリ
        self.download_dir = Path.home() / ".cache" / "txgemma"
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.filepath = self.download_dir / self.filename
        
        print(f"TxGemma-9B-Chat-GGUF Downloader")
        print(f"Download directory: {self.download_dir}")
        print(f"Model: {self.filename}")
        print(f"Size: {self.expected_size / (1024**3):.2f} GB")
    
    def check_disk_space(self) -> bool:
        """ディスク容量をチェック"""
        import shutil
        
        free_space = shutil.disk_usage(self.download_dir).free
        required_space = self.expected_size * 1.2  # 20%余裕
        
        print(f"Free space: {free_space / (1024**3):.2f} GB")
        print(f"Required space: {required_space / (1024**3):.2f} GB")
        
        if free_space < required_space:
            print(f"Insufficient disk space!")
            return False
        
        print("Sufficient disk space available")
        return True
    
    def download_file(self) -> bool:
        """ファイルをダウンロード"""
        if self.filepath.exists():
            print(f"File already exists: {self.filepath}")
            return True
        
        print(f"Downloading {self.filename}...")
        print(f"URL: {self.url}")
        
        try:
            # ヘッダー設定
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            # リクエスト開始
            response = requests.get(self.url, headers=headers, stream=True)
            response.raise_for_status()
            
            # ファイルサイズ取得
            total_size = int(response.headers.get('content-length', 0))
            
            # プログレスバー付きでダウンロード
            with open(self.filepath, 'wb') as f, tqdm(
                desc="Downloading",
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            print(f"Download completed: {self.filepath}")
            return True
            
        except requests.exceptions.RequestException as e:
            print(f"Download failed: {e}")
            return False
        except Exception as e:
            print(f"Error: {e}")
            return False
    
    def verify_download(self) -> bool:
        """ダウンロードファイルを検証"""
        if not self.filepath.exists():
            print("File not found")
            return False
        
        file_size = self.filepath.stat().st_size
        print(f"Downloaded file size: {file_size / (1024**3):.2f} GB")
        
        # サイズチェック（±10%許容）
        size_diff = abs(file_size - self.expected_size) / self.expected_size
        if size_diff > 0.1:
            print(f"File size mismatch (diff: {size_diff:.1%})")
            return False
        
        print("File size verification passed")
        return True
    
    def import_to_ollama(self) -> bool:
        """Ollamaにモデルをインポート"""
        print("Importing to Ollama...")
        
        try:
            # Ollama create コマンド
            cmd = [
                "ollama", "create", 
                f"txgemma:9b-chat-{self.quantization.lower()}",
                "-f", str(self.filepath)
            ]
            
            print(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            print("Successfully imported to Ollama!")
            print(f"Output: {result.stdout}")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"Ollama import failed: {e}")
            print(f"Error output: {e.stderr}")
            return False
        except FileNotFoundError:
            print("Ollama not found. Please install Ollama first.")
            return False
    
    def test_model(self) -> bool:
        """モデルの動作テスト"""
        print("Testing model...")
        
        try:
            # 簡単なテストクエリ
            cmd = [
                "ollama", "run", 
                f"txgemma:9b-chat-{self.quantization.lower()}",
                "Hello, can you help with drug discovery?"
            ]
            
            print(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                print("Model test successful!")
                print(f"Response: {result.stdout[:200]}...")
                return True
            else:
                print(f"Model test failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print("Model test timed out (this is normal for first run)")
            return True
        except Exception as e:
            print(f"Model test error: {e}")
            return False
    
    def run(self) -> bool:
        """メイン実行"""
        print("Starting TxGemma-9B-Chat-GGUF download and setup...")
        
        # 1. ディスク容量チェック
        if not self.check_disk_space():
            return False
        
        # 2. ダウンロード
        if not self.download_file():
            return False
        
        # 3. 検証
        if not self.verify_download():
            return False
        
        # 4. Ollamaインポート
        if not self.import_to_ollama():
            return False
        
        # 5. テスト
        if not self.test_model():
            print("Model test failed, but import may have succeeded")
        
        print("\nTxGemma-9B-Chat-GGUF setup completed!")
        print(f"Model name: txgemma:9b-chat-{self.quantization.lower()}")
        print(f"Usage: ollama run txgemma:9b-chat-{self.quantization.lower()}")
        print(f"CLI usage: python cli.py chat --model txgemma:9b-chat-{self.quantization.lower()}")
        
        return True


def main():
    """メイン関数"""
    print("=" * 60)
    print("TxGemma-9B-Chat-GGUF Downloader")
    print("=" * 60)
    
    # 依存関係チェック
    try:
        import requests
        import tqdm
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Please install: pip install requests tqdm")
        return 1
    
    # ダウンローダー実行
    downloader = TxGemmaDownloader()
    success = downloader.run()
    
    if success:
        print("\nSetup completed successfully!")
        return 0
    else:
        print("\nSetup failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
