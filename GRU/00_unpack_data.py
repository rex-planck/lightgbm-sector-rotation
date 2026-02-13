import os
import tarfile
import shutil
from pathlib import Path
from tqdm import tqdm

# --- 配置 ---
# 压缩包所在位置 (刚才你手动放进去的)
SOURCE_FILE = Path(r"E:\Quant_program\Qlib-Cache\qlib_bin.tar.gz")
# 解压目标位置
DEST_DIR = Path(r"E:\Quant_program\Qlib-Cache\cn_data")


def extract_and_fix():
    print(f"📦 正在检查数据包: {SOURCE_FILE}")

    if not SOURCE_FILE.exists():
        print(f"❌ 错误：找不到文件！\n请确保你已经手动下载并把它放在了: {SOURCE_FILE}")
        return

    # 1. 准备目录
    if DEST_DIR.exists():
        print("⚠️ 目标目录已存在，正在清理旧数据以防止冲突...")
        try:
            shutil.rmtree(DEST_DIR)
        except Exception as e:
            print(f"⚠️ 清理失败 (可能文件被占用): {e}")
            # 如果清理失败，尝试直接解压覆盖

    DEST_DIR.mkdir(parents=True, exist_ok=True)

    # 2. 解压
    print("🚀 开始解压 (这可能需要 1-2 分钟)...")
    try:
        with tarfile.open(SOURCE_FILE, "r:gz") as tar:
            members = tar.getmembers()
            for member in tqdm(members, desc="Extracting"):
                # 为了防止解压出 qlib_bin/xxx 这种嵌套结构，我们做个路径重映射
                # 无论包里怎么套娃，我们都把文件平铺到 DEST_DIR
                tar.extract(member, path=DEST_DIR)

        print("✅ 解压完成！正在检查目录结构...")

        # 3. 智能修正目录结构 (Flatten)
        # 社区数据包解压后通常会在外面套一层 "qlib_bin" 文件夹
        # 我们需要把它里面的内容（calendars, features, instruments）提到最外层
        nested_folder = DEST_DIR / "qlib_bin"

        if nested_folder.exists():
            print("🔧 检测到嵌套文件夹，正在提升目录层级...")
            # 移动所有内容到上一级
            for item in nested_folder.iterdir():
                shutil.move(str(item), str(DEST_DIR))
            # 删除空的 qlib_bin
            nested_folder.rmdir()
            print("✅ 目录结构修正完毕！")

        # 4. 最终验证
        if (DEST_DIR / "calendars").exists() and (DEST_DIR / "features").exists():
            print("-" * 50)
            print(f"🎉 恭喜！数据已成功部署到: {DEST_DIR}")
            print("✅ 你的数据环境现在是 100% 完美的。")
        else:
            print("⚠️ 警告：解压后的文件结构似乎不对，请检查文件夹内容。")

    except Exception as e:
        print(f"❌ 解压过程中出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    extract_and_fix()