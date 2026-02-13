import qlib
from qlib.data import D
import os

# 🔥 关键修改：指向你的 E 盘数据路径
provider_uri = r"E:\Quant_program\Qlib-Cache\cn_data"

# 检查一下路径真的存在吗
if not os.path.exists(provider_uri):
    print(f"❌ 严重错误：路径不存在 -> {provider_uri}")
    print("请先执行上面的解压脚本！")
else:
    try:
        qlib.init(provider_uri=provider_uri, region="cn")
        print(f"✅ Qlib 初始化成功！数据源: {provider_uri}")

        # 拉取茅台数据
        print("📊 读取 贵州茅台(SH600519) 测试...")
        df = D.features(['SH600519'], ['$close', '$volume'], start_time='2020-01-01', end_time='2020-01-05')
        print(df)
        print("\n🎉🎉🎉 全链路跑通！你现在是一名拥有本地数据仓库的量化工程师了！")

    except Exception as e:
        print(f"❌ 运行报错: {e}")