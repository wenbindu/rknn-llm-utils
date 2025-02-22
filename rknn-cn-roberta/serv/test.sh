sudo systemctl kill sentiment-api && journalctl -u sentiment-api -f

# 发送超过1000字符的请求
curl -X POST "http://192.168.0.19:8080/predict" \
-H "Content-Type: application/json" \
-d '{"text":"特斯拉皮卡，辅助泊车撞墙了！"}'

# 生成大日志
for i in {1..100}; do
  curl -X POST "http://192.168.0.19:8080/predict" \
  -H "Content-Type: application/json" \
  -d '{"text":"交警队出盖章文件了！hw智驾连环车祸+行车记录仪内存卡被偷最新动态！ '$i'"}'
done
