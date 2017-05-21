# tensorflow test  
Sparse Autoencoder used Kullback–Leibler divergence  
  
誤差関数に正則化項としてカルバック・ライブラーダイバージェンスを加えました。  
  
Autoencoderの実装について以下のサイトを参考にしました。  
<http://qiita.com/mokemokechicken/items/8216aaad36709b6f0b5c>

# Execution result
誤差の推移  
![loss graph](./images/weight2)

最終的な重み  
![a weight of encoder](./images/weight1)

![a weight of encoder](./images/weight2)

７を復元した結果  
![inpout image](./images/input7)
![output image](./images/output7)
